#!/usr/bin/env python3
"""Stage 1 — data ingestion & cleaning (deterministic).

Discovers the ``Malabar_Dataset/<class>(count)/*.jpg`` tree, derives a **proxy
``source_domain``** from image metadata (resolution / aspect / JPEG
quantization), runs a **perceptual-hash duplicate / near-duplicate audit**,
then produces a **leakage-free stratified split** by (class x domain) plus a
held-out **external_domain_test** (a real proxy domain when one separates
cleanly, else a synthetic device-shift holdout).

Emits the four state artifacts the graph threads downstream:
``class_domain_counts``, ``dedup_report``, ``cleaning_log``, ``split_manifest``.

Self-contained pHash (numpy + scipy DCT) is used when ``imagehash`` is absent
(it is, in this env); ``imagehash`` is preferred automatically when installed.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from augmentation_pipeline import IMAGE_EXTENSIONS
from common import RunPaths, cfg_get, get_logger, profile_value, save_json, strip_class_count

try:  # prefer the real library when present; fall back to self-contained pHash
    import imagehash as _imagehash

    _HAS_IMAGEHASH = True
except Exception:  # pragma: no cover
    _HAS_IMAGEHASH = False

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class ImageRecord:
    path: str
    label: str
    width: int
    height: int
    fmt: str
    quant_mean: float
    domain: str = "unknown"
    label_index: int = -1
    group_id: int = -1

    @property
    def sort_key(self) -> str:
        return self.path

    def manifest_entry(self, synthetic: bool) -> Dict[str, Any]:
        return {
            "path": self.path,
            "label": self.label,
            "label_index": self.label_index,
            "domain": self.domain,
            "group_id": self.group_id,
            "synthetic_corruption": synthetic,
        }


# ---------------------------------------------------------------------------
# Perceptual hash (pHash) — self-contained
# ---------------------------------------------------------------------------
def compute_phash_bits(gray_square: np.ndarray, hash_size: int) -> np.ndarray:
    """DCT-based perceptual hash → flat boolean array of length hash_size**2."""
    from scipy.fft import dct

    coeffs = dct(dct(gray_square, axis=0, norm=None), axis=1, norm=None)
    low = coeffs[:hash_size, :hash_size]
    med = float(np.median(low))
    return (low > med).flatten()


def read_meta_and_gray(path: Path, phash_input: int) -> Tuple[Dict[str, Any], np.ndarray]:
    """Single decode → (metadata, grayscale square for pHash)."""
    with Image.open(path) as im:
        fmt = im.format or path.suffix.lstrip(".").upper()
        width, height = im.size
        quant = getattr(im, "quantization", None)
        if quant and 0 in quant:
            quant_mean = float(np.mean(quant[0]))
        else:
            quant_mean = -1.0  # non-JPEG / no tables
        gray = im.convert("L").resize((phash_input, phash_input), Image.BILINEAR)
    return (
        {"width": width, "height": height, "fmt": fmt, "quant_mean": quant_mean},
        np.asarray(gray, dtype=np.float32),
    )


def hamming_pairs(packed: np.ndarray, threshold: int) -> List[Tuple[int, int, int]]:
    """All (i, j, distance) with Hamming distance <= threshold. Vectorized per row."""
    pairs: List[Tuple[int, int, int]] = []
    n = packed.shape[0]
    for i in range(n - 1):
        xor = np.bitwise_xor(packed[i + 1 :], packed[i])
        dist = _POPCOUNT[xor].sum(axis=1)
        hits = np.nonzero(dist <= threshold)[0]
        for off in hits:
            pairs.append((i, i + 1 + int(off), int(dist[off])))
    return pairs


# ---------------------------------------------------------------------------
# Union-find for near-duplicate grouping
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# ---------------------------------------------------------------------------
# Proxy-domain derivation
# ---------------------------------------------------------------------------
def _resolution_bucket(width: int, height: int, thresholds: List[int]) -> str:
    max_dim = max(width, height)
    labels = ["S", "M", "L", "XL"]
    for i, thr in enumerate(thresholds):
        if max_dim <= thr:
            return labels[i]
    return labels[len(thresholds)] if len(thresholds) < len(labels) else labels[-1]


def _aspect_bucket(width: int, height: int) -> str:
    ratio = width / max(1, height)
    if ratio >= 1.15:
        return "wide"
    if ratio <= 0.87:
        return "tall"
    return "square"


def _quant_bucket(quant_mean: float) -> str:
    if quant_mean < 0:
        return "qNA"
    if quant_mean <= 3.0:
        return "qA"
    if quant_mean <= 10.0:
        return "qB"
    return "qC"


def derive_domain(meta: Dict[str, Any], res_thresholds: List[int]) -> str:
    return "-".join(
        (
            _resolution_bucket(meta["width"], meta["height"], res_thresholds),
            _aspect_bucket(meta["width"], meta["height"]),
            _quant_bucket(meta["quant_mean"]),
        )
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def discover_images(
    root: Path,
    canonical_classes: List[str],
    cap: Optional[int],
    logger,
) -> Tuple[List[ImageRecord], List[str], List[str], List[str]]:
    """Return (records, active_classes, declared_unpopulated, corrupt). Corrupt files skipped."""
    present: Dict[str, List[Path]] = {}
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        label = strip_class_count(class_dir.name)
        files = sorted(
            (p for p in class_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTENSIONS),
            key=lambda p: p.name.lower(),
        )
        if files:
            present.setdefault(label, []).extend(files)

    active = [c for c in canonical_classes if c in present]
    # any on-disk class not in the canonical list is still trained, appended stably
    for label in sorted(present):
        if label not in active:
            active.append(label)
    declared_unpopulated = [c for c in canonical_classes if c not in present]

    records: List[ImageRecord] = []
    corrupt: List[str] = []
    for label in active:
        files = present[label]
        if cap is not None:
            files = files[:cap]
        for path in files:
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception as exc:  # noqa: BLE001
                corrupt.append(f"{path.name}: {exc}")
                continue
            records.append(ImageRecord(path=str(path), label=label, width=0, height=0, fmt="", quant_mean=-1.0))
    logger.info("Discovered %d images across %d active classes (%d corrupt skipped)",
                len(records), len(active), len(corrupt))
    return records, active, declared_unpopulated, corrupt


# ---------------------------------------------------------------------------
# Split assembly
# ---------------------------------------------------------------------------
@dataclass
class Group:
    gid: int
    label: str
    domain: str
    members: List[int] = field(default_factory=list)  # record indices

    @property
    def size(self) -> int:
        return len(self.members)


def _stable_order(keys: List[str]) -> List[int]:
    """Deterministic pseudo-random order via md5 of a stable key."""
    return sorted(range(len(keys)), key=lambda i: hashlib.md5(keys[i].encode()).hexdigest())


def stratified_split_groups(
    groups: List[Group],
    records: List[ImageRecord],
    train_frac: float,
    val_frac: float,
) -> Dict[int, str]:
    """Assign each group (atomic) to train/val/test, stratified by (class, domain)."""
    strata: Dict[Tuple[str, str], List[Group]] = defaultdict(list)
    for g in groups:
        strata[(g.label, g.domain)].append(g)

    assignment: Dict[int, str] = {}
    for _key, gs in strata.items():
        keys = [records[g.members[0]].sort_key for g in gs]
        order = _stable_order(keys)
        total = sum(gs[i].size for i in order)
        t_target = train_frac * total
        v_target = val_frac * total
        acc = 0
        for i in order:
            if acc < t_target:
                split = "train"
            elif acc < t_target + v_target:
                split = "val"
            else:
                split = "test"
            assignment[gs[i].gid] = split
            acc += gs[i].size
    return assignment


def select_external(
    groups: List[Group],
    active_classes: List[str],
    mode: str,
    holdout_fraction: float,
) -> Tuple[str, Optional[str], List[int], List[int]]:
    """Pick external_domain_test groups. Returns (mode_used, domain, ext_gids, rest_gids)."""
    counts: Dict[str, int] = defaultdict(int)
    cover: Dict[str, set] = defaultdict(set)
    for g in groups:
        counts[g.domain] += g.size
        cover[g.domain].add(g.label)
    total = sum(counts.values()) or 1
    min_classes = min(2, len(active_classes))

    if mode in ("auto", "real"):
        candidates = [
            d for d in counts
            if len(cover[d]) >= min_classes and 0.05 <= counts[d] / total <= 0.40
        ]
        if candidates:
            dstar = min(candidates, key=lambda d: abs(counts[d] / total - holdout_fraction))
            ext = [g.gid for g in groups if g.domain == dstar]
            rest = [g.gid for g in groups if g.domain != dstar]
            return "real_domain_holdout", dstar, ext, rest
        if mode == "real":
            dstar = max(counts, key=lambda d: (len(cover[d]), counts[d]))
            ext = [g.gid for g in groups if g.domain == dstar]
            rest = [g.gid for g in groups if g.domain != dstar]
            return "real_domain_holdout", dstar, ext, rest

    # synthetic: whole groups, stratified by class, ~holdout_fraction of images
    by_class: Dict[str, List[Group]] = defaultdict(list)
    for g in groups:
        by_class[g.label].append(g)
    ext_gids: List[int] = []
    rest_gids: List[int] = []
    for _label, gs in by_class.items():
        order = _stable_order([str(g.gid) for g in gs])
        cls_total = sum(g.size for g in gs)
        target = holdout_fraction * cls_total
        acc = 0
        for i in order:
            if acc < target:
                ext_gids.append(gs[i].gid)
                acc += gs[i].size
            else:
                rest_gids.append(gs[i].gid)
    return "synthetic_corruption", None, ext_gids, rest_gids


# ---------------------------------------------------------------------------
# Main stage entry
# ---------------------------------------------------------------------------
def run_data_stage(
    config: Dict[str, Any],
    run_paths: RunPaths,
    profile: str,
    logger=None,
) -> Dict[str, Any]:
    logger = logger or get_logger("stage1.data", run_paths.stage("stage1_data") / "data.log")
    data_cfg = config["data"]
    root = Path(data_cfg["root"])
    if not root.is_absolute():
        from common import PROJECT_ROOT

        root = PROJECT_ROOT / root
    canonical = list(data_cfg["canonical_classes"])
    cap = profile_value(data_cfg.get("max_images_per_class"), profile)
    hash_size = int(cfg_get(config, "data.dedup.hash_size", 16))
    near_dup = int(cfg_get(config, "data.dedup.near_dup_hamming", 12))
    res_thresholds = list(cfg_get(config, "data.domain.proxy.resolution_buckets", [512, 1024, 2048]))
    min_domain_fraction = float(cfg_get(config, "data.domain.proxy.min_domain_fraction", 0.05))
    ext_mode = cfg_get(config, "data.domain.external.mode", "auto")
    holdout_fraction = float(cfg_get(config, "data.domain.external.holdout_fraction", 0.15))
    corruptions = list(cfg_get(config, "data.domain.external.corruptions", []))
    splits_cfg = data_cfg["splits"]

    cleaning_log: List[str] = []

    # 1) discover -----------------------------------------------------------
    records, active_classes, declared_unpopulated, corrupt = discover_images(root, canonical, cap, logger)
    if not records:
        raise RuntimeError(f"No images found under {root}")
    if corrupt:
        cleaning_log.append(f"Skipped {len(corrupt)} corrupt/unreadable image(s).")
    if cap is not None:
        cleaning_log.append(f"Applied fast-profile cap of {cap} images/class before hashing.")
    if declared_unpopulated:
        cleaning_log.append(
            f"{len(declared_unpopulated)} of {len(canonical)} canonical classes have 0 images "
            f"(declared but unpopulated): {', '.join(declared_unpopulated)}."
        )
    class_to_index = {c: i for i, c in enumerate(active_classes)}
    for r in records:
        r.label_index = class_to_index[r.label]

    # 2) metadata + pHash ---------------------------------------------------
    phash_input = hash_size * 4
    bits_list: List[np.ndarray] = []
    for r in records:
        meta, gray = read_meta_and_gray(Path(r.path), phash_input)
        r.width, r.height, r.fmt, r.quant_mean = meta["width"], meta["height"], meta["fmt"], meta["quant_mean"]
        if _HAS_IMAGEHASH:
            with Image.open(r.path) as im:
                h = _imagehash.phash(im, hash_size=hash_size)
            bits_list.append(h.hash.flatten())
        else:
            bits_list.append(compute_phash_bits(gray, hash_size))
    bits = np.stack(bits_list).astype(np.uint8)
    packed = np.packbits(bits, axis=1)
    logger.info("Hashed %d images (%s, hash_size=%d, %d bits)",
                len(records), "imagehash" if _HAS_IMAGEHASH else "self-contained pHash",
                hash_size, bits.shape[1])

    # 3) domains + rare merge ----------------------------------------------
    for r in records:
        r.domain = derive_domain(
            {"width": r.width, "height": r.height, "quant_mean": r.quant_mean}, res_thresholds
        )
    domain_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        domain_counts[r.domain] += 1
    total_imgs = len(records)
    merged_domains = [d for d, c in domain_counts.items() if c / total_imgs < min_domain_fraction]
    if merged_domains:
        for r in records:
            if r.domain in merged_domains:
                r.domain = "misc"
        cleaning_log.append(
            f"Merged {len(merged_domains)} rare proxy-domain(s) (<{min_domain_fraction:.0%} each) into 'misc'."
        )
    final_domains = sorted({r.domain for r in records})
    cleaning_log.append(f"Derived {len(final_domains)} proxy source-domain(s): {', '.join(final_domains)}.")

    # 4) dedup: exact-dup removal + near-dup grouping ----------------------
    pairs = hamming_pairs(packed, near_dup)
    exact_uf = UnionFind(len(records))
    near_uf = UnionFind(len(records))
    exact_pairs = 0
    for i, j, d in pairs:
        near_uf.union(i, j)
        if d == 0:
            exact_uf.union(i, j)
            exact_pairs += 1

    # remove all-but-first in each exact-duplicate component
    exact_components: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(records)):
        exact_components[exact_uf.find(idx)].append(idx)
    removed: List[int] = []
    label_conflicts = 0
    for _root_idx, members in exact_components.items():
        if len(members) <= 1:
            continue
        members_sorted = sorted(members, key=lambda k: records[k].sort_key)
        labels = {records[k].label for k in members_sorted}
        if len(labels) > 1:
            label_conflicts += 1
        removed.extend(members_sorted[1:])  # keep the first
    removed_set = set(removed)
    if removed:
        cleaning_log.append(
            f"Removed {len(removed)} exact-duplicate image(s) (pHash Hamming=0), keeping one per group."
        )
    if label_conflicts:
        cleaning_log.append(
            f"WARNING: {label_conflicts} exact-duplicate group(s) span >1 class (possible label noise)."
        )

    kept = [idx for idx in range(len(records)) if idx not in removed_set]

    # near-dup groups over kept set (atomic split units)
    group_of: Dict[int, int] = {}
    groups_map: Dict[int, Group] = {}
    next_gid = 0
    for idx in kept:
        root_idx = near_uf.find(idx)
        # root may be a removed exact-dup; remap to a stable representative among kept
        if root_idx not in group_of:
            group_of[root_idx] = next_gid
            groups_map[next_gid] = Group(gid=next_gid, label=records[idx].label, domain=records[idx].domain)
            next_gid += 1
        gid = group_of[root_idx]
        records[idx].group_id = gid
        groups_map[gid].members.append(idx)
    # set each group's (label, domain) to its majority member (stable)
    for g in groups_map.values():
        g.label = records[g.members[0]].label
        dom_counts: Dict[str, int] = defaultdict(int)
        for m in g.members:
            dom_counts[records[m].domain] += 1
        g.domain = max(sorted(dom_counts), key=lambda d: dom_counts[d])
    groups = list(groups_map.values())
    near_dup_groups = sum(1 for g in groups if g.size > 1)
    cleaning_log.append(
        f"pHash near-duplicate audit: {len(pairs)} pair(s) <= Hamming {near_dup}; "
        f"{near_dup_groups} multi-image near-dup group(s) kept intact within a single split."
    )

    # 5) external holdout + split ------------------------------------------
    mode_used, ext_domain, ext_gids, rest_gids = select_external(
        groups, active_classes, ext_mode, holdout_fraction
    )
    ext_gid_set = set(ext_gids)
    rest_groups = [g for g in groups if g.gid in set(rest_gids)]
    assignment = stratified_split_groups(
        rest_groups, records, splits_cfg["train"], splits_cfg["val"]
    )
    synthetic = mode_used == "synthetic_corruption"
    cleaning_log.append(
        f"External domain test via {mode_used}"
        + (f" (held-out proxy domain '{ext_domain}')." if ext_domain else
           f" ({', '.join(corruptions)} corruptions on {len(ext_gids)} held-out group(s)).")
    )

    # 6) assemble manifest --------------------------------------------------
    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": [], "external_domain_test": []}
    for g in groups:
        target = "external_domain_test" if g.gid in ext_gid_set else assignment.get(g.gid, "train")
        for m in g.members:
            splits[target].append(records[m].manifest_entry(synthetic and target == "external_domain_test"))
    split_counts = {k: len(v) for k, v in splits.items()}
    cleaning_log.append(
        "Split sizes — " + ", ".join(f"{k}={v}" for k, v in split_counts.items()) + "."
    )

    # 7) class_domain_counts (audited by Agent 1; includes 0-count canonicals)
    class_domain_counts: Dict[str, Dict[str, int]] = {c: {} for c in canonical}
    for c in active_classes:
        class_domain_counts.setdefault(c, {})
    for idx in kept:
        r = records[idx]
        class_domain_counts[r.label][r.domain] = class_domain_counts[r.label].get(r.domain, 0) + 1

    # 8) leakage verification (must be all-zero by construction) -----------
    gid_split: Dict[int, str] = {}
    for g in groups:
        gid_split[g.gid] = "external_domain_test" if g.gid in ext_gid_set else assignment.get(g.gid, "train")
    split_names = ["train", "val", "test", "external_domain_test"]
    leakage_matrix = {a: {b: 0 for b in split_names} for a in split_names}
    kept_set = set(kept)
    for i, j, d in pairs:
        if i not in kept_set or j not in kept_set:
            continue
        si = gid_split.get(records[i].group_id)
        sj = gid_split.get(records[j].group_id)
        if si is None or sj is None or si == sj:
            continue
        leakage_matrix[si][sj] += 1
        leakage_matrix[sj][si] += 1
    leakage_remaining = sum(leakage_matrix[a][b] for a in split_names for b in split_names) // 2

    dedup_report = {
        "hash": "imagehash.phash" if _HAS_IMAGEHASH else "self_contained_phash",
        "hash_size": hash_size,
        "hash_bits": int(bits.shape[1]),
        "near_dup_hamming_threshold": near_dup,
        "images_hashed": len(records),
        "near_dup_pairs_detected": len(pairs),
        "exact_duplicate_pairs": exact_pairs,
        "exact_duplicates_removed": len(removed),
        "label_conflict_groups": label_conflicts,
        "near_duplicate_groups": near_dup_groups,
        "cross_split_leakage_matrix": leakage_matrix,
        "leakage_pairs_remaining": leakage_remaining,
        "removed_files": [Path(records[k].path).name for k in removed[:50]],
    }

    split_manifest = {
        "classes": active_classes,
        "class_to_index": class_to_index,
        "declared_unpopulated": declared_unpopulated,
        "domains": final_domains,
        "external_mode": mode_used,
        "external_domain": ext_domain,
        "corruptions": corruptions if synthetic else [],
        "counts": split_counts,
        "working_size": int(cfg_get(config, "data.working_size", 640)),
        "image_size": int(cfg_get(config, "data.image_size", 224)),
        "splits": splits,
    }

    # persist stage artifacts
    stage_dir = run_paths.stage("stage1_data")
    save_json(class_domain_counts, stage_dir / "class_domain_counts.json")
    save_json(dedup_report, stage_dir / "dedup_report.json")
    save_json(split_manifest, stage_dir / "split_manifest.json")
    save_json(cleaning_log, stage_dir / "cleaning_log.json")

    logger.info("Stage 1 done. leakage_pairs_remaining=%d (expect 0); external=%s n=%d",
                leakage_remaining, mode_used, split_counts["external_domain_test"])
    if leakage_remaining != 0:
        logger.error("LEAKAGE DETECTED: %d cross-split near-dup pair(s) — this is a bug", leakage_remaining)

    return {
        "class_domain_counts": class_domain_counts,
        "dedup_report": dedup_report,
        "cleaning_log": cleaning_log,
        "split_manifest": split_manifest,
    }
