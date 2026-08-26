#!/usr/bin/env python3
"""Stage 6 — final report, dataset card, and pipeline diagram (Part B, node 6).

Deterministic. Renders three artifacts from the finished ``PipelineState``:

* ``final_report.md`` — config snapshot, CV / in-domain / external metric tables
  with bootstrap CIs, per-class P/R/F1, confusion matrices, **both agents' full
  findings including every failure**, and the limitations checklist as a table.
* ``dataset_card.md`` — HF-style card: active + declared-unpopulated classes,
  proxy/synthetic domains, split sizes, and the biases/limitations both agents
  surfaced.
* ``pipeline_diagram.png`` (+ ``pipeline_diagram.mmd``) — the workflow graph,
  purple = LLM agents, gray = deterministic, amber = the human-review gate.

Everything degrades gracefully: any missing state key renders as "n/a" so a
partial/standalone run still produces a readable report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from common import cfg_get, save_text

# ---------------------------------------------------------------------------
# small formatters
# ---------------------------------------------------------------------------
def _f(v: Optional[float], nd: int = 3) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _pct(v: Optional[float], nd: int = 1) -> str:
    return "n/a" if v is None else f"{float(v) * 100:.{nd}f}%"


def _ci(block: Optional[Dict[str, Any]], key: str) -> str:
    """Render a bootstrap CI entry as ``point [lo, hi]``."""
    if not block or key not in block:
        return "n/a"
    b = block[key]
    return f"{_f(b.get('point'))} [{_f(b.get('lo'))}, {_f(b.get('hi'))}]"


def _get(state: Dict[str, Any], key: str, default: Any = None) -> Any:
    val = state.get(key)
    return default if val is None else val


# ---------------------------------------------------------------------------
# metric renderers
# ---------------------------------------------------------------------------
def _per_class_table(metrics: Dict[str, Any]) -> str:
    per_class = (metrics or {}).get("per_class") or {}
    if not per_class:
        return "_no per-class metrics_\n"
    lines = ["| Class | Precision | Recall | F1 | Support |",
             "|---|---|---|---|---|"]
    for name, m in per_class.items():
        lines.append(f"| {name} | {_f(m.get('precision'))} | {_f(m.get('recall'))} | "
                     f"{_f(m.get('f1'))} | {int(m.get('support', 0))} |")
    return "\n".join(lines) + "\n"


def _headline_table(test: Dict[str, Any], ext: Dict[str, Any], cv: Dict[str, Any]) -> str:
    lines = ["| Metric | Cross-validation | In-domain test | External-domain test |",
             "|---|---|---|---|"]
    cv = cv or {}
    test = test or {}
    ext = ext or {}
    cv_acc = (f"{_f(cv.get('mean_accuracy'))} ± {_f(cv.get('std_accuracy'))}"
              if cv.get("mean_accuracy") is not None else "n/a")
    cv_f1 = (f"{_f(cv.get('mean_macro_f1'))} ± {_f(cv.get('std_macro_f1'))}"
             if cv.get("mean_macro_f1") is not None else "n/a")
    lines.append(f"| Accuracy | {cv_acc} | {_f(test.get('accuracy'))} | {_f(ext.get('accuracy'))} |")
    lines.append(f"| Macro-F1 | {cv_f1} | {_f(test.get('macro_f1'))} | {_f(ext.get('macro_f1'))} |")
    lines.append(f"| Weighted-F1 | n/a | {_f(test.get('weighted_f1'))} | {_f(ext.get('weighted_f1'))} |")
    lines.append(f"| N samples | {cv.get('n_folds', 'n/a')} folds | {test.get('n', 'n/a')} | {ext.get('n', 'n/a')} |")
    lines.append(f"| Accuracy 95% CI | — | {_ci(test.get('bootstrap'), 'accuracy')} | "
                 f"{_ci(ext.get('bootstrap'), 'accuracy')} |")
    lines.append(f"| Macro-F1 95% CI | — | {_ci(test.get('bootstrap'), 'macro_f1')} | "
                 f"{_ci(ext.get('bootstrap'), 'macro_f1')} |")
    return "\n".join(lines) + "\n"


def _confusion_md(labels: List[str], matrix: List[List[int]]) -> str:
    if not labels or not matrix:
        return "_not available_\n"
    header = "| true ↓ / pred → | " + " | ".join(labels) + " |"
    sep = "|" + "---|" * (len(labels) + 1)
    rows = [header, sep]
    for i, label in enumerate(labels):
        row = matrix[i] if i < len(matrix) else [0] * len(labels)
        rows.append(f"| **{label}** | " + " | ".join(str(int(x)) for x in row) + " |")
    return "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# agent renderers
# ---------------------------------------------------------------------------
def _agent1_section(audit: Dict[str, Any]) -> str:
    if not audit:
        return "_Agent 1 did not run (deterministic-only invocation)._\n"
    out = [f"**Risk level:** `{audit.get('risk_level', 'n/a')}` · "
           f"**Proceed:** `{audit.get('proceed')}`"]
    guard = audit.get("_guardrail") or {}
    if guard.get("proceed_forced_false"):
        out.append("\n> ⚠️ **Guardrail engaged:** a `blocking` finding forced `proceed=False` "
                   "regardless of the LLM's own decision.")
    out.append("\n**Recommendation:** " + str(audit.get("recommendation", "n/a")))
    out.append("\n\n| Severity | Finding | Evidence |\n|---|---|---|")
    for f in audit.get("findings", []):
        sev = f.get("severity", "?")
        badge = {"blocking": "🔴 blocking", "warning": "🟡 warning", "info": "🟢 info"}.get(sev, sev)
        out.append(f"| {badge} | {_clean(f.get('finding'))} | {_clean(f.get('evidence'))} |")
    return "\n".join(out) + "\n"


def _agent2_section(audit: Dict[str, Any]) -> str:
    if not audit:
        return "_Agent 2 did not run (deterministic-only invocation)._\n"
    det = audit.get("_deterministic") or {}
    out = [f"**Overall status:** `{audit.get('overall_status', 'n/a')}`"]
    if det.get("loopback_node"):
        out.append(f" · looped back to `{det['loopback_node']}` "
                   f"(retries remaining at decision: {det.get('retries_remaining', 'n/a')})")
    overrides = det.get("overrides") or []
    if overrides:
        ov = ", ".join(f"{o['gap']} (LLM={o['llm']}→{o['deterministic']})" for o in overrides)
        out.append(f"\n\n> 🛡️ **Deterministic reconciliation** overrode the LLM on: {ov}. "
                   "Gap statuses below are recomputed from the numbers; the LLM supplies the prose.")
    out.append("\n\n**Summary:** " + str(audit.get("summary", "n/a")))
    out.append("\n\n| # | Gap | Status | Evidence | Suggested fix |\n|---|---|---|---|---|")
    for i, g in enumerate(audit.get("checklist", []), start=1):
        status = g.get("status", "?")
        badge = {"pass": "✅ pass", "fail": "❌ fail"}.get(status, status)
        out.append(f"| {i} | `{g.get('gap_name')}` | {badge} | {_clean(g.get('evidence'))} | "
                   f"{_clean(g.get('suggested_fix'))} |")
    return "\n".join(out) + "\n"


def _clean(text: Any) -> str:
    """Make a string safe for a single Markdown table cell."""
    if text is None:
        return ""
    return str(text).replace("\n", " ").replace("|", "\\|").strip()


# ---------------------------------------------------------------------------
# final_report.md
# ---------------------------------------------------------------------------
def build_final_report(state: Dict[str, Any]) -> str:
    config = _get(state, "config", {})
    sm = _get(state, "split_manifest", {})
    dedup = _get(state, "dedup_report", {})
    test = _get(state, "test_metrics", {})
    ext = _get(state, "external_domain_metrics", {})
    cv = _get(state, "cv_metrics", {})
    cms = _get(state, "confusion_matrices", {})
    expl = _get(state, "explainability_report", {})
    exp = _get(state, "export_report", {})
    aug = _get(state, "augmentation_summary", {})

    classes = sm.get("classes", [])
    declared = sm.get("declared_unpopulated", [])
    counts = sm.get("counts", {})
    backbone = cfg_get(config, "model.backbone", "n/a")
    faith = (expl or {}).get("faithfulness", {})
    lat = (exp or {}).get("latency_ms", {})

    md: List[str] = []
    md.append(f"# Final Report — Malabar Plant-Disease Classifier\n")
    md.append(f"**Run ID:** `{state.get('run_id', 'n/a')}`  ·  "
              f"**Profile:** `{state.get('profile', 'n/a')}`  ·  "
              f"**Seed:** `{config.get('seed', 'n/a')}`\n")
    if state.get("halted"):
        md.append(f"\n> ⛔ **Run halted:** {state.get('halt_reason', 'unknown reason')}\n")

    # --- 1. configuration ---
    md.append("\n## 1. Configuration\n")
    md.append(f"- **Backbone:** `{backbone}` (pretrained={cfg_get(config, 'model.pretrained', True)}, "
              f"SimSiam={cfg_get(config, 'model.simsiam.enabled', False)})")
    md.append(f"- **Loss:** CE (w={cfg_get(config, 'loss.ce_weight', 1.0)}) + "
              f"SupCon (w={cfg_get(config, 'loss.supcon_weight', 0.5)}); "
              f"class-balancing = `{cfg_get(config, 'loss.class_balancing', 'n/a')}`")
    md.append(f"- **Image size:** {sm.get('image_size', cfg_get(config, 'data.image_size', 'n/a'))} "
              f"(working {sm.get('working_size', cfg_get(config, 'data.working_size', 'n/a'))})")
    md.append(f"- **Augmentation:** {aug.get('num_train_ops', 'n/a')} train ops; "
              f"domain-randomization {(aug.get('domain_randomization') or {}).get('group', [])}")
    md.append(f"- **Active classes ({len(classes)}):** {', '.join(classes) if classes else 'n/a'}")
    if declared:
        md.append(f"- **Declared but unpopulated ({len(declared)}):** {', '.join(declared)}")

    # --- 2. data & leakage ---
    md.append("\n## 2. Data & leakage check\n")
    md.append(f"- **Images hashed:** {dedup.get('images_hashed', 'n/a')} "
              f"({dedup.get('hash', 'phash')} @ {dedup.get('hash_bits', 'n/a')} bits)")
    md.append(f"- **Exact duplicates removed:** {dedup.get('exact_duplicates_removed', 'n/a')}")
    md.append(f"- **Near-duplicate pairs detected:** {dedup.get('near_dup_pairs_detected', 'n/a')} "
              f"(Hamming ≤ {dedup.get('near_dup_hamming_threshold', 'n/a')})")
    leak = dedup.get("leakage_pairs_remaining")
    leak_badge = "✅ 0 (no cross-split leakage)" if leak == 0 else f"🔴 {leak}"
    md.append(f"- **Cross-split leakage pairs remaining:** {leak_badge}")
    md.append(f"- **Split sizes:** train={counts.get('train', 'n/a')}, val={counts.get('val', 'n/a')}, "
              f"test={counts.get('test', 'n/a')}, external_domain_test={counts.get('external_domain_test', 'n/a')}")
    md.append(f"- **External-domain test:** mode=`{sm.get('external_mode', 'n/a')}`, "
              f"domain=`{sm.get('external_domain', 'n/a')}`")
    md.append(f"- **Proxy domains:** {', '.join(map(str, sm.get('domains', []))) or 'n/a'}")

    # --- 3. Agent 1 ---
    md.append("\n## 3. Agent 1 — Explainable Data Audit\n")
    md.append(_agent1_section(_get(state, "data_audit", {})))

    # --- 4. results ---
    md.append("\n## 4. Results\n")
    md.append("### 4.1 Headline metrics (with bootstrap 95% CIs)\n")
    md.append(_headline_table(test, ext, cv))
    # in-domain → external drop (recomputed from the two metric blocks for display)
    if test.get("accuracy") is not None and ext.get("accuracy") is not None:
        acc_drop = (test["accuracy"] - ext["accuracy"]) * 100.0
        f1_drop = (test.get("macro_f1", 0) - ext.get("macro_f1", 0)) * 100.0
        md.append(f"\n**In-domain → external drop:** accuracy {acc_drop:+.1f} pts, "
                  f"macro-F1 {f1_drop:+.1f} pts "
                  f"(threshold {cfg_get(config, 'thresholds.domain_gap_pts', 10.0)} pts).\n")

    md.append("\n### 4.2 Per-class metrics (in-domain test)\n")
    md.append(_per_class_table(test))
    md.append("\n### 4.3 Per-class metrics (external-domain test)\n")
    md.append(_per_class_table(ext))

    md.append("\n### 4.4 Confusion matrices\n")
    md.append("**In-domain test**\n")
    md.append(_confusion_md(cms.get("labels", classes), cms.get("in_domain_test", [])))
    md.append("\n**External-domain test**\n")
    md.append(_confusion_md(cms.get("labels", classes), cms.get("external_domain_test", [])))

    # --- 5. explainability ---
    md.append("\n## 5. Explainability (Grad-CAM++ + faithfulness)\n")
    if expl:
        md.append(f"- **Method:** `{expl.get('method', 'n/a')}`, {expl.get('samples_per_class', 'n/a')} sample(s)/class")
        md.append(f"- **Coverage:** {'✅ complete' if expl.get('coverage_complete') else '⚠️ incomplete'} "
                  f"({len(expl.get('classes_covered', []))}/{expl.get('num_active_classes', 'n/a')} active classes)"
                  + (f"; missing: {', '.join(expl.get('coverage_missing', []))}" if expl.get('coverage_missing') else ""))
        md.append(f"- **Faithfulness:** score={_f(faith.get('faithfulness_score'))} "
                  f"(insertion {_f(faith.get('insertion_auc_mean'))} − deletion {_f(faith.get('deletion_auc_mean'))}), "
                  f"over {faith.get('n_images', 'n/a')} images "
                  f"(min {cfg_get(config, 'thresholds.faithfulness_min', 0.05)})")
        overlays = expl.get("overlays", [])
        if overlays:
            md.append(f"- **Overlays ({len(overlays)}):**")
            for ov in overlays:
                rel = Path(ov.get("path", "")).name
                md.append(f"  - `{ov.get('class')}` (pred `{ov.get('pred')}`) → `explain/{rel}`")
    else:
        md.append("_explainability stage did not run_")

    # --- 6. deployment ---
    md.append("\n## 6. Deployment (ONNX export + CPU benchmark)\n")
    if exp:
        md.append(f"- **ONNX exported:** {'✅' if exp.get('onnx_exported') else '❌'} "
                  + (f"(checked={exp.get('onnx_checked')})" if exp.get("onnx_exported") else
                     f"— `{_clean(exp.get('onnx_export_error'))}`"))
        md.append(f"- **Runtime:** `{exp.get('runtime', 'n/a')}` @ {exp.get('cpu_threads', 'n/a')} thread(s), "
                  f"opset {exp.get('opset', 'n/a')}")
        md.append(f"- **Params:** {exp.get('num_params', 'n/a'):,}" if isinstance(exp.get("num_params"), int)
                  else f"- **Params:** {exp.get('num_params', 'n/a')}")
        md.append(f"- **Latency:** mean {_f(lat.get('mean'), 1)} ms, p95 {_f(lat.get('p95'), 1)} ms "
                  f"(budget {cfg_get(config, 'export.device_budget.latency_ms', 200.0)} ms)")
        md.append(f"- **Peak memory:** {_f(exp.get('peak_mem_mb'), 1)} MB "
                  f"(budget {cfg_get(config, 'export.device_budget.peak_mem_mb', 512.0)} MB)")
        wb = exp.get("within_budget", {})
        md.append(f"- **Within budget:** latency {'✅' if wb.get('latency') else '❌'}, "
                  f"memory {'✅' if wb.get('peak_mem') else '❌'}, "
                  f"overall {'✅' if wb.get('overall') else '❌'}")
    else:
        md.append("_export stage did not run_")

    # --- 7. Agent 2 ---
    md.append("\n## 7. Agent 2 — Limitations Coverage Audit\n")
    md.append(_agent2_section(_get(state, "limitations_audit", {})))

    # --- 8. orchestration trace ---
    md.append("\n## 8. Orchestration trace\n")
    hist = state.get("history", [])
    if hist:
        md.append("Ordered node visits and edge decisions:\n")
        for h in hist:
            kind = h.get("kind")
            mark = "▸" if kind == "node" else "→"
            detail = h.get("detail") or {}
            detail_str = ", ".join(f"{k}={v}" for k, v in detail.items()) if detail else ""
            md.append(f"- {mark} `{h.get('name')}`" + (f" — {detail_str}" if detail_str else ""))
    else:
        md.append("_no history recorded_")

    md.append(f"\n---\n_Pipeline diagram: `reports/pipeline_diagram.png`. "
              f"Engine, per-stage artifacts, and agent-call logs live under the run directory._\n")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# dataset_card.md
# ---------------------------------------------------------------------------
def build_dataset_card(state: Dict[str, Any]) -> str:
    config = _get(state, "config", {})
    sm = _get(state, "split_manifest", {})
    dedup = _get(state, "dedup_report", {})
    cdc = _get(state, "class_domain_counts", {})
    counts = sm.get("counts", {})
    classes = sm.get("classes", [])
    declared = sm.get("declared_unpopulated", [])
    canonical = cfg_get(config, "data.canonical_classes", [])

    md: List[str] = []
    md.append("# Dataset Card — Malabar Spinach (Basella alba) Leaf Disease\n")
    md.append("Auto-generated by the pipeline's stage-6 reporter. Numbers are copied from the "
              "run's data-stage bookkeeping and the two audit agents.\n")

    md.append("\n## Classes\n")
    md.append(f"**Canonical classes ({len(canonical)}):** fixed vocabulary, never renamed.\n")
    md.append("\n| Class | Status | Total images |\n|---|---|---|")
    for c in canonical:
        total = sum((cdc.get(c) or {}).values()) if cdc.get(c) else 0
        status = "active" if c in classes else "🚫 declared, unpopulated (0 images)"
        md.append(f"| {c} | {status} | {total} |")
    if declared:
        md.append(f"\n> The pipeline trains and evaluates on the **{len(classes)} active** classes. "
                  f"The **{len(declared)}** declared-but-unpopulated classes "
                  f"({', '.join(declared)}) are a documented coverage limitation — the model "
                  "cannot predict a class it never saw.")

    md.append("\n## Source domains (proxy)\n")
    md.append("No real capture-device metadata exists in the source folders, so a deterministic "
              "**proxy `source_domain`** is derived per image from resolution / aspect-ratio / "
              "JPEG-quantization signatures. This drives the (class × domain) stratified split and "
              "the held-out external-domain test.\n")
    md.append(f"- **Proxy domains:** {', '.join(map(str, sm.get('domains', []))) or 'n/a'}")
    md.append(f"- **External-domain test:** mode=`{sm.get('external_mode', 'n/a')}`, "
              f"domain=`{sm.get('external_domain', 'n/a')}`, n={counts.get('external_domain_test', 'n/a')}")
    if sm.get("external_mode") == "synthetic_corruption":
        md.append(f"- **Synthetic corruptions:** {', '.join(sm.get('corruptions', [])) or 'n/a'} "
                  "(applied to build a device-shift holdout when no proxy domain separates cleanly)")

    md.append("\n## Splits\n")
    md.append("Stratified by (class × proxy-domain), 70/15/15, with perceptual-hash duplicate groups "
              "kept intra-split so no near-duplicate crosses a boundary.\n")
    md.append("\n| Split | Images |\n|---|---|")
    for name in ("train", "val", "test", "external_domain_test"):
        md.append(f"| {name} | {counts.get(name, 'n/a')} |")

    md.append("\n### Per-class × domain counts\n")
    if cdc:
        domains = sm.get("domains", [])
        header = "| Class | " + " | ".join(map(str, domains)) + " | Total |"
        sep = "|" + "---|" * (len(domains) + 2)
        md.append(header)
        md.append(sep)
        for c in classes:
            row = cdc.get(c, {})
            cells = [str(row.get(d, 0)) for d in domains]
            md.append(f"| {c} | " + " | ".join(cells) + f" | {sum(row.values())} |")
    else:
        md.append("_not available_")

    md.append("\n## Data cleaning\n")
    md.append(f"- Perceptual hash: `{dedup.get('hash', 'phash')}` @ {dedup.get('hash_bits', 'n/a')} bits")
    md.append(f"- Exact duplicate pairs: {dedup.get('exact_duplicate_pairs', 'n/a')} "
              f"(removed {dedup.get('exact_duplicates_removed', 'n/a')})")
    md.append(f"- Near-duplicate pairs: {dedup.get('near_dup_pairs_detected', 'n/a')}")
    md.append(f"- **Cross-split leakage after cleaning:** {dedup.get('leakage_pairs_remaining', 'n/a')} "
              "(0 = leakage-free)")
    clog = state.get("cleaning_log", [])
    if clog:
        md.append("\n<details><summary>Cleaning log</summary>\n")
        for entry in clog:
            md.append(f"- {entry}")
        md.append("\n</details>")

    # --- biases & limitations from both agents ---
    md.append("\n## Known biases & limitations\n")
    da = _get(state, "data_audit", {})
    if da:
        md.append("**From Agent 1 (data audit):**\n")
        for f in da.get("findings", []):
            if f.get("severity") in ("warning", "blocking"):
                md.append(f"- ({f.get('severity')}) {f.get('finding')} — {f.get('evidence')}")
    la = _get(state, "limitations_audit", {})
    if la:
        md.append(f"\n**From Agent 2 (limitations audit)** — overall `{la.get('overall_status', 'n/a')}`:\n")
        for g in la.get("checklist", []):
            if g.get("status") == "fail":
                md.append(f"- ❌ `{g.get('gap_name')}`: {g.get('evidence')} — fix: {g.get('suggested_fix')}")
        if all(g.get("status") == "pass" for g in la.get("checklist", [])) and la.get("checklist"):
            md.append("- ✅ All five limitation gaps passed.")
    if not da and not la:
        md.append("_agents did not run; limitations not audited_")

    md.append("\n## Intended use & caveats\n")
    md.append("- **Intended use:** research/decision-support for Malabar spinach leaf-disease "
              "screening. Not a substitute for expert diagnosis.")
    md.append("- **Out of scope:** species other than *Basella alba*; the declared-unpopulated "
              "classes; deployment domains far from the training proxy domains (see the "
              "domain-generalization gap in the final report).")
    md.append(f"\n---\n_Generated for run `{state.get('run_id', 'n/a')}`._\n")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# pipeline diagram (matplotlib PNG + mermaid source)
# ---------------------------------------------------------------------------
def render_diagram(png_path: Path) -> bool:
    """Draw the workflow as a PNG (purple=agents, gray=deterministic, amber=gate).

    Returns True on success; False if matplotlib is unavailable (the .mmd source
    is still written by the caller so the graph is never lost).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except Exception:
        return False

    AGENT = "#8e6fd1"
    DET = "#d9d9d9"
    GATE = "#f2c14e"

    # (label, y, color, text_color)
    nodes = [
        ("1 · Data ingestion & cleaning", 11, DET, "black"),
        ("2 · Agent 1 · Data Audit", 9.5, AGENT, "white"),
        ("await_human_review (interrupt)", 8.2, GATE, "black"),
        ("3 · Preprocessing & augmentation", 6.8, DET, "black"),
        ("4 · Train · Evaluate · Explain", 5.3, DET, "black"),
        ("4b · Export ONNX + CPU benchmark", 3.9, DET, "black"),
        ("5 · Agent 2 · Limitations Audit", 2.5, AGENT, "white"),
        ("6 · Final report & dataset card", 1.0, DET, "black"),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 11.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    cx, w, h = 5.0, 5.2, 0.82
    centers: Dict[str, float] = {}
    for label, y, color, tc in nodes:
        box = FancyBboxPatch((cx - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.08,rounding_size=0.12",
                             linewidth=1.4, edgecolor="#555555", facecolor=color)
        ax.add_patch(box)
        ax.text(cx, y, label, ha="center", va="center", fontsize=10.5,
                color=tc, weight="bold" if color == AGENT else "normal")
        centers[label] = y

    def arrow(y0, y1, color="#333333", rad=0.0, x0=cx, x1=cx, label=None, lx=None, ly=None, style="-|>"):
        a = FancyArrowPatch((x0, y0 - h / 2), (x1, y1 + h / 2),
                            connectionstyle=f"arc3,rad={rad}", arrowstyle=style,
                            mutation_scale=16, linewidth=1.5, color=color)
        ax.add_patch(a)
        if label:
            ax.text(lx if lx is not None else cx, ly if ly is not None else (y0 + y1) / 2,
                    label, ha="center", va="center", fontsize=8.5, color=color,
                    style="italic", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    ys = [n[1] for n in nodes]
    # main vertical spine
    for i in range(len(ys) - 1):
        arrow(ys[i], ys[i + 1])
    # gate label on the 2->interrupt edge
    ax.text(cx + 0.4, (ys[1] + ys[2]) / 2, "proceed=false", ha="left", va="center",
            fontsize=8, color="#8a6d00", style="italic")

    # loop-back edges (right side): Agent 2 -> train (gap 2) and -> export (gap 5)
    arrow(ys[6], ys[4], color="#a0522d", rad=-0.55, x0=cx + w / 2 - 0.3, x1=cx + w / 2 - 0.3,
          label="needs_rework · gap 2", lx=cx + w / 2 + 1.0, ly=(ys[6] + ys[4]) / 2)
    arrow(ys[6], ys[5], color="#a0522d", rad=-0.35, x0=cx + w / 2 - 0.6, x1=cx + w / 2 - 0.6,
          label="gap 5", lx=cx + w / 2 + 0.5, ly=(ys[6] + ys[5]) / 2 - 0.2)

    # legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=AGENT, edgecolor="#555", label="LLM agent"),
              Patch(facecolor=DET, edgecolor="#555", label="Deterministic stage"),
              Patch(facecolor=GATE, edgecolor="#555", label="Human-review gate")]
    ax.legend(handles=legend, loc="upper right", fontsize=9, frameon=True)
    ax.set_title("Malabar Plant-Disease · Agentic Pipeline (LangGraph / MiniGraph)",
                 fontsize=12, weight="bold", pad=12)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# stage entry point
# ---------------------------------------------------------------------------
def run_report_stage(state: Dict[str, Any], run_paths, logger=None) -> Dict[str, Any]:
    from common import get_logger
    logger = logger or get_logger("stage6.report", run_paths.reports / "report.log")

    report_md = build_final_report(state)
    card_md = build_dataset_card(state)
    report_path = run_paths.reports / "final_report.md"
    card_path = run_paths.reports / "dataset_card.md"
    save_text(report_md, report_path)
    save_text(card_md, card_path)

    # mermaid source (always) + PNG (best effort)
    try:
        from workflow_graph import mermaid_source
        save_text(mermaid_source(), run_paths.reports / "pipeline_diagram.mmd")
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not write mermaid source: %s", exc)

    png_path = run_paths.reports / "pipeline_diagram.png"
    diagram_ok = render_diagram(png_path)
    if not diagram_ok:
        logger.warning("matplotlib unavailable — wrote pipeline_diagram.mmd only (no PNG).")

    logger.info("Stage 6 done. report=%s card=%s diagram=%s",
                report_path.name, card_path.name, png_path.name if diagram_ok else "(mmd only)")
    return {
        "final_report_path": str(report_path),
        "dataset_card_path": str(card_path),
        "diagram_path": str(png_path) if diagram_ok else str(run_paths.reports / "pipeline_diagram.mmd"),
    }
