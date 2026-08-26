# Malabar Plant-Disease Classifier + Agentic Audit Workflow

A production-style **supervised leaf-disease classifier** for Malabar spinach
(*Basella alba*), wrapped in a **6-node agentic workflow** (LangGraph, with an
offline fallback engine) whose two LLM agents audit the data and the trained
model and gate the pipeline.

The project is **two layers, both delivered**:

| Layer | What it is |
|---|---|
| **Part A — CV pipeline** | Data ingestion → proxy-domain derivation → perceptual-hash dedup + leakage check → stratified (class×domain) split + held-out external-domain test → 12-technique augmentation → EfficientNetV2-S (CE + SupCon, class-balanced/focal) → 5-fold CV + in-domain & external test metrics with bootstrap CIs → Grad-CAM++ + deletion/insertion faithfulness → ONNX export + CPU latency/memory benchmark. |
| **Part B — agentic layer** | A `StateGraph` threading a single typed `PipelineState` through six logical nodes. **Agent 1** (Explainable Data Audit) reasons over the data bookkeeping JSON and can halt the run for human review. **Agent 2** (Limitations Coverage Audit) walks a fixed 5-gap checklist and gates ship / ship-with-limitations / rework. |

### Reconciling the numbers (7 / 5 / 12)

- **7 canonical classes** — the fixed label vocabulary, never renamed:
  `Healthy-Leaf, Anthracnose, Pest-Damage, Bacterial-Spot, Downy-Mildew, Dichotomophthora-Leaf-Spot, Bipolaris-Leaf-Spot`.
- **5 active classes** — those actually present on disk. The **2** remaining
  (`Dichotomophthora-Leaf-Spot`, `Bipolaris-Leaf-Spot`) have zero images; Agent 1
  reports them and the dataset card lists them as *declared-but-unpopulated*.
- **12** refers to the **augmentation-technique pool size**, not a class count.

---

## The workflow graph

```
1 · data_ingestion        (deterministic)   discover · proxy-domains · dedup · split
        │
2 · data_audit            ● Agent 1 (LLM)    proceed=false ─▶ await_human_review ⏸ (interrupt)
        │                                                          │  (edit data/config, then --resume)
3 · preprocess_augment    (deterministic)   ◀─────────────────────┘
        │
4 · train_and_evaluate    (deterministic)   train + CV + in-domain/external test + Grad-CAM++
        │
4b· export_model          (deterministic)   ONNX export + CPU latency/memory benchmark
        │
5 · limitations_audit     ● Agent 2 (LLM)    needs_rework ─▶ loop back to train (gap 2) or export (gap 5)
        │                                                     (bounded by retry.max_retries, then force-ship)
6 · final_report          (deterministic)   final_report.md + dataset_card.md + pipeline_diagram.png
```

Purple nodes are LLM agents, gray are deterministic, amber is the human-review
gate. The diagram is regenerated every run (`reports/pipeline_diagram.png` +
`.mmd` source).

**Agent safety is deterministic, not vibes.** Agent 1's *any blocking finding ⇒
`proceed=false`* rule and Agent 2's five gap statuses are **recomputed from the
numbers** and override whatever the LLM says — the LLM supplies prose and fixes,
the code owns the arithmetic and the routing. The rework loop is bounded, so the
graph always terminates.

---

## Setup

Use the project virtual environment at `venv`:

```bash
python3 -m pip install -r requirements.txt
```

`requirements.txt` has a **core** tier (pinned, required) and an **optional**
tier (commented). Everything runs on the core tier alone; optional libraries are
*preferred when present* and the code degrades gracefully without them:

| Optional lib | Without it, falls back to |
|---|---|
| `timm` | torchvision `efficientnet_v2_s` → `efficientnet_b0` (cached ImageNet weights) |
| `langgraph` (+ checkpoint-sqlite) | built-in `MiniGraph` — identical node/edge/interrupt/resume/retry semantics |
| `onnx` / `onnxruntime` | torch-CPU benchmark (ONNX export error recorded, non-fatal) |
| `imagehash` | self-contained DCT pHash (numpy/scipy) |
| `grad-cam` | from-scratch Grad-CAM++ in `explain.py` |
| `mlflow` / `wandb` | local file tracker under `runs/<id>/tracking/` (always on) |

---

## Running it

### 1. Full agentic pipeline

The two agent nodes need an LLM. Provide a key (see [LLM config](#llm-provider--keys))
**or** use `--mock` for a fully offline demo with canned, grounded agents:

```bash
# offline demo — no key, canned agents, exercises the whole graph
python run_pipeline.py --profile fast --mock
```

```bash
# live agents (Gemini is the default provider) — put the key in .env (see below)
GEMINI_API_KEY=... python run_pipeline.py --profile fast
```

`--profile fast` runs in a few CPU-minutes (image subset, 1–3 epochs, 2 CV
folds). `--profile full` is the spec-scale run (40 epochs, 5 folds, 1000-sample
bootstrap) for a GPU / full-deps machine.

### 2. When Agent 1 halts the run (human review)

If Agent 1 emits a **blocking** finding it sets `proceed=false`, the graph pauses
at `await_human_review`, and the state is checkpointed to
`runs/<id>/pipeline_state.json`. Fix the data/config, then resume — stage 1 is
**not** re-run:

```bash
python run_pipeline.py --resume --run-id <id>          # or just --resume for the latest paused run
python run_pipeline.py --resume --run-id <id> --mock   # offline
```

### 3. Standalone Part-A stages (deterministic, no key)

Run any single stage; missing prerequisites are auto-run in the **same** run
directory and their artifacts reused:

```bash
python run_pipeline.py --stage data     --profile fast   # discover + dedup + split only
python run_pipeline.py --stage evaluate --profile fast   # auto-runs data→augment→train→evaluate
```

Stages: `data · augment · train · evaluate · explain · export`.

### Common flags

`--config <path>` · `--profile {fast,full}` · `--override a.b.c=value` (repeatable)
· `--device {auto,cpu,cuda,mps}` · `--simsiam` (SimSiam SSL pretrain) ·
`--run-id <id>` · `--run-dir <path>`.

---

## Outputs

Everything for a run lives under `runs/<run_id>/`:

| Path | Contents |
|---|---|
| `reports/final_report.md` | config, metric tables (CV / in-domain / external + bootstrap CIs), per-class P/R/F1, confusion matrices, **both agents' full findings incl. failures**, the 5-gap checklist as a table, and the ordered orchestration trace |
| `reports/dataset_card.md` | HF-style card: active + declared-unpopulated classes, proxy domains, split sizes, per-class×domain counts, cleaning log, and biases/limitations from both agents |
| `reports/pipeline_diagram.png` / `.mmd` | the workflow graph (purple = agents, gray = deterministic, amber = gate) |
| `tracking/` | `metrics.csv`, `params.json`, `agent_calls.jsonl`, `edge_decisions.jsonl` — the agent reasoning sits right next to the metrics |
| `explain/` · `onnx/` · `checkpoints/` | Grad-CAM++ overlays, exported model, best checkpoint |
| `pipeline_state.json` | the checkpoint used for `--resume` |

---

## LLM provider & keys

Configured under `agents:` in `config.yaml`; the provider is swappable
(`gemini | anthropic | openai`) and Gemini is the default. Providers are called
directly over REST via `requests` (no SDK required) with enforced JSON structured
output validated against Pydantic schemas.

```yaml
agents:
  provider: gemini
  model: gemini-2.5-flash          # gemini-2.5-pro for stronger reasoning
  api_key_env: GEMINI_API_KEY      # falls back to GOOGLE_API_KEY
  temperature: 0.1                 # agents audit; they do not create
```

Put the key in a **gitignored `.env`** at the repo root (loaded automatically at
startup) — never commit it:

```
GEMINI_API_KEY=your-key-here
```

Swap providers with an override, e.g.:

```bash
python run_pipeline.py --override agents.provider=anthropic --override agents.model=claude-sonnet-4-5
```

Missing key → a clear error telling you to set it or pass `--mock`.

---

## Tests

Smoke tests exercise the orchestration with an injected offline mock LLM (no key,
no network) plus the real data-stage leakage invariant:

```bash
pytest -q
```

Coverage: Agent 1's blocking-finding guardrail, Agent 2's gap arithmetic and
bounded-retry termination, the `await_human_review` interrupt + `--resume` round
trip, the rework loop, report/card/diagram generation, and the real dedup
no-leakage invariant (skips if the dataset is absent).

> No pytest installed and can't reach PyPI? A dependency-free runner mirrors the
> suite: `python tests/run_smoke.py`.

---

## Configuration

`config.yaml` is the single source of truth for both layers. Profile-scoped
values look like `{fast: X, full: Y}` and are selected with `--profile`. Key
sections: `data` (classes, splits, proxy-domain + external-holdout strategy,
dedup), `augment` (12 techniques + domain-randomization), `model` (backbone,
SimSiam), `loss` (CE + SupCon + class-balancing), `train`, `eval` (CV folds,
bootstrap), `explain`, `export` (ONNX opset, device budget), `agents`,
`thresholds` (Agent 2's gap floors), and `retry.max_retries`.

---

## Notes

- **The external-domain test and the leakage check are always run** — they are
  required, not optional. No real capture-device metadata exists in the source
  folders, so a deterministic proxy `source_domain` is derived per image
  (resolution / aspect-ratio / JPEG-quantization signatures) to drive the
  stratified split and the held-out external-domain test.
- The pre-existing Fuzzy C-Means clustering scripts (`fuzzy_cmeans_pipeline.py`,
  `feature_extraction.py`, `build_manifest.py`, …) are untouched and still work
  against the legacy `fcm:` block in `config.yaml`.
