# RDMonitor Project Checkpoint — 2026-07-04

Runtime-agent-safety-monitoring research (representation drift). Everything through
the scaled evaluation campaign + paper aggregation is complete. No processes running.
This checkpoint precedes Intro/Related-Work drafting and the optional leave-one-attack-out.

## Environments (on the Mac Studio M3 Ultra, 96 GB)
- `.venv311/`  — uv-managed CPython 3.11; has **mlx-lm + agentdojo + fastapi**. USE THIS for
  all rollouts/scoring. Run scripts as `./.venv311/bin/python <script>`.
- `.venv/`     — system py3.9; mlx-lm only; used by `server.py` (introspection API).
- 7 models cached in `~/.cache/huggingface/hub/` (Qwen3-4B/8B/14B/32B/30B-A3B, Llama-3.1-8B, Mistral-Small-24B), all `mlx-community *-8bit`.

## Headline results — mean leave-one-suite-out AUROC (transition form)
| Model | AUROC | | Cross-dataset (R-Judge, LOSO-domain) |
|---|---|---|---|
| Qwen3-8B | 0.92 | | Qwen3-8B 0.77 |
| Qwen3-14B | 0.96 | | Qwen3-32B 0.80 |
| Qwen3-32B | 0.93 | | |
| Qwen3-30B-A3B | 0.94 | | Ablation (state vs transition, avg): |
| Llama-3.1-8B | 0.96 | | state 0.884 vs transition 0.925 |
| Mistral-24B | 0.85 | | |

## The definitive scorer & dataset
- `traces_campaign.jsonl` — **1164 real agent-rollout traces** (6 models × 4 AgentDojo suites, clean+injected). THE evaluation dataset. Regenerate any number from it.
- `score_final.py <model_id> traces_campaign.jsonl` — unified per-suite state+transition LOSO AUROC. Supersedes score_campaign.py (state only) and score_ablation.py (mean only).
- `rjudge_eval.py <model_id>` — non-injection eval (needs `rjudge_data/`, already downloaded, 530 trajectories).

## File map
Core method:  introspect.py (Introspector: hidden-state capture) · repeng.py (encode/concept_direction) · steer.py (activation steering) · server.py (FastAPI /introspect, on .venv)
Mechanistic probes (constructed contexts, RQ4): agentdojo_pairs.py · agentdojo_probe.py · monitor.py · evidence.py · benign_control.py · intent_probe.py · action_matched_probe.py · injection_defense.py
Campaign (real traces): run_rollout.py (generalized, resumable) · drive_rollouts.sh (fleet driver) · score_final.py (definitive) · final_driver.sh · score_campaign.py · score_all.sh · score_ablation.py · ablation_driver.sh · rjudge_eval.py
Evidence: traces_campaign.jsonl (main) · traces.jsonl, traces_30b.jsonl (pilot) · rjudge_data/ · results/ (scoring logs) · *.npz (saved directions) · introspect_runs/
Paper: `paper/` (RDMonitor LaTeX; also lives at ~/Downloads/RDMonitor_paper/). eval.tex = full Experiments (RQ1 cross-domain transition, RQ2 ablation, RQ3 R-Judge, RQ4 structure-vs-intent). approach.tex = RepE method. intro.tex + background.tex = EMPTY stubs.

## Honest caveats baked into the paper
- LOSO varies **domain + goal** but holds the **attack template fixed** (shared important_instructions wrapper) — NOT cross-attack.
- Direction is **largely structural** ("injected instruction present"), weaker harm axis (RQ4). Explains the strong transfer.
- **AUROC ≫ accuracy** — threshold doesn't transfer across domains.
- Transition > state, and it stabilizes weak folds (fixes travel collapse, rescues thin-data Mistral).
- Scorer = linear projection on activations (NOT pattern matching; marker only labels ground truth).

## Next steps (not started)
1. **Leave-one-attack-out** — fit on some attack templates, test on held-out ones (tool_knowledge, system_message, dos). The missing hard axis; RQ4 predicts a drop. ~1 h.
2. **Intro + Related Work** — draft the two empty stubs.
3. **AgentHarm** — HF-gated; needs user to accept terms first.

## How to resume from scratch
```
cd "/Users/haoyu/LLM playground" && source .venv311/bin/activate
# re-score any model:
python score_final.py mlx-community/Qwen3-14B-8bit traces_campaign.jsonl
# new rollout: start server then run_rollout
python -m mlx_lm server --model <id> --port 8082 &   # wait for /v1/models
python run_rollout.py <id> 8082 traces_campaign.jsonl workspace,banking,slack,travel
```
