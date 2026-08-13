# DART — Detect, Attribute, Redact

Representation-drift monitoring for LLM-agent defense.

## Layout
```
src/
  dart/                  the package (modular by concern)
    agent.py               MAIN AGENT + model: HiddenLM (generation, hidden states, steering hook),
                           tool-call parsing, trace→chat conversion, closed-loop MonitoredAgent
    monitor.py             MONITOR (detection): read(C)/fit_um/pick_layer, transition scoring (Monitor),
                           detection AUROC, the reference fit_direction
    mitigation.py          MITIGATION levers: redact (+leave-one-span-out attribution), reminders
                           (verbalize/directive/attributed), steer_vec (RepE actuator), mit_cell (τ-gating)
    datasets.py            executable adapters: AgentDojo + MT rollouts, dataset fits (ad_fit/ad_fit_steer),
                           run_agentdojo / run_mt  (wires agent+monitor+mitigation+baselines per dataset)
    baselines.py           AgentSpec (rule-based action enforcement)
    eval.py                grid harness: record store (emit) + report + run entry
  rdeval.py              CLI → dart.eval  (eval grid / --report)
  rd_agent.py            CLI → MonitoredAgent demo
  augment_{redact,attributed,steering}.py   add one mitigation column to the records (import dart)
  {attr,steer}_val_*.py  fast single-cell validations
  diag_*.py              mechanism diagnostics
  redact_summary.py      analysis over records
  span_attribute.py      marker-free span-attribution validation
  run_*.sh               multi-model grid runners
  appendix/              archive (RQ5 exploration + the unrelated dxscore·judge·fireworks strand)
log/                     all run logs
*.jsonl                  data + results (rdeval_records.jsonl is the record store;
                         traces_campaign.jsonl / mt_agentrisk.jsonl / mt_benign_multi.jsonl are inputs)
.venv311/                interpreter (mlx + agentdojo)
```

## Dependency direction
`agent` → (`monitor`, `mitigation`) → `datasets` → `eval`; `baselines` standalone; `eval` imports
`datasets` lazily (in `main`) so there is no cycle.

## Run convention (always from repo root)
Scripts live at `src/` top level, so `python src/<script>.py` puts `src/` on `sys.path` and
`from dart.<module> import …` resolves; data files are read relative to the root cwd; logs go to `log/`.
```bash
.venv311/bin/python src/rdeval.py --report rdeval_records.jsonl
.venv311/bin/python src/augment_steering.py mlx-community/Qwen3-8B-8bit 10 2
nohup zsh src/run_augment_steer.sh > log/augment_steer_grid.log 2>&1 &
```
