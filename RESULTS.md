# RDMonitor — evaluation (6 models × 3 detection benchmarks + closed-loop mitigation)

White-box transition-level representation-drift monitoring. u = difference-of-means over contrastive pairs, scored on hidden-state transitions. Deterministic (greedy, seed 0). Reproduce: detection `detect.py`; AgentDojo mitigation `mitigate.py`; MT mitigation `mitigate_mt.py`; then `aggregate_results.py`.

## AgentDojo — prompt-injection detection (leave-one-suite-out AUROC)

| Model | workspace | banking | slack | travel | **Mean** |
|---|---|---|---|---|---|
| Qwen3-8B | 0.871 | 0.958 | 0.907 | 1.000 | **0.934** |
| Qwen3-14B | 0.880 | 1.000 | 0.996 | 1.000 | **0.969** |
| Qwen3-32B | 0.907 | 0.929 | 0.975 | 1.000 | **0.953** |
| Qwen3-30B-A3B | 0.811 | 1.000 | 1.000 | 1.000 | **0.953** |
| Llama-3.1-8B | 0.820 | 0.970 | 0.950 | 1.000 | **0.935** |
| Mistral-24B | 0.837 | 0.949 | 1.000 | 1.000 | **0.946** |

## R-Judge — operational-risk detection (leave-one-domain-out AUROC)

| Model | Finance | Application | Program | Web | IoT | **Mean** |
|---|---|---|---|---|---|---|
| Qwen3-8B | 0.970 | 0.788 | 0.775 | 0.681 | 0.595 | **0.762** |
| Qwen3-14B | 0.973 | 0.808 | 0.805 | 0.703 | 0.605 | **0.779** |
| Qwen3-32B | 0.962 | 0.829 | 0.870 | 0.742 | 0.611 | **0.803** |
| Qwen3-30B-A3B | 0.929 | 0.835 | 0.838 | 0.676 | 0.558 | **0.767** |
| Llama-3.1-8B | 0.941 | 0.729 | 0.795 | 0.676 | 0.479 | **0.724** |
| Mistral-24B | 0.947 | 0.823 | 0.770 | 0.764 | 0.537 | **0.768** |

## MT-AgentRisk — multi-turn (real tool+length-matched benign-decomposition control)

| Model | SINGLE | transfer drift (Add/Dec) | **in-domain DRIFT** | in-domain per-turn |
|---|---|---|---|---|
| Qwen3-8B | 1.000 | 0.576/0.518 | **1.000** | 0.783 |
| Qwen3-14B | 1.000 | 0.709/0.694 | **1.000** | 0.776 |
| Qwen3-32B | 0.996 | 0.402/0.360 | **1.000** | 0.750 |
| Qwen3-30B-A3B | 0.999 | 0.588/0.469 | **1.000** | 0.723 |
| Llama-3.1-8B | 1.000 | 0.648/0.584 | **1.000** | 0.722 |
| Mistral-24B | 1.000 | 0.597/0.539 | **1.000** | 0.751 |

SINGLE = committed 1-turn harm vs 1-turn benign.  transfer = single-turn direction applied to decomposition (≈0.5 = no transfer).  in-domain = fit on the decomposition contrast, leave-one-tool-out; DRIFT=(h_n−h_0)·u, per-turn=max single transition.  Negative is a real benign task of the same tool decomposed into the same #turns.

## Closed-loop mitigation — Algorithm 1, online (AgentDojo, attack success rate + utility)

| Model / suite | pairs | ASR off→on | Δ ASR | utility off→on | flags/run |
|---|---|---|---|---|---|
| Qwen3-14B-8bit / slack | 12 | 0.583 → **0.083** | +0.500 | 0.667 → 0.583 | 3.0 |
| Qwen3-32B-8bit / slack | 12 | 0.250 → **0.000** | +0.250 | 0.500 → 0.417 | 2.2 |
| Qwen3-30B-A3B-8bit / slack | 12 | 0.250 → **0.000** | +0.250 | 0.583 → 0.417 | 3.9 |
| Qwen3-32B-8bit / banking | 12 | 0.500 → **0.250** | +0.250 | 0.500 → 0.417 | 0.8 |
| Qwen3-8B-8bit / slack | 12 | 0.083 → **0.000** | +0.083 | 0.333 → 0.583 | 2.3 |
| Meta-Llama-3.1-8B-Instruct-8bit / slack | 12 | 0.250 → **0.000** | +0.250 | 0.417 → 0.500 | 7.0 |
| Mistral-Small-24B-Instruct-2501-8bit / slack | 12 | 0.833 → **0.750** | +0.083 | 0.500 → 0.500 | 6.3 |
| Qwen3-8B-8bit / banking | 12 | 0.083 → **0.000** | +0.083 | 0.583 → 0.500 | 1.8 |
| Qwen3-14B-8bit / banking | 12 | 0.250 → **0.250** | +0.000 | 0.333 → 0.417 | 0.2 |
| Qwen3-30B-A3B-8bit / banking | 12 | 0.083 → **0.000** | +0.083 | 0.333 → 0.417 | 0.7 |
| Meta-Llama-3.1-8B-Instruct-8bit / banking | 12 | 0.083 → **0.000** | +0.083 | 0.417 → 0.333 | 2.5 |
| Mistral-Small-24B-Instruct-2501-8bit / banking | 12 | 0.833 → **0.667** | +0.167 | 0.333 → 0.417 | 1.5 |

ASR = injection_task.security (attack succeeded; lower better).  utility = user_task.utility (benign task completed; preserve).  OFF = monitor disabled; ON = transition monitor + reminder (the closed loop). Scored by AgentDojo's own checkers.

## Closed-loop mitigation — MT-AgentRisk decomposition (Algorithm 1, per-turn drift)

| Model | pairs | ASR off→on (harmful compliance) | Δ ASR | benign completion off→on |
|---|---|---|---|---|
| Qwen3-8B-8bit | 12 | 1.000 → **0.750** | +0.250 | 0.833 → 0.833 |
| Qwen3-14B-8bit | 12 | 0.917 → **0.583** | +0.333 | 0.917 → 0.917 |
| Qwen3-30B-A3B-8bit | 12 | 1.000 → **0.417** | +0.583 | 0.833 → 0.750 |
| Meta-Llama-3.1-8B-Instruct-8bit | 12 | 0.833 → **0.583** | +0.250 | 1.000 → 1.000 |
| Mistral-Small-24B-Instruct-2501-8bit | 12 | 0.833 → **0.083** | +0.750 | 0.833 → 0.833 |
| Qwen3-32B-8bit | 12 | 1.000 → **0.667** | +0.333 | 0.917 → 1.000 |

ASR = harmful-decomposition compliance (1 − refusal; lower better).  benign completion = 1 − benign-refusal (preserve; over-refusal would drop it).  ON = cumulative user-turn drift monitor + decomposition-aware reminder. Refusal via classifier; direction fit on a disjoint split.
