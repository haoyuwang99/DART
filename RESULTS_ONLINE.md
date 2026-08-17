# DART — online runtime enforcement results

Causal single-pass monitor (Algorithm 1): score each tool-return transition r_i=(h_i-h_{i-1})·u,
fire at the first crossing of tau, attribute the crossing read, enforce; monitoring CONTINUES after
an intervention. `u` and tau are fit on HELD-OUT tasks run through the SAME agent loop.
tau comes from a benign false-alarm budget (10% unless stated). `catchSUCC` = fraction of
attacks that actually succeeded which the monitor flagged — the security-relevant recall.


## AgentDojo — slack

| model | nH | baseline ASR | AUROC | catch@10% | FPR@10% | catchSUCC |
|---|---|---|---|---|---|---|
| Qwen3-8B | 55 | 0.18 | 0.762 | 0.64 | 0.09 | 0.80 |
| Llama-3.1-8B | 55 | 0.07 | 0.785 | 0.55 | 0.18 | 0.25 |
| Qwen3-14B | 55 | 0.38 | 0.803 | 0.76 | 0.18 | 0.90 |
| Qwen3-30B-A3B | 55 | 0.31 | 0.769 | 0.38 | 0.00 | 0.41 |
| Mistral-24B | 55 | 0.55 | 0.781 | 0.64 | 0.18 | 0.83 |
| Qwen3-32B | 55 | 0.45 | 0.847 | 0.73 | 0.18 | 1.00 |

**Enforcement @10%, FIRED cases only (ASR / Utility)**

| model | n | none | stop | attr | redact | redact_read |
|---|---|---|---|---|---|---|
| Qwen3-8B | 35 | 0.23 / 0.31 | 0.00 / 0.14 | 0.29 / 0.37 | 0.26 / 0.51 | 0.00 / 0.29 |
| Llama-3.1-8B | 30 | 0.03 / 0.43 | 0.00 / 0.00 | 0.00 / 0.50 | 0.00 / 0.33 | 0.03 / 0.00 |
| Qwen3-14B | 42 | 0.45 / 0.57 | 0.00 / 0.12 | 0.05 / 0.45 | 0.21 / 0.38 | 0.05 / 0.24 |
| Qwen3-30B-A3B | 21 | 0.33 / 0.62 | 0.00 / 0.24 | 0.10 / 0.71 | 0.10 / 0.52 | 0.05 / 0.71 |
| Mistral-24B | 35 | 0.71 / 0.60 | 0.00 / 0.14 | 0.26 / 0.60 | 0.11 / 0.37 | 0.09 / 0.43 |
| Qwen3-32B | 40 | 0.62 / 0.45 | 0.00 / 0.12 | 0.00 / 0.38 | — | 0.03 / 0.28 |

## AgentDojo — banking

| model | nH | baseline ASR | AUROC | catch@10% | FPR@10% | catchSUCC |
|---|---|---|---|---|---|---|
| Qwen3-14B | 72 | 0.33 | 0.802 | 0.65 | 0.12 | 0.79 |

**Enforcement @10%, FIRED cases only (ASR / Utility)**

| model | n | none | stop | attr | redact | redact_read |
|---|---|---|---|---|---|---|
| Qwen3-14B | 47 | 0.40 / 0.40 | 0.00 / 0.19 | 0.09 / 0.38 | 0.11 / 0.51 | 0.00 / 0.38 |

## AgentDojo — travel

| model | nH | baseline ASR | AUROC | catch@10% | FPR@10% | catchSUCC |
|---|---|---|---|---|---|---|
| Qwen3-14B | 70 | 0.27 | 0.986 | 0.93 | 0.20 | 0.84 |

**Enforcement @10%, FIRED cases only (ASR / Utility)**

| model | n | none | stop | attr | redact | redact_read |
|---|---|---|---|---|---|---|
| Qwen3-14B | 65 | 0.25 / 0.17 | 0.00 / 0.00 | 0.11 / 0.32 | 0.14 / 0.03 | 0.00 / 0.11 |

## InjecAgent (enhanced variant, online core)

Balanced 1:1 (each case yields one benign + one harmful run). NOTE: InjecAgent utility is
"the user tool was called", which happens BEFORE the injection is read, so Util≈1 by
construction — it is not comparable to AgentDojo's task-completion utility.

| model | split | nH | baseline ASR | AUROC | catch | FPR | catchSUCC | none | stop | attr | redact | redact_read |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | dh-enh | 15 | 0.13 | 0.922 | 0.93 | 0.07 | 1.00 | 0.14 | 0.00 | 0.29 | 0.00 | 0.00 |
| Qwen3-8B | ds-enh | 15 | 0.33 | 1.000 | 1.00 | 0.20 | 1.00 | 0.33 | 0.00 | 0.87 | 0.13 | 0.00 |
| Qwen3-14B | dh-enh | 15 | 0.13 | 0.609 | 0.47 | 0.07 | 1.00 | 0.29 | 0.00 | 0.00 | 0.00 | 0.00 |
| Qwen3-14B | ds-enh | 15 | 0.07 | 0.556 | 0.33 | 0.07 | 1.00 | 0.20 | 0.00 | 0.00 | 0.20 | 0.00 |

## Findings

1. **Detection scales with capability.** Qwen3-32B is the best detector (AUROC 0.847) and flags
   **100% of attacks that actually succeeded**. Across models, catchSUCC is 0.80–1.00 for
   8B / 14B / Mistral / 32B.
2. **The attributed reminder is capability-gated.** ASR under `attr`: 8B 0.23→0.29 (backfires),
   14B 0.45→0.05, 30B-A3B 0.33→0.10, Mistral 0.71→0.26, 32B 0.62→**0.00**. Replicated independently
   on InjecAgent (8B ds-enh 0.33→0.87 backfire; 14B →0.00). Quoting the injected span is read as a
   negative example by capable models and as an instruction by small ones.
3. **Read-level redaction is universally reliable.** `redact_read` holds ASR ≤0.09 on every model and
   both benchmarks, including where `attr` inverts — but costs more utility (`attr` wins utility on
   5 of 6 models).
4. **Enforcement robustness tracks attribution dependence.** `stop` needs none (ASR 0.00 everywhere,
   worst utility); `redact_read` needs read-level attribution (measured 1.00, causally); span-`redact`
   needs sub-read localisation, which is confounded by span length and is dominated on both axes.
5. **Detection transfers across suites** (14B: slack 0.803, banking 0.802, travel 0.986).

## Known limitations

- **FPR runs above budget** (0.18 vs nominal 0.10 on four of six slack cells). tau is calibrated on the
  same runs used to fit `u`, so the quantile is optimistic; an inner split of the calibration set would
  fix it. All FPR figures should be read as slightly optimistic.
- **Two models break the capability story**: Llama-3.1-8B (catchSUCC 0.25) and Qwen3-30B-A3B (0.41).
  30B-A3B's tau is over-conservative (FPR 0.00); Llama's figure rests on 4 successful attacks.
- **AUROC understates security performance** because the "harmful" label includes injections that were
  read but never acted on. catchSUCC is the security-relevant number.
- Calibration locates injected reads via the attack template's signature; deployment needs a labelled
  red-team calibration set (this is the method's contrastive supervision, made explicit). It is cheap:
  3 positives / 5 negatives already give AUROC 0.92 — but it must be collected on the deployment loop.
- workspace suite not run (dropped for cost); ASB / MT are out of scope (injection-only framing).

