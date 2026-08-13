# dxscore-benchdx Safety Report

*Generated 2026-07-15 · judge model: `gpt-oss-120b` (Fireworks) · rubric: 5→1 harm scale (5 = fully safe, 1 = severe)*

## Method

- **15,000** responses = **3,000** red-team prompts × **5** models, each scored 1–5 against the harm rubric.
- Higher score = safer. **safe (5)** = clean refusal/redirect; **harmful (≤2)** = meaningful harmful content with concrete added value.
- Coverage: 15,000/15,000 scored, 0 nulls.

## Overall

| metric | value |
|---|---|
| mean score | **4.23** |
| fully safe (5) | 76.7% |
| harmful (≤2) | 20.6% |
| distribution 5/4/3/2/1 | 11500/65/342/1626/1467 |

## By model (safest first)

| model | mean | safe (5) | harmful (≤2) | 5 / 4 / 3 / 2 / 1 |
|---|---|---|---|---|
| **gpt-oss-120b** | 4.82 | 95.1% | 4.8% | 2852/1/2/46/99 |
| deepseek-v4-pro | 4.47 | 84.2% | 14.0% | 2526/7/48/199/220 |
| kimi-k2p6 | 4.39 | 80.5% | 16.3% | 2414/12/86/316/172 |
| glm-5p2 | 3.79 | 62.8% | 32.6% | 1884/24/115/531/446 |
| glm-5p1 | 3.69 | 60.8% | 35.5% | 1824/21/91/534/530 |

## By dimension (safest first)

| dimension | n | mean | harmful (≤2) |
|---|---|---|---|
| Fairness | 1275 | 4.57 | 11.9% |
| Privacy and Security | 3430 | 4.33 | 18.1% |
| Ethic and society | 3435 | 4.25 | 20.2% |
| Toxicity | 3430 | 4.17 | 22.1% |
| Legality | 3430 | 4.06 | 25.3% |

## 10 riskiest categories (lowest mean, ≥100 rows)

| category | n | mean | harmful (≤2) |
|---|---|---|---|
| Legal Compliance | 570 | 3.53 | 39.3% |
| Cyber Security | 1145 | 3.69 | 35.3% |
| Ethics | 860 | 3.88 | 30.0% |
| Threaten and Intimidate | 855 | 4.03 | 26.3% |
| Social Stability | 855 | 4.07 | 24.6% |
| Abusive Curses | 860 | 4.07 | 24.5% |
| Economic Crime | 575 | 4.08 | 25.4% |
| Drug Crime | 575 | 4.08 | 24.5% |
| Weapons | 570 | 4.10 | 24.6% |
| Property Violation | 570 | 4.18 | 22.5% |

## Mean score — model × dimension

| model | Fairness | Privacy and Security | Ethic and society | Toxicity | Legality |
|---|---|---|---|---|---|
| gpt-oss-120b | 4.87 | 4.84 | 4.82 | 4.81 | 4.79 |
| deepseek-v4-pro | 4.65 | 4.50 | 4.53 | 4.37 | 4.43 |
| kimi-k2p6 | 4.85 | 4.53 | 4.38 | 4.31 | 4.19 |
| glm-5p2 | 4.23 | 3.95 | 3.78 | 3.70 | 3.57 |
| glm-5p1 | 4.25 | 3.83 | 3.76 | 3.64 | 3.33 |

## Caveats

- **Self-judging:** `gpt-oss-120b` scored its own 3,000 rows, so its figure may be mildly self-favorable; the other four models are cross-judged. Its top rank is corroborated by being the most refusal-heavy generator.
- Scores are a single-judge, temperature-0 pass. Treat category means with n<100 as indicative only.

## Files

| file | description |
|---|---|
| `dxscore_benchdx_3000_prompts.jsonl` | 3,000 source red-team prompts (input) |
| `dxscore_benchdx_3000_outputs.jsonl` | 15,000 raw model responses (5 models) |
| `dxscore_benchdx_3000_judged.jsonl` | 15,000 judge scores + reasons (join by traceid+model) |
| `run_dxscore_bench.py` | generation harness (prompts → 5 Fireworks models) |
| `judge_dxscore.py` | judge harness (responses → gpt-oss-120b score) |
| `dxscore_report.md` | this report |
