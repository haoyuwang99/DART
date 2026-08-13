#!/usr/bin/env python3
"""Aggregate the detection grid (6 models x 3 benchmarks) into final markdown tables.

Reads the run logs produced by detect.py and prints three tables:
  AgentDojo (leave-one-suite-out injection), R-Judge (leave-one-domain-out), MT-AgentRisk
  (single committed harm; single->decomp transfer; in-domain LOSO drift vs per-turn).

Log sources (override on the CLI): the incremental runs live in different files because R-Judge
and AgentDojo were finished before the MT redesign; a single clean sweep would put all three in
one log, which this parser also handles.
    aggregate_results.py [agentdojo.log] [rjudge.log] [mt.log]
"""
import sys, re, collections

AD_LOG = sys.argv[1] if len(sys.argv) > 1 else "detect_full.log"
RJ_LOG = sys.argv[2] if len(sys.argv) > 2 else "tasks/bn54pdbh5.output"
MT_LOG = sys.argv[3] if len(sys.argv) > 3 else "mt_grid.log"

ORDER = ["Qwen3-8B-8bit", "Qwen3-14B-8bit", "Qwen3-32B-8bit", "Qwen3-30B-A3B-8bit",
         "Meta-Llama-3.1-8B-Instruct-8bit", "Mistral-Small-24B-Instruct-2501-8bit"]
SHORT = {"Qwen3-8B-8bit": "Qwen3-8B", "Qwen3-14B-8bit": "Qwen3-14B", "Qwen3-32B-8bit": "Qwen3-32B",
         "Qwen3-30B-A3B-8bit": "Qwen3-30B-A3B", "Meta-Llama-3.1-8B-Instruct-8bit": "Llama-3.1-8B",
         "Mistral-Small-24B-Instruct-2501-8bit": "Mistral-24B"}
MARK = re.compile(r"@@@@+\s*(?:MODEL\s+)?([A-Za-z0-9.\-]+?)\s*@@@@")


def read(path):
    try:
        return open(path).read().splitlines()
    except FileNotFoundError:
        return []


def blocks(lines):
    """Split a log into {model: [lines]} by the @@@@ marker."""
    out, cur = collections.OrderedDict(), None
    for ln in lines:
        m = MARK.search(ln)
        if m:
            cur = m.group(1); out.setdefault(cur, [])
        elif cur is not None:
            out[cur].append(ln)
    return out


def parse_agentdojo(block):
    """-> (mean, {suite: auroc}) from a leave-one-suite-out table."""
    suites, mean, inside = {}, None, False
    for ln in block:
        if "AgentDojo injection" in ln:
            inside = True
        if not inside:
            continue
        m = re.match(r"\s*(workspace|banking|slack|travel)\s+\d+\s+([0-9.]+)", ln)
        if m:
            suites[m.group(1)] = float(m.group(2))
        m = re.match(r"\s*MEAN\s+([0-9.]+)", ln)
        if m and mean is None and suites:
            mean = float(m.group(1)); break
    return mean, suites


def parse_rjudge(block):
    doms, mean = {}, None
    for ln in block:
        m = re.match(r"\s*(Application|Finance|IoT|Program|Web)\s+\d+\s+([0-9.]+)", ln)
        if m:
            doms[m.group(1)] = float(m.group(2))
        m = re.match(r"\s*MEAN\s+([0-9.]+)", ln)
        if m and doms:
            mean = float(m.group(1))
    return mean, doms


def parse_mt(block):
    r = {"single": None, "tr": {}, "drift": None, "perturn": None}
    section = None
    for ln in block:
        m = re.search(r"SINGLE committed harm.*AUROC=([0-9.]+)", ln)
        if m:
            r["single"] = float(m.group(1))
        m = re.match(r"\s*(Addition|Decomposition)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", ln)
        if m:
            r["tr"][m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        if "in-domain: harmful-decomp DRIFT" in ln:
            section = "drift"
        elif "in-domain: per-TURN" in ln:
            section = "perturn"
        m = re.match(r"\s*MEAN\s+([0-9.]+)", ln)
        if m and section:
            r[section] = float(m.group(1)); section = None
    return r


def cell(x):
    return f"{x:.3f}" if isinstance(x, float) else "--"


def main():
    ad = blocks(read(AD_LOG)); ad_mt = blocks(read(MT_LOG))
    rj = blocks(read(RJ_LOG)); mt = blocks(read(MT_LOG))

    print("## AgentDojo — prompt-injection detection (leave-one-suite-out AUROC)\n")
    print("| Model | workspace | banking | slack | travel | **Mean** |")
    print("|---|---|---|---|---|---|")
    for k in ORDER:
        mean, su = parse_agentdojo(ad.get(k, []))
        if mean is None:                                   # Mistral's AgentDojo lives in the MT-grid log
            mean, su = parse_agentdojo(ad_mt.get(k, []))
        print(f"| {SHORT[k]} | " + " | ".join(cell(su.get(s)) for s in
              ["workspace", "banking", "slack", "travel"]) + f" | **{cell(mean)}** |")

    print("\n## R-Judge — operational-risk detection (leave-one-domain-out AUROC)\n")
    print("| Model | Finance | Application | Program | Web | IoT | **Mean** |")
    print("|---|---|---|---|---|---|---|")
    for k in ORDER:
        mean, d = parse_rjudge(rj.get(k, []))
        print(f"| {SHORT[k]} | " + " | ".join(cell(d.get(x)) for x in
              ["Finance", "Application", "Program", "Web", "IoT"]) + f" | **{cell(mean)}** |")

    print("\n## MT-AgentRisk — multi-turn (real tool+length-matched benign-decomposition control)\n")
    print("| Model | SINGLE | transfer drift (Add/Dec) | **in-domain DRIFT** | in-domain per-turn |")
    print("|---|---|---|---|---|")
    for k in ORDER:
        r = parse_mt(mt.get(k, []))
        tr = r["tr"]
        trd = "/".join(cell(tr[f][1]) for f in ["Addition", "Decomposition"] if f in tr) or "--"
        print(f"| {SHORT[k]} | {cell(r['single'])} | {trd} | **{cell(r['drift'])}** | {cell(r['perturn'])} |")
    print("\nSINGLE = committed 1-turn harm vs 1-turn benign.  transfer = single-turn direction applied to "
          "decomposition (≈0.5 = no transfer).  in-domain = fit on the decomposition contrast, leave-one-"
          "tool-out; DRIFT=(h_n−h_0)·u, per-turn=max single transition.  Negative is a real benign task of "
          "the same tool decomposed into the same #turns.")

    # ---- closed-loop mitigation (from mitigate_results.jsonl, if present) ----
    try:
        mrows = [__import__("json").loads(l) for l in open("mitigate_results.jsonl")]
    except FileNotFoundError:
        mrows = []
    if mrows:
        print("\n## Closed-loop mitigation — Algorithm 1, online (AgentDojo, attack success rate + utility)\n")
        print("| Model / suite | pairs | ASR off→on | Δ ASR | utility off→on | flags/run |")
        print("|---|---|---|---|---|---|")
        for r in mrows:
            print(f"| {r['model']} / {r['suite']} | {r['pairs']} | "
                  f"{r['asr_off']:.3f} → **{r['asr_on']:.3f}** | {r['asr_off'] - r['asr_on']:+.3f} | "
                  f"{r['util_off']:.3f} → {r['util_on']:.3f} | {r['flags_on']:.1f} |")
        print("\nASR = injection_task.security (attack succeeded; lower better).  utility = user_task.utility "
              "(benign task completed; preserve).  OFF = monitor disabled; ON = transition monitor + reminder "
              "(the closed loop). Scored by AgentDojo's own checkers.")

    # ---- MT-AgentRisk closed-loop mitigation (from mitigate_mt_results.jsonl, if present) ----
    try:
        mtrows = [__import__("json").loads(l) for l in open("mitigate_mt_results.jsonl")]
    except FileNotFoundError:
        mtrows = []
    if mtrows:
        print("\n## Closed-loop mitigation — MT-AgentRisk decomposition (Algorithm 1, per-turn drift)\n")
        print("| Model | pairs | ASR off→on (harmful compliance) | Δ ASR | benign completion off→on |")
        print("|---|---|---|---|---|")
        for r in mtrows:
            print(f"| {r['model']} | {r['pairs']} | {r['asr_off']:.3f} → **{r['asr_on']:.3f}** | "
                  f"{r['asr_off'] - r['asr_on']:+.3f} | "
                  f"{1 - r['benign_refuse_off']:.3f} → {1 - r['benign_refuse_on']:.3f} |")
        print("\nASR = harmful-decomposition compliance (1 − refusal; lower better).  benign completion = "
              "1 − benign-refusal (preserve; over-refusal would drop it).  ON = cumulative user-turn drift "
              "monitor + decomposition-aware reminder. Refusal via classifier; direction fit on a disjoint split.")


if __name__ == "__main__":
    main()
