#!/usr/bin/env python
"""Tables of the paper's appendix, as LaTeX rows, from results/.

  tables/monitors.tex             Table "AUROC of the 48 monitors"   (results/monitor_auroc.csv)
  tables/compliance_exp1.tex      Table "Compliance in Experiment 1"  (results/exp1.npz)
  tables/compliance_exp4.tex      Table "Chain of thought in Experiment 4" (results/exp4.npz)
  tables/compliance_preflight.tex Table "Compliance test before the monitor experiments"
                                  (results/monitor_preflight.jsonl)
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from neurofeedback.analysis import RESULTS, load  # noqa: E402
from neurofeedback.monitors import CONCEPTS  # noqa: E402

OUT = REPO / "tables"
NAME = {"qwen3.5-27b": "Qwen3.5-27B", "gemma-4-31b": "Gemma-4-31B", "gpt-oss-20b": "GPT-OSS-20B",
        "mistral-small-3.2-24b": "Mistral-Small-3.2-24B", "olmo-3.1-32b": "OLMo-3.1-32B",
        "qwen3-32b": "Qwen3-32B", "qwen3-14b": "Qwen3-14B", "gemma-4-12b": "Gemma-4-12B",
        "qwen3.5-9b": "Qwen3.5-9B", "qwen3.6-27b": "Qwen3.6-27B"}
# Status of a turn in results/exp1.npz.
VALID, SHORT, LONG, EMPTY, FAILED, UNDEFINED = 0, 1, 2, 3, 4, 5


def f1(x):
    return f"{x:.1f}"


def monitors_table(results):
    rows = {(r["model"], r["monitor"]): r for r in csv.DictReader(open(Path(results) / "monitor_auroc.csv"))}
    order = ["gemma-4-12b", "gemma-4-31b", "qwen3-14b", "qwen3-32b", "qwen3.5-9b", "qwen3.5-27b",
             "qwen3.6-27b", "gpt-oss-20b"]
    lines = []
    for m in order:
        cells = [f"{float(rows[(m, c)]['auroc']):.2f}\\,{{\\scriptsize[{float(rows[(m, c)]['ci_lo']):.2f},"
                 f"\\,{float(rows[(m, c)]['ci_hi']):.2f}]}}" for c in CONCEPTS]
        lines.append(f"{NAME[m]:12s} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def exp1_counts(status, repeat):
    """Counters of one session's generated turns (turn 1 is the opening sentence)."""
    c = Counter(sessions=1)
    st, rp = status[1:], repeat[1:]
    if np.all(st == FAILED):                  # every call to the writer's endpoint failed
        c["failed_sessions"] += 1
        return c
    prev_invalid = False
    for s, r in zip(st, rp):
        if prev_invalid:
            c["after_invalid"] += 1
        if s == FAILED:
            continue
        c["turns"] += 1
        if s != VALID:
            c["invalid"] += 1
            c[{SHORT: "short", LONG: "long", EMPTY: "empty", UNDEFINED: "other"}[int(s)]] += 1
            prev_invalid = True
            continue
        if prev_invalid:
            c["recovered"] += 1
        prev_invalid = False
        c["valid"] += 1
        c["repeats"] += int(r)
    return c


def compliance_exp1_table(results):
    r = load("exp1", results)
    models = [str(x) for x in r["subjects"]]
    conds = [str(x) for x in r["conditions"]]
    status, repeat = r["status"], r["repeat"]
    agg = {}
    # one session per index: [subject, writer, readout, objective, condition, arm, session]
    for idx in np.ndindex(status.shape[:-1]):
        c = exp1_counts(status[idx], repeat[idx])
        w, cond, arm = models[idx[1]], conds[idx[4]], ("real", "control")[idx[5]]
        for k in [(w, "all", "all"), (w, arm, "all"), (w, "all", cond),
                  ("all", "all", "all"), ("all", arm, "all"), ("all", "all", cond)]:
            agg.setdefault(k, Counter()).update(c)

    def valid(c):
        return 100 * c["valid"] / c["turns"]

    lines = []
    for w in models + ["all"]:
        a = agg[(w, "all", "all")]
        cells = [NAME.get(w, "All writers"), f1(valid(a)), f1(valid(agg[(w, "real", "all")])),
                 f1(valid(agg[(w, "control", "all")])), f1(valid(agg[(w, "all", "hidden")])),
                 f1(valid(agg[(w, "all", "disclosed")])), f1(100 * a["short"] / a["turns"]),
                 f1(100 * a["long"] / a["turns"]), f1(100 * a["empty"] / a["turns"]),
                 f1(100 * a["recovered"] / a["after_invalid"]) if a["after_invalid"] else "--",
                 f1(100 * a["repeats"] / a["valid"]), f1(100 * a["failed_sessions"] / a["sessions"])]
        if w == "all":
            lines.append("\\midrule")
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines)


def compliance_exp4_table(results):
    r = load("exp4", results)
    unit_model = r["unit_model"].astype(str)
    cond = r["unit_condition"].astype(str)
    closed, tokens = r["cot_closed"], r["cot_tokens"]       # [unit, objective, session, turn]
    order = ["qwen3.5-9b", "qwen3.5-27b", "qwen3.6-27b", "qwen3-14b", "qwen3-32b", "gemma-4-12b",
             "gemma-4-31b", "gpt-oss-20b"]
    lines = []
    for m in order:
        u = unit_model == m
        pct = lambda x: f1(100 * x.mean())          # noqa: E731
        cells = [NAME[m], pct(closed[u]), pct(closed[u][:, :, :1]), pct(closed[u][:, :, 1:]),
                 pct(closed[u & (cond == "hidden")]), pct(closed[u & (cond == "disclosed")]),
                 f"{np.median(tokens[u]):.0f}"]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines)


def compliance_preflight_table(results):
    recs = [json.loads(line) for line in open(Path(results) / "monitor_preflight.jsonl")]
    # Only cases whose reply finished and was judged.
    recs = [r for r in recs if not r.get("error") and r.get("finish_reason") != "length"
            and r.get("complied") is not None]
    order = ["qwen3.5-9b", "qwen3.5-27b", "qwen3-14b", "qwen3-32b", "gemma-4-31b", "gpt-oss-20b"]
    lines = []

    def row(name, g):
        return " & ".join([name, str(len(g)), f1(100 * np.mean([x["complied"] for x in g])),
                           f1(100 * np.mean([x["repeated"] for x in g]))]) + " \\\\"
    for m in order:
        lines.append(row(NAME[m], [x for x in recs if x["model"] == m]))
    lines += ["\\midrule", row("All models", recs)]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(RESULTS), help="directory of the results (default: results/)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for name, fn in [("monitors", monitors_table), ("compliance_exp1", compliance_exp1_table),
                     ("compliance_exp4", compliance_exp4_table),
                     ("compliance_preflight", compliance_preflight_table)]:
        tex = fn(a.results)
        (OUT / f"{name}.tex").write_text(tex + "\n")
        print(f"% {name}\n{tex}\n")


if __name__ == "__main__":
    main()
