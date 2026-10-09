"""scripts/collect_results.py turns session records into the arrays of results/."""
import importlib.util

import numpy as np

from conftest import REPO
from fakes import fake_backend
from neurofeedback import prompts
from neurofeedback.analysis import READOUT_ORDER
from neurofeedback.experiments import fixed_sentence, monitor
from neurofeedback.io import load_sentences, read_jsonl, write_jsonl
from neurofeedback.models import EXP2_MODELS
from neurofeedback.monitors import load_monitor
from neurofeedback.readouts import ReadoutSet

spec = importlib.util.spec_from_file_location("collect_results", REPO / "scripts" / "collect_results.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)
MODEL = "qwen3.5-9b"


def backend():
    rs = ReadoutSet(MODEL)
    return fake_backend(num_layers=rs.num_layers, hidden=int(rs.readouts["probe_truth_d33"].mu.shape[0]))


def test_collect_exp2(tmp_path):
    be = backend()
    rs = ReadoutSet(MODEL, "exp2", readout_ids=["probe_truth_d33", "step_change_d66"])
    sentences = load_sentences(REPO / "data" / "exp2_fixed_sentences.jsonl")
    cfg = fixed_sentence.Exp2Config(horizon=3, n_sentences=2, n_control_hidden=2, n_control_disclosed=1)
    fixed_sentence.run(subject=MODEL, objective="min", condition="hidden", sentences=sentences, backend=be,
                       readouts=rs, out_root=tmp_path, cfg=cfg)
    recs = read_jsonl(fixed_sentence.run_path(tmp_path, MODEL, "min", "hidden", 0))
    # the collector expects all 42 readouts in the records of hidden-criterion controls
    for r in recs:
        r["scores"] = {rid: r["scores"].get(rid, r["scores"]["probe_truth_d33"]) for rid in READOUT_ORDER}
    write_jsonl(fixed_sentence.run_path(tmp_path, MODEL, "min", "hidden", 0), recs)
    out = collect.collect_exp2(tmp_path, n_sentences=2, horizon=3)
    si, ri = EXP2_MODELS.index(MODEL), READOUT_ORDER.index("step_change_d66")
    real = next(r for r in recs if r["arm"] == "real" and r["readout"] == "step_change_d66" and r["sentence"] == 1)
    assert np.allclose(out["real"][si, ri, 1, 0, 1], real["scores"]["step_change_d66"])
    ctrl = [r["scores"]["step_change_d66"] for r in recs if r["arm"] == "control" and r["sentence"] == 1]
    assert np.allclose(out["control"][si, ri, 1, 0, 1], np.mean(ctrl, axis=0))


def test_collect_monitor(tmp_path):
    be = backend()
    mon = load_monitor(MODEL, "emotion")
    sents = read_jsonl(REPO / "data" / "monitor_sentences" / MODEL / "emotion.jsonl")[:2]
    specs = monitor.session_specs(sents, 3, lambda t, s: prompts.monitor_system("emotion", *t.split("_"), s["system"]))
    recs = monitor.run_fixed(be, mon, specs, horizon=2)
    write_jsonl(tmp_path / MODEL / "emotion.shard0.jsonl", recs)
    out = collect.collect_monitor(tmp_path, horizon=2, cot=False)
    assert out["real_milli"].shape == (4, 2, 2) and out["control_milli"].shape == (4, 2, 3, 2)
    assert out["unit_condition"].tolist() == ["hidden", "disclosed", "hidden", "disclosed"]
    first = next(r for r in recs if r["sentence"] == sents[0]["id"] and r["template"] == "disclosed_min"
                 and r["arm"] == "real")
    assert np.allclose(out["real_milli"][1, 1] / 1000, first["scores"])
