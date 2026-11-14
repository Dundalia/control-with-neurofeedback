"""scripts/collect_results.py turns session records into the arrays of results/."""
import importlib.util

import numpy as np
import pytest

from conftest import REPO
from fakes import fake_backend
from neurofeedback import prompts
from neurofeedback.analysis import READOUT_ORDER
from neurofeedback.experiments import fixed_sentence, free_generation, monitor
from neurofeedback.io import load_sentences, read_jsonl, write_jsonl
from neurofeedback.models import EXP1_MODELS, EXP2_MODELS
from neurofeedback.monitors import load_monitor
from neurofeedback.readouts import ReadoutSet

spec = importlib.util.spec_from_file_location("collect_results", REPO / "scripts" / "collect_results.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)
MODEL = "qwen3.5-9b"


def backend(model=MODEL):
    rs = ReadoutSet(model)
    return fake_backend(num_layers=rs.num_layers, hidden=int(rs.readouts["probe_truth_d33"].mu.shape[0]))


def same_format(out, name):
    """The collector writes the keys and dtypes of the released results/<name>.npz (strings may
    differ in width)."""
    released = np.load(REPO / "results" / f"{name}.npz")
    return set(out) == set(released.files) and all(
        out[k].dtype == released[k].dtype or out[k].dtype.kind == released[k].dtype.kind == "U" for k in out)


class FakeWriter:
    async def write(self, conversation):
        return " ".join(["word"] * 12), {"provider": "Fake", "completion_tokens": 12, "finish_reason": "stop"}

    def describe(self):
        return {"type": "fake"}


def test_collect_exp1(tmp_path, monkeypatch):
    subject, writer, rid = "qwen3.5-27b", "gemma-4-31b", "curvature_step3_d33"
    monkeypatch.setattr(free_generation, "api_writer", lambda key, max_tokens: FakeWriter())
    opening = [s["text"] for s in load_sentences(REPO / "data" / "exp1_opening_sentences.jsonl")]
    free_generation.run_cell(subject=subject, readout_id=rid, objective="min", condition="disclosed",
                             opening=opening, backend=backend(subject), readouts=ReadoutSet(subject, readout_ids=[rid]),
                             out_root=tmp_path, cfg=free_generation.Exp1Config(horizon=4, n_sessions=2),
                             writers=[subject, writer])
    out = collect.collect_exp1(tmp_path, n_sessions=2, horizon=4)
    assert same_format(out, "exp1")
    for w in (subject, writer):
        idx = (EXP1_MODELS.index(subject), EXP1_MODELS.index(w), READOUT_ORDER.index(rid), 1, 1)
        for arm in ("real", "control"):
            for r in free_generation.load_cell(tmp_path, subject, rid, "min", "disclosed", w, arm):
                scores = [np.nan if t["score"] is None else t["score"] for t in r["turns"]]
                assert np.allclose(out[arm][idx + (r["session"],)], scores, equal_nan=True)
                assert out[f"status_{arm}"][idx + (r["session"],)].tolist() == [collect.turn_status(t) for t in r["turns"]]


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
    assert same_format(out, "exp2")
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
    assert same_format(out, "exp3")
    assert out["real"].shape == (4, 2, 2) and out["control"].shape == (4, 2, 3, 2)
    assert out["real"].dtype == out["control"].dtype == np.float32
    assert out["unit_condition"].tolist() == ["hidden", "disclosed", "hidden", "disclosed"]
    first = next(r for r in recs if r["sentence"] == sents[0]["id"] and r["template"] == "disclosed_min"
                 and r["arm"] == "real")
    assert np.allclose(out["real"][1, 1], first["scores"])


def _exp2_sessions(tmp_path, arms, conditions, objectives=("max", "min"), readouts=("probe_truth_d33", "step_change_d66"),
                   n_sentences=2, horizon=3, plans=None):
    be = backend()
    rs = ReadoutSet(MODEL, "exp2", readout_ids=list(readouts))
    sentences = load_sentences(REPO / "data" / "exp2_fixed_sentences.jsonl")
    cfg = fixed_sentence.Exp2Config(horizon=horizon, n_sentences=n_sentences, n_control_hidden=2, n_control_disclosed=1,
                                    n_shuffled=2)
    for o in objectives:
        for c in conditions:
            fixed_sentence.run(subject=MODEL, objective=o, condition=c, sentences=sentences, backend=be, readouts=rs,
                               out_root=tmp_path, cfg=cfg, arms=arms, plans=plans(o, c) if plans else None)
    return [r for o in objectives for c in conditions
            for r in read_jsonl(fixed_sentence.run_path(tmp_path, MODEL, o, c, 0))]


def test_collect_exp2_shuffled(tmp_path):
    def plans(o, c):
        shown = {(rid, s): [10 * s + t for t in range(3)] for rid in ("probe_truth_d33", "step_change_d66") for s in (0, 1)}
        return {a: fixed_sentence.shuffled_plans(shown, condition=c, n_shuffled=2) for a in ("shuffled", "crossed")}
    recs = _exp2_sessions(tmp_path, ("shuffled", "crossed"), ("hidden", "disclosed"), plans=plans)
    out = collect.collect_exp2_shuffled([tmp_path], subjects=[MODEL], n_sentences=2, k_shuffled=2, turn=3)
    assert same_format(out, "exp2_shuffled")
    assert out["shuffled"].shape == (1, 42, 2, 2, 2, 2)
    for r in recs:
        idx = (0, READOUT_ORDER.index(r["readout"]), ["max", "min"].index(r["objective"]),
               ["hidden", "disclosed"].index(r["condition"]), r["sentence"], r["stream"])
        assert out[r["arm"]][idx] == r["scores"][r["readout"]][2]
    with pytest.raises(ValueError):                 # a session recorded twice must be identical
        bad = dict(recs[0], scores={k: [v[0] + 1] + v[1:] for k, v in recs[0]["scores"].items()})
        write_jsonl(tmp_path / "copy" / MODEL / "x.jsonl", [bad])
        collect.collect_exp2_shuffled([tmp_path, tmp_path / "copy"], subjects=[MODEL], n_sentences=2, k_shuffled=2,
                                      turn=3)


def test_collect_exp2_targeted(tmp_path):
    def plans(o, c):
        return {"replay": fixed_sentence.replay_plans(
            {(rid, s): [3 * s + t for t in range(3)] for rid in ("probe_truth_d33", "step_change_d66") for s in (0, 1)})}
    recs = _exp2_sessions(tmp_path, ("control", "replay"), ("hidden",), plans=plans)
    ids = [str(x) for x in np.load(REPO / "artifacts" / "readouts" / f"{MODEL}.npz")["readout_ids"]]
    for r in recs:                                  # the collector reads all 42 readouts of every session
        r["scores"] = {rid: r["scores"].get(rid, r["scores"]["probe_truth_d33"]) for rid in ids}
    write_jsonl(tmp_path / MODEL / "all.jsonl", recs)
    for f in (tmp_path / MODEL).glob("*.shard0.jsonl"):
        f.unlink()
    out = collect.collect_exp2_targeted([tmp_path], subjects=[MODEL], n_sentences=2, n_control=2, horizon=3)
    assert same_format(out, "exp2_targeted")
    x, ri = ids.index("step_change_d66"), ids.index("probe_truth_d33")
    rep = next(r for r in recs if r["arm"] == "replay" and r["readout"] == "step_change_d66" and r["sentence"] == 1
               and r["objective"] == "min")
    assert out["real"][0, x, ri, 1, 1] == rep["scores"]["probe_truth_d33"][2]
    ctrl = [r["scores"]["step_change_d66"][2] for r in recs if r["arm"] == "control" and r["sentence"] == 0
            and r["objective"] == "max"]
    assert np.isclose(out["control"][0, x, 0, 0], np.mean(ctrl))
    own = [r["scores"]["step_change_d66"][:2] for r in recs if r["arm"] == "replay" and r["readout"] == "step_change_d66"]
    assert np.isclose(out["level"][0, x], np.mean(own))
    assert np.isnan(out["real"][0, ids.index("encres_pc1_d33")]).all()


def test_collect_exp2_depths(tmp_path):
    from neurofeedback.readouts import DEPTH_READOUTS_DIR, depth_readout_ids
    subject, ids = "qwen3.5-27b", depth_readout_ids()
    rs = ReadoutSet(subject, "exp2", readouts_dir=DEPTH_READOUTS_DIR, readout_ids=ids[:2])
    be = fake_backend(num_layers=rs.num_layers, hidden=int(rs.readouts[ids[0]].mu.shape[0]))
    sentences = load_sentences(REPO / "data" / "exp2_fixed_sentences.jsonl")
    cfg = fixed_sentence.Exp2Config(horizon=3, n_sentences=1, n_control_hidden=2, n_shuffled=2)
    for o in ("max", "min"):
        fixed_sentence.run(subject=subject, objective=o, condition="hidden", sentences=sentences, backend=be,
                           readouts=rs, out_root=tmp_path / "real", cfg=cfg)
    recs = [r for o in ("max", "min") for r in read_jsonl(fixed_sentence.run_path(tmp_path / "real", subject, o,
                                                                                  "hidden", 0))]
    for r in recs:                                  # the control sessions are read with all 75 readouts
        r["scores"] = {rid: r["scores"].get(rid, r["scores"][ids[0]]) for rid in ids}
    write_jsonl(tmp_path / "real" / subject / "all.jsonl", recs)
    for f in (tmp_path / "real" / subject).glob("*.shard0.jsonl"):
        f.unlink()
    shown = {o: {(r["readout"], 0): r["shown"] for r in recs if r["arm"] == "real" and r["objective"] == o}
             for o in ("max", "min")}
    for o, other in (("max", "min"), ("min", "max")):
        fixed_sentence.run(subject=subject, objective=o, condition="hidden", sentences=sentences, backend=be,
                           readouts=rs, out_root=tmp_path / "shuffled", cfg=cfg, arms=("shuffled", "crossed"),
                           plans={"shuffled": fixed_sentence.shuffled_plans(shown[o], condition="hidden", n_shuffled=2),
                                  "crossed": fixed_sentence.shuffled_plans(shown[other], condition="hidden",
                                                                           n_shuffled=2)})
    out = collect.collect_exp2_depths([tmp_path / "real", tmp_path / "shuffled"], subjects=[subject], n_sentences=1,
                                      n_control=2, k_shuffled=2, horizon=3)
    assert same_format(out, "exp2_depths")
    real = next(r for r in recs if r["arm"] == "real" and r["readout"] == ids[1] and r["objective"] == "min")
    assert np.allclose(out["real"][0, 1, 1, 0, 0], real["scores"][ids[1]])
    assert np.isfinite(out["shuffled"][0, :2]).all() and np.isnan(out["shuffled"][0, 2:]).all()
    assert np.isfinite(out["control"][0]).all()
