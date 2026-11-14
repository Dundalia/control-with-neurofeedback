"""The four session loops, on the fake engine."""
import json
import random

import numpy as np
import pytest

from conftest import REPO
from fakes import fake_backend
from neurofeedback import prompts
from neurofeedback.experiments import fixed_sentence, free_generation, monitor
from neurofeedback.io import load_sentences, read_jsonl
from neurofeedback.monitors import load_monitor
from neurofeedback.readouts import ReadoutSet

MODEL = "qwen3.5-9b"
READOUTS = ["probe_truth_d33", "encres_pc1_d66", "curvature_step3_d33", "curvature_diff_66_33"]


@pytest.fixture
def backend():
    rs = ReadoutSet(MODEL)
    return fake_backend(num_layers=rs.num_layers, hidden=int(rs.readouts["probe_truth_d33"].mu.shape[0]))


def test_prefix_cached_reads_use_one_sampling_params_per_request(backend):
    backend.prefix_cache_reads = True
    items = [([{"role": "user", "content": f"q{i}"}], "an answer") for i in range(3)]
    backend.read_residual_multi_batch(items, [3, 5], allow_prefix_cache=True)
    sps = backend.llm.calls[-1]
    assert len({id(s) for s in sps}) == 3 and all(s.extra_args["span_prefix_cache"] for s in sps)
    backend.read_residual_multi_batch(items, [3, 5])
    assert "span_prefix_cache" not in backend.llm.calls[-1][0].extra_args


def test_exp2_run_and_resume(backend, tmp_path):
    rs = ReadoutSet(MODEL, "exp2", readout_ids=READOUTS)
    rs.check_backend(backend)
    sentences = load_sentences(REPO / "data" / "exp2_fixed_sentences.jsonl")
    cfg = fixed_sentence.Exp2Config(horizon=3, n_sentences=2, n_control_hidden=2, n_control_disclosed=1, chunk=4)
    for cond, n_ctrl in (("hidden", 2 * 2), ("disclosed", 2 * len(READOUTS))):
        fixed_sentence.run(subject=MODEL, objective="max", condition=cond, sentences=sentences, backend=backend,
                           readouts=rs, out_root=tmp_path, cfg=cfg)
        recs = read_jsonl(fixed_sentence.run_path(tmp_path, MODEL, "max", cond, 0))
        real = [r for r in recs if r["arm"] == "real"]
        ctrl = [r for r in recs if r["arm"] == "control"]
        assert len(real) == 2 * len(READOUTS) and len(ctrl) == n_ctrl
        for r in real:
            assert r["shown"] == [int(round(x)) for x in r["scores"][r["readout"]]]
        S = 2 if cond == "hidden" else 1
        for r in ctrl:
            rng = random.Random(r["sentence"] * S + r["stream"])
            assert r["shown"] == [rng.randint(0, 100) for _ in range(3)]
            assert (r["readout"] is None) == (cond == "hidden")
        fixed_sentence.run(subject=MODEL, objective="max", condition=cond, sentences=sentences, backend=backend,
                           readouts=rs, out_root=tmp_path, cfg=cfg)
        assert len(read_jsonl(fixed_sentence.run_path(tmp_path, MODEL, "max", cond, 0))) == len(recs)


def test_shuffled_plans_permute_the_displayed_numbers():
    shown = {("a", 0): list(range(10)), ("b", 1): [5, 5, 7, 1, 9, 2, 2, 8, 0, 4]}
    plans = fixed_sentence.shuffled_plans(shown, condition="hidden", n_shuffled=3)
    assert len(plans) == 6
    for (rid, sent, k), plan in plans.items():
        seq = shown[(rid, sent)]
        assert sorted(plan[:-1]) == sorted(seq[:-1]) and plan[-1] == seq[-1]
    assert plans[("a", 0, 0)] != plans[("a", 0, 1)] != list(range(10))
    again = fixed_sentence.shuffled_plans({("a", 0): [x + 100 for x in range(10)]}, condition="hidden", n_shuffled=3)
    assert [x - 100 for x in again[("a", 0, 2)]] == plans[("a", 0, 2)]
    other = fixed_sentence.shuffled_plans(shown, condition="disclosed", n_shuffled=3)
    assert other[("a", 0, 0)] != plans[("a", 0, 0)]


def test_exp2_planned_arms(backend, tmp_path):
    """Shuffled, crossed and replay sessions show their plans; a replay reproduces the real session."""
    rs = ReadoutSet(MODEL, "exp2", readout_ids=READOUTS)
    rs.check_backend(backend)
    sentences = load_sentences(REPO / "data" / "exp2_fixed_sentences.jsonl")
    cfg = fixed_sentence.Exp2Config(horizon=4, n_sentences=2, n_control_hidden=2, n_control_disclosed=1, chunk=5,
                                    n_shuffled=2)
    for cond in ("hidden", "disclosed"):
        kw = dict(subject=MODEL, objective="min", condition=cond, sentences=sentences, backend=backend, readouts=rs,
                  cfg=cfg)
        fixed_sentence.run(out_root=tmp_path / "real", arms=("real",), **kw)
        real = {(r["readout"], r["sentence"]): r
                for r in read_jsonl(fixed_sentence.run_path(tmp_path / "real", MODEL, "min", cond, 0))}
        assert len(real) == 2 * len(READOUTS)
        shown = {k: r["shown"] for k, r in real.items()}

        fixed_sentence.run(out_root=tmp_path / "replay", arms=("replay",),
                           plans={"replay": fixed_sentence.replay_plans(shown)}, **kw)
        replayed = read_jsonl(fixed_sentence.run_path(tmp_path / "replay", MODEL, "min", cond, 0))
        assert len(replayed) == len(real) and {r["arm"] for r in replayed} == {"replay"}
        for r in replayed:
            assert r["stream"] == 0 and r["scores"] == real[(r["readout"], r["sentence"])]["scores"]

        plans = fixed_sentence.shuffled_plans(shown, condition=cond, n_shuffled=2)
        fixed_sentence.run(out_root=tmp_path / "shuf", arms=("shuffled",), plans={"shuffled": plans}, **kw)
        path = fixed_sentence.run_path(tmp_path / "shuf", MODEL, "min", cond, 0)
        recs = read_jsonl(path)
        assert len(recs) == 2 * 2 * len(READOUTS) and {r["arm"] for r in recs} == {"shuffled"}
        for r in recs:
            assert r["shown"] == plans[(r["readout"], r["sentence"], r["stream"])]
            assert sorted(r["shown"]) == sorted(shown[(r["readout"], r["sentence"])])
            first = real[(r["readout"], r["sentence"])]["scores"]
            assert all(r["scores"][rid][0] == first[rid][0] for rid in READOUTS)
        fixed_sentence.run(out_root=tmp_path / "shuf", arms=("shuffled",), plans={"shuffled": plans}, **kw)
        assert len(read_jsonl(path)) == len(recs)

        other = {(rid, sent): [int(x) + 7 for x in seq] for (rid, sent), seq in shown.items()}
        cross = fixed_sentence.shuffled_plans(other, condition=cond, n_shuffled=2)
        fixed_sentence.run(out_root=tmp_path / "shuf", arms=("shuffled", "crossed"),
                           plans={"shuffled": plans, "crossed": cross}, **kw)
        recs = read_jsonl(path)
        crossed = [r for r in recs if r["arm"] == "crossed"]
        assert len(crossed) == 2 * 2 * len(READOUTS) and len(recs) == 2 * len(crossed)
        for r in crossed:
            key = (r["readout"], r["sentence"], r["stream"])
            assert r["shown"] == cross[key] == [x + 7 for x in plans[key]]
    with pytest.raises(ValueError):
        fixed_sentence.run(out_root=tmp_path / "x", arms=("crossed",), plans={"shuffled": plans}, **kw)


def test_exp2_depth_ablation():
    """The readouts at the extra depths: their anchors are read, and only theirs."""
    from neurofeedback.readouts import DEPTH_READOUTS_DIR, depth_readout_ids
    subject = "qwen3.5-27b"
    ids = depth_readout_ids(["d45", "d85"])
    rs = ReadoutSet(subject, "exp2", readouts_dir=DEPTH_READOUTS_DIR, readout_ids=ids)
    assert rs.ids == ids and len(ids) == 30
    be = fake_backend(num_layers=rs.num_layers, hidden=int(rs.readouts[ids[0]].mu.shape[0]))
    rs.check_backend(be)
    paper = ReadoutSet(subject, "exp2")
    assert rs.read_layers == sorted(set(paper.read_layers) | {rs.layers["d45"], rs.layers["d85"]})
    sentences = load_sentences(REPO / "data" / "exp2_fixed_sentences.jsonl")
    cfg = fixed_sentence.Exp2Config(horizon=2, n_sentences=1, n_control_hidden=2, chunk=40)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        fixed_sentence.run(subject=subject, objective="max", condition="hidden", sentences=sentences, backend=be,
                           readouts=rs, out_root=tmp, cfg=cfg)
        recs = read_jsonl(fixed_sentence.run_path(tmp, subject, "max", "hidden", 0))
    assert len(recs) == 30 + 2 and all(set(r["scores"]) == set(ids) for r in recs)
    assert sorted(be.llm.calls[-1][0].extra_args["output_residual_stream"]) == rs.read_layers
    for r in recs:
        if r["arm"] == "real":
            assert r["shown"] == [int(round(x)) for x in r["scores"][r["readout"]]]


class FakeAPIWriter:
    def __init__(self, key):
        self.key = key

    async def write(self, conversation):
        n = len(conversation)
        if n == 5:
            raise RuntimeError("API failure")
        return " ".join(["word"] * (8 + n)), {"provider": "Fake", "completion_tokens": 9, "finish_reason": "stop"}

    def describe(self):
        return {"type": "fake", "model": self.key}


def test_exp1_cell(backend, tmp_path, monkeypatch):
    monkeypatch.setattr(free_generation, "api_writer", lambda key, max_tokens: FakeAPIWriter(key))
    rs = ReadoutSet(MODEL, readout_ids=READOUTS)
    opening = [s["text"] for s in load_sentences(REPO / "data" / "exp1_opening_sentences.jsonl")]
    cfg = free_generation.Exp1Config(horizon=5, n_sessions=3)
    writers = [MODEL, "gemma-4-31b"]
    free_generation.run_cell(subject=MODEL, readout_id="curvature_step3_d33", objective="min", condition="disclosed",
                             opening=opening, backend=backend, readouts=rs, out_root=tmp_path, cfg=cfg, writers=writers)
    assert free_generation.cell_done(tmp_path, MODEL, "curvature_step3_d33", "min", "disclosed", writers)
    for w in writers:
        for arm in ("real", "control"):
            recs = free_generation.load_cell(tmp_path, MODEL, "curvature_step3_d33", "min", "disclosed", w, arm)
            assert [r["session"] for r in recs] == [0, 1, 2]
            for r in recs:
                assert r["turns"][0]["text"] == opening[r["session"]]
                rng = random.Random(r["session"])
                for t in r["turns"]:
                    valid = t["invalid"] is None
                    assert valid == (t["score"] is not None) == (t["shown"] is not None)
                    if t["n_words"] and not 10 <= t["n_words"] <= 20 and t is not r["turns"][0]:
                        assert t["invalid"] == "word_band"
                    if valid and arm == "control":
                        assert t["shown"] == rng.randint(0, 100)
                    if valid and arm == "real":
                        assert t["shown"] == int(round(t["score"]))
                if w != MODEL:
                    assert r["turns"][2]["api_error"] and r["turns"][2]["invalid"] == "word_band"
    assert "Qwen3.5-9B's internal activations" in recs[0]["instructions"]


def test_exp3_and_exp4(backend):
    mon = load_monitor(MODEL, "truth")
    sents = read_jsonl(REPO / "data" / "monitor_sentences" / MODEL / "truth.jsonl")[:2]
    specs = monitor.session_specs(sents, 2, lambda t, s: prompts.monitor_system("truth", *t.split("_"), s["system"]))
    recs = monitor.run_fixed(backend, mon, specs, horizon=3, sub_batch=5)
    assert len(recs) == 2 * 4 * 3
    for r in recs:
        if r["arm"] == "real":
            assert r["shown"] == [int(round(x)) for x in r["scores"]]
        else:
            rng = np.random.default_rng(r["sentence"] * 131 + r["stream"] + 1)
            assert r["shown"] == [int(round(float(rng.integers(0, 101)))) for _ in range(3)]
    specs = monitor.session_specs(sents, 1, lambda t, s: prompts.monitor_cot_system(
        "truth", *t.split("_"), s["system"], s["answer"]))
    recs = monitor.run_cot(backend, mon, specs, horizon=2, gen_batch=3, read_batch=2)
    assert len(recs) == 2 * 4 * 2
    assert all(len(r["cots"]) == len(r["cot_closed"]) == len(r["scores"]) == 2 for r in recs)


def test_writer_cleaning():
    from neurofeedback.writers import clean_reply
    assert clean_reply('  "A quoted  reply."  ') == "A quoted reply."
    assert clean_reply("<think>x</think>Answer here.") == "Answer here."
    assert clean_reply("Answer.<|end|><|start|>assistant") == "Answer."


def test_exp1_ffill_effects_are_paired():
    from neurofeedback.analysis import ffill
    x = np.array([[1.0, np.nan, 3.0, np.nan], [np.nan, 2.0, np.nan, np.nan]])
    assert np.array_equal(ffill(x), np.array([[1, 1, 3, 3], [np.nan, 2, 2, 2]]), equal_nan=True)
    assert json.dumps({"a": None})
