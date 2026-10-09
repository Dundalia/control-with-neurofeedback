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
