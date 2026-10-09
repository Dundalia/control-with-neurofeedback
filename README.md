# Control with neurofeedback

Code and data for *Cross-Model In-Context Neurofeedback Reveals (Limited) Activation Control in LLMs* (Davide Baldelli, Damiano Fornasiere, Mirko Bronzi).

In an in-context neurofeedback session, a model is asked to maximize or minimize a score, and at every turn it is shown the score of its last reply: a *readout* of its own activations. Control sessions are identical except that the scores shown are random.

- **Experiment 1 (free generation).** A writer generates the sentences and a subject's activations are scored, for 5 writers x 5 subjects (each model also writing for itself), 42 readouts, 20 turns.
- **Experiment 2 (fixed sentence).** The subject's reply is a fixed sentence, teacher-forced at every turn; 10 models, 42 readouts, 30 turns.
- **Experiment 3 (monitors).** As Experiment 2, but the score is an activation monitor (truthfulness, deception, sentiment, emotion, morality, formality) on sentences it flags; 8 models, 20 turns.
- **Experiment 4 (monitors, chain of thought).** As Experiment 3, but the model reasons before the fixed sentence; 10 turns.

## Contents

```
neurofeedback/            the library
  models.py               the 10 models (Hugging Face ids, display names, how they write in Experiment 1)
  backend.py              vLLM engine: greedy generation and teacher-forced activation reads (vllm-lens)
  readouts.py             the 42 readouts
  monitors.py             the attention-probe monitors
  prompts.py              rendering of prompts/*.yaml
  writers.py              writers of Experiment 1 (the subject itself, OpenRouter, a vLLM server)
  experiments/            session loops: free_generation.py (1), fixed_sentence.py (2), monitor.py (3, 4)
  analysis.py             estimators and bootstraps of the paper
prompts/                  every model-facing string
scripts/
  run_exp{1,2,3,4}.py     run an experiment for one subject
  serve_olmo.sh           OLMo as a writer of Experiment 1 (it is not on OpenRouter)
  collect_results.py      session records -> results/
  figures/                the figures of the paper
  tables.py               the tables of the appendix
  check_monitor_compliance.py   the compliance test before Experiments 3 and 4
  build/                  how the readouts, the monitors and the sentence sets were built
data/                     sentence sets and datasets (see data/README.md)
artifacts/readouts/       the readouts of each model (directions, means, calibrations)
artifacts/monitors/       the monitors of each model
results/                  the processed results of the paper's runs (see below)
vendor/                   patch of vllm-lens for the prefix-cached reads of Experiment 2
tests/                    tests (CPU only)
```

## Installation

To reproduce the figures and tables from the released results (CPU, Python >= 3.10):

```bash
pip install -e .
```

To run the experiments and the build (Python 3.12, NVIDIA GPUs; the paper used one A100 80GB per model):

```bash
pip install -r requirements.txt
bash vendor/apply_vllm_lens_patch.sh      # only needed for the prefix-cached reads of Experiment 2
```

Experiment 1 calls OpenRouter: put `OPENROUTER_API_KEY` in the environment or in a `.env` file at the root of the repository. Model weights are downloaded from the Hugging Face Hub.

## Reproducing the figures and tables

```bash
python scripts/figures/exp1.py          # ~1 min
python scripts/figures/exp2.py
python scripts/figures/exp3_exp4.py
python scripts/tables.py
pytest                                  # includes checks against the numbers of the paper
```

| Paper | Output |
|---|---|
| Figure "Experiment 1 (free generation)" | `figures/exp1_results.pdf` |
| Figure "Experiment 2 (fixed sentence)" | `figures/exp2_results.pdf` |
| Figure "Experiments 3 and 4 (activation monitors)" | `figures/exp34_results.pdf` |
| Figure "Experiment 1, additive fit" | `figures/exp1_fit.pdf` |
| Figure "Experiment 1, every writer x subject pair" | `figures/exp1_grid.pdf` |
| Figure "Experiment 2, per model and objective" | `figures/exp2_effects.pdf` |
| Figure "Experiment 2, per readout" | `figures/exp2_readouts.pdf` |
| Figure "Experiments 3 and 4, per model and objective" | `figures/exp34_effects.pdf` |
| Figures "Experiment 3 / 4, per monitor" | `figures/exp3_monitors.pdf`, `figures/exp4_monitors.pdf` |
| Table "AUROC of the 48 monitors" | `tables/monitors.tex` |
| Tables "Compliance in Experiment 1", "Chain of thought in Experiment 4", "Compliance test" | `tables/compliance_*.tex` |

With matplotlib 3.10.9, the figures are identical to those of the paper, except for tiny shifts of the points of `exp1_fit.pdf`, as the results of Experiment 1 are stored in single precision, and the tick labels of `exp1_grid.pdf`, which now use the model names.

### The released results

`results/` holds, for every session of the paper's runs, the score at every turn:

- `exp1.npz`: `scores[subject, writer, readout, objective, condition, arm, session, turn]` (NaN where the reply had no readout), the compliance `status` of every turn (0 valid, 1 too short, 2 too long, 3 empty, 4 failed API call, 5 readout undefined) and whether a valid reply `repeat`ed the previous one;
- `exp2.npz`: `real` and `control[subject, readout, objective, condition, sentence, turn]`, the control being the mean of the control sessions of the sentence;
- `exp3.npz`, `exp4.npz`: one unit per (model, monitor, sentence, condition), with the real and the 3 control sessions of each objective, in milli-points; for Experiment 4 also whether each chain of thought finished within the budget and its length;
- `monitor_auroc.csv`, `monitor_preflight.jsonl`: the monitor evaluation and the compliance test.

Readouts and units are stored in the order in which the bootstraps of the paper resample them, so that the figures reproduce the paper's intervals exactly.

## Running the experiments

Each script loads one subject on one GPU, resumes where an interrupted run stopped, and accepts `--shard i --num-shards n` to split the work over several GPUs. Outputs are JSON lines, one per session.

```bash
# Experiment 1: the writers other than the subject are called through OpenRouter, and OLMo through a vLLM server
CUDA_VISIBLE_DEVICES=1 bash scripts/serve_olmo.sh &  export OLMO_URL=http://localhost:8000/v1
python scripts/run_exp1.py --subject qwen3.5-27b --out outputs/exp1
# Experiment 2
python scripts/run_exp2.py --subject qwen3.5-27b --out outputs/exp2
# Experiments 3 and 4
python scripts/run_exp3.py --subject qwen3.5-27b --out outputs/exp3
python scripts/run_exp4.py --subject qwen3.5-27b --out outputs/exp4
# then, for each experiment
python scripts/collect_results.py --exp 2 --outputs outputs/exp2
```

Models: `qwen3.5-27b`, `gemma-4-31b`, `gpt-oss-20b`, `mistral-small-3.2-24b`, `olmo-3.1-32b` (Experiments 1 and 2), `qwen3-32b`, `qwen3-14b`, `gemma-4-12b`, `qwen3.5-9b`, `qwen3.6-27b` (Experiment 2), and the 8 models with a native chain of thought for Experiments 3 and 4 (all but Mistral and OLMo).

Settings of the paper, set by default:

- All generations are greedy. Experiment 1 caps the subject's own replies at 64 tokens and the API writers' at 128.
- Experiment 2 reads with prefix caching (`vendor/`) for Qwen3.5-27B, GPT-OSS-20B, Mistral-Small-3.2-24B, OLMo-3.1-32B, Qwen3.5-9B and Qwen3.6-27B, and with full recomputation for the other four, after checking per subject that the two agree; Gemma-4-12B also ran with vLLM's prefix caching disabled (`run_exp2.py --read-mode`).
- Experiment 4 budgets 2048 reasoning tokens per turn; GPT-OSS-20B reasons with `reasoning_effort=medium`.

Greedy decoding with vLLM is not bitwise reproducible across batch compositions, and OpenRouter routes the same request to different providers (logged per turn), so a re-run reproduces the paper's sessions only up to this noise.

## Rebuilding the readouts, the monitors and the sentence sets

The released `artifacts/` and `data/` are what the runs used; `scripts/build/` documents how they were made and can rebuild them. Intermediates go to `build/` (several GB per model).

```bash
# shared (CPU, except the embedding)
python scripts/build/build_corpora.py reference          # data/c4_reference_256.jsonl
python scripts/build/concept_probes.py --prepare         # data/concept_probe_sentences.json
python scripts/build/build_corpora.py corpus             # data/c4_corpus_20k.jsonl
python scripts/build/embed_corpus.py                     # Qwen3-Embedding-8B embeddings (GPU)
# per model
python scripts/build/concept_probes.py --subject M       # GPU, then CPU fit
python scripts/build/read_corpus.py --subject M          # GPU, a few hours
python scripts/build/encoder_residual.py --subject M     # CPU
python scripts/build/readouts.py --stage directions --subject M
python scripts/build/calibration_reads.py --subject M    # GPU
python scripts/build/readouts.py --stage finalize --subject M     # -> artifacts/readouts/M.npz
python scripts/build/calibrate_exp2.py --subject M       # GPU -> artifacts/readouts/M.exp2_calibration.npz
# sentences of Experiments 1 and 2, from the five models of Experiment 1
python scripts/build/select_sentences.py
# monitors and their sentences (GPU)
python scripts/build/train_monitors.py --subject M
python scripts/build/evaluate_monitors.py --subject M
python scripts/build/select_monitor_sentences.py --subject M
```

Some steps of the original build are only partly reproducible:

- Qwen3.5-27B, Gemma-4-31B and GPT-OSS-20B were read without the turn-20 log-probabilities (`read_corpus.py --logprob-depths 0 5`), so their encoder-residual directions were screened without the log-probability criterion.
- For the same three models, the diagnostics of how predictable each geometry readout is from the text (`pub_*` in the readouts, used only to select the sentences of Experiments 1 and 2) came from earlier exploratory runs with the same formulas; rebuilding them with `calibration_reads.py` gives values that differ at the level of vLLM's numerical noise.
- As in the original build, the PCA of the corpus reads accumulates its Gram matrix in single precision, so the low-variance end of the spectrum depends on the BLAS library and its thread count, and the random directions of the band [1000, 3000) can come out different. For Qwen3.5-9B at d33, 1, 2 and 4 threads (`OMP_NUM_THREADS`) reproduce the released directions on our machines, and 3 threads selects two other candidates.
- The monitors are trained with early stopping on a validation AUROC, so vLLM's numerical noise in the reads changes the trained probes slightly, and can change the selected layer when two layers are close: retraining the formality monitor of Qwen3.5-9B selects layer 22 instead of 25, with per-layer AUROCs within 0.013 of the released ones.
- The last cut of the monitor sentences (the N sentences whose score at the first turn of Experiment 4 is closest to 60) was done by hand; `select_monitor_sentences.py` implements the rule, with N taken from the released pools, but not the order of the sentences.

## Data

See `data/README.md` for the sources of the sentence sets and datasets.

## License

The code is released under the MIT license (`LICENSE`). The results (`results/`) and the readouts and monitors (`artifacts/`) are released under CC BY 4.0. The sentences in `data/` keep the licenses of their sources (see `data/README.md`).

## Citation

The citation will be added upon publication.
