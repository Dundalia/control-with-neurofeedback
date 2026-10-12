# Data

| File | Content | Built by |
|---|---|---|
| `c4_reference_256.jsonl` | 256 low-entity declarative sentences from C4 (English), the reference set on which every readout is calibrated | `scripts/build/build_corpora.py reference` |
| `c4_corpus_20k.jsonl` | 20,000 C4 sentences (at most one per document), on which the readout directions are built and the corpus means computed | `scripts/build/build_corpora.py corpus` |
| `concept_probe_sentences.json` | 600 balanced sentences per concept for the concept probes: SST-2 (sentiment), ETHICS commonsense (morality), Azaria & Mitchell true/false statements (truthfulness) | `scripts/build/concept_probes.py --prepare` |
| `exp1_opening_sentences.jsonl` | the 20 opening sentences of Experiment 1 (`rank` = session index), from the C4 corpus | `scripts/build/select_sentences.py` |
| `exp2_fixed_sentences.jsonl` | the 40 fixed sentences of Experiment 2 (the 20 of Experiment 1 and 20 more) | `scripts/build/select_sentences.py` |
| `monitor_sentences/<model>/<monitor>.jsonl` | the fixed sentences of Experiments 3 and 4, with their scenario and question, and the scores used to select them | `scripts/build/select_monitor_sentences.py` |
| `monitor_preflight_sentences.json` | the 10 flagged sentences per monitor of the compliance test | (fixed) |
| `monitor_training/` | deception data from the Apollo Research [`deception-detection`](https://github.com/ApolloResearch/deception-detection) repository (commit [`f8ec4010`](https://github.com/ApolloResearch/deception-detection/tree/f8ec4010)): RepE true/false facts, AI-Liar goal-directed lying, and roleplaying dialogues graded by deceptiveness (>= 5 deceptive, <= 3 honest, 4 dropped) | (vendored) |

The other datasets of the monitors are downloaded from the Hugging Face Hub by `scripts/build/train_monitors.py`: [`pminervini/true-false`](https://huggingface.co/datasets/pminervini/true-false), [`copenlu/fever_gold_evidence`](https://huggingface.co/datasets/copenlu/fever_gold_evidence), [`fancyzhx/yelp_polarity`](https://huggingface.co/datasets/fancyzhx/yelp_polarity), [`cornell-movie-review-data/rotten_tomatoes`](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes), [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion), [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions), [`iproskurina/hendrycks_ethics_commonsense`](https://huggingface.co/datasets/iproskurina/hendrycks_ethics_commonsense), [`LabHC/moral_stories`](https://huggingface.co/datasets/LabHC/moral_stories), [`osyvokon/pavlick-formality-scores`](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores).

## Licenses

The sentences in this directory are taken from the datasets below and keep the license of their source (the MIT license of this repository does not apply to them).

| Sentences | Source | License |
|---|---|---|
| `c4_*.jsonl`, `exp1_opening_sentences.jsonl`, `exp2_fixed_sentences.jsonl` | [C4](https://huggingface.co/datasets/allenai/c4) (`allenai/c4`) | ODC-BY |
| `concept_probe_sentences.json`, sentiment | [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) | none stated |
| `concept_probe_sentences.json`, morality | [ETHICS](https://huggingface.co/datasets/hendrycks/ethics) | MIT |
| `concept_probe_sentences.json`, truthfulness | [Azaria & Mitchell true/false statements](https://huggingface.co/datasets/pminervini/true-false) | none stated |
| `monitor_sentences/*/truth.jsonl` | [FEVER](https://huggingface.co/datasets/copenlu/fever_gold_evidence) | CC BY-SA 3.0 |
| `monitor_sentences/*/deception.jsonl` | [AI-Liar](https://github.com/LoryPack/LLM-LieDetector) (Pacchiardi et al.) and the Apollo Research [roleplaying dialogues](https://github.com/ApolloResearch/deception-detection) | BSD-3-Clause (AI-Liar), none stated (roleplaying) |
| `monitor_sentences/*/sentiment.jsonl` | [Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) (Pang & Lee) | none stated |
| `monitor_sentences/*/emotion.jsonl` | [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Apache 2.0 |
| `monitor_sentences/*/morality.jsonl` | [Moral Stories](https://huggingface.co/datasets/LabHC/moral_stories) | MIT |
| `monitor_sentences/*/formality.jsonl` | [Pavlick & Tetreault formality scores](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores) | CC BY 3.0 |
| `monitor_preflight_sentences.json` | the monitor sentences above | as above |
| `monitor_training/repe_true_false_facts.csv` | [RepE](https://github.com/andyzoujm/representation-engineering) (Zou et al.) | MIT |
| `monitor_training/ai_liar_goal_directed.json` | [AI-Liar](https://github.com/LoryPack/LLM-LieDetector) (Pacchiardi et al.) | BSD-3-Clause |
| `monitor_training/roleplaying.jsonl` | Apollo Research [`deception-detection`](https://github.com/ApolloResearch/deception-detection) | none stated |
