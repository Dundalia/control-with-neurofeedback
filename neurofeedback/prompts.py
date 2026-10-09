"""Rendering of the prompts in ``prompts/``."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
CONDITIONS = ("hidden", "disclosed")
OBJECTIVES = ("max", "min")
_PLACEHOLDER = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
# Removed from the monitor instructions in Experiment 4, where cot_answer_line replaces it.
_KEEP_CLAUSE = re.compile(r"\s*Keep (?:asserting the same statement|giving the human the same "
                          r"answer), word for word\.\s*")


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    with open(PROMPTS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def _filled(text: str) -> str:
    leftover = _PLACEHOLDER.findall(text)
    if leftover:
        raise ValueError(f"unfilled placeholders {leftover} in prompt: {text[:80]}...")
    return text


def _check(condition, objective):
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective!r}")


def feedback(score: int) -> str:
    return _filled(_load("neurofeedback")["feedback"].format(score=score))


def invalid_reply() -> str:
    return _load("neurofeedback")["invalid_reply"]


def readout_description(readout_id: str) -> str:
    return _load("neurofeedback")["readout_descriptions"][readout_id]


def build_instructions(objective: str) -> str:
    """Instructions in whose context the readouts were built (``scripts/build``)."""
    p = _load("neurofeedback")
    return _filled(p["build_context"].format(objective_line=p["objective_line"][objective]))


def exp1_instructions(condition: str, objective: str, subject_name: str, readout_id: str) -> str:
    """First user message of an Experiment 1 session."""
    _check(condition, objective)
    p = _load("neurofeedback")
    fields = {"objective_line": p["objective_line"][objective]}
    if condition == "disclosed":
        fields.update(subject=subject_name, readout_description=readout_description(readout_id))
    return _filled(p["exp1"][condition].format(**fields))


def exp2_instructions(condition: str, objective: str, fixed_sentence: str, readout_id: str) -> str:
    """First user message of an Experiment 2 session."""
    _check(condition, objective)
    if not fixed_sentence.strip():
        raise ValueError("fixed_sentence must be non-empty")
    p = _load("neurofeedback")
    fields = {"objective_line": p["objective_line"][objective], "fixed_sentence": fixed_sentence}
    if condition == "disclosed":
        fields["readout_description"] = readout_description(readout_id)
    return _filled(p["exp2"][condition].format(**fields))


def monitor_system(concept: str, condition: str, objective: str, scenario: str) -> str:
    """System prompt of an Experiment 3 session: the scenario, then the instructions."""
    _check(condition, objective)
    para = _load("monitors")["instructions"][concept][f"{condition}_{objective}"]
    return scenario.rstrip() + " " + para


def monitor_cot_system(concept: str, condition: str, objective: str, scenario: str,
                       answer: str) -> str:
    """System prompt of an Experiment 4 session: the scenario, the instructions without the
    "keep asserting" sentence, and the line asking to reply with the fixed answer."""
    _check(condition, objective)
    p = _load("monitors")
    para = " ".join(_KEEP_CLAUSE.sub(" ", p["instructions"][concept][f"{condition}_{objective}"]).split())
    return scenario.rstrip() + " " + para + " " + p["cot_answer_line"].replace("{answer}", answer)
