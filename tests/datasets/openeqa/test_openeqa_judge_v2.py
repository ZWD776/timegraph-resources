from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path


SCRIPT_DIR = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir()) / "scripts" / "openeqa"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

evaluator = import_module("scripts.datasets.openeqa.audit.evaluate_open_eqa_predictions")


def test_judge_v2_is_an_explicit_prompt_variant() -> None:
    args = evaluator.build_parser().parse_args(
        ["--predictions", "predictions.json", "--prompt-variant", "judge-v2"]
    )

    assert args.prompt_variant == "judge-v2"
    assert "judge-v2" in evaluator.PROMPT_VARIANTS


def test_judge_v2_is_leniency_only() -> None:
    prompt = evaluator.load_prompt("judge-v2")

    assert "leniency-only revision" in prompt
    assert "must not introduce a\nstricter requirement" in prompt
    assert "accept a short answer when it reasonably supplies" in prompt
    assert "omitted nonessential detail is allowed" in prompt
    assert "do not apply a special penalty to yes/no answers" in prompt
    assert "partial credit rather than being collapsed to the minimum" in prompt
    assert "a bare \"yes\" or \"no\" is ambiguous" not in prompt


def test_judge_prompt_hash_is_stable_for_identical_text() -> None:
    prompt = evaluator.load_prompt("judge-v2")

    assert evaluator.prompt_sha256(prompt) == evaluator.prompt_sha256(prompt)
    assert len(evaluator.prompt_sha256(prompt)) == 64


def test_judge_v2_messages_include_extra_acceptable_answers() -> None:
    messages = evaluator.build_messages(
        prompt=evaluator.load_prompt("judge-v2"),
        question="Where is the mirror?",
        gold_answer="Above the cabinet by the stairs",
        predicted_answer="By the stairs",
        extra=["Next to the staircase"],
    )

    assert messages[0]["role"] == "system"
    assert "Extra acceptable answers: Next to the staircase" in messages[1]["content"]
