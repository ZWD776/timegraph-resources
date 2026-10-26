from __future__ import annotations

from pathlib import Path

import pytest

from vlkg.datasets.navqa.retrieval.synonym_lexicon import (
    SYNONYM_LEXICON_CONTRACT,
    expand_question_ir_with_synonyms,
    load_synonym_lexicon,
    synonym_lexicon_from_payload,
)
from vlkg.datasets.navqa.retrieval.visual_entity_candidate_packs import visual_entity_pack_query_phrase_records
from vlkg.datasets.navqa.retrieval.visual_faiss_grounding import _visual_query_phrase_records


REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file() and (p / 'registry').is_dir())


def _payload() -> dict:
    return {
        "contract": SYNONYM_LEXICON_CONTRACT,
        "entries": [
            {
                "canonical": "couch",
                "aliases": ["couches", "sofa", "seating"],
                "level": "L0",
                "roles": ["target"],
                "source": "general_language",
                "source_evidence": ["general language"],
                "uses_gold_answer": False,
            },
            {
                "canonical": "building",
                "aliases": ["entrance", "entryway"],
                "level": "L1",
                "roles": ["constraint"],
                "source": "hsgg_cg_observed",
                "source_evidence": ["observed HSGG/CG context wording"],
                "uses_gold_answer": False,
            },
            {
                "canonical": "art installation",
                "aliases": ["sculpture", "artwork"],
                "level": "L0",
                "roles": ["target"],
                "source": "general_language",
                "source_evidence": ["general language"],
                "uses_gold_answer": False,
            },
        ],
    }


def test_synonym_lexicon_rejects_gold_answer_entries() -> None:
    payload = _payload()
    payload["entries"][0]["uses_gold_answer"] = True

    with pytest.raises(ValueError, match="uses gold answers"):
        synonym_lexicon_from_payload(payload)


def test_synonym_lexicon_requires_source_evidence() -> None:
    payload = _payload()
    payload["entries"][0]["source_evidence"] = []

    with pytest.raises(ValueError, match="missing source_evidence"):
        synonym_lexicon_from_payload(payload)


def test_expand_question_ir_keeps_target_aliases_separate_from_constraints() -> None:
    lexicon = synonym_lexicon_from_payload(_payload())
    expanded = expand_question_ir_with_synonyms(
        {
            "target_entities": ["red couches"],
            "attribute_dimension": "color",
            "hard_constraints": ["entered building"],
        },
        lexicon,
        question="What were the color of the couches near where I entered the building?",
    )

    assert expanded is not None
    assert "red sofa" in expanded["target_entity_aliases"]
    assert "sofa" in expanded["target_entity_aliases"]
    assert "entered entrance" in expanded["hard_constraint_aliases"]
    assert "entrance" in expanded["hard_constraint_aliases"]
    assert "entrance" not in expanded["target_entity_aliases"]
    assert expanded["synonym_expansion"]["uses_gold_answers"] is False


def test_visual_phrase_records_preserve_synonym_roles() -> None:
    lexicon = synonym_lexicon_from_payload(_payload())
    expanded = expand_question_ir_with_synonyms(
        {
            "target_entities": ["couches"],
            "hard_constraints": ["entered building"],
        },
        lexicon,
        question="What were the color of the couches near where I entered the building?",
    )

    records = visual_entity_pack_query_phrase_records(
        question="What were the color of the couches near where I entered the building?",
        question_ir=expanded or {},
        limit=12,
    )
    by_text = {record["text"]: record for record in records}

    assert by_text["sofa"]["role"] == "primary_target"
    assert by_text["sofa"]["source"] == "target_entity_aliases"
    assert by_text["entrance"]["role"] == "hard_constraint"
    assert by_text["entrance"]["source"] == "hard_constraint_aliases"


def test_role_aware_ger_phrase_records_include_aliases_without_promoting_constraints() -> None:
    lexicon = synonym_lexicon_from_payload(_payload())
    expanded = expand_question_ir_with_synonyms(
        {
            "target_entities": ["art installation"],
            "event_constraints": ["entered building"],
        },
        lexicon,
        question="What was the large art installation near the building?",
    )

    records = _visual_query_phrase_records(
        question="What was the large art installation near the building?",
        question_ir=expanded or {},
        limit=12,
    )
    by_text = {record["text"]: record for record in records}

    assert by_text["sculpture"]["role"] == "primary_target"
    assert by_text["sculpture"]["source"] == "target_entity_aliases"
    assert by_text["entrance"]["role"] == "event_constraint"
    assert by_text["entrance"]["source"] == "event_constraint_aliases"


def test_project_synonym_lexicon_loads_and_expands_known_examples() -> None:
    lexicon = load_synonym_lexicon(REPO_ROOT / "configs/datasets/navqa/synonyms.json")
    expanded = expand_question_ir_with_synonyms(
        {"target_entities": ["art installation"]},
        lexicon,
        question="What was the art installation made of?",
    )

    assert expanded is not None
    assert "sculpture" in expanded["target_entity_aliases"]
    assert expanded["synonym_expansion"]["contract"] == SYNONYM_LEXICON_CONTRACT
