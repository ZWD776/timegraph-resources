from __future__ import annotations

import json
from pathlib import Path

from vlkg.core.temporal.processed_attributes import (
    PROCESSED_ATTRIBUTES_CONTRACT,
    AttributeRecordInput,
    OpenAICompatibleAttributeStructurer,
    PassthroughAttributeStructurer,
    ProcessedAttributesBuildConfig,
    SimpleAttributeStructurer,
    build_processed_attributes,
    collect_attribute_inputs,
    filter_priority_attribute_records,
    priority_attribute_flags,
    load_processed_attributes,
    processed_attribute_record,
)


class FailingOnceStructurer(PassthroughAttributeStructurer):
    def __init__(self) -> None:
        self.calls = 0

    def structure_batch(
        self,
        records: list[AttributeRecordInput],
    ) -> dict[str, dict[str, object]]:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("planned batch failure")
        return super().structure_batch(records)


def test_collect_attribute_inputs_keeps_object_anchors(tmp_path: Path) -> None:
    hsgg_dir = _write_sample_hsgg(tmp_path)

    records = collect_attribute_inputs(hsgg_dir)

    assert len(records) == 2
    assert records[0].record_id == "frame_000001:object:0"
    assert records[0].label == "person"
    assert records[0].bbox == (10.0, 20.0, 50.0, 80.0)
    assert records[0].raw_attributes == ("person wearing blue jacket",)


def test_build_processed_attributes_writes_persistent_sidecar(tmp_path: Path) -> None:
    hsgg_dir = _write_sample_hsgg(tmp_path)
    output_path = hsgg_dir / "processed_attributes.json"

    summary = build_processed_attributes(
        ProcessedAttributesBuildConfig(
            hsgg_input_dir=hsgg_dir,
            output_path=output_path,
            batch_size=2,
            force=True,
            generator_mode="passthrough",
        ),
        structurer=PassthroughAttributeStructurer(),
    )

    assert summary["record_count"] == 2
    assert summary["unique_attribute_count"] == 2
    payload = load_processed_attributes(output_path)
    assert payload["contract"] == PROCESSED_ATTRIBUTES_CONTRACT
    assert payload["generation"]["uses_questions"] is False
    assert payload["generation"]["uses_gold_answers"] is False
    assert payload["generation"]["writes_hsgg_source_files"] is False
    assert payload["generation"]["dedupe_strategy"] == "label_plus_normalized_raw_attributes"
    record = processed_attribute_record(
        payload,
        frame_name="frame_000001",
        object_index=0,
    )
    assert record is not None
    assert record["attribute_key"] in payload["unique_attributes"]
    assert record["raw_attributes"] == ["person wearing blue jacket"]
    assert record["structured"]["summary"] == "person wearing blue jacket"
    assert record["structured"]["identity_signature"] == []
    assert record["structured"]["facts"] == []
    assert record["structured"]["uncertain"] == [
        "person wearing blue jacket"
    ]


def test_build_processed_attributes_simple_mode_structures_locally(tmp_path: Path) -> None:
    hsgg_dir = _write_sample_hsgg(tmp_path)
    output_path = hsgg_dir / "processed_attributes.json"

    summary = build_processed_attributes(
        ProcessedAttributesBuildConfig(
            hsgg_input_dir=hsgg_dir,
            output_path=output_path,
            batch_size=2,
            force=True,
            generator_mode="simple",
            generator_model=None,
        ),
        structurer=SimpleAttributeStructurer(),
    )

    assert summary["mode"] == "simple"
    payload = load_processed_attributes(output_path)
    assert payload["generation"]["generator_mode"] == "simple"
    assert payload["generation"]["generator_model"] is None
    record = processed_attribute_record(
        payload,
        frame_name="frame_000001",
        object_index=0,
    )
    assert record is not None
    assert record["structured"]["summary"] == "person wearing blue jacket"
    assert "blue jacket" in record["structured"]["identity_signature"]
    assert record["structured"]["facts"]


def test_priority_attribute_filter_selects_text_and_number_only(tmp_path: Path) -> None:
    hsgg_dir = _write_sample_hsgg(tmp_path)
    records = collect_attribute_inputs(hsgg_dir)
    from vlkg.core.temporal.processed_attributes import deduplicate_attribute_inputs

    unique_records, _ = deduplicate_attribute_inputs(records)
    selected = filter_priority_attribute_records(unique_records, "text-number")

    assert len(selected) == 1
    selected_record = next(iter(selected.values()))
    assert selected_record.label == "door"
    assert priority_attribute_flags(selected_record) == ("text_logo_sign",)


def test_build_processed_attributes_resume_uses_progress_checkpoint(tmp_path: Path) -> None:
    hsgg_dir = _write_sample_hsgg(tmp_path)
    output_path = hsgg_dir / "processed_attributes.json"
    progress_path = hsgg_dir / "processed_attributes.progress.json"

    try:
        build_processed_attributes(
            ProcessedAttributesBuildConfig(
                hsgg_input_dir=hsgg_dir,
                output_path=output_path,
                progress_path=progress_path,
                batch_size=1,
                force=True,
                generator_mode="passthrough",
                max_retries=0,
            ),
            structurer=FailingOnceStructurer(),
        )
    except RuntimeError as exc:
        assert "planned batch failure" in str(exc)
    else:
        raise AssertionError("expected planned batch failure")

    assert progress_path.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["summary"]["completed_unique_attribute_count"] == 1
    assert progress["summary"]["failed_batch_count"] == 1
    assert not output_path.exists()

    summary = build_processed_attributes(
        ProcessedAttributesBuildConfig(
            hsgg_input_dir=hsgg_dir,
            output_path=output_path,
            progress_path=progress_path,
            batch_size=1,
            force=True,
            generator_mode="passthrough",
            resume=True,
            max_retries=0,
        ),
        structurer=PassthroughAttributeStructurer(),
    )

    assert summary["record_count"] == 2
    assert summary["unique_attribute_count"] == 2
    assert output_path.exists()
    assert not progress_path.exists()
    payload = load_processed_attributes(output_path)
    assert len(payload["unique_attributes"]) == 2


def test_openai_attribute_structurer_disables_thinking_by_default() -> None:
    structurer = OpenAICompatibleAttributeStructurer()
    messages = structurer._messages(
        [
            AttributeRecordInput(
                record_id="pa_test",
                frame_name="frame_000001",
                object_index=0,
                label="person",
                raw_attributes=("person wearing blue jacket",),
            )
        ]
    )
    payload = {
        "model": structurer.model,
        "messages": messages,
    }
    from vlkg.core.transport.openai_compatible_payload import apply_reasoning_options

    apply_reasoning_options(
        payload,
        reasoning_effort=structurer.reasoning_effort,
        thinking_enabled=structurer.thinking_enabled,
    )

    assert payload["thinking"] == {"type": "disabled"}


def _write_sample_hsgg(tmp_path: Path) -> Path:
    hsgg_dir = tmp_path / "hsgg_local_paths"
    hsgg_dir.mkdir()
    objects = {
        "frame_000001": {
            "image_path": str(tmp_path / "frame_000001.jpg"),
            "labels": ["person", "door"],
            "boxes": [[10, 20, 50, 80], [45, 20, 90, 85]],
            "scores": [0.9, 0.8],
        }
    }
    attributes = {
        "frame_000001": {
            "0": "person wearing blue jacket",
            "1": "white door with window and sign",
        }
    }
    (hsgg_dir / "objects.json").write_text(
        json.dumps(objects),
        encoding="utf-8",
    )
    (hsgg_dir / "attributes.json").write_text(
        json.dumps(attributes),
        encoding="utf-8",
    )
    return hsgg_dir
