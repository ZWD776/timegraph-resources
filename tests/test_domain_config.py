from __future__ import annotations

from pathlib import Path

from vlkg.core.config.config import load_domain_config
from vlkg.core.models.temporal_graph import TemporalGraphDocument, Entity


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]




def test_archived_qwen_graph_qa_config_uses_default_deepseek_provider() -> None:
    config = load_domain_config(
        "qwen_graph_qa.yaml",
        config_dir=_repo_root() / "configs",
    )

    assert config.graph_qa.provider == "deepseek_openai_compatible"
    assert config.graph_qa.model == "deepseek-v4-flash"
    assert config.graph_qa.api_key_env_var == "DEEPSEEK_API_KEY"


def test_filtered_copy_keeps_protected_fragment_labels() -> None:
    document = TemporalGraphDocument(
        video_id="video_001",
        source_dir=_repo_root(),
        frame_count=1,
        observations=[],
        entities=[
            Entity(
                entity_id="ball_1",
                video_id="video_001",
                category="basketball",
                first_seen_frame=0,
                last_seen_frame=0,
                first_seen_ts=0.0,
                last_seen_ts=0.0,
                observation_count=1,
                is_fragment=True,
            ),
            Entity(
                entity_id="cloth_1",
                video_id="video_001",
                category="shirt",
                first_seen_frame=0,
                last_seen_frame=0,
                first_seen_ts=0.0,
                last_seen_ts=0.0,
                observation_count=1,
                is_fragment=True,
            ),
        ],
        relation_episodes=[],
        state_episodes=[],
    )

    exported = document.filtered_copy(
        exclude_fragments=True,
        keep_fragment_labels={"basketball"},
    )

    assert [entity.entity_id for entity in exported.entities] == ["ball_1"]
