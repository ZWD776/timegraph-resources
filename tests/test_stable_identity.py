from __future__ import annotations

from pathlib import Path

from vlkg.core.models.temporal_graph import BoundingBox, Entity, Observation, TemporalGraphDocument
from vlkg.core.transformers import build_stable_identity_document


def _entity(
    *,
    entity_id: str,
    category: str,
    first_seen_frame: int,
    last_seen_frame: int,
    canonical_embedding: list[float],
    labels_seen: list[str] | None = None,
    states_seen: list[str] | None = None,
    is_static: bool = False,
    is_noise: bool = False,
    is_fragment: bool = False,
) -> Entity:
    return Entity(
        entity_id=entity_id,
        video_id="video_stable_identity",
        category=category,
        first_seen_frame=first_seen_frame,
        last_seen_frame=last_seen_frame,
        first_seen_ts=first_seen_frame / 10.0,
        last_seen_ts=last_seen_frame / 10.0,
        observation_count=max(1, last_seen_frame - first_seen_frame + 1),
        canonical_embedding=canonical_embedding,
        labels_seen=labels_seen or [category],
        states_seen=states_seen or [],
        parent_labels_seen=[],
        node_role="static" if is_static else "dynamic",
        is_static=is_static,
        is_noise=is_noise,
        is_fragment=is_fragment,
    )


def _observation(
    *,
    observation_id: str,
    entity_id: str,
    frame_index: int,
    label: str,
    bbox: tuple[float, float, float, float],
) -> Observation:
    return Observation(
        observation_id=observation_id,
        video_id="video_stable_identity",
        entity_id=entity_id,
        frame_name=f"frame_{frame_index:06d}",
        frame_rank=frame_index,
        frame_index=frame_index,
        timestamp_sec=frame_index / 10.0,
        image_path="synthetic.png",
        object_index=0,
        bbox=BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
        label=label,
        score=0.99,
    )


def _document(
    *,
    entities: list[Entity],
    observations: list[Observation],
) -> TemporalGraphDocument:
    return TemporalGraphDocument(
        video_id="video_stable_identity",
        source_dir=Path("data/examples/video_stable_identity"),
        frame_count=100,
        observations=observations,
        entities=entities,
        relation_episodes=[],
        state_episodes=[],
    )


def test_stable_identity_merges_disjoint_human_tracks_with_shared_signature() -> None:
    document = _document(
        entities=[
            _entity(
                entity_id="man_track_a",
                category="man",
                first_seen_frame=0,
                last_seen_frame=5,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["man"],
                states_seen=["red jersey", "number 23"],
            ),
            _entity(
                entity_id="person_track_b",
                category="person",
                first_seen_frame=9,
                last_seen_frame=14,
                canonical_embedding=[0.99, 0.01, 0.0],
                labels_seen=["person"],
                states_seen=["red jersey", "number 23"],
            ),
        ],
        observations=[
            _observation(
                observation_id="obs_a",
                entity_id="man_track_a",
                frame_index=0,
                label="man",
                bbox=(10.0, 20.0, 60.0, 140.0),
            ),
            _observation(
                observation_id="obs_b",
                entity_id="person_track_b",
                frame_index=9,
                label="person",
                bbox=(12.0, 22.0, 62.0, 142.0),
            ),
        ],
    )

    stable_document = build_stable_identity_document(document)

    assert stable_document.identity_count == 1
    identity = stable_document.identities[0]
    assert identity.identity_type == "human"
    assert identity.canonical_label == "person"
    assert identity.labels_seen == ["man", "person"]
    assert identity.signature_tokens == ["color:red", "num:23", "wear:jersey"]
    assert [member.entity_id for member in identity.member_entities] == [
        "man_track_a",
        "person_track_b",
    ]
    assert identity.attributes["member_count"] == 2


def test_stable_identity_none_merge_mode_keeps_tracks_isolated() -> None:
    document = _document(
        entities=[
            _entity(
                entity_id="man_track_a",
                category="man",
                first_seen_frame=0,
                last_seen_frame=5,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["man"],
                states_seen=["red jersey", "number 23"],
            ),
            _entity(
                entity_id="person_track_b",
                category="person",
                first_seen_frame=9,
                last_seen_frame=14,
                canonical_embedding=[0.99, 0.01, 0.0],
                labels_seen=["person"],
                states_seen=["red jersey", "number 23"],
            ),
        ],
        observations=[
            _observation(
                observation_id="obs_a",
                entity_id="man_track_a",
                frame_index=0,
                label="man",
                bbox=(10.0, 20.0, 60.0, 140.0),
            ),
            _observation(
                observation_id="obs_b",
                entity_id="person_track_b",
                frame_index=9,
                label="person",
                bbox=(12.0, 22.0, 62.0, 142.0),
            ),
        ],
    )

    stable_document = build_stable_identity_document(
        document,
        merge_mode="none",
    )

    assert stable_document.identity_count == 2
    assert [
        [member.entity_id for member in identity.member_entities]
        for identity in stable_document.identities
    ] == [["man_track_a"], ["person_track_b"]]
    assert [identity.identity_type for identity in stable_document.identities] == [
        "object",
        "object",
    ]
    assert [identity.canonical_label for identity in stable_document.identities] == [
        "man",
        "person",
    ]
    assert all(
        not identity.signature_tokens
        for identity in stable_document.identities
    )


def test_stable_identity_keeps_conflicting_human_numbers_separate() -> None:
    document = _document(
        entities=[
            _entity(
                entity_id="player_7",
                category="man",
                first_seen_frame=0,
                last_seen_frame=5,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["man"],
                states_seen=["blue jersey", "number 7"],
            ),
            _entity(
                entity_id="player_12",
                category="person",
                first_seen_frame=7,
                last_seen_frame=12,
                canonical_embedding=[0.99, 0.01, 0.0],
                labels_seen=["person"],
                states_seen=["blue jersey", "number 12"],
            ),
        ],
        observations=[
            _observation(
                observation_id="obs_7",
                entity_id="player_7",
                frame_index=0,
                label="man",
                bbox=(10.0, 20.0, 60.0, 140.0),
            ),
            _observation(
                observation_id="obs_12",
                entity_id="player_12",
                frame_index=7,
                label="person",
                bbox=(10.0, 20.0, 60.0, 140.0),
            ),
        ],
    )

    stable_document = build_stable_identity_document(document)

    assert stable_document.identity_count == 2
    assert {
        tuple(member.entity_id for member in identity.member_entities)
        for identity in stable_document.identities
    } == {
        ("player_12",),
        ("player_7",),
    }


def test_stable_identity_merges_sparse_ball_tracks() -> None:
    document = _document(
        entities=[
            _entity(
                entity_id="ball_track_a",
                category="basketball",
                first_seen_frame=10,
                last_seen_frame=10,
                canonical_embedding=[0.0, 1.0, 0.0],
                labels_seen=["basketball"],
            ),
            _entity(
                entity_id="ball_track_b",
                category="basketball",
                first_seen_frame=25,
                last_seen_frame=25,
                canonical_embedding=[0.0, 1.0, 0.0],
                labels_seen=["basketball"],
            ),
        ],
        observations=[
            _observation(
                observation_id="ball_obs_a",
                entity_id="ball_track_a",
                frame_index=10,
                label="basketball",
                bbox=(100.0, 100.0, 112.0, 112.0),
            ),
            _observation(
                observation_id="ball_obs_b",
                entity_id="ball_track_b",
                frame_index=25,
                label="basketball",
                bbox=(101.0, 101.0, 113.0, 113.0),
            ),
        ],
    )

    stable_document = build_stable_identity_document(document)

    assert stable_document.identity_count == 1
    identity = stable_document.identities[0]
    assert identity.identity_type == "ball"
    assert identity.canonical_label == "basketball"
    assert identity.attributes["member_count"] == 2
    assert identity.time_range.start_frame == 10
    assert identity.time_range.end_frame == 25


def test_stable_identity_excludes_noise_and_fragments_by_default() -> None:
    document = _document(
        entities=[
            _entity(
                entity_id="stable_person",
                category="woman",
                first_seen_frame=0,
                last_seen_frame=8,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["woman"],
                states_seen=["white jersey"],
            ),
            _entity(
                entity_id="fragment_person",
                category="woman",
                first_seen_frame=9,
                last_seen_frame=9,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["woman"],
                states_seen=["white jersey"],
                is_fragment=True,
            ),
            _entity(
                entity_id="noise_person",
                category="people",
                first_seen_frame=10,
                last_seen_frame=10,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["people"],
                states_seen=["white jersey"],
                is_noise=True,
            ),
        ],
        observations=[
            _observation(
                observation_id="stable_obs",
                entity_id="stable_person",
                frame_index=0,
                label="woman",
                bbox=(20.0, 20.0, 70.0, 150.0),
            ),
            _observation(
                observation_id="fragment_obs",
                entity_id="fragment_person",
                frame_index=9,
                label="woman",
                bbox=(21.0, 20.0, 71.0, 150.0),
            ),
            _observation(
                observation_id="noise_obs",
                entity_id="noise_person",
                frame_index=10,
                label="people",
                bbox=(22.0, 20.0, 72.0, 150.0),
            ),
        ],
    )

    stable_document = build_stable_identity_document(document)

    assert stable_document.identity_count == 1
    identity = stable_document.identities[0]
    assert [member.entity_id for member in identity.member_entities] == [
        "stable_person",
    ]


def test_stable_identity_can_include_noise_and_fragments_for_strict_identity() -> None:
    document = _document(
        entities=[
            _entity(
                entity_id="stable_person",
                category="woman",
                first_seen_frame=0,
                last_seen_frame=8,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["woman"],
            ),
            _entity(
                entity_id="fragment_person",
                category="woman",
                first_seen_frame=9,
                last_seen_frame=9,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["woman"],
                is_fragment=True,
            ),
            _entity(
                entity_id="noise_person",
                category="people",
                first_seen_frame=10,
                last_seen_frame=10,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["people"],
                is_noise=True,
            ),
        ],
        observations=[
            _observation(
                observation_id="stable_obs",
                entity_id="stable_person",
                frame_index=0,
                label="woman",
                bbox=(20.0, 20.0, 70.0, 150.0),
            ),
            _observation(
                observation_id="fragment_obs",
                entity_id="fragment_person",
                frame_index=9,
                label="woman",
                bbox=(21.0, 20.0, 71.0, 150.0),
            ),
            _observation(
                observation_id="noise_obs",
                entity_id="noise_person",
                frame_index=10,
                label="people",
                bbox=(22.0, 20.0, 72.0, 150.0),
            ),
        ],
    )

    stable_document = build_stable_identity_document(
        document,
        include_fragments=True,
        include_noise=True,
        merge_mode="none",
    )

    assert stable_document.identity_count == 3
    assert {
        tuple(member.entity_id for member in identity.member_entities)
        for identity in stable_document.identities
    } == {
        ("stable_person",),
        ("fragment_person",),
        ("noise_person",),
    }


def test_stable_identity_does_not_merge_player_like_with_appearance_only_human() -> None:
    document = _document(
        entities=[
            _entity(
                entity_id="player_track",
                category="man",
                first_seen_frame=0,
                last_seen_frame=12,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["man"],
                states_seen=[
                    "basketball player in yellow jersey with number 23",
                    "dribbling basketball",
                ],
            ),
            _entity(
                entity_id="appearance_track",
                category="woman",
                first_seen_frame=15,
                last_seen_frame=20,
                canonical_embedding=[0.98, 0.02, 0.0],
                labels_seen=["woman"],
                states_seen=[
                    "wearing black dress",
                    "wearing white sneakers",
                ],
            ),
        ],
        observations=[
            _observation(
                observation_id="player_obs_1",
                entity_id="player_track",
                frame_index=0,
                label="man",
                bbox=(20.0, 20.0, 80.0, 160.0),
            ).model_copy(
                update={
                    "state_values": [
                        "basketball player in yellow jersey with number 23",
                        "dribbling basketball",
                    ]
                }
            ),
            _observation(
                observation_id="player_obs_2",
                entity_id="player_track",
                frame_index=5,
                label="man",
                bbox=(22.0, 22.0, 82.0, 162.0),
            ).model_copy(
                update={
                    "state_values": [
                        "basketball player in yellow jersey with number 23",
                        "guarding opponent",
                    ]
                }
            ),
            _observation(
                observation_id="appearance_obs_1",
                entity_id="appearance_track",
                frame_index=15,
                label="woman",
                bbox=(21.0, 21.0, 81.0, 161.0),
            ).model_copy(
                update={"state_values": ["wearing black dress", "wearing white sneakers"]}
            ),
            _observation(
                observation_id="appearance_obs_2",
                entity_id="appearance_track",
                frame_index=20,
                label="woman",
                bbox=(23.0, 23.0, 83.0, 163.0),
            ).model_copy(
                update={"state_values": ["wearing black dress", "wearing white sneakers"]}
            ),
        ],
    )

    stable_document = build_stable_identity_document(document)

    assert stable_document.identity_count == 2
    assert {
        tuple(member.entity_id for member in identity.member_entities)
        for identity in stable_document.identities
    } == {
        ("appearance_track",),
        ("player_track",),
    }


def test_stable_identity_ignores_rejected_raw_action_object_for_human_context() -> None:
    document = _document(
        entities=[
            _entity(
                entity_id="appearance_track_a",
                category="woman",
                first_seen_frame=0,
                last_seen_frame=5,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["woman"],
                states_seen=["wearing black dress"],
            ).model_copy(
                update={
                    "raw_states_seen": [
                        "woman wearing black dress dribbling basketball"
                    ]
                }
            ),
            _entity(
                entity_id="appearance_track_b",
                category="woman",
                first_seen_frame=8,
                last_seen_frame=13,
                canonical_embedding=[0.99, 0.01, 0.0],
                labels_seen=["woman"],
                states_seen=["wearing black dress"],
            ),
        ],
        observations=[
            _observation(
                observation_id="appearance_a_1",
                entity_id="appearance_track_a",
                frame_index=0,
                label="woman",
                bbox=(20.0, 20.0, 80.0, 160.0),
            ).model_copy(
                update={
                    "state_values": ["wearing black dress"],
                    "raw_state_values": [
                        "woman wearing black dress dribbling basketball"
                    ],
                }
            ),
            _observation(
                observation_id="appearance_a_2",
                entity_id="appearance_track_a",
                frame_index=5,
                label="woman",
                bbox=(22.0, 22.0, 82.0, 162.0),
            ).model_copy(
                update={
                    "state_values": ["wearing black dress"],
                    "raw_state_values": [
                        "woman wearing black dress dribbling basketball"
                    ],
                }
            ),
            _observation(
                observation_id="appearance_b_1",
                entity_id="appearance_track_b",
                frame_index=8,
                label="woman",
                bbox=(21.0, 21.0, 81.0, 161.0),
            ).model_copy(update={"state_values": ["wearing black dress"]}),
            _observation(
                observation_id="appearance_b_2",
                entity_id="appearance_track_b",
                frame_index=13,
                label="woman",
                bbox=(23.0, 23.0, 83.0, 163.0),
            ).model_copy(update={"state_values": ["wearing black dress"]}),
        ],
    )

    stable_document = build_stable_identity_document(document)

    assert stable_document.identity_count == 1
    identity = stable_document.identities[0]
    assert [member.entity_id for member in identity.member_entities] == [
        "appearance_track_a",
        "appearance_track_b",
    ]
    assert "appearance:black:dress" not in identity.signature_tokens
    assert identity.signature_tokens == ["color:black", "wear:dress"]


def test_stable_identity_human_labels_use_dominant_observation_labels() -> None:
    document = _document(
        entities=[
            _entity(
                entity_id="noisy_track",
                category="man",
                first_seen_frame=0,
                last_seen_frame=3,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["man", "woman"],
                states_seen=["basketball player in yellow jersey"],
            ),
        ],
        observations=[
            _observation(
                observation_id="obs_0",
                entity_id="noisy_track",
                frame_index=0,
                label="man",
                bbox=(20.0, 20.0, 80.0, 160.0),
            ),
            _observation(
                observation_id="obs_1",
                entity_id="noisy_track",
                frame_index=1,
                label="man",
                bbox=(22.0, 22.0, 82.0, 162.0),
            ),
            _observation(
                observation_id="obs_2",
                entity_id="noisy_track",
                frame_index=2,
                label="woman",
                bbox=(24.0, 24.0, 84.0, 164.0),
            ),
            _observation(
                observation_id="obs_3",
                entity_id="noisy_track",
                frame_index=3,
                label="man",
                bbox=(26.0, 26.0, 86.0, 166.0),
            ),
        ],
    )

    stable_document = build_stable_identity_document(document)

    assert stable_document.identity_count == 1
    identity = stable_document.identities[0]
    assert identity.labels_seen == ["man"]
    assert identity.attributes["raw_labels_seen"] == ["man", "woman"]


def test_stable_identity_merges_sparse_humans_with_two_shared_appearance_pairs() -> None:
    document = _document(
        entities=[
            _entity(
                entity_id="player_a",
                category="man",
                first_seen_frame=0,
                last_seen_frame=3,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["man"],
                states_seen=["red jersey", "white shorts"],
            ),
            _entity(
                entity_id="player_b",
                category="person",
                first_seen_frame=8,
                last_seen_frame=11,
                canonical_embedding=[0.99, 0.01, 0.0],
                labels_seen=["person"],
                states_seen=["red jersey", "white shorts"],
            ),
        ],
        observations=[
            _observation(
                observation_id="player_a_obs_1",
                entity_id="player_a",
                frame_index=0,
                label="man",
                bbox=(20.0, 20.0, 80.0, 160.0),
            ).model_copy(
                update={"state_values": ["red jersey", "white shorts"]}
            ),
            _observation(
                observation_id="player_a_obs_2",
                entity_id="player_a",
                frame_index=3,
                label="man",
                bbox=(22.0, 22.0, 82.0, 162.0),
            ).model_copy(
                update={"state_values": ["red jersey", "white shorts"]}
            ),
            _observation(
                observation_id="player_b_obs_1",
                entity_id="player_b",
                frame_index=8,
                label="person",
                bbox=(21.0, 21.0, 81.0, 161.0),
            ).model_copy(
                update={"state_values": ["red jersey", "white shorts"]}
            ),
            _observation(
                observation_id="player_b_obs_2",
                entity_id="player_b",
                frame_index=11,
                label="person",
                bbox=(23.0, 23.0, 83.0, 163.0),
            ).model_copy(
                update={"state_values": ["red jersey", "white shorts"]}
            ),
        ],
    )

    stable_document = build_stable_identity_document(document)

    assert stable_document.identity_count == 1
    identity = stable_document.identities[0]
    assert [member.entity_id for member in identity.member_entities] == [
        "player_a",
        "player_b",
    ]


def test_stable_identity_keeps_sparse_humans_separate_without_distinctive_shared_appearance() -> None:
    document = _document(
        entities=[
            _entity(
                entity_id="player_a",
                category="man",
                first_seen_frame=0,
                last_seen_frame=3,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["man"],
                states_seen=[
                    "yellow jersey",
                    "orange shorts",
                    "black shoes",
                    "dribbling basketball",
                ],
            ),
            _entity(
                entity_id="player_b",
                category="man",
                first_seen_frame=8,
                last_seen_frame=11,
                canonical_embedding=[0.99, 0.01, 0.0],
                labels_seen=["man"],
                states_seen=[
                    "yellow uniform",
                    "blue shorts",
                    "black sneakers",
                    "running",
                ],
            ),
        ],
        observations=[
            _observation(
                observation_id="player_a_obs_1",
                entity_id="player_a",
                frame_index=0,
                label="man",
                bbox=(20.0, 20.0, 80.0, 160.0),
            ).model_copy(
                update={
                    "state_values": [
                        "yellow jersey",
                        "orange shorts",
                        "dribbling basketball",
                    ]
                }
            ),
            _observation(
                observation_id="player_a_obs_2",
                entity_id="player_a",
                frame_index=3,
                label="man",
                bbox=(22.0, 22.0, 82.0, 162.0),
            ).model_copy(
                update={
                    "state_values": [
                        "yellow jersey",
                        "black shoes",
                        "dribbling basketball",
                    ]
                }
            ),
            _observation(
                observation_id="player_b_obs_1",
                entity_id="player_b",
                frame_index=8,
                label="man",
                bbox=(21.0, 21.0, 81.0, 161.0),
            ).model_copy(
                update={
                    "state_values": [
                        "yellow uniform",
                        "blue shorts",
                        "running",
                    ]
                }
            ),
            _observation(
                observation_id="player_b_obs_2",
                entity_id="player_b",
                frame_index=11,
                label="man",
                bbox=(23.0, 23.0, 83.0, 163.0),
            ).model_copy(
                update={
                    "state_values": [
                        "black sneakers",
                        "blue shorts",
                        "running",
                    ]
                }
            ),
        ],
    )

    stable_document = build_stable_identity_document(document)

    assert stable_document.identity_count == 2
    assert {
        tuple(member.entity_id for member in identity.member_entities)
        for identity in stable_document.identities
    } == {
        ("player_a",),
        ("player_b",),
    }


def test_stable_identity_merges_strong_sparse_human_tail_continuation() -> None:
    document = _document(
        entities=[
            _entity(
                entity_id="long_yellow_track",
                category="man",
                first_seen_frame=0,
                last_seen_frame=40,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["man"],
                states_seen=[
                    "yellow jersey number 62",
                    "blue shorts",
                    "black and white sneakers",
                ],
            ),
            _entity(
                entity_id="short_tail_track",
                category="man",
                first_seen_frame=55,
                last_seen_frame=70,
                canonical_embedding=[0.99, 0.01, 0.0],
                labels_seen=["man"],
                states_seen=[
                    "blue cotton shorts",
                    "blue shirt",
                    "gray shorts",
                    "black and white sneakers",
                    "white sneakers",
                ],
            ),
        ],
        observations=[
            _observation(
                observation_id="long_0",
                entity_id="long_yellow_track",
                frame_index=0,
                label="man",
                bbox=(20.0, 20.0, 80.0, 160.0),
            ).model_copy(
                update={
                    "state_values": [
                        "yellow jersey number 62",
                        "blue shorts",
                        "black and white sneakers",
                    ]
                }
            ),
            _observation(
                observation_id="long_10",
                entity_id="long_yellow_track",
                frame_index=10,
                label="man",
                bbox=(21.0, 20.0, 81.0, 160.0),
            ).model_copy(
                update={
                    "state_values": [
                        "yellow jersey number 62",
                        "blue shorts",
                        "black and white sneakers",
                    ]
                }
            ),
            _observation(
                observation_id="long_20",
                entity_id="long_yellow_track",
                frame_index=20,
                label="man",
                bbox=(22.0, 20.0, 82.0, 160.0),
            ).model_copy(
                update={
                    "state_values": [
                        "yellow jersey number 62",
                        "blue shorts",
                        "white sneakers",
                    ]
                }
            ),
            _observation(
                observation_id="long_40",
                entity_id="long_yellow_track",
                frame_index=40,
                label="man",
                bbox=(23.0, 20.0, 83.0, 160.0),
            ).model_copy(
                update={
                    "state_values": [
                        "yellow jersey number 62",
                        "blue shorts",
                        "black and white sneakers",
                    ]
                }
            ),
            _observation(
                observation_id="tail_55",
                entity_id="short_tail_track",
                frame_index=55,
                label="man",
                bbox=(24.0, 22.0, 84.0, 162.0),
            ).model_copy(
                update={
                    "state_values": [
                        "blue cotton shorts",
                        "black and white sneakers",
                    ]
                }
            ),
            _observation(
                observation_id="tail_60",
                entity_id="short_tail_track",
                frame_index=60,
                label="man",
                bbox=(25.0, 22.0, 85.0, 162.0),
            ).model_copy(
                update={
                    "state_values": [
                        "blue shirt",
                        "white sneakers",
                    ]
                }
            ),
            _observation(
                observation_id="tail_65",
                entity_id="short_tail_track",
                frame_index=65,
                label="man",
                bbox=(26.0, 22.0, 86.0, 162.0),
            ).model_copy(
                update={
                    "state_values": [
                        "gray shorts",
                        "black and white sneakers",
                    ]
                }
            ),
            _observation(
                observation_id="tail_70",
                entity_id="short_tail_track",
                frame_index=70,
                label="man",
                bbox=(27.0, 22.0, 87.0, 162.0),
            ).model_copy(update={"state_values": ["white sneakers"]}),
        ],
    )

    stable_document = build_stable_identity_document(document)

    assert stable_document.identity_count == 1
    identity = stable_document.identities[0]
    assert [member.entity_id for member in identity.member_entities] == [
        "long_yellow_track",
        "short_tail_track",
    ]


def test_stable_identity_keeps_sparse_humans_separate_when_overlap_is_only_footwear() -> None:
    document = _document(
        entities=[
            _entity(
                entity_id="player_a",
                category="man",
                first_seen_frame=0,
                last_seen_frame=3,
                canonical_embedding=[1.0, 0.0, 0.0],
                labels_seen=["man"],
                states_seen=[
                    "yellow jersey",
                    "black sneakers",
                    "white sneakers",
                    "dribbling basketball",
                ],
            ),
            _entity(
                entity_id="player_b",
                category="man",
                first_seen_frame=8,
                last_seen_frame=11,
                canonical_embedding=[0.99, 0.01, 0.0],
                labels_seen=["man"],
                states_seen=[
                    "blue shirt",
                    "black sneakers",
                    "white sneakers",
                    "running",
                ],
            ),
        ],
        observations=[
            _observation(
                observation_id="player_a_obs_1",
                entity_id="player_a",
                frame_index=0,
                label="man",
                bbox=(20.0, 20.0, 80.0, 160.0),
            ).model_copy(
                update={
                    "state_values": [
                        "yellow jersey",
                        "black sneakers",
                        "white sneakers",
                    ]
                }
            ),
            _observation(
                observation_id="player_a_obs_2",
                entity_id="player_a",
                frame_index=3,
                label="man",
                bbox=(22.0, 22.0, 82.0, 162.0),
            ).model_copy(
                update={
                    "state_values": [
                        "yellow jersey",
                        "black sneakers",
                        "white sneakers",
                    ]
                }
            ),
            _observation(
                observation_id="player_b_obs_1",
                entity_id="player_b",
                frame_index=8,
                label="man",
                bbox=(21.0, 21.0, 81.0, 161.0),
            ).model_copy(
                update={
                    "state_values": [
                        "blue shirt",
                        "black sneakers",
                        "white sneakers",
                    ]
                }
            ),
            _observation(
                observation_id="player_b_obs_2",
                entity_id="player_b",
                frame_index=11,
                label="man",
                bbox=(23.0, 23.0, 83.0, 163.0),
            ).model_copy(
                update={
                    "state_values": [
                        "blue shirt",
                        "black sneakers",
                        "white sneakers",
                    ]
                }
            ),
        ],
    )

    stable_document = build_stable_identity_document(document)

    assert stable_document.identity_count == 2
