from __future__ import annotations

import json
from email.message import Message
from urllib.error import HTTPError

import pytest
from vlkg.core.config.config import GraphQAConfig
from vlkg.datasets.openeqa import graph_qa_answering as answering
from vlkg.datasets.openeqa.qa.graph_qa_answering import (
    OpenAICompatibleGraphQAResponder,
    OpenEQAHTTPRetryError,
)


class _JSONResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "_JSONResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _http_error(status_code: int, *, retry_after: str | None = None) -> HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return HTTPError(
        url="https://example.invalid/v1/chat/completions",
        code=status_code,
        msg="temporary test error",
        hdrs=headers,
        fp=None,
    )


def _responder() -> OpenAICompatibleGraphQAResponder:
    return OpenAICompatibleGraphQAResponder(
        GraphQAConfig(
            api_base_url="https://example.invalid/v1",
            endpoint_path="/chat/completions",
            timeout_sec=1.0,
        )
    )


def test_open_target_relation_images_are_hard_disabled() -> None:
    bundle = {
        "open_target_relation_contract": "openeqa_open_target_relation_graph_only.v1",
        "open_target_relation_image_frames": [
            {
                "frame_index": 7,
                "image_path": "C:/historical/frame.png",
                "source": "relation_pair_context",
                "view_mode": "annotated_relation_context",
                "overlay_contract": "openeqa_relation_role_overlay.v1",
                "candidate_rank": 1,
                "target_label": "box",
                "relation": "on",
                "anchor_label": "shelf",
                "crop_box": [0, 0, 320, 240],
                "target_bbox": [20, 40, 120, 140],
                "anchor_bboxes": [[10, 170, 310, 225]],
                "uses_gold_answer": False,
            }
        ],
    }

    parts, metadata = answering._open_target_relation_image_parts(
        bundle,
        evidence_demand={"strategy_variant": "open_target_relation"},
    )
    isolated_parts, isolated_metadata = answering._open_target_relation_image_parts(
        bundle,
        evidence_demand={"strategy_variant": "target_location"},
    )

    assert parts == []
    assert metadata == []
    assert isolated_parts == []
    assert isolated_metadata == []


def test_anchor_bound_identity_uses_separate_supplemental_prompt_channel() -> None:
    bundle = {
        "visual_evidence_contract": "openeqa_visual_entity_candidate_packs.v1",
        "visual_entity_candidate_pack_lines": ["kind=frozen_visual_candidate"],
        "question_route_anchor_bound_identity_lines": [
            "rank=1 kind=openeqa_anchor_bound_identity_candidate "
            "target='plant' relation=left_of anchor='mirror' "
            "relation_verified=true"
        ],
        "evidence_demand": {
            "strategy_variant": "anchor_bound_open_target_identity"
        },
    }
    payload = _responder()._build_payload(
        question="What is to the left of the mirror?",
        llm_ready_bundle=bundle,
        graph_result={"llm_ready_bundle": bundle},
    )
    prompt = json.loads(payload["messages"][-1]["content"])

    assert prompt["graph_summary"]["visual_entity_candidate_pack_lines"] == [
        "kind=frozen_visual_candidate"
    ]
    assert prompt["graph_summary"][
        "question_route_anchor_bound_identity_lines"
    ][0].startswith("rank=1 kind=openeqa_anchor_bound_identity_candidate")
    assert prompt["instructions"][
        "use_anchor_bound_identity_candidates_when_available"
    ] is True


def test_specialized_evidence_uses_separate_supplemental_prompt_channel() -> None:
    bundle = {
        "visual_evidence_contract": "openeqa_visual_entity_candidate_packs.v1",
        "visual_entity_candidate_pack_lines": ["kind=frozen_visual_candidate"],
        "question_route_specialized_evidence_lines": [
            "kind=surface_capacity_guidance contract=surface.v1",
            "rank=1 kind=owner_bound_surface_observation owner='desk-1'",
        ],
        "evidence_demand": {"strategy_variant": "surface_capacity"},
    }
    payload = _responder()._build_payload(
        question="Is there room on my desk?",
        llm_ready_bundle=bundle,
        graph_result={"llm_ready_bundle": bundle},
    )
    prompt = json.loads(payload["messages"][-1]["content"])

    assert prompt["graph_summary"]["visual_entity_candidate_pack_lines"] == [
        "kind=frozen_visual_candidate"
    ]
    assert prompt["graph_summary"][
        "question_route_specialized_evidence_lines"
    ] == bundle["question_route_specialized_evidence_lines"]
    assert prompt["instructions"][
        "use_question_route_specialized_evidence_when_available"
    ] is True
    assert "specialized_evidence_policy" in prompt["answering_policy"]


@pytest.mark.parametrize(
    ("status_code", "retry_after", "expected_delay"),
    [
        (429, "0", 0.0),
        (503, None, 5.0),
    ],
)
def test_post_payload_retries_transient_http_error_without_changing_payload(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
    retry_after: str | None,
    expected_delay: float,
) -> None:
    calls: list[bytes | None] = []
    responses: list[object] = [
        _http_error(status_code, retry_after=retry_after),
        _JSONResponse({"id": "ok", "choices": []}),
    ]

    def fake_urlopen(request: object, *, timeout: float) -> _JSONResponse:
        del timeout
        calls.append(getattr(request, "data", None))
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, _JSONResponse)
        return response

    sleeps: list[float] = []
    monkeypatch.setattr(answering, "urlopen", fake_urlopen)
    monkeypatch.setattr(answering.time, "sleep", sleeps.append)

    payload = {"model": "test-model", "messages": [{"role": "user", "content": "x"}]}
    raw_response = _responder()._post_payload(payload, api_key="secret-test-key")

    assert len(calls) == 2
    assert [json.loads(item.decode("utf-8")) for item in calls if item is not None] == [
        payload,
        payload,
    ]
    assert sleeps == [expected_delay]
    assert raw_response[answering._TRANSPORT_METADATA_KEY] == {"http_retry_count": 1}


def test_post_payload_does_not_retry_non_transient_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_urlopen(request: object, *, timeout: float) -> _JSONResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        raise _http_error(400)

    sleeps: list[float] = []
    monkeypatch.setattr(answering, "urlopen", fake_urlopen)
    monkeypatch.setattr(answering.time, "sleep", sleeps.append)

    with pytest.raises(HTTPError) as exc_info:
        _responder()._post_payload({"model": "test-model"}, api_key="secret-test-key")

    assert exc_info.value.code == 400
    assert calls == 1
    assert sleeps == []


def test_post_payload_reports_exhausted_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_urlopen(request: object, *, timeout: float) -> _JSONResponse:
        nonlocal calls
        del request, timeout
        calls += 1
        raise _http_error(429, retry_after="0")

    sleeps: list[float] = []
    monkeypatch.setattr(answering, "urlopen", fake_urlopen)
    monkeypatch.setattr(answering.time, "sleep", sleeps.append)
    monkeypatch.setattr(answering, "_HTTP_RETRY_BACKOFF_SECONDS", (0.0, 0.0))

    with pytest.raises(OpenEQAHTTPRetryError) as exc_info:
        _responder()._post_payload({"model": "test-model"}, api_key="secret-test-key")

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_count == 2
    assert "after 2 retries" in str(exc_info.value)
    assert calls == 3
    assert sleeps == [0.0, 0.0]


def test_response_metadata_exposes_transport_retry_count() -> None:
    responder = _responder()
    metadata = responder._response_metadata(
        {
            "id": "response-1",
            "choices": [{"finish_reason": "stop"}],
            answering._TRANSPORT_METADATA_KEY: {"http_retry_count": 2},
        }
    )

    assert metadata["transport"] == {"http_retry_count": 2}
