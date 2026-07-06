from __future__ import annotations

import json
from typing import Any

import pytest

from joint_decode.runtime_state import runtime_state
from joint_decode.tpu.decision_client import DecisionClient


def _client() -> DecisionClient:
    return DecisionClient(
        decision_url="http://127.0.0.1:1/a",
        side="a",
        timeout=1.0,
        eos_token_id=2,
    )


def test_unknown_rid_raises_before_any_post() -> None:
    runtime_state.reset()
    runtime_state.live_rids.add("r0")

    with pytest.raises(RuntimeError, match="VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"):
        _client()(["r0", "r0-a1b2c3d4"], {"r0": _topk(), "r0-a1b2c3d4": _topk()})


def test_pending_rows_are_forced_without_a_post() -> None:
    runtime_state.reset()
    runtime_state.live_rids.update({"r0", "r1"})
    runtime_state.pending_tokens["r0"] = [7]
    runtime_state.pending_tokens["r1"] = [8, 9]

    # No urlopen patch: any coordinator post would fail loudly.
    forced = _client()(["r0", "r1"], {"r0": _topk(), "r1": _topk()})

    assert forced == {"r0": 7, "r1": 8}
    assert runtime_state.pending_tokens == {"r1": [9]}


def test_undecided_rows_post_only_their_rids(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_state.reset()
    runtime_state.live_rids.update({"r0", "r1"})
    runtime_state.pending_tokens["r0"] = [7]
    posted: dict[str, Any] = {}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"tokens": {"r1": [21]}, "force_stop": [], "admit": [], "abort": None}).encode()

    def fake_urlopen(request: Any, *, timeout: float) -> _Response:
        del timeout
        posted.update(json.loads(request.data))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    forced = _client()(["r0", "r1"], {"r0": _topk(), "r1": _topk()})

    assert forced == {"r0": 7, "r1": 21}
    assert posted["request_ids"] == ["r1"]
    assert list(posted["topk"]) == ["r1"]


def _topk() -> list[dict[str, Any]]:
    return [{"token_id": 1, "logit": 1.0}]
