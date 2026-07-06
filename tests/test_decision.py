from __future__ import annotations

import json
from typing import Any

import pytest

from joint_decode.decision import resolve_undecided, split_pending
from joint_decode.runtime_state import runtime_state


def test_split_pending_consumes_one_token_per_rid_in_order() -> None:
    runtime_state.reset()
    runtime_state.pending_tokens["r1"] = [7, 8]
    runtime_state.pending_tokens["r2"] = [9]

    forced, undecided = split_pending(["r0", "r1", "r2"])

    assert forced == {"r1": 7, "r2": 9}
    assert undecided == ["r0"]
    assert runtime_state.pending_tokens == {"r1": [8]}


def test_resolve_undecided_forces_first_token_and_queues_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_state.reset()
    _respond(monkeypatch, {"tokens": {"r0": [11, 12], "r1": 21}, "force_stop": [], "admit": ["r9"], "abort": None})

    forced = _resolve({"r0": _topk(), "r1": _topk()})

    assert forced == {"r0": 11, "r1": 21}
    assert runtime_state.pending_tokens == {"r0": [12]}
    assert runtime_state.drain_commands().admit == ["r9"]


def test_resolve_undecided_forces_eos_for_force_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_state.reset()
    runtime_state.pending_tokens["r0"] = [5]
    _respond(monkeypatch, {"tokens": {}, "force_stop": ["r0"], "admit": [], "abort": None})

    forced = _resolve({"r0": _topk()})

    assert forced == {"r0": 2}
    assert runtime_state.pending_tokens == {}


def test_resolve_undecided_raises_and_publishes_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_state.reset()
    _respond(monkeypatch, {"tokens": {}, "force_stop": [], "admit": [], "abort": "boom"})

    with pytest.raises(RuntimeError, match="boom"):
        _resolve({"r0": _topk()})
    assert runtime_state.drain_commands().abort == "boom"


def test_resolve_undecided_raises_on_uncovered_rid(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime_state.reset()
    _respond(monkeypatch, {"tokens": {"r0": [11]}, "force_stop": [], "admit": [], "abort": None})

    with pytest.raises(RuntimeError, match="did not return tokens or force_stop"):
        _resolve({"r0": _topk(), "r1": _topk()})


def _topk() -> list[dict[str, Any]]:
    return [{"token_id": 1, "logit": 1.0}]


def _resolve(topk_by_rid: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return resolve_undecided(
        topk_by_rid,
        decision_url="http://127.0.0.1:1/a",
        side="a",
        timeout=1.0,
        eos_token_id=2,
    )


def _respond(monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]) -> None:
    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(response).encode()

    def fake_urlopen(request: object, *, timeout: float) -> _Response:
        del request, timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
