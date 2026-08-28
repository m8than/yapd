"""CPU-only contract tests for the shared heterogeneous scheduler."""
from __future__ import annotations

from dataclasses import dataclass

from server.models import (
    FLASH_MODEL,
    KOKORO_MODEL,
    TURBO_MODEL,
    canonical_model,
    model_owner,
    parse_model_options,
    worker_capabilities,
)
from server.scheduler import SchedulerQueue


@dataclass
class Job:
    rid: int
    priority: int = 0
    shape: int = 1


def test_priority_admission_is_stable() -> None:
    queue: SchedulerQueue[Job] = SchedulerQueue()
    queue.submit(Job(1, priority=1))
    queue.submit(Job(2, priority=0))
    queue.submit(Job(3, priority=0))

    selected = queue.take(
        2,
        select_key=lambda request: (request.priority, request.rid),
    )

    assert [request.rid for request in selected] == [2, 3]
    assert [request.rid for request in queue.take(2)] == [1]


def test_compatible_group_preserves_other_waiters() -> None:
    queue: SchedulerQueue[Job] = SchedulerQueue()
    for job in (Job(1, shape=4), Job(2, shape=8), Job(3, shape=4)):
        queue.submit(job)

    selected = queue.take_group(
        select_key=lambda request: request.rid,
        limit_for=lambda _: 8,
        compatible=lambda seed, request: seed.shape == request.shape,
    )

    assert [request.rid for request in selected] == [1, 3]
    assert [request.rid for request in queue.take(8)] == [2]


def test_group_can_absorb_new_compatible_work() -> None:
    queue: SchedulerQueue[Job] = SchedulerQueue()
    queue.submit(Job(1, shape=4))
    selected = queue.take_group(
        select_key=lambda request: request.rid,
        limit_for=lambda _: 3,
        compatible=lambda seed, request: seed.shape == request.shape,
    )
    queue.submit(Job(2, shape=8))
    queue.submit(Job(3, shape=4))

    added = queue.extend_group(
        selected,
        limit=3,
        compatible=lambda seed, request: seed.shape == request.shape,
    )

    assert added == 1
    assert [request.rid for request in selected] == [1, 3]
    assert [request.rid for request in queue.take(8)] == [2]


def test_model_registry_normalizes_builtin_aliases() -> None:
    assert canonical_model("kokoro") == KOKORO_MODEL
    assert canonical_model("turbo") == TURBO_MODEL
    assert canonical_model("flash") == FLASH_MODEL
    assert canonical_model("kokoro/af_heart") == "kokoro/af_heart"
    assert model_owner(KOKORO_MODEL) == "kokoro"
    assert model_owner(TURBO_MODEL) == "chatterbox"


def test_model_options_are_nested_and_strict() -> None:
    assert parse_model_options(
        {"text": "hello", "model_options": {"voice": "af_heart"}},
        {"voice", "speed"},
    ) == {"voice": "af_heart"}
    for body in (
        {"text": "hello", "voice": "af_heart"},
        {"text": "hello", "model_options": []},
        {"text": "hello", "model_options": {"temperature": 0.5}},
        {"text": "hello", "temperature": 0.5},
    ):
        try:
            parse_model_options(body, {"voice", "speed"})
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid model_options accepted: {body}")


def test_capabilities_describe_request_and_pcm_output() -> None:
    capabilities = worker_capabilities(
        KOKORO_MODEL,
        voices=["af_heart"],
        max_input=4096,
        input_unit="characters",
        model_options={"voice": {"type": "string", "enum": ["af_heart"]}},
    )
    text = capabilities["request"]["common"]["text"]
    assert capabilities["request"]["required"] == ["text"]
    assert text == {
        "type": "string",
        "max_length": 4096,
        "length_unit": "characters",
    }
    assert capabilities["output"]["format"] == "pcm_s16le"
    assert capabilities["output"]["streaming"] is True
