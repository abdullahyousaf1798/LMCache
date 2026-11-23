# SPDX-License-Identifier: Apache-2.0
"""``L1_KEYS_EVICTED`` reports every removed resident key and why it left."""

# Standard
from collections.abc import Callable, Iterator

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1ManagerConfig, L1MemoryManagerConfig
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import DramL1Manager
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventBusConfig,
    init_event_bus,
)
import lmcache.v1.memory_management as memory_management

LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([1024])], dtypes=[torch.uint8])
KEYS = [
    ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="m", kv_rank=0)
    for i in range(4)
]


@pytest.fixture
def fake_pinned_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Back the L1 arena with plain host memory so the test needs no GPU."""
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)


@pytest.fixture
def evicted_bus() -> Iterator[tuple[EventBus, list[Event]]]:
    """A fresh enabled global bus capturing ``L1_KEYS_EVICTED``.

    The manager binds the global bus at construction, so build it after
    this fixture; stop the bus (which drains it) before reading events.
    """
    bus = init_event_bus(EventBusConfig(enabled=True, max_queue_size=10_000))
    events: list[Event] = []
    bus.subscribe(EventType.L1_KEYS_EVICTED, events.append)
    bus.start()
    try:
        yield bus, events
    finally:
        bus.stop()
        init_event_bus(EventBusConfig(enabled=False))


@pytest.mark.parametrize(
    ("remove", "reason"),
    [
        pytest.param(lambda m: m.delete(KEYS), "evict", id="delete"),
        pytest.param(lambda m: m.clear(), "clear", id="clear"),
        pytest.param(lambda m: m.clear(force=True), "clear", id="force-clear"),
    ],
)
def test_removed_keys_published_with_reason(
    fake_pinned_memory: None,
    evicted_bus: tuple[EventBus, list[Event]],
    remove: Callable[[DramL1Manager], object],
    reason: str,
) -> None:
    """Subscribers that clean up after removed keys (coordinator cache
    events, KV events, blend fingerprints) hear every key from a clear as
    from an eviction, while the eviction metrics can tell the two apart."""
    bus, events = evicted_bus
    manager = DramL1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(size_in_bytes=1 << 20, use_lazy=False, shm_name="")
        )
    )
    try:
        reserved = manager.reserve_write(KEYS, [False] * len(KEYS), LAYOUT)
        assert all(err == L1Error.SUCCESS for err, _ in reserved.values())
        assert all(e == L1Error.SUCCESS for e in manager.finish_write(KEYS).values())

        remove(manager)
    finally:
        manager.close()
    bus.stop()

    removals = [e for e in events if e.metadata["keys"]]
    assert len(removals) == 1
    assert set(removals[0].metadata["keys"]) == set(KEYS)
    assert removals[0].metadata["reason"] == reason
