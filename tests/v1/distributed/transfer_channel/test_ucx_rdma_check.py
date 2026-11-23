# SPDX-License-Identifier: Apache-2.0
"""Tests for ``check_ucx_rdma`` against a fake sysfs and /dev tree."""

# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.transfer_channel.impl.ucx_rdma_check import (
    check_ucx_rdma,
)


def _make_host(
    root: Path,
    ports: dict[str, str],
    uverbs: bool = True,
) -> tuple[str, str]:
    """Create fake ``/sys/class/infiniband`` and ``/dev/infiniband`` trees.

    Args:
        root: Directory to create them in.
        ports: Device name to the state of its port 1 (``ACTIVE``/``DOWN``).
        uverbs: Whether to create an accessible ``uverbs0`` device file.

    Returns:
        ``(sysfs_root, dev_root)``.
    """
    sysfs = root / "sys"
    dev = root / "dev"
    sysfs.mkdir()
    dev.mkdir()
    states = {"ACTIVE": "4: ACTIVE", "DOWN": "1: DOWN"}
    for name, state in ports.items():
        port = sysfs / name / "ports" / "1"
        port.mkdir(parents=True)
        (port / "state").write_text(states[state] + "\n")
    if uverbs:
        (dev / "uverbs0").touch()
    return str(sysfs), str(dev)


_TWO_DEVICES = {"mlx5_0": "ACTIVE", "mlx5_1": "ACTIVE"}


def test_no_rdma_devices_is_not_a_problem(tmp_path):
    sysfs, dev = _make_host(tmp_path, {}, uverbs=False)

    status = check_ucx_rdma({}, sysfs_root=sysfs, dev_root=dev)

    assert status.rdma_devices == ()
    assert not status.rdma_usable
    assert status.reason == ""


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"UCX_TLS": "all"},
        {"UCX_TLS": "rc,sm,self,cuda_copy"},
        {"UCX_TLS": "rc_x,self"},
        {"UCX_TLS": "^tcp"},
        {"UCX_NET_DEVICES": "all"},
    ],
)
def test_rdma_usable_under_default_and_rdma_settings(tmp_path, env):
    sysfs, dev = _make_host(tmp_path, _TWO_DEVICES)

    status = check_ucx_rdma(env, sysfs_root=sysfs, dev_root=dev)

    assert status.rdma_devices == ("mlx5_0", "mlx5_1")
    assert status.usable_devices == ("mlx5_0", "mlx5_1")
    assert status.rdma_usable
    assert status.reason == ""


def test_net_devices_restricts_usable_devices(tmp_path):
    sysfs, dev = _make_host(tmp_path, _TWO_DEVICES)

    status = check_ucx_rdma(
        {"UCX_NET_DEVICES": "mlx5_1:1,eth0"}, sysfs_root=sysfs, dev_root=dev
    )

    assert status.usable_devices == ("mlx5_1",)


def test_devices_with_no_active_port_are_not_usable(tmp_path):
    sysfs, dev = _make_host(tmp_path, {"mlx5_0": "ACTIVE", "mlx5_1": "DOWN"})

    status = check_ucx_rdma({}, sysfs_root=sysfs, dev_root=dev)

    assert status.rdma_devices == ("mlx5_0", "mlx5_1")
    assert status.usable_devices == ("mlx5_0",)


@pytest.mark.parametrize(
    "ports, uverbs, env, cause",
    [
        (_TWO_DEVICES, True, {"UCX_TLS": "tcp,self,sm,cuda_copy"}, "UCX_TLS"),
        (_TWO_DEVICES, True, {"UCX_TLS": "^ib"}, "UCX_TLS"),
        (_TWO_DEVICES, True, {"UCX_NET_DEVICES": "eth0"}, "UCX_NET_DEVICES"),
        (_TWO_DEVICES, False, {}, "uverbs"),
        ({"mlx5_0": "DOWN", "mlx5_1": "DOWN"}, True, {}, "ACTIVE"),
        (
            {"mlx5_0": "ACTIVE", "mlx5_1": "DOWN"},
            True,
            {"UCX_NET_DEVICES": "mlx5_1:1"},
            "ACTIVE",
        ),
    ],
    ids=[
        "tls-tcp-only",
        "tls-excludes-ib",
        "net-devices-no-rdma",
        "no-uverbs-access",
        "all-ports-down",
        "selected-port-down",
    ],
)
def test_rdma_present_but_unusable_names_the_cause(tmp_path, ports, uverbs, env, cause):
    sysfs, dev = _make_host(tmp_path, ports, uverbs=uverbs)

    status = check_ucx_rdma(env, sysfs_root=sysfs, dev_root=dev)

    assert status.rdma_devices == tuple(sorted(ports))
    assert not status.rdma_usable
    assert cause in status.reason
