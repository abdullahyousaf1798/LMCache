# SPDX-License-Identifier: Apache-2.0
"""Tell whether UCX can use this host's RDMA devices.

UCX picks its transports on its own and falls back to TCP without an error
when no RDMA device is usable. NIXL does not report which transport an
endpoint selected, so the NIXL transfer channel uses this check to tell the
operator what UCX can use, and to warn when RDMA devices exist but UCX cannot
use them.
"""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass
import glob
import os

SYSFS_INFINIBAND = "/sys/class/infiniband"
DEV_INFINIBAND = "/dev/infiniband"

# UCX transport names (and their prefixes, e.g. rc_x, rc_mlx5, dc_mlx5,
# ud_verbs) that move data over an RDMA device. "ib" is UCX's alias for all
# of them.
_RDMA_TRANSPORTS = ("rc", "dc", "ud", "srd", "gga")


############################################################
# Helper functions
############################################################
def _is_rdma_transport(name: str) -> bool:
    return name == "ib" or name.startswith(_RDMA_TRANSPORTS)


def _tls_enables_rdma(tls: str) -> bool:
    """Whether a ``UCX_TLS`` value leaves at least one RDMA transport enabled."""
    if tls == "all":
        return True
    if tls.startswith("^"):
        excluded = set(tls[1:].split(","))
        return "ib" not in excluded and not excluded.issuperset(_RDMA_TRANSPORTS)
    return any(_is_rdma_transport(name) for name in tls.split(","))


def _has_active_port(sysfs_root: str, device: str) -> bool:
    for state_file in glob.glob(
        os.path.join(sysfs_root, device, "ports", "*", "state")
    ):
        with open(state_file) as f:
            if "ACTIVE" in f.read():
                return True
    return False


@dataclass(frozen=True)
class UcxRdmaStatus:
    """What UCX can use for RDMA on this host under a given environment.

    Attributes:
        rdma_devices: Every RDMA device found (``mlx5_0``, ``efa_0``, ...),
            sorted by name.
        usable_devices: The subset of ``rdma_devices`` that UCX can send over:
            allowed by ``UCX_TLS`` and ``UCX_NET_DEVICES``, openable by this
            process, and with at least one ACTIVE port.
        reason: Why none of ``rdma_devices`` is usable, with the fix. Empty
            when ``usable_devices`` is non-empty or when there are no RDMA
            devices at all.
    """

    rdma_devices: tuple[str, ...]
    usable_devices: tuple[str, ...]
    reason: str

    @property
    def rdma_usable(self) -> bool:
        """True if UCX can use at least one RDMA device."""
        return bool(self.usable_devices)


def check_ucx_rdma(
    env: Mapping[str, str],
    sysfs_root: str = SYSFS_INFINIBAND,
    dev_root: str = DEV_INFINIBAND,
) -> UcxRdmaStatus:
    """Check whether UCX can use the RDMA devices of this host.

    The check mirrors how UCX filters its resources; it does not open a UCX
    context, so it reports what UCX can use, not what an endpoint selected.
    Set ``UCX_PROTO_INFO=y`` to have UCX print its exact per-operation choice.

    A device is usable when all of the following hold:

    - ``UCX_TLS`` is unset, ``all``, or enables an RDMA transport (``rc``,
      ``dc``, ``ud``, ``srd``, ``gga`` and their variants, or ``ib``); a
      ``^``-prefixed list enables everything it does not exclude;
    - ``UCX_NET_DEVICES`` is unset, ``all``, or lists the device
      (``mlx5_0:1`` lists ``mlx5_0``);
    - this process can read and write a ``uverbs*`` file in ``dev_root``;
    - one of the device's ports is ACTIVE.

    Args:
        env: The environment UCX reads its configuration from, usually
            ``os.environ``.
        sysfs_root: Directory with one entry per RDMA device.
        dev_root: Directory with the RDMA device files.

    Returns:
        The devices found, the usable ones, and why none is usable if so.
    """
    devices = tuple(sorted(os.listdir(sysfs_root))) if os.path.isdir(sysfs_root) else ()
    if not devices:
        return UcxRdmaStatus(rdma_devices=(), usable_devices=(), reason="")

    tls = env.get("UCX_TLS", "all")
    if not _tls_enables_rdma(tls):
        return UcxRdmaStatus(
            devices,
            (),
            f"UCX_TLS={tls} enables no RDMA transport; unset UCX_TLS or add "
            "rc, e.g. UCX_TLS=rc,sm,self,cuda_copy",
        )

    net_devices = env.get("UCX_NET_DEVICES", "all")
    allowed = devices
    if net_devices != "all":
        listed = {entry.split(":", 1)[0] for entry in net_devices.split(",")}
        allowed = tuple(d for d in devices if d in listed)
    if not allowed:
        return UcxRdmaStatus(
            devices,
            (),
            f"UCX_NET_DEVICES={net_devices} lists none of them; unset it or "
            f"list them as <device>:<port>, e.g. {devices[0]}:1",
        )

    uverbs = glob.glob(os.path.join(dev_root, "uverbs*"))
    if not any(os.access(path, os.R_OK | os.W_OK) for path in uverbs):
        return UcxRdmaStatus(
            devices,
            (),
            f"this process cannot open any {dev_root}/uverbs* device file "
            "(e.g. a container without the RDMA device plugin)",
        )

    usable = tuple(d for d in allowed if _has_active_port(sysfs_root, d))
    if not usable:
        return UcxRdmaStatus(
            devices,
            (),
            f"no port is ACTIVE on {', '.join(allowed)}; check the link and the "
            "InfiniBand/RoCE setup",
        )
    return UcxRdmaStatus(devices, usable, "")
