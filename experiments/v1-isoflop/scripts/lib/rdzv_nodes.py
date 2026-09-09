"""Parse DS_RDZV_NODES=host,nproc,cvd,ip;host2,... for multi-node torchrun."""

from __future__ import annotations

import os
import sys


def load_rdzv_nodes(env_key: str = "DS_RDZV_NODES") -> tuple[tuple[str, int, str, str], ...]:
    raw = os.environ.get(env_key, "").strip()
    if not raw:
        raise SystemExit(
            f"Set {env_key}='host,nproc,cvd,ip;host2,nproc,cvd,ip;...' "
            "(SSH alias, GPU count, CUDA_VISIBLE_DEVICES, rendezvous IPv4)."
        )
    nodes = []
    for spec in raw.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        parts = [p.strip() for p in spec.split(",")]
        # host, nproc, cvd, ip — cvd may itself contain commas (0,1,2,3).
        if len(parts) < 4:
            raise SystemExit(f"bad {env_key} entry: {spec!r} (want host,nproc,cvd,ip)")
        host, nproc, ip = parts[0], parts[1], parts[-1]
        cvd = ",".join(parts[2:-1])
        nodes.append((host, int(nproc), cvd, ip))
    return tuple(nodes)


def master_addr(nodes: tuple[tuple[str, int, str, str], ...], env_key: str) -> str:
    override = os.environ.get(env_key, "").strip()
    return override or nodes[0][3]


if __name__ == "__main__":
    print(load_rdzv_nodes(), file=sys.stderr)
