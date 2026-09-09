"""Reading what the card *actually* holds, as opposed to what a backend claims.

Why this exists, measured on an RTX 2070 SUPER (8GB):

- With Ollama's model loaded, its container held ~6GB of the 8GB card. Asking
  Ollama to release with `keep_alive: 0` did not give that memory back — only
  stopping the container did. So a backend can truthfully report "no models
  resident" while its process still holds gigabytes.
- Because Ollama ran in Docker, it was also invisible to
  `nvidia-smi --query-compute-apps`, which showed only
  `pid 524 [Insufficient Permissions]`. Per-process attribution is unreliable;
  the total is not.
- A request that exceeds free VRAM does NOT fail. A 10.08GB peak on this 8GB
  card ran to completion in 103s where a 5.28GB peak took 2.3s — a 45x
  slowdown, silent, with no error. Slow inference on a small card means "over
  VRAM", not "compute-bound".

The conclusion the arbiter acts on: a backend's own bookkeeping cannot be the
only check. Before granting a claim we ask the driver how much is genuinely
free, because the memory held against us may belong to something onecard does
not manage at all.
"""

import shutil
import subprocess
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

NVIDIA_SMI_QUERY = "memory.total,memory.used,memory.free"


@dataclass(frozen=True)
class VramReading:
    """A point-in-time reading of the whole card, in MiB."""

    total_mb: int
    used_mb: int
    free_mb: int


@runtime_checkable
class VramProbe(Protocol):
    """Reads total/used/free VRAM.

    `read()` returns None when no reading can be taken — no GPU, no driver
    tool, or a call that failed. None means "unknown", never "zero" and never
    "fine": callers must say so rather than assuming the card is empty.
    """

    def read(self) -> VramReading | None: ...


class NvidiaSmiProbe:
    """Reads VRAM totals from `nvidia-smi`.

    Deliberately reads the card total rather than per-process usage: a backend
    running inside Docker does not show up in per-process output, and its
    memory is exactly the memory we most need to notice.
    """

    def __init__(self, executable: str = "nvidia-smi", timeout_s: float = 5.0) -> None:
        self.executable = executable
        self.timeout_s = timeout_s

    def read(self) -> VramReading | None:
        if shutil.which(self.executable) is None:
            return None
        try:
            proc = subprocess.run(
                [
                    self.executable,
                    f"--query-gpu={NVIDIA_SMI_QUERY}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        return parse_nvidia_smi(proc.stdout)


def parse_nvidia_smi(output: str) -> VramReading | None:
    """Parse `total, used, free` in MiB from nvidia-smi CSV output.

    Only the first GPU is read. onecard's whole subject is the single-card
    machine, and a multi-GPU rig does not have the problem this project exists
    to solve.
    """
    for line in output.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            total, used, free = (int(f) for f in fields)
        except ValueError:
            continue
        return VramReading(total_mb=total, used_mb=used, free_mb=free)
    return None


class StaticVramProbe:
    """A probe with a fixed answer.

    For tests, and for a machine where the operator would rather state the
    truth than have it discovered.
    """

    def __init__(self, reading: VramReading | None) -> None:
        self._reading = reading

    def read(self) -> VramReading | None:
        return self._reading
