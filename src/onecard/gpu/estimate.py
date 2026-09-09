import math

DEFAULT_KV_MB_PER_1K = 64


def estimate_kv_mb(num_ctx: int, mb_per_1k: int = DEFAULT_KV_MB_PER_1K) -> int:
    """Estimate KV-cache VRAM for a context window.

    A deliberate over-estimate: partial 1k blocks round up, because
    under-estimating causes a spill to system RAM, which is the failure
    this project exists to prevent. The rate is configurable per deployment
    (`kv_mb_per_1k_ctx`) because it varies by model architecture.
    """
    if num_ctx < 0:
        raise ValueError("num_ctx must be non-negative")
    return math.ceil(num_ctx / 1024) * mb_per_1k
