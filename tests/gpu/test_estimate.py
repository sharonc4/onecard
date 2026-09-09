import pytest

from onecard.gpu.estimate import estimate_kv_mb


def test_scales_linearly_with_context():
    assert estimate_kv_mb(1024, 64) == 64
    assert estimate_kv_mb(4096, 64) == 256


def test_partial_blocks_round_up():
    """Never under-estimate: a partial 1k block still costs a full block."""
    assert estimate_kv_mb(1500, 64) == 128


def test_zero_context_costs_nothing():
    assert estimate_kv_mb(0, 64) == 0


def test_rate_is_configurable():
    assert estimate_kv_mb(2048, 100) == 200


def test_negative_context_is_rejected():
    with pytest.raises(ValueError):
        estimate_kv_mb(-1, 64)
