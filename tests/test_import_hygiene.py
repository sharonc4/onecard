import subprocess
import sys


def test_package_imports_no_gpu_libraries():
    """The onecard package must never pull in torch/CUDA. It talks HTTP only."""
    code = (
        "import onecard, sys;"
        "bad=[m for m in sys.modules if m.split('.')[0] in "
        "{'torch','tensorflow','pycuda','cupy','nvidia'}];"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "", f"GPU libraries imported: {out.stdout}"


def test_error_hierarchy():
    from onecard.errors import (
        BackendError,
        BudgetError,
        ConfigError,
        EvictionError,
        OneCardError,
    )

    for cls in (ConfigError, BudgetError, BackendError, EvictionError):
        assert issubclass(cls, OneCardError)
