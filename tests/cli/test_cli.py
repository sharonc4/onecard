import textwrap
from pathlib import Path

from typer.testing import CliRunner

from onecard.cli.main import app

runner = CliRunner()

GOOD = textwrap.dedent(
    """
    vram_budget_mb: 8000
    models:
      coder: { ref: "qwen2.5-coder:7b", footprint_mb: 4500 }
    tasks:
      code: { model: coder }
    """
)

TOO_BIG = textwrap.dedent(
    """
    vram_budget_mb: 1000
    models:
      coder: { ref: "qwen2.5-coder:7b", footprint_mb: 4500 }
    tasks:
      code: { model: coder }
    """
)


def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "onecard.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_validate_accepts_a_good_config(tmp_path: Path):
    result = runner.invoke(app, ["validate", "--config", str(write(tmp_path, GOOD))])
    assert result.exit_code == 0
    assert "ok" in result.stdout.lower()


def test_validate_rejects_an_overcommitted_config(tmp_path: Path):
    result = runner.invoke(app, ["validate", "--config", str(write(tmp_path, TOO_BIG))])
    assert result.exit_code == 1
    assert "can never fit" in result.stdout


def test_validate_reports_missing_config_clearly(tmp_path: Path):
    result = runner.invoke(app, ["validate", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 1
    assert "not found" in result.stdout


def test_run_with_unknown_task_fails_clearly(tmp_path: Path):
    result = runner.invoke(
        app, ["run", "ghost", "hi", "--config", str(write(tmp_path, GOOD))]
    )
    assert result.exit_code == 1
    assert "unknown task 'ghost'" in result.stdout
