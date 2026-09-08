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

WORKFLOW = textwrap.dedent(
    """
    vram_budget_mb: 6800
    tasks:
      picture: { workflow: "workflows/sd15.json", exclusive: true }
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


def test_run_on_a_workflow_task_fails_clearly_instead_of_a_traceback(tmp_path: Path):
    result = runner.invoke(
        app, ["run", "picture", "a cat", "--config", str(write(tmp_path, WORKFLOW))]
    )
    assert result.exit_code == 1
    assert "image tasks" in result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_validate_warns_about_workflow_tasks_but_still_says_ok(tmp_path: Path):
    result = runner.invoke(app, ["validate", "--config", str(write(tmp_path, WORKFLOW))])
    assert result.exit_code == 0
    assert "warning" in result.stdout.lower()
    assert "workflow" in result.stdout.lower()
    assert "cannot run" in result.stdout.lower()
    assert "ok" in result.stdout.lower()


def test_validate_honours_onecard_config_env_var(tmp_path: Path):
    config_path = write(tmp_path, GOOD)
    result = runner.invoke(app, ["validate"], env={"ONECARD_CONFIG": str(config_path)})
    assert result.exit_code == 0
    assert "ok" in result.stdout.lower()


def test_ps_with_a_malformed_ollama_url_fails_clearly(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "ps",
            "--config", str(write(tmp_path, GOOD)),
            "--ollama", "http://[::1",
        ],
    )
    assert result.exit_code == 1
    assert "http://[::1" in result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)
