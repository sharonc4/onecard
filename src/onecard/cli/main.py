import asyncio
from collections.abc import Callable
from pathlib import Path

import httpx
import typer

from onecard.backend.ollama import OllamaConsumer
from onecard.config.loader import load_config
from onecard.config.schema import Config, MemorySpec, TaskSpec
from onecard.config.validate import validate_config
from onecard.errors import BackendError, OneCardError
from onecard.gpu.arbiter import Arbiter
from onecard.router.execute import Executor
from onecard.router.plan import build_plan
from onecard.store.footprints import FootprintStore

app = typer.Typer(help="A local AI assistant for one small GPU.", no_args_is_help=True)

ConfigOpt: Path = typer.Option(
    Path("onecard.yaml"), "--config", "-c", envvar="ONECARD_CONFIG", help="Path to onecard.yaml"
)
OllamaOpt: str = typer.Option(
    "http://localhost:11434", "--ollama", envvar="ONECARD_OLLAMA", help="Ollama base URL"
)
DataOpt: Path = typer.Option(
    Path("data"), "--data", envvar="ONECARD_DATA", help="Directory for the SQLite stores"
)
ExplainOpt: bool = typer.Option(False, "--explain", help="Show routing decisions")
TaskArg: str = typer.Argument(..., help="Task name from the config")
TextArg: str = typer.Argument(..., help="Input for the task")


def _fail(exc: Exception) -> None:
    typer.echo(str(exc))
    raise typer.Exit(code=1)


def _make_client(ollama: str) -> httpx.AsyncClient:
    """Build the Ollama HTTP client, turning a malformed --ollama URL into a BackendError.

    httpx validates the base URL inside AsyncClient's constructor and raises
    httpx.InvalidURL there -- before OllamaConsumer ever sees it -- and that
    exception subclasses plain Exception rather than httpx.HTTPError, so it
    would otherwise slip past the existing OneCardError handling as a traceback.
    """
    try:
        return httpx.AsyncClient(base_url=ollama)
    except httpx.InvalidURL as exc:
        raise BackendError(f"invalid --ollama URL '{ollama}': {exc}") from exc


def _deferred_warnings(cfg: Config) -> list[str]:
    """Warn about keys this build parses and then ignores.

    These are validly configured -- it is this build that lacks the features --
    so they belong here rather than in `validate_config`. Saying "ok" to a
    config whose permission allowlist does nothing would be a lie in a project
    whose pitch is that a task provably cannot read the filesystem.
    """
    warnings: list[str] = []
    default_memory = MemorySpec()

    def named(predicate: Callable[[TaskSpec], bool]) -> str:
        return ", ".join(sorted(n for n, t in cfg.tasks.items() if predicate(t)))

    workflow_tasks = named(lambda t: t.workflow is not None)
    if workflow_tasks:
        warnings.append(
            f"task(s) {workflow_tasks} declare 'workflow' and cannot run in this "
            "build (no ComfyUI consumer)"
        )
    permission_tasks = named(lambda t: bool(t.permissions))
    if permission_tasks or cfg.defaults.permissions:
        where = permission_tasks or "defaults"
        warnings.append(
            f"'permissions' is declared ({where}) but is not enforced by this build: "
            "no sandbox is applied and every task has the same access this process has"
        )
    tool_tasks = named(lambda t: bool(t.tools))
    if tool_tasks:
        warnings.append(
            f"'tools' is declared (task(s) {tool_tasks}) but this build has no tool "
            "runtime; the allowlist has no effect"
        )
    memory_tasks = named(lambda t: t.memory != default_memory)
    if memory_tasks or cfg.defaults.memory != default_memory:
        where = f"task(s) {memory_tasks}" if memory_tasks else "defaults"
        warnings.append(
            f"'memory' is declared ({where}) but this build has no memory store; "
            "nothing is read or written"
        )
    exclusive_tasks = named(lambda t: t.exclusive and t.workflow is None)
    if exclusive_tasks:
        warnings.append(
            f"'exclusive: true' on model task(s) {exclusive_tasks} is ignored by this "
            "build; only workflow tasks take the card exclusively"
        )
    if cfg.disk_budget_gb is not None:
        warnings.append(
            f"'disk_budget_gb: {cfg.disk_budget_gb}' is recorded but not enforced: "
            "this build never pulls or prunes models"
        )
    if cfg.defaults.model is not None:
        warnings.append(
            f"'defaults.model: {cfg.defaults.model}' is ignored; every task must name "
            "its own model in this build"
        )
    return warnings


@app.command()
def validate(config: Path = ConfigOpt) -> None:
    """Check the config before anything tries to use it."""
    try:
        cfg = load_config(config)
        warnings = validate_config(cfg, config.parent)
    except OneCardError as exc:
        _fail(exc)
        return
    warnings.extend(_deferred_warnings(cfg))

    for w in warnings:
        typer.echo(f"warning: {w}")
    typer.echo(
        f"ok: {len(cfg.tasks)} tasks, {len(cfg.models)} models, "
        f"budget {cfg.vram_budget_mb}MB"
    )


@app.command()
def ps(config: Path = ConfigOpt, ollama: str = OllamaOpt) -> None:
    """Show what is currently resident on the card, and what budget remains."""

    async def _run() -> None:
        cfg = load_config(config)
        async with _make_client(ollama) as client:
            residents = await OllamaConsumer(base_url=ollama, client=client).residents()
        used = sum(r.footprint_mb for r in residents)
        for r in residents:
            typer.echo(f"{r.consumer:8} {r.key:40} {r.footprint_mb:6}MB")
        typer.echo(f"{'':8} {'TOTAL':40} {used:6}MB of {cfg.vram_budget_mb}MB")

    try:
        asyncio.run(_run())
    except OneCardError as exc:
        _fail(exc)


@app.command()
def run(
    task: str = TaskArg,
    text: str = TextArg,
    config: Path = ConfigOpt,
    ollama: str = OllamaOpt,
    data: Path = DataOpt,
    explain: bool = ExplainOpt,
) -> None:
    """Run a named task."""

    async def _run() -> None:
        cfg = load_config(config)
        validate_config(cfg, config.parent)
        plan = build_plan(cfg, task, config.parent)
        if explain:
            typer.echo(f"task={plan.task} models={plan.load_sequence}", err=True)
        async with _make_client(ollama) as client:
            consumer = OllamaConsumer(base_url=ollama, client=client)
            arbiter = Arbiter(budget_mb=cfg.vram_budget_mb, consumers={"ollama": consumer})
            executor = Executor(
                arbiter=arbiter,
                chat_fn=consumer.chat,
                store=FootprintStore(data / "footprints.db"),
                on_event=lambda msg: typer.echo(msg, err=True),
            )
            async for chunk in executor.run(plan, text):
                typer.echo(chunk, nl=False)
        typer.echo("")

    try:
        asyncio.run(_run())
    except (OneCardError, NotImplementedError) as exc:
        _fail(exc)


if __name__ == "__main__":
    app()
