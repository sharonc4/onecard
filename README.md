# onecard

**A local AI assistant for one small GPU.**

Every local-AI tool assumes you have 24GB of VRAM. On an 8GB card you can hold
*one* 7B model — so you get one generalist, and generalists are worse at every
specific job than a specialist would be.

onecard lets you declare a specialist per task and swaps them for you:

```yaml
tasks:
  chat:      { model: reasoner }
  summarize: { model: fast }
```

```bash
onecard run summarize "some long text"
```

The shipped `profiles/8gb-developer.yaml` adds a `coder` model and a `code`
task for exactly this kind of prompt:

```bash
onecard run code "why is this function slow?" --config profiles/8gb-developer.yaml
```

It is a **GPU arbiter**, not just a model router: one component owns the card
and decides who holds it, so nothing ever quietly spills to system RAM.

## Quickstart

```bash
docker compose up -d
docker compose run --rm onecard validate
docker compose run --rm onecard run summarize "some long text"
```

No NVIDIA card? Everything still runs, slowly, on CPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

## Status

Early. The core router, arbiter, and CLI work. Image generation, voice, memory,
and tools are designed but not yet built — see
[the design doc](docs/superpowers/specs/2026-09-08-onecard-design.md).

**No VRAM figure in this repo has been measured on real 8GB hardware yet.** If
you have such a card, correcting the numbers in `profiles/` is the single most
useful contribution you can make.

## Development

Install dev dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -v
```

## License

Apache-2.0
