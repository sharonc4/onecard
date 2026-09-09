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

It is a **GPU arbiter**, not just a model router: one component owns the card,
enforces the VRAM budget you declare, and confirms against the backend that an
eviction actually happened before loading anything on top of it.

## Quickstart

```bash
docker compose up -d
docker compose exec ollama ollama pull qwen2.5:1.5b-instruct-q4_K_M
docker compose run --rm onecard validate
docker compose run --rm onecard run summarize "some long text"
```

The `pull` is the model `onecard.yaml`'s `summarize` task uses; without it the
run fails with a model-not-found error from Ollama. `onecard run code` from the
developer profile additionally needs `ollama pull qwen2.5-coder:7b-instruct-q4_K_M`.

No NVIDIA card? Everything still runs, slowly, on CPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

## Status

Early. The core router, arbiter, and CLI work. Image generation, voice, memory,
and tools are designed but not yet built — see
[the design doc](docs/superpowers/specs/2026-09-08-onecard-design.md).

**What is measured and what is not.** The VRAM budgets come from a real
RTX 2070 SUPER (8192 MiB): 1630 MiB sits idle when the card drives both
monitors, 894 MiB with one moved to the motherboard, 662 MiB headless. Moving
your displays off the GPU buys nearly a gigabyte, which is routinely the
difference between a model fitting and silently spilling. The per-model
`footprint_mb` numbers in `profiles/` are still estimates from published
quantization sizes — onecard measures each model on first load and remembers
the real value, so correcting those in a PR is the single most useful
contribution you can make.

**Why "silently" is the right word.** On that same card a workload peaking at
10.08 GB did not fail — it finished in 103s where a 5.28 GB peak took 2.3s. And
with Ollama holding 6 GB, a Flux render degraded to ~916 s/step. No error, no
warning, just a machine that looks broken. onecard reads free VRAM from the
driver before every claim and refuses one that would not fit, because a backend
can report "nothing loaded" while its process still holds gigabytes.

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
