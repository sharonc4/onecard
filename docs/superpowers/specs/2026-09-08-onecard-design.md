# onecard — Design

**Date:** 2026-09-08
**Status:** Approved design, pre-implementation
**License:** Apache-2.0

## Thesis

Every local-AI assistant on the market tacitly assumes you have 16–24GB of VRAM.
On an 8GB card you can hold roughly **one** 7–8B model at Q4 (~4.5–5.5GB) plus its
KV cache. That means you get one generalist, and generalists are worse at every
specific job than a specialist would be.

onecard's premise: **you have one small card and you want five specialists.**
Tasks are declared in YAML, each pinned to a specific model, and the harness
swaps models in and out of VRAM so the user experiences a multi-model assistant
on hardware that can only hold one at a time.

Model switching is therefore not a feature. It is the only mechanism by which a
low-VRAM user can have specialists at all, and its cost — load latency — is the
central engineering problem of the project.

The same argument extends past language models. An image generator wants the
whole card too, and if a 5GB LLM is already resident when ComfyUI loads a
checkpoint, the checkpoint spills to system RAM and generation slows by two
orders of magnitude — **silently**, with no error, looking exactly like a hang.
So onecard is not a model router with an image plugin. It is a **GPU arbiter**:
one component owns the card and decides who holds it, whether the claimant is an
Ollama model or a ComfyUI workflow.

That is the part no other tool does, because on 24GB the question never comes up.

## Scope

**Target user:** a developer or enthusiast with a single consumer GPU, 8GB VRAM,
running Linux or Windows with Docker and an NVIDIA card.

**In v1:**

- YAML-declared tasks mapping to specific models
- GPU arbitration: VRAM budget accounting and swap policy across *all* GPU consumers
- Image generation via ComfyUI, arbitrated against the LLM backend
- Voice in and out (Whisper, Piper), CPU-only by design
- Memory (durable facts + retrieved conversation history)
- Tool actions with per-task permission allowlists
- Plugin interface, with two reference connectors (Obsidian, HTTP/webhook)
- CLI, HTTP API, OpenAI-compatible endpoint, streaming responses
- First-run wizard: detect the card, pick a profile, pull the models
- Disk budget accounting for downloaded models
- Docker Compose packaging with a CPU-only override

**Explicitly out of v1:**

- A web chat UI (Open WebUI already exists and can point at the compatible endpoint)
- LLM backends other than Ollama (the backend is behind an interface, but only one implementation ships)
- Document RAG over a user-chosen folder (overlaps the memory layer; deferred to v2)
- Model auto-download on demand — pulling 5GB mid-request is a hostile surprise
- Autonomous multi-step agent loops — small models plan badly enough that this
  would only ever be a demo
- Speculative decoding — requires draft and target models resident together,
  the one thing 8GB cannot do
- Multi-user support, auth, or anything network-facing beyond localhost
- Fine-tuning, training, or model conversion

## Architecture

Three containers. Two of them want the GPU; exactly one may hold it at a time,
and `onecard` decides which.

| Service | Responsibility | GPU |
|---|---|---|
| `ollama` | Holds language models, performs inference | Yes (arbitrated) |
| `comfyui` | Image generation workflows | Yes (arbitrated) |
| `onecard` | Arbiter, router, memory, tools, voice, API, CLI | No |

`comfyui` is optional: a `docker-compose.noimage.yml` override omits it, and a
config with no image tasks never starts it.

The `onecard` container never imports torch or CUDA. It talks to Ollama over
HTTP. This keeps the image small, keeps GPU concerns entirely in Ollama's
domain, and means the full harness can be developed and tested on a machine
with no GPU at all, against a CPU-hosted small model — the code path is
identical, only slower.

Internal module boundaries, each independently testable:

```
onecard/
  config/     # schema, parsing, validation
  router/     # task -> model resolution, pipeline scheduling
  gpu/        # arbiter: budget accounting, residency policy, eviction, claims
  backend/    # GpuConsumer implementations: Ollama, ComfyUI
  voice/      # Whisper STT + Piper TTS, CPU-only
  memory/     # SQLite facts + embedded history, retrieval
  tools/      # tool registry, call loop, argument validation
  permissions/# allowlist matcher
  plugins/    # entry-point discovery and contract
  api/        # FastAPI app, OpenAI-compatible shim, streaming
  cli/        # command-line entrypoint, first-run wizard
  disk/       # model disk accounting
```

## GPU arbitration: budget and swap policy

Every consumer of the card implements one `GpuConsumer` interface: report what
you hold, load this, release everything. Ollama and ComfyUI are the two v1
implementations. The arbiter reasons about claims, not about model families.

**The budget is declared, never inferred.** `vram_budget_mb` is set by the user.
An 8GB card realistically offers ~6500–7000MB after the display driver. The
harness reads actual free VRAM at boot only to emit a warning when the declared
budget looks optimistic. It never silently adjusts the budget: a limit that
moves under the user is the precise class of silent failure this project exists
to avoid.

**Two residency tiers.** Each model declares `residency: pinned | on_demand`.

- `pinned` — loaded at startup and never evicted. Intended for small models
  (0.5–1.5B, ~400MB–1GB) and embedding models (~100–300MB).
- `on_demand` — loaded when a task needs it, evicted under pressure.

On 8GB the workable shape is one on-demand 7–8B specialist plus one pinned small
model plus one pinned embedding model. The pinned small model handles memory
query rewriting, tool-argument extraction, and titling, so the cheap steps of a
job never trigger a swap.

**Fit calculation.** Before invoking a task the harness computes
`footprint(model) + kv_estimate(num_ctx)` and evicts the least-recently-used
`on_demand` model until it fits. KV cache size scales with the task's declared
`num_ctx`, which is why `num_ctx` is a per-task parameter and part of the fit
calculation rather than a global.

**Eviction mechanism.** The harness does not manage GPU memory directly. It
expresses policy through Ollama's per-request `keep_alive` (`-1` to pin, `0` to
evict immediately, a duration otherwise) and reads `/api/ps` for ground truth
about what is actually resident.

**Footprints self-correct.** Declared footprints are estimates. On first load
the harness records the observed VRAM delta from `/api/ps` into SQLite and uses
the measured value from then on. There is no hand-maintained size table to rot.

**Swap-aware scheduling.** A task may be a multi-step pipeline. The scheduler
groups consecutive steps sharing a model so a three-step job performs one swap,
not three. Every swap is logged with its duration so users can see why something
felt slow.

**Overcommit is a config error.** If pinned models exceed the budget, or a task's
model plus KV cannot fit even alone, `onecard validate` fails. This is caught at
validate time, not at request time.

### Exclusive claims

Some consumers cannot share. A ComfyUI workflow declares `exclusive: true`,
meaning the arbiter must evict **every** other resident model — pinned ones
included — before the claim is granted, and restore them afterward.

The sequence for an image task:

1. Snapshot current residency (including pinned models, so they can be restored).
2. Evict everything via `keep_alive: 0`, and confirm against `/api/ps` that the
   card is actually clear. **Confirm, not assume** — proceeding on an
   unconfirmed eviction is what produces the silent spill this design exists to
   prevent.
3. Grant the claim; ComfyUI loads its checkpoint and runs.
4. Release, then restore the pinned set.

This is genuinely slow — two swaps plus a checkpoint load. The CLI and API must
**report the arbitration steps as they happen**, so an image request reads as
"evicting reasoner → loading checkpoint → generating" rather than a thirty-second
silence the user interprets as a crash.

Exclusive claims are serialized behind a single lock. A second image request
waits rather than racing; concurrent exclusive claims are the failure mode that
produces two half-loaded checkpoints and a thrashing card.

## Configuration

One file, `onecard.yaml`. It is the entire user-facing surface — preferences,
routing, and permissions — and is intended to be shared and forked.

```yaml
vram_budget_mb: 6800
disk_budget_gb: 60          # refuse pulls that would exceed this

voice:                       # CPU-only; excluded from the VRAM budget
  stt: { engine: whisper, model: base }
  tts: { engine: piper,   voice: en_US-amy-medium }

image:
  backend: comfyui
  url: http://comfyui:8188

models:
  fast:      { ref: "qwen2.5:1.5b-instruct-q4_K_M", residency: pinned }
  embed:     { ref: "nomic-embed-text",             residency: pinned }
  reasoner:  { ref: "llama3.1:8b-instruct-q4_K_M",  residency: on_demand }
  coder:     { ref: "qwen2.5-coder:7b-q4_K_M",      residency: on_demand }

defaults:
  model: reasoner
  memory: { read: true, write: false }
  permissions: []

tasks:
  chat:
    model: reasoner
    memory: { read: true, write: true }

  code:
    model: coder
    params: { temperature: 0.2, num_ctx: 16384 }
    permissions: [ "fs.read:./**", "shell.run:git *" ]
    tools: [ read_file, list_files, git ]

  summarize:
    model: fast
    prompt: prompts/summarize.md
    permissions: []

  # An image task claims the GPU exclusively: everything else is evicted first.
  picture:
    workflow: workflows/sd15_txt2img.json
    exclusive: true
    params: { steps: 25, width: 512, height: 512 }
    permissions: [ "fs.write:./data/images/**" ]

  # A task may instead declare ordered steps. Each step names a model; the
  # scheduler groups consecutive steps sharing a model into one load.
  review:
    steps:
      - { model: coder,    prompt: prompts/find_issues.md }
      - { model: coder,    prompt: prompts/rank_issues.md }   # no swap: same model
      - { model: fast,     prompt: prompts/format_report.md } # one swap here
    permissions: [ "fs.read:./**" ]
```

A task declares either `model` (single step) or `steps` (ordered pipeline), never
both; declaring both is a validation error. Each step inherits the task's
permissions, tools, and memory settings — steps cannot widen a task's authority.

Models are family-agnostic: any reference Ollama can pull is valid, and nothing
in the router assumes a family. Mixing families is the intent — a strong coder
model for `code`, a different general model for `chat`, a small vision model for
`describe`.

**Tasks are the unit of everything.** A task binds a model, sampling parameters,
a prompt, a tool allowlist, a permission set, and memory access. This is why
`code` can read the filesystem and `summarize` provably cannot.

**Permissions are explicit glob allowlists**, empty by default. A tool call
outside the allowlist is a hard, logged refusal — never a silent no-op.

**`onecard validate`** checks that every model exists in Ollama, pinned
footprints fit the budget, every task's model is defined, every permission
string parses, and every referenced plugin loads.

## Profiles

The repo ships `profiles/` — ready-made configs per VRAM class and use case:
`8gb-developer.yaml`, `8gb-writer.yaml`, `8gb-vision.yaml`, `12gb-general.yaml`.
Each selects a different model per task, across families.

Every profile carries a dated header naming the models it pins, the evidence for
those choices, and when it was last reviewed. "Model X is best at Y" is a claim
with a shelf life of months. Profiles are versioned, dated, and PR-able — which
also gives contributors an obvious low-friction way to participate.

No "best model per task" table is baked into the router's code, where it could
not be corrected without a release.

## Image generation

Image tasks name a **ComfyUI workflow JSON**, not a model. onecard submits the
workflow via ComfyUI's HTTP API with parameter overrides, polls for completion,
and writes the result to a permitted path. It does not build workflows, expose a
node graph, or attempt to be a ComfyUI frontend — ComfyUI already is one.

**Be honest about speed.** At 8GB an image request costs an eviction, a
checkpoint load, generation, and a restore. SD1.5-class checkpoints (~2–4GB) are
the realistic default; SDXL is tight; Flux is impractical without aggressive
quantization and patience. The shipped profile uses SD1.5 and the docs say why
rather than letting users discover it as a disappointment.

**Failure is loud.** If ComfyUI is unreachable, the workflow JSON is invalid, or
a node is missing, the task errors with the cause. There is no fallback to a
different workflow, and a generation that silently produced nothing is an error,
not an empty result — the same rule as everywhere else in this design.

## Voice

Two independent capabilities, both **CPU-only and excluded from the VRAM
budget**:

- **Speech in** — Whisper (`base`/`small`), transcribing to text before routing.
- **Speech out** — Piper, synthesizing a task's text response.

The exclusion is deliberate. These models are small enough to run acceptably on
CPU, and the moment voice competes for the card it reintroduces the arbitration
cost for the sake of a few hundred milliseconds. Voice must never cause a swap.

Voice is a transport, not a task type: `onecard chat --voice` transcribes input,
routes it through the normal task machinery, and speaks the result. Any task can
be driven by voice; no task is voice-specific.

## First-run wizard

`onecard init` detects the installed GPU and its VRAM, recommends a matching
profile, shows the total disk cost of that profile's models, and — on
confirmation — pulls them and writes an `onecard.yaml`.

This exists because the target user wants an assistant, not a VRAM budgeting
exercise. Everything the wizard does is also doable by hand; the wizard writes a
plain config file the user can then read and edit. It is a starting point, not a
layer of magic that owns the configuration.

If no supported GPU is detected, the wizard says so plainly and offers the
CPU-only profile rather than writing a config that cannot work.

## Disk budget

Models are large and multiply quietly: five specialists at ~5GB each is 25GB,
before checkpoints and voice models. `disk_budget_gb` caps the total.

`onecard pull` reports what it is about to download and the resulting total
before starting, and refuses to exceed the budget. `onecard ps --disk` shows
current usage per model. Nothing is ever downloaded as a side effect of a
request — pulls are always explicit.

## Memory

SQLite, one file on a bind mount. Two layers:

- **Facts** — small, explicitly written, always injectable. Plain rows a human
  can read and delete. No opaque vector blob as the only copy.
- **History** — conversation chunks, embedded and retrieved.

Retrieval uses a local embedding model through the same Ollama instance.
Embedding models are small and pinned, so recall never triggers a swap.

Writes are gated per task via `memory.write`, so a summarizer cannot quietly
rewrite what the assistant believes about the user.

## Tools and permissions

Tools are Python functions with typed signatures; the harness generates the
schema and runs the call loop.

**Small models call tools unreliably.** A 7B at Q4 produces malformed arguments
regularly. Mitigations, in order:

1. Constrained decoding via a JSON grammar where the backend supports it.
2. A validation layer that rejects malformed arguments and re-prompts once.
3. After a second failure, a loud logged error — never a shrug.

Every tool call is recorded with task, arguments, permission decision, and
result.

## Plugins

One entry-point contract registers both tools and memory sources. Two reference
implementations ship: an Obsidian vault connector (read notes, append notes) and
a generic HTTP/webhook connector.

Connectors are deliberately not built in beyond these two. Each hard-coded
connector is a permanent maintenance tail on a single-maintainer repo; a stable
extension point is the sustainable alternative.

## Interfaces

The router is a library; every surface is a thin adapter over it.

**CLI** (primary, and what the README demos):

- `onecard init` — first-run wizard: detect card, pick profile, pull models
- `onecard run <task> "<input>"` — streams tokens by default
- `onecard chat [--voice]`
- `onecard validate`
- `onecard ps [--disk]` — what is resident, measured footprint, remaining budget
- `onecard pull --profile 8gb-developer` — fetch every model a profile needs,
  reporting disk cost first

Arbitration progress is printed to stderr as it happens (`evicting reasoner…`,
`loading checkpoint…`), so slow operations never look like a hang. Token
streaming is on by default in both CLI and API; `/v1/chat/completions` honours
the standard `stream` parameter.

**HTTP API** (FastAPI):

- `POST /task/{name}`
- `POST /v1/chat/completions` — OpenAI-compatible, with the `model` field
  carrying the task name. This makes onecard a drop-in for any existing
  OpenAI-speaking client with no client changes.

**`--explain`** on any run prints the routing decision, model chosen, permissions
granted, tools offered, and memory injected.

## Packaging

```yaml
services:
  ollama:
    image: ollama/ollama
    volumes: [ ollama-models:/root/.ollama ]
    deploy:
      resources:
        reservations:
          devices: [ { driver: nvidia, count: 1, capabilities: [gpu] } ]
  comfyui:
    image: comfyui
    volumes: [ ./data/comfyui:/data ]
    deploy:
      resources:
        reservations:
          devices: [ { driver: nvidia, count: 1, capabilities: [gpu] } ]
  onecard:
    build: .
    depends_on: [ ollama ]
    volumes:
      - ./onecard.yaml:/config/onecard.yaml:ro
      - ./data:/data
    ports: [ "8080:8080" ]
```

Both GPU services reserve the same device. That is intentional and safe *only*
because onecard guarantees they are never resident simultaneously — the
arbitration lock, not Docker, is what enforces exclusivity. This is the single
most important invariant in the system, and it is enforced in one place.

A `docker-compose.noimage.yml` override omits `comfyui` entirely for users who
do not want image generation. A `docker-compose.cpu.yml` override drops the device reservation so contributors
without an NVIDIA card can run tests and submit PRs. A project targeting people
with modest hardware must not require good hardware to contribute to.

Config is mounted read-only; data is a visible bind mount at `./data`. For a
tool whose pitch is "runs on your own infrastructure," being able to inspect and
delete your own data with `ls` and `rm` is part of the argument.

## Failure behavior

Nothing degrades quietly. Three explicit commitments:

1. A model that will not fit the declared budget fails at `onecard validate`,
   not mid-request.
2. A tool call violating a permission is a hard, logged refusal — never a silent
   skip, and never an empty result passed downstream as if it succeeded.
3. If a task's model cannot be pulled, or Ollama is unreachable, the request
   errors with the actual cause. **There is no automatic fallback to a different
   model.** A harness whose premise is "this task uses *this* model" must never
   silently substitute another and hand back output the user will misattribute.

## Testing

- **Unit** — router, VRAM accounting, permission matcher, and config validator
  are pure logic, tested against a fake Ollama backend. No GPU, no model
  downloads; fast enough for CI on GitHub's free runners.
- **Integration** — one suite against a real Ollama with a single tiny CPU
  model, exercising the actual load and evict path.
- **Swap policy specifically** — a three-step pipeline must produce exactly one
  swap; eviction must select LRU; overcommitted configs must fail validation.
- **Arbitration specifically** — an exclusive claim must evict pinned models and
  restore them afterward; a claim must not be granted until eviction is
  *confirmed*, not merely requested; two concurrent exclusive claims must
  serialize rather than overlap. The fake backend simulates a consumer that
  reports itself still resident after an evict request, so the confirm-don't-
  assume path is actually exercised.

## Repository

- Apache-2.0 (permissive, with a patent grant — the right default for
  infrastructure intended for adoption).
- CI runs lint, type checks, and the fake-backend suite on every PR.
- README leads with the 8GB thesis and a 60-second quickstart.
- `CONTRIBUTING.md` points at `profiles/` as the easiest first contribution.

## Open items

- **onecard itself has not run on 8GB hardware, but the card has been measured.**
  The budget and headroom figures below come from an RTX 2070 SUPER (8GB) that
  ran Flux, ComfyUI, Ollama and a VLM detector for months. What remains
  unmeasured is onecard's own swap timing and per-model Ollama footprints.

  | Measurement | Value |
  |---|---|
  | Card total | 8192 MiB |
  | Idle, both displays driven by the card | 1630 MiB used → 6562 free |
  | Idle, one display moved to the motherboard | 894 MiB used → 7298 free |
  | Idle, headless (both on the motherboard) | 662 MiB used → 7530 free |
  | Largest Flux UNet that fits headless | Q4_K_S, 6.33 GB |
  | Next quantization up | Q5_K_S, 7.71 GB — does **not** fit |
  | Render peak, headless | 6844 MiB (84%) |
  | Render peak, display attached | 7682 MiB (94%) |

  Two conclusions the design now rests on. First, **the display tax is real and
  large**: moving both monitors to the integrated GPU bought 968 MiB, which is
  the difference between a model fitting and not. Second, **weights fitting is
  what matters, not raw speed** — losing a second GPU cost no time at all
  (219.7s → 227.4s per frame) because nothing had to stream over PCIe.

- **Spilling is silent and enormous, and that is now quantified.** On the same
  card, a workload peaking at 10.08 GB did not fail — it completed in 103s
  where a 5.28 GB peak took 2.3s. A 45x slowdown, no error, no warning. Ollama
  holding 6 GB during a Flux render degraded it to ~916 s/step. This is the
  entire justification for the project: on a small card, *slow means over-VRAM,
  not compute-bound*, and nothing in the stack tells you.
- ~~**Eviction confirmation needs a real-hardware answer.**~~ **ANSWERED —
  yes, and more strongly than expected.** Measured on an RTX 2070 SUPER (8GB):
  Ollama's container held ~6GB while a model was loaded, and a `keep_alive: 0`
  release **did not give that memory back** — only stopping the container did.
  Because it ran in Docker it was also invisible to
  `nvidia-smi --query-compute-apps`, which showed only
  `pid 524 [Insufficient Permissions]`. So a backend can truthfully answer "no
  models resident" while still holding gigabytes, and per-process attribution
  cannot be trusted either. Asking the consumer is necessary but not
  sufficient. The arbiter now also reads total free VRAM from the driver and
  refuses any claim larger than it (`gpu/probe.py`).
- **ComfyUI container image is unpinned.** The compose sketch names `comfyui`
  generically; v1 must pin a specific published image and version, or ship a
  Dockerfile.
- **Profile model selections are unvalidated.** Initial profiles are seeded from
  general reputation, not measurement. Before v1.0 each profile's claims should
  be checked on real 8GB hardware, and the dated headers filled in honestly.
- **Entering a spill is now prevented; detecting one already in progress is
  not.** The arbiter reads free VRAM from the driver before granting a claim
  and refuses anything larger, which covers the measured failure — memory held
  by a process onecard does not manage. What is still missing is the *other*
  direction: `/api/ps` reports both `size` and `size_vram`, and comparing them
  would reveal a model that is already partly in system RAM. onecard does not
  compare them, because on the CPU-only compose override every model
  legitimately reports `size_vram: 0`, so detection has to distinguish "no GPU
  expected" from "GPU expected but unused". Worth doing; no longer the most
  urgent gap.
- **`onecard validate` never contacts Ollama.** The Configuration section above
  claims validation checks that every model exists in Ollama. It does not: it
  checks the config's internal consistency and its VRAM arithmetic only, and
  never opens a connection. A misspelled model ref is therefore caught at first
  run, not at validate time. Either implement the check behind a flag or amend
  the claim.
- **Validation under-reserves for an unmeasured pinned model.** A model declared
  `residency: pinned` with no `footprint_mb` contributes nothing to the pinned
  total, so `validate` can pass a config that overcommits at run time. Since the
  arbiter now reconciles against `/api/ps` before every claim, the consequence is
  a clear `BudgetError` rather than a silent spill — but validate should still
  catch it.
