# Study plan — rev 4 (2026-10-09)

**Original:** 2026-09-03 → 2026-09-16. **Revised:** now runs 2026-09-10 → 2026-09-20,
with dubpls continuing past it. v1 preserved as `study_plan_2wk_v1_superseded.md`.

**Budget:** 2–3 h weekdays, ~5 h weekend days.
**Scope decision: one thing, properly.** Unchanged.

---

## What changed in rev 4 (2026-10-09)

- **Week 2 now has two parts.** Part A is rev 3's quantisation week, re-dated. **Part B adds
  a VLM project**: fine-tune Qwen2.5-VL-3B on soccer broadcast frames, publish the adapter on
  HuggingFace, and add **Ch 11 "What a picture costs"** to `llm-serve-bench`. Qwen2.5-VL-3B
  shares Part A's exact language backbone, so the two halves measure the same model with and
  without vision.
- **Why:** closes four of the seven requirements in Tether's *AI Research Engineer
  (Multi-Modal & Vision)* posting — PEFT, a VLM with measurable results, resource-constrained
  deployment, open-source multimodal work on HF — while building on your deployment strength.
- **Speech vocabulary is now floating**, triggered by a speech-company reply.
- **Decision point moves to ~2026-10-21.**

### Calendar

| Session | Date | Hours | Work |
|---|---|--:|---|
| **A1** | Fri 2026-10-09 | 2.5 | 3B bf16 baseline + eager vs CUDA graphs *(rev 3 "Day 1")* |
| **A2** | Sat 2026-10-10, first half | 2.5 | AWQ / GPTQ *(rev 3 "Day 2")* |
| **A3** | Sat 2026-10-10 second half + Sun 2026-10-11 morning | ~4.5 | FP8 KV cache, then FP8 W8A8 *(rev 3 "Day 3")* |
| **A4** | Sun 2026-10-11 afternoon + Mon 2026-10-12 | ~5.5 | Write Ch 8–10, clean up Ch 1–7, push *(rev 3 "Day 4")* |
| **B1** | Tue 2026-10-13 | 2.5 | Audit the dataset, split without leakage |
| **B2** | Wed 2026-10-14 | 2.5 | Hand-verify the test set, write the eval |
| **B3** | Thu 2026-10-15 | 2.5 | Zero-shot baselines: 3B, and a 7B teacher |
| **B4** | Fri 2026-10-16 | 2.5 | Training env + first QLoRA run |
| **B5** | Sat 2026-10-17 | 5 | Evaluate, iterate once, merge, quantise |
| **B6** | Sun 2026-10-18 | 5 | Ch 11 measurements: what a picture costs |
| **B7** | Mon 2026-10-19 + Tue 2026-10-20 | 5 | Publish: HF model card, GitHub, Ch 11 |

## What changed in rev 3 (2026-10-01)

- **Week 1 shipped** as `pra-dan/llm-serve-bench` (Ch 1–7) — beyond the rev 2 spec:
  Prometheus/Grafana, a client-bound finding, forced preemption.
- **Week 2 rewritten** around what Week 1 found: no 3B baseline existed yet (Day 1 builds
  it), client-measured numbers capped at concurrency 64, eager mode tested first.
- **TensorRT-LLM stays cut.** Current applications barely use it.
- **dubpls is no longer automatically next** — see the decision point after Week 2.

## What changed in rev 2, and why

1. **TensorRT-LLM is cut.** v1 pre-authorised shipping vLLM-only if the build fought back.
   Deciding that up front instead of after losing two days: TRT-LLM demand is 3 in
   `upskilling_priorities.md`, you already know TensorRT, and on a 12 GB card engine
   builds are harder than on the datacenter parts the docs assume (build-time memory).
   Extra days do not change the version-matrix risk — they just give it more room.
2. **The freed days go to dubpls**, which `positioning_brief.md` ranked as the single
   highest-leverage gap and which is currently a live liability: your speech CV leads
   with it, so reviewers who click through see work-in-progress.
3. **Hardware resolved: RTX 4070 Ti** (Ada, sm_89, 12 GB). FP8 is on; the v1 "if Ampere,
   do INT8-KV instead" contingency is dead. But 12 GB constrains far more than the
   architecture does — see model pins below.
4. **Day 3 audit changed Thursday.** You already have preemption evidence, so the paper
   reading now explains data you own rather than preparing you to collect it.

---

## Where you actually are (audited 2026-09-09)

Repo: `pra-dan/transformers_and_cuda_kernels`, subfolder `vllm/`. One commit,
"feat: add vLLM latency and throughput benchmarking harness (D1 & D2)".

**Good — better than the plan asked for:**

- `bench_vllm.py` measures TTFT, mean ITL, per-request and aggregate tokens/sec, with
  p50/p90/p99. Percentiles weren't in the v1 spec.
- Async concurrency, configurable warmup + 5 measurement rounds, token-count fallback
  when the server omits usage stats.
- README already identifies the knee (128 → 256 gives ~5%) and correctly reads
  diverging median vs p99 TTFT as scheduler preemption under KV pressure — and notes
  ITL is the weaker detector. That diagnosis is right, and it is Thursday's lesson
  arriving on Wednesday.

**Three things that undercut it as an artifact:**

| # | Problem | Cost to fix |
|---|---|---|
| 1 | `results.csv` header is `concurrency,round,ttft,mean_itl,completion_tokens,total_time,tokens_per_sec` — **no model, no quantisation, no config, no GPU**. Survivable with one run; fatal the moment Week 2 compares AWQ vs GPTQ vs FP8, because no row can be attributed. | 20 min |
| 2 | Everything so far is **Qwen2.5-0.5B-Instruct-AWQ**. Correct on day 1 (iterate in seconds), wrong as a headline: a 0.5B on 12 GB is ~all KV cache and represents no real serving regime — and because it is *already* AWQ, you have no FP16 baseline for Week 2 to measure against. | Thursday |
| 3 | README quotes 128 → 256; the committed CSV stops at 128. **A claim not backed by the data.** Either save the 256 run or drop the sentence. | 5 min |

---

## The deliverable

A **standalone public repo — `llm-serving-bench`** — measuring vLLM on a 12 GB Ada card:
TTFT, inter-token latency and throughput across concurrency levels, quantisation schemes
and KV-cache dtypes.

Keep `transformers_and_cuda_kernels` as your scratchpad, but extract before publishing.
Two reasons: a reviewer clicking a CV link should land *on* the benchmark rather than
navigate into a subfolder, and `positioning_brief.md` §8 says explicitly not to present
as a CUDA kernel engineer — a repo name leading with "cuda_kernels" does that
advertising for you and invites the shared-memory-bank-conflict question you can't yet
answer. Don't fight git history; a fresh repo is fine.

**Resume line this earns:**
> Benchmarked vLLM on 12 GB Ada hardware: measured TTFT, inter-token latency and
> throughput across concurrency, quantisation (AWQ/GPTQ/FP8) and FP8 KV cache;
> predicted the KV-pressure preemption cliff from first-principles cache arithmetic
> and verified it against the scheduler's own behaviour.

---

## Hardware pins — RTX 4070 Ti, 12 GB, ~504 GB/s

KV per token = `2 × layers × kv_heads × head_dim × dtype_bytes`.
Budget assumes `gpu-memory-utilization 0.90` (~10.8 GB) minus weights minus ~0.8 GB
activation/CUDA-graph/cuBLAS overhead.

| Model | Layers | KV heads | KV/token | FP16 fits? | KV tokens | Seqs @2k |
|---|--:|--:|--:|:--:|--:|--:|
| **Qwen2.5-3B-Instruct** | 36 | 2 | **36 KiB** | yes (6.2 GB) | ~103,000 | **~50** |
| Llama-3.2-3B-Instruct | 28 | 8 | 112 KiB | yes (6.4 GB) | ~31,000 | ~15 |
| Qwen2.5-7B-Instruct | 28 | 4 | 56 KiB | no (15.2 GB) | AWQ: ~77,000 | ~37 |
| Llama-3.1-8B-Instruct | 32 | 8 | 128 KiB | no (16.1 GB) | AWQ: ~33,000 | ~16 |

**Week 1 model: `Qwen/Qwen2.5-3B-Instruct` in bfloat16.** Not Llama-3.2-3B — identical
parameter count, 3× the reachable concurrency, purely from the GQA ratio (2 KV heads vs
8). That contrast is itself a result worth publishing.

**Week 2 step-up: `Qwen/Qwen2.5-7B-Instruct-AWQ`.** 7B FP16 doesn't fit at all, so
quantisation *unlocks a model class* rather than merely measuring one — a far better
story than "I measured a 4-bit speedup".

**Speculative-decoding draft: `Qwen/Qwen2.5-0.5B-Instruct`** — same tokenizer family,
already on disk. A 1B draft would eat KV budget you don't have.

**Bandwidth roofline for sanity checks** (batch-1 decode is bandwidth-bound):
3B bf16 ≈ 12.3 ms/token (~81 tok/s ceiling) · 7B AWQ ≈ 11.1 ms (~90) · 3B FP8 ≈ 6.2 ms (~163).
Measured far under ceiling ⇒ you are not yet bandwidth-bound, and knowing which regime
you're in is the difference between a benchmark and a table of numbers.

---

## Standing rules — apply to every run from here

1. **Every CSV row carries its own config.** model, quantisation, dtype, kv_cache_dtype,
   max_model_len, gpu_memory_utilization, max_num_seqs, block_size, vllm_version,
   gpu_name, driver, prompt_tokens, max_tokens, seed, timestamp. No exceptions.
2. **Log telemetry per row**, not per README sentence:
   `nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader`.
   Data that shows you controlled for throttling beats prose asserting it.
3. **Never report a concurrency without its context length.** "concurrency 64 @ 1024
   tokens", never bare "concurrency 64". Half of published serving benchmarks skip this
   and it's the first thing a good interviewer checks.
4. **Discard the warmup round.** Report p50/p90/p99, never means alone.
5. **Every claim in the README must be reproducible from committed data.**
6. **Change one variable at a time.** If you tune quantisation and max_num_seqs together,
   you have learned nothing.

---

# Week 1 — vLLM for real

## Thursday 2026-09-10 (2.5 h) — Real baseline

**Goal.** Replace toy numbers with a defensible FP16 baseline and make every row
self-describing. This is the highest-value 2.5 hours in the whole plan, because
everything downstream compares against it.

**Do.**

1. Extend `bench_vllm.py` with the config + telemetry columns from Standing Rule 1–2.
   Pass server config in as CLI args and echo into every row; sample `nvidia-smi`
   once per round.
2. Lock clocks properly. Persistence mode first, then pick a clock you can *hold*:
   ```bash
   sudo nvidia-smi -pm 1
   nvidia-smi -q -d SUPPORTED_CLOCKS | head -40     # pick a sustainable SM clock
   sudo nvidia-smi -lgc <chosen>,<chosen>
   ```
   Aim for roughly 85–90% of max boost, not max — a 285 W board in a desktop case will
   not hold peak boost through a 30-minute sweep, and a clock it can't hold is worse
   than a lower one it can.
3. Serve the real baseline:
   ```bash
   vllm serve Qwen/Qwen2.5-3B-Instruct \
     --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.90
   ```
   bfloat16, not float16 — it's the model's native training dtype and avoids overflow
   surprises. KV arithmetic is unchanged at 2 bytes.
4. Re-run the ladder at fixed shape: **1024-token prompts, 256 output tokens**,
   concurrency 1 / 8 / 32 / 64 / 128 / 256.
5. Reset clocks when done: `sudo nvidia-smi -rgc`.

**Done when.** `results.csv` rows are individually interpretable, and you can point at
one number and say "that is 3B bf16 at concurrency 32, 1k context, on a clock-locked
4070 Ti."

**If it goes wrong.** At `max-model-len 4096` you have ~25 full-length sequences of KV.
Concurrency 128 and 256 *will* preempt hard, and may OOM. **That is the result, not a
failure** — record where it breaks. Friday explains it.

---

## Friday 2026-09-11 (2.5 h) — PagedAttention, then KV arithmetic

**Goal.** Explain Thursday's cliff from first principles, then *predict* it. You already
have the evidence; this is the day it stops being a mystery.

**Do.**

1. Read the vLLM paper §3–4 (PagedAttention and the KV cache manager). One pass.
2. Hand-compute KV per token for Qwen2.5-3B:
   `2 × 36 layers × 2 kv_heads × 128 head_dim × 2 bytes = 36,864 B = 36 KiB`.
3. Check it against reality: vLLM logs a `# GPU blocks:` line at startup.
   `blocks × block_size(16) × 36 KiB` should land near your free-VRAM-after-weights
   estimate. Reconcile the gap — that reconciliation *is* the exercise.
4. **Predict** the preemption onset for `max-model-len 4096`:
   `free_kv_bytes / (36 KiB × 4096)`. Write the number down before you measure.
5. Verify by sweeping `--gpu-memory-utilization` 0.95 / 0.90 / 0.80 / 0.70 and watching
   the cliff move. Grep the server log for preemption/swap messages and correlate with
   the p99 TTFT spike you already observed.

**Done when.** Predicted and measured cliff agree within ~20%, and you can say in one
sentence what PagedAttention actually solves — memory fragmentation and over-reservation,
**not** raw speed. This is a standard interview question and doing it once makes it yours.

**If the numbers disagree**, the usual culprits in order: block-size rounding; non-KV
overhead you forgot to subtract (activations, CUDA graphs, cuBLAS workspace); chunked
prefill reserving extra. Identifying *which* one is worth more than getting it right
first try.

---

## Saturday 2026-09-12 (5 h) — Prefix caching and speculative decoding

**Goal.** Two features that trade memory or compute for latency. Find where each *loses* —
that's the half nobody publishes.

**Do.**

1. **Prefix caching** (`--enable-prefix-caching`). Build two workloads: (a) 800 shared
   system-prompt tokens + 200 unique, (b) all-unique prompts of the same length. Measure
   TTFT with and without the flag on both.
   Expect: large TTFT win on shared-prefix, ~nothing on unique, small hashing overhead
   you should be able to see and quantify.
2. **Speculative decoding.** Draft `Qwen2.5-0.5B-Instruct` against the 3B target, at
   concurrency 1, 8 and 32. **Record the acceptance rate** — without it the latency
   numbers are uninterpretable.
   Expect a crossover: spec decoding wins at low concurrency where the GPU has spare
   compute, and loses at high concurrency where the batch already saturates it and every
   rejected draft token is wasted work. Plus the draft weights eat KV budget.

**Done when.** You have a chart with a crossover point and can name the mechanism behind
it in one sentence.

**Timebox.** Speculative-decoding configuration has churned across vLLM releases
(`--speculative-model` flags vs a `--speculative-config` JSON blob). Check *your*
installed version's docs, not a blog post. **Hard stop at 90 minutes** — if the API
fights you, ship prefix caching alone and say so in the README. "I timeboxed it against
a moving config API" is a credible engineering answer.

---

## Sunday 2026-09-13 (5 h) — Extract, write up, publish v1

**Do.**

1. Create standalone `llm-serving-bench`. Copy in the harness, results, plots, README.
2. **Charts** (these five, in this order of importance):
   - TTFT p50 vs p99 against concurrency — the divergence is the money chart
   - aggregate throughput vs concurrency, with the knee marked
   - per-request tokens/sec vs concurrency (the collapse you already found)
   - preemption cliff vs `gpu-memory-utilization`
   - spec-decoding crossover, with acceptance rate on a second axis
3. **README structure:** what and why · hardware + software version table · methodology
   (clock locking, warmup, percentiles, exact repro commands) · results with charts ·
   **"what surprised me"** · **"what this does not measure"**.
4. The last two sections are what make it read as engineering rather than a tutorial.
   The honest limitations section is not a weakness — it's the strongest signal in the
   whole document.

**Done when.** A stranger clones it, runs one command, and gets comparable numbers.

---

# Week 2, Part A — Quantisation and FP8 on a real model (re-dated 2026-10-09)

Week 1 shipped as **`pra-dan/llm-serve-bench`** (Ch 1–7). Week 2 adds chapters to the
same repo instead of starting a new one. TensorRT-LLM stays cut (see "Cut, and why").

## Rules carried over from Week 1's own findings

1. **Client-measured numbers only up to concurrency 64.** Ch 2 showed the single asyncio
   loop saturates around 10–11k tok/s; above that you are measuring the harness, and a
   client ceiling flattens any server-side gain rather than just shrinking it. For
   concurrency ≥ 128, use vLLM's Prometheus metrics with `@`-pinned PromQL windows (Ch 2).
2. **Prefix caching off** (`--no-enable-prefix-caching`) for every comparison, unless
   caching is the variable. Ch 2 found it was silently on.
3. **No `--enforce-eager`** unless eager mode is the variable — see A1.
4. **Predict before measuring.** Write the number down first. It is the strongest habit
   in Ch 1–7 and the most interview-legible thing in the repo.
5. Every run's label carries model, quantisation, KV dtype, eager on/off, prefix caching,
   `gpu-memory-utilization`, `max-model-len`.
6. One variable at a time.

---

## A1 — Fri 2026-10-09 (2.5 h) — 3B baseline, and what eager mode cost you

**Goal.** A real-size baseline everything else compares against, and an answer to why
Week 1's low-batch numbers sat so far above the hardware floor.

**Why eager mode first — computed from your own Week 1 data.** The 0.5B AWQ weights were
0.43 GB → batch-1 bandwidth floor ≈ 0.43 GB / 504 GB/s ≈ **0.85 ms/token**. Measured ITL
p50 at concurrency 1 was **7.9 ms — about 9× the floor.** Week 1's low-batch numbers were
overhead-dominated, and `--enforce-eager` (no CUDA graphs, so every decode step pays full
kernel-launch and Python overhead) is the prime suspect. If it stays on, A2's main
effect — fewer weight bytes per step at low batch — disappears under the same overhead.

**Predict first.**

- KV per token for Qwen2.5-3B: `2 × 36 × 2 × 128 × 2 = 36 KiB` (3× the 0.5B's 12 KiB).
  Predict `kv_cache_size_tokens` at util 0.90 before launching. Subtract bf16 weights
  (~6.2 GB), activations, and — new now that eager is off — **CUDA-graph capture memory**.
- Batch-1 floor: 6.2 GB / 504 GB/s ≈ **12.2 ms/token**.

**Do.**

1. `nvidia-smi` for free VRAM first. If your display runs on this GPU, 0.90 may not
   fit; drop to 0.85 and note it.
2. **Eager vs CUDA graphs, 0.5B, concurrency 1 only.** Same model as Week 1, toggle
   `--enforce-eager`, read server-side ITL from Prometheus. ~20 min. This feeds straight
   back into Ch 5's "overhead-bound?" question.
3. Serve the baseline:
   ```bash
   vllm serve Qwen/Qwen2.5-3B-Instruct --dtype bfloat16 \
     --max-model-len 4096 --gpu-memory-utilization 0.90 --no-enable-prefix-caching
   ```
4. Ladder 1 / 8 / 32 / 64 with the harness, same prompt and `--max-tokens` as Week 1 so
   the chapters compare. Pull server-side TTFT/ITL for the same windows.

**Done when.** Predicted vs actual `kv_cache_size_tokens`, measured batch-1 ITL against
the 12.2 ms floor, and a number for how much eager mode cost at batch 1.

**If the KV prediction is off**, the usual suspects: block rounding, activation
workspace, and CUDA-graph memory you didn't budget for.

---

## A2 — Sat 2026-10-10, first half (2.5 h) — AWQ and GPTQ: where weight-only quantisation helps

**Check your Week 1 server log first.** You launched with `--quantization awq`
explicitly. On Ada, vLLM can run AWQ checkpoints through the faster `awq_marlin` kernel,
and when you force plain `awq` it typically logs a warning saying so. If that warning is
in your Week 1 log, part of the 9× gap above is the slower kernel, not eager mode — and
A1's A/B will tell you how to split the blame.

**Predict first.**

- **AWQ weight VRAM is not 6.2 / 4.** Embeddings and `lm_head` usually stay 16-bit;
  Qwen2.5-3B's embedding table alone is 151,936 × 2,048 ≈ 311M params ≈ 0.6 GB. Expect
  roughly **2 GB**, not 1.5.
- KV tokens gained = freed GB / 36 KiB. KV per token is **unchanged** — Ch 3's
  "quantising weights does nothing for KV", now quantified on a real model.
- Batch-1 floor ≈ 2 GB / 504 GB/s ≈ **4 ms/token**.

**Do.**

1. `Qwen/Qwen2.5-3B-Instruct-AWQ` and `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4`, same flags as
   A1. **Don't pass `--quantization`** — let vLLM pick, and record which kernel it
   reports.
2. Optional, 10 minutes, very instructive: same AWQ weights, `--quantization awq` vs
   `awq_marlin`, concurrency 1. Same bytes, different kernel.
3. Ladder 1–64 (harness + server). Batch-1 ITL vs the 4 ms floor.
4. Quality: a fixed set of ~10 prompts, greedy decoding, diff against bf16 outputs.

**The finding to look for.** A large ITL win at concurrency 1–8 (bandwidth-bound: fewer
bytes per step), shrinking or reversing by 32–64 as decode turns compute-bound and
dequantisation becomes overhead. If the crossover sits above 64, locate it with
Prometheus at 128/256 rather than the harness.

**Done when.** A bf16 / AWQ / GPTQ table — weight VRAM, KV tokens, TTFT, ITL,
throughput — and either a crossover or a clear statement that you didn't find one.

---

## A3 — Sat 2026-10-10 second half + Sun 2026-10-11 morning (~4.5 h) — FP8 KV cache first, then FP8 weights

**Order matters.** FP8 KV cache is a runtime flag — no checkpoint, no new dependency.
FP8 W8A8 needs llm-compressor, which may not install cleanly beside Python 3.14 and
PyTorch 2.13. Do the zero-risk headline first.

### Part A — FP8 KV cache (the headline)

**Predict first.** `kv_cache_size_tokens` ≈ 2× A1 (36 → 18 KiB/token). Then predict
the preemption change: 256 sequences × ~1,070 tokens each against each capacity.

**Do.** Your Ch 7 forced-preemption setup on the 3B, twice, back to back — bf16 KV, then
`--kv-cache-dtype fp8`. Concurrency 256, `--max-tokens 1024`, `--ignore-eos`, prefix
caching off. **Server-side only:** `vllm:num_preemptions_total`, KV usage peak,
throughput via PromQL. The client bottleneck never enters into it — and Ch 7 already
showed the preemption counter is the only instrument that sees preemption at all.

Quality check: greedy outputs on the fixed prompt set, bf16 KV vs FP8 KV.

**Framing.** Expect little speed change at these short contexts — vLLM's own April 2026
writeup says FP8 KV gains thin out below ~7k context. **The result is capacity, not
speed**: fewer evictions, more concurrent sequences on 12 GB. Say that plainly. Published
FP8-KV numbers are overwhelmingly Hopper/Blackwell; consumer-Ada numbers are yours.

### Part B — FP8 W8A8 weights

1. Separate conda env for quantising — it's an offline step; don't touch the vLLM env.
   llm-compressor `FP8_DYNAMIC` recipe on Qwen2.5-3B-Instruct, save, serve from the
   vLLM env.
2. **Hard stop: 60 minutes on install + quantise.** Fallback: a prebuilt FP8-dynamic
   checkpoint on the Hub if one exists (check the uploader — it'll be community), or ship
   Part A alone and say so.
3. Measure as A2. Batch-1 floor ≈ 3.1 GB / 504 GB/s ≈ 6.1 ms.

**Why it's worth the attempt.** W8A8 is the one scheme here that cuts weight bytes *and*
runs the matmuls on FP8 tensor cores — so it's where you may **not** see the high-batch
regression AWQ shows. If A2 found a crossover, Part B is the test of why.

---

## A4 — Sun 2026-10-11 afternoon + Mon 2026-10-12 (~5.5 h) — Write chapters 8–10, clean up 1–7, push

New chapters, same narrative style as Ch 1–7 (wrong turns left in):

- **Ch 8** — Eager mode vs CUDA graphs (and `awq` vs `awq_marlin`): what Week 1's
  low-batch numbers were really measuring.
- **Ch 9** — Weight-only quantisation: bf16 vs AWQ vs GPTQ, and where the crossover sits.
- **Ch 10** — FP8: KV-cache capacity and preemption as the headline; W8A8 if Part B landed.

Cleanup of Ch 1–7:

- Typos: numebers, numebrs, paass, montoring, quantizied, lenth, drifing; `2,62,144` → `262,144`.
- Ch 2: reconcile the 68.3% hit rate at the top with the 69.6% "observed" later.
- Ch 5: cite vLLM's server-side ITL (18.7 ms at concurrency 256, from Ch 2) instead of the
  harness's, so the "decode got slower" argument doesn't rest on your own client.
- Make the repo name and the README title agree on what this is.

Ch 11 is reserved for Part B. Then update the CV line. *"vLLM (scheduler & KV-cache benchmarking)"* undersells it.

---

# Week 2, Part B — Fine-tune a small VLM, publish it, measure what a picture costs

**Deliverables.**

1. **HuggingFace:** a LoRA adapter for `Qwen/Qwen2.5-VL-3B-Instruct` that classifies soccer
   broadcast frames (camera view + match event), with a model card carrying a real eval table.
2. **GitHub:** split manifests, eval script and training config, so the numbers reproduce.
3. **`llm-serve-bench` Ch 11 — "What a picture costs":** the same model served under Part A's
   methodology.

**Why this model.** Qwen2.5-VL-3B's language backbone is *identical* to Part A's Qwen2.5-3B —
36 layers, 16 query heads, 2 KV heads, hidden 2048 (checked against the HF `config.json`) —
so the same **36 KiB of KV per token**. Part A hands you the text-only baseline for free, and
Ch 11 can isolate exactly what the vision side adds. Licence is `qwen-research`
(non-commercial): fine for a portfolio, your adapter inherits it, and the model card says so.

**Why this data.** *A comprehensive dataset of soccer event images for advancing automatic
recognition systems* (Data in Brief vol. 60, 2025, doi:10.1016/j.dib.2025.111518; open copy
on PMC: PMC11999477). 640×360 JPGs sampled from ~45–50 YouTube match videos, two label sets:

- **view:** Long · Medium · Shorter · Outer
- **event:** Goal · Player celebration · Plenty stock (sic — read the paper's definition) ·
  Goal attempt · Free kick · Corner · Red card · Offside · Yellow card · Spectator

The view task is your QuickFlip scene classifier's job, so the story writes itself: *my
production pipeline runs a dedicated scene classifier — can one 3B VLM call replace it, and
what does that cost to serve?*

**Three cautions, baked in from day one.**

- **Never upload the images.** YouTube-sourced, "research and educational purposes" only, no
  dataset licence stated. Publish manifests (filenames + labels) and scripts, not pixels.
- **The paper's own numbers don't reconcile.** The abstract claims 137,196 view images;
  Table 1 sums to 99,717. Event percentages don't match event counts. Treat labels as noisy
  until audited — and the audit is itself a model-card finding.
- **Frames from one video are near-duplicates.** A random split leaks and inflates your
  fine-tuning gain. Split by video.

**Predict first — from the config, before touching data.**

- Qwen2.5-VL turns each 28×28-pixel block into one token (14-px patches, 2×2 merge). A
  640×360 frame resizes to 644×364 → 23 × 13 = **299 visual tokens** → 299 × 36 KiB ≈
  **10.5 MiB of KV per image**, about 6.5× your entire Week 1 text prompt (46 tokens, 1.6 MiB).
- Capped at `max_pixels = 57,600` → 308×168 → **66 tokens** (2.3 MiB). Upscaled to 1280×720 →
  **1,196 tokens** (42 MiB).
- A classification answer is ~5 tokens. **This workload is almost all vision encoder +
  prefill — the opposite regime from Week 1, which was decode-bound.** Predict which latency
  metric dominates before you measure it.

---

## B1 — Tue 2026-10-13 (2.5 h) — Audit the data, split without leakage

1. Download from Kaggle (link in the paper; needs a Kaggle account). Keep it **outside**
   any git repo.
2. Count what is actually there per class, against both of the paper's tables. Write the
   discrepancy down.
3. **Find the video grouping.** Check whether filenames encode a video or clip ID. If they
   don't, cluster near-duplicates with a perceptual hash (`imagehash` pHash, small Hamming
   threshold) and treat each cluster as a group.
4. Balanced subset, **split by group**: ~300 train / 50 val / 40 test per class, 14 classes.
   Small on purpose — balance matters more than volume for a QLoRA run on 12 GB.
5. Write `train.jsonl` / `val.jsonl` / `test.jsonl` (path, task, label, group id). These are
   what you publish.

**Done when.** Manifests exist, an assertion proves no group appears in two splits, counts
are recorded.

**If short.** The rarest classes (Spectator ~841, Yellow card ~890 by the paper's counts) may
not fill 390 after grouping. Take what's there and report per-class n honestly.

---

## B2 — Wed 2026-10-14 (2.5 h) — Hand-verify the test set, write the eval

1. **Hand-check all ~560 test labels** (14 × 40). A tiny script — show image, show label,
   keypress y / n / fix — at ~5 s each is ~45 minutes. A test set labelled by the same noisy
   process you're auditing can't tell you whether the model is right.
2. Read the paper's class definitions and put them **in the prompt**. **The zero-shot
   baselines get the same definitions** — otherwise the fine-tuning "gain" is mostly the
   model learning your label vocabulary, not seeing better.
3. Eval script: strict output parsing (an unparseable answer is wrong, and you report how
   many there were); **macro-F1 per task as the headline** — accuracy would just reward
   predicting "Long view"; per-class F1; confusion matrix. Point it at any OpenAI-compatible
   endpoint, i.e. your vLLM server.

**Done when.** Verified test manifest + an eval script that runs end to end.

---

## B3 — Thu 2026-10-15 (2.5 h) — Zero-shot baselines: the 3B, and a 7B teacher

**Predict first.** Which classes will zero-shot get right (Red card? Corner?) and which will
it fail (the view boundaries — Medium vs Shorter)? Write it down.

1. Serve `Qwen/Qwen2.5-VL-3B-Instruct` in vLLM (Part A flags, prefix caching off). Run the eval.
2. Serve `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` as teacher and upper reference — Apache-2.0,
   ~6.9 GB on disk, vision tower kept in bf16. Tight on 12 GB: `--max-model-len 2048`, cap
   `max_pixels`. If your display shares the GPU and it won't fit, drop it, note why, move on.
3. **Label-noise audit with the teacher.** On the *train* split, list images where the 7B
   disagrees with the label. Eyeball ~50. If a meaningful fraction of dataset labels are
   wrong, fix or drop them and record the rate.

That's the honest form of "distillation" here: gold labels exist, so training on teacher
pseudo-labels is a stretch goal, not the core.

**Done when.** 3B and 7B zero-shot macro-F1 for both tasks, plus a noise-rate estimate.

---

## B4 — Fri 2026-10-16 (2.5 h) — Training environment, first QLoRA run

1. **Separate conda env on Python 3.11 or 3.12** — bitsandbytes and TRL may not ship wheels
   for the vLLM env's Python 3.14. **Hard stop: 60 minutes** on environment. Fallback:
   Unsloth, which supports Qwen2.5-VL QLoRA on small GPUs with its own kernels.
2. Stack: `transformers` + `peft` + `bitsandbytes` (4-bit NF4 base) + TRL `SFTTrainer`.
3. **LoRA on the language model only, vision encoder frozen.** Cheaper, standard, one
   variable at a time. Gradient checkpointing on. Native 299 tokens per image.
4. Start at rank 16, 1–2 epochs, small batch with gradient accumulation. Log loss and val
   macro-F1 to Weights & Biases.

**Done when.** An adapter is saved and val macro-F1 beats the 3B zero-shot.

**If it OOMs.** Lower `max_pixels` for training first — fewer visual tokens, smaller
activations — then batch size.

---

## B5 — Sat 2026-10-17 (5 h) — Evaluate, iterate once, merge, quantise

1. Test-set eval of the adapter against B3.
2. **One iteration, one variable** — whichever the confusion matrix points at (balance,
   epochs, or rank). Not three.
3. Merge into the **bf16** base (`merge_and_unload`), not the 4-bit one. Serve the merged
   model in vLLM and re-run the eval; it should match the adapter within noise. If it
   doesn't, that's a finding.
4. **Quantise the language model only; keep the vision tower in bf16** — the same choice
   Qwen's own AWQ checkpoint makes (its config excludes `visual`). Reuse Part A's
   llm-compressor FP8 recipe with the vision modules on the ignore list. Eval again.

**Done when.** The headline table:

| Model | View macro-F1 | Event macro-F1 | Weights VRAM |
|---|---|---|---|
| 3B zero-shot | | | |
| 7B-AWQ zero-shot | | | |
| 3B + LoRA (bf16) | | | |
| 3B + LoRA, FP8 LM | | | |

Row 3 vs row 2 is the question an edge-deployment interviewer actually cares about:
**does a fine-tuned 3B beat a zero-shot 7B at roughly half the memory?**

---

## B6 — Sun 2026-10-18 (5 h) — Ch 11 measurements: what a picture costs

Part A's rules throughout: client numbers only up to concurrency 64, Prometheus above,
prefix caching off, predict first.

1. **Resolution sweep:** 66 / 299 / 1,196 visual tokens via `max_pixels`
   (`--mm-processor-kwargs`). At each point: TTFT, KV per request, **and test-set
   macro-F1** — the accuracy-vs-cost curve. Does view classification need 299 tokens, or
   does 66 do?
2. **Same backbone, with and without eyes:** text-only vs image request on the same model.
   Splits TTFT into vision-encoder time and LLM prefill.
3. **Concurrency 1–64 with image requests.** Predict first: KV capacity ÷ ~350 tokens per
   request says preemption isn't the limit. So what is?
4. Part A's FP8 KV cache on image requests. Predict: barely matters when outputs are 5
   tokens. Say why.

**Done when.** Three charts: macro-F1 vs visual tokens, TTFT vs visual tokens, and the TTFT
breakdown text vs image.

---

## B7 — Mon 2026-10-19 + Tue 2026-10-20 (2 × 2.5 h) — Publish

1. **HuggingFace model repo** — e.g. `pra-dan/qwen2.5-vl-3b-soccer-views-lora` — with the
   adapter (optionally the merged FP8 model too). Model card: task; data provenance with the
   paper link and **no images**; the leakage-safe split method; the dataset discrepancies
   you found; the eval table and per-class F1; serving cost per image; licence
   (`qwen-research`, inherited); limitations (one sport, one broadcast style, 640×360 source).
2. **GitHub:** manifests, eval script, training config.
3. **`llm-serve-bench` Ch 11, "What a picture costs"**, linking to the HF card and back.
4. One CV line under Open Source & Projects; a LinkedIn post.

**Done when.** A stranger can reproduce your eval table from the manifests, the public base
model and your adapter.

---

## Minimum viable deliverable

If Part B slips, what **must** ship is: **the adapter on HF + a model card showing zero-shot
vs fine-tuned macro-F1 on a leakage-safe, hand-verified test set.** Everything else is bonus.

## Stretch — only if B1–B7 land

- GGUF export, run in llama.cpp on the 4070 Ti — the on-device path, and the QVAC angle. It's
  also a cheap first step toward the llama.cpp option at the ~2026-10-21 decision point.
- True distillation: train on 7B pseudo-labels for unlabelled frames, compare with gold.
- LoRA on the vision tower as well.

---

## Speech vocabulary sprint — floating (2.5 h)

No longer on the calendar. Do it the day a speech company replies, before the screen.

**Vocabulary** (one-sentence definition each, written down):
`TTFT / TTFA` · `RTF (real-time factor)` · `streaming vs batch ASR` · `VAD` ·
`endpointing` · `barge-in` · `WER` · `diarization` · `chunked / incremental decoding` ·
`jitter buffer` · `MOS` · `phoneme alignment` · `warm-start & model residency`.

**Then do, don't read.** Run `faster-whisper` in streaming mode on the 4070 Ti and
measure your own RTF and TTFA.

**The connection to make explicit** — rehearse it, it's the interview answer: these are
the metrics you just spent two weeks measuring, renamed. TTFT is TTFA. Throughput is RTF.
Your client-bound finding is the same trap a streaming-ASR load test falls into.

---

## After Week 2 — decision point (~2026-10-21)

The choice is three-way now, not dubpls-vs-study:

| Option | For | Against |
|---|---|---|
| **dubpls** | The portfolio gap the positioning brief ranks highest; speech CV leads with it | Packaging and env work, not learning |
| **llama.cpp / ggml** | Where applications have drifted (Qualcomm ×2, Sarvam on-device, Cyient Jetson); Qualcomm's JD names it; edge is your strongest pillar | Narrower market than LLM-infra |
| **Kubernetes** | Highest demand score in the table (10.0) | Scored against the global-remote LLM-infra set that has gone quiet and mostly needs visas |

Decide with that week's application responses in hand, not before.

---

# dubpls — block one

## Dates set at the ~2026-10-21 decision point, plus extension days

**Goal.** From "1–2 weeks of real work" to *a stranger clones it and sees the demo*.
Scope to **defensible, not done** in this block; the extension days finish the punch list.

| Day | Work |
|---|---|
| **Thu** | **Audit only — fix nothing.** Fresh clone into a clean environment. Follow your own README literally. Log every step that breaks, every undocumented assumption, every missing weight file. The written punch list *is* Thursday's deliverable. Resisting the urge to fix as you go is what makes the list complete. |
| **Fri** | Environment consolidation — your own README TODO, "reduce multiple environments to one". Pin versions. One env file or one Dockerfile that actually builds from scratch. TIGER, WhisperX, CosyVoice and IndexTTS-2 having incompatible dependency trees is the likely reason there are several; if they genuinely cannot merge, document *why* and ship two clearly-labelled envs rather than pretending. |
| **Sat** | Make the demo path work end to end from a clean clone: one command, one sample input, one output artifact. Commit the sample input. |
| **Sun** | README rewrite — what it does, architecture, the demo, and an honest **"what isn't finished"** section. Push. |
| **Extension** | Whatever Thursday's punch list still says. |

**Done when** you would be happy for an interviewer to click the link — which is the
standard `positioning_brief.md` §8 actually sets, and the reason this block exists.

---

## Cut, and why

| Cut | Reason |
|---|---|
| **TensorRT-LLM engine build + comparison** (v1 Week 2 Wed/Thu) | Demand 3; you already know TensorRT; version-matrix risk is unchanged by having spare days; 12 GB makes engine builds harder, not easier. v1 already pre-authorised shipping vLLM-only. Say "timeboxed it" in the README. |
| **Kubernetes** | Still the top-ranked gap by demand (value 10.0) but helps none of the 12 live applications. Next block after dubpls — and if it slips a third time, that's a pattern, not a scheduling accident. |

---

## Risks

**Speculative decoding config churn** — Saturday's likeliest sinkhole. Hard stop at
90 minutes, ship prefix caching alone if needed.

**The 12 GB ceiling is the binding constraint all fortnight.** Every "it OOMed" is a
data point about serving economics, not a setback. Record them; they're half the value
of benchmarking on a consumer card rather than a rented H100.

**dubpls scope creep.** The block-one goal is *runnable*, not *good*. Hindi support and
better tone prompting are on your TODO list; both are v2. Shipping a clean repo that
does one language well beats a broader one nobody can run.

**Don't skip the write-up days.** An unpublished benchmark is worth roughly nothing to a
reviewer — the same rule that made the kernel-ladder plan insist on the profiling ritual.

---

## Watch signal

As of 2026-09-09: **12 applications sent, zero recruiter responses.** Too early to read
anything into it — the 09-07 batch is two days old. But if it's still zero across 12+
applications around 09-20, the binding constraint is probably not knowledge. It's more
likely geography (most of the current shortlist is flagged relocation/visa) or the
profile itself. Both are dubpls-and-sourcing problems rather than study problems, which
would make this rescope look prescient rather than lucky. Revisit the diagnosis then
rather than reflexively starting another study block.
