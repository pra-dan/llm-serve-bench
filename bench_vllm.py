#!/usr/bin/env python3
"""vLLM measurement harness: TTFT, inter-token latency, and tokens/sec across concurrency levels.

Usage:
    python3 bench_vllm.py --model Qwen/Qwen2.5-3B-Instruct --concurrency-levels 1,8,32,64
"""
import argparse
import asyncio
import csv
import os
import statistics
import time
from datetime import datetime, timezone

from openai import AsyncOpenAI

DEFAULT_PROMPT = (
    "Explain how a hash table resolves collisions, in about 150 words."
)


async def run_single_request(client, model, prompt, max_tokens, ignore_eos=False):
    start = time.perf_counter()
    first_token_time = None
    last_token_time = None
    itl = []
    completion_tokens = None

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
        # vLLM extension: keep generating past EOS so every request really emits
        # max_tokens. Without it, max_tokens is only a cap and the model stops
        # wherever the prompt's natural answer ends.
        extra_body={"ignore_eos": True} if ignore_eos else None,
    )

    async for chunk in stream:
        now = time.perf_counter()
        if chunk.usage is not None:
            completion_tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None or not delta.content:
            continue
        if first_token_time is None:
            first_token_time = now
        else:
            itl.append(now - last_token_time)
        last_token_time = now

    end = time.perf_counter()
    total_time = end - start
    ttft = (first_token_time - start) if first_token_time is not None else None

    if completion_tokens is None:
        # Fallback for servers that don't return usage on stream: count content chunks as tokens.
        completion_tokens = len(itl) + (1 if first_token_time is not None else 0)

    tokens_per_sec = completion_tokens / total_time if total_time > 0 else 0.0

    return {
        "ttft": ttft,
        "mean_itl": statistics.mean(itl) if itl else 0.0,
        "completion_tokens": completion_tokens,
        "total_time": total_time,
        "tokens_per_sec": tokens_per_sec,
    }


async def run_batch(client, model, prompt, max_tokens, concurrency, ignore_eos=False):
    batch_start = time.perf_counter()
    results = await asyncio.gather(
        *[run_single_request(client, model, prompt, max_tokens, ignore_eos) for _ in range(concurrency)]
    )
    batch_time = time.perf_counter() - batch_start
    return results, batch_time


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return s[idx]


async def bench_concurrency(client, model, prompt, max_tokens, concurrency, rounds, warmup_rounds,
                            ignore_eos=False):
    for _ in range(warmup_rounds):
        await run_batch(client, model, prompt, max_tokens, concurrency, ignore_eos)

    all_results = []
    total_tokens = 0
    total_wall_time = 0.0
    for _ in range(rounds):
        results, batch_time = await run_batch(client, model, prompt, max_tokens, concurrency, ignore_eos)
        all_results.extend(results)
        total_tokens += sum(r["completion_tokens"] for r in results)
        total_wall_time += batch_time

    ttfts = [r["ttft"] for r in all_results if r["ttft"] is not None]
    itls = [r["mean_itl"] for r in all_results if r["mean_itl"]]
    per_request_tps = [r["tokens_per_sec"] for r in all_results]
    aggregate_tps = total_tokens / total_wall_time if total_wall_time > 0 else 0.0

    summary = {
        "concurrency": concurrency,
        "n_requests": len(all_results),
        "n_ttft_samples": len(ttfts),
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p90": percentile(ttfts, 90),
        "ttft_p99": percentile(ttfts, 99),
        "itl_p50": percentile(itls, 50),
        "itl_p90": percentile(itls, 90),
        "itl_p99": percentile(itls, 99),
        "req_tps_mean": statistics.mean(per_request_tps) if per_request_tps else 0.0,
        "aggregate_tps": aggregate_tps,
    }
    return summary, all_results


def print_summary_table(summaries):
    header = f"{'conc':>5} {'TTFT p50':>9} {'TTFT p90':>9} {'TTFT p99':>9} {'ITL p50':>8} {'ITL p90':>8} {'req tok/s':>10} {'agg tok/s':>10}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        print(
            f"{s['concurrency']:>5} "
            f"{s['ttft_p50']*1000:>8.1f}m "
            f"{s['ttft_p90']*1000:>8.1f}m "
            f"{s['ttft_p99']*1000:>8.1f}m "
            f"{s['itl_p50']*1000:>7.1f}m "
            f"{s['itl_p90']*1000:>7.1f}m "
            f"{s['req_tps_mean']:>10.1f} "
            f"{s['aggregate_tps']:>10.1f}"
        )


def write_csv(path, all_rows):
    fieldnames = ["run_id", "label", "concurrency", "round", "ttft", "mean_itl",
                  "completion_tokens", "total_time", "tokens_per_sec"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


SUMMARY_FIELDS = [
    "run_id", "timestamp", "label", "model",
    "max_num_seqs", "max_num_batched_tokens", "prefix_caching",
    "rounds", "warmup_rounds", "max_tokens", "ignore_eos",
    "concurrency", "n_requests", "n_ttft_samples",
    "ttft_p50", "ttft_p90", "ttft_p99",
    "itl_p50", "itl_p90", "itl_p99",
    "req_tps_mean", "aggregate_tps",
]


def append_summary_csv(path, rows):
    """Append one summary row per concurrency level, accumulating across runs.

    A sweep restarts the server per config, so each config is a separate process
    and cannot hold the results of the others. Appending here is what lets a
    sweep end up in one file; `label` and the knob columns are what make the
    rows distinguishable once they are all in it.
    """
    existing = os.path.exists(path) and os.path.getsize(path) > 0
    if existing:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
        if header != SUMMARY_FIELDS:
            raise SystemExit(
                f"{path} has columns {header}, expected {SUMMARY_FIELDS}.\n"
                "Appending would misalign it. Move the old file aside or pass "
                "a different --summary-csv."
            )
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not existing:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _silence_asyncgen_teardown_noise():
    """Drop the httpcore2 pool-teardown traceback that buries the summary table.

    On Python 3.14 + httpcore2, asyncio.run()'s final shutdown_asyncgens() races
    the connection pool's PoolByteStream.__aiter__ generator and reports
    "RuntimeError: generator didn't stop after athrow()" via the loop exception
    handler. It fires after all results are collected (exit status stays 0) and
    is intermittent -- roughly 6 runs in 10 here. Closing the client or wrapping
    each stream in `async with` does not prevent it; both were measured.

    Only asyncgen-finalisation reports are dropped. Real failures propagate out
    of main() untouched.
    """

    def handler(loop, context):
        if "asynchronous generator" in context.get("message", ""):
            return
        loop.default_exception_handler(context)

    asyncio.get_running_loop().set_exception_handler(handler)


async def main():
    _silence_asyncgen_teardown_noise()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency-levels", default="1,8,32,64")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--output-csv", default="results.csv")
    parser.add_argument("--summary-csv", default="summary.csv",
                        help="Per-concurrency summary rows, appended across runs.")
    parser.add_argument("--label", default="",
                        help="Config tag for this run, e.g. 'seqs64'. Distinguishes "
                             "sweep rows in --summary-csv.")
    # Recorded, not applied: these must match how the server was actually
    # launched. The sweep driver knows the values it used; the harness cannot
    # introspect them, so it takes them on trust and writes them down.
    parser.add_argument("--max-num-seqs", default="",
                        help="Server's max_num_seqs, recorded in the summary.")
    parser.add_argument("--max-num-batched-tokens", default="",
                        help="Server's max_num_batched_tokens, recorded in the summary.")
    parser.add_argument("--prefix-caching", choices=["on", "off", "unknown"],
                        default="unknown",
                        help="Whether the server ran with prefix caching enabled.")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="Generate exactly --max-tokens per request (vLLM ignore_eos).")
    args = parser.parse_args()

    started = datetime.now(timezone.utc)
    run_id = started.strftime("%Y%m%dT%H%M%SZ")

    prompt = DEFAULT_PROMPT
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()

    concurrency_levels = [int(c) for c in args.concurrency_levels.split(",")]
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)

    summaries = []
    csv_rows = []
    try:
        for concurrency in concurrency_levels:
            print(f"\n== concurrency={concurrency} ==")
            summary, all_results = await bench_concurrency(
                client, args.model, prompt, args.max_tokens, concurrency, args.rounds, args.warmup_rounds,
                args.ignore_eos,
            )
            summaries.append(summary)
            for i, r in enumerate(all_results):
                csv_rows.append({
                    "run_id": run_id,
                    "label": args.label,
                    "concurrency": concurrency,
                    "round": i // concurrency,
                    "ttft": r["ttft"],
                    "mean_itl": r["mean_itl"],
                    "completion_tokens": r["completion_tokens"],
                    "total_time": r["total_time"],
                    "tokens_per_sec": r["tokens_per_sec"],
                })
    finally:
        # Release pooled connections deterministically rather than leaving them
        # to interpreter shutdown. This is hygiene, not a fix for the teardown
        # traceback -- see _silence_asyncgen_teardown_noise().
        await client.close()

    print()
    print_summary_table(summaries)
    write_csv(args.output_csv, csv_rows)

    summary_rows = [
        {
            "run_id": run_id,
            "timestamp": started.isoformat(timespec="seconds"),
            "label": args.label,
            "model": args.model,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "prefix_caching": args.prefix_caching,
            "rounds": args.rounds,
            "warmup_rounds": args.warmup_rounds,
            "max_tokens": args.max_tokens,
            "ignore_eos": args.ignore_eos,
            **sm,
        }
        for sm in summaries
    ]
    append_summary_csv(args.summary_csv, summary_rows)

    print(f"\nRaw per-request results written to {args.output_csv}")
    print(f"Summary rows appended to {args.summary_csv} (run_id={run_id}, label={args.label or '-'})")


if __name__ == "__main__":
    asyncio.run(main())
