#!/usr/bin/env python3
"""vLLM measurement harness: TTFT, inter-token latency, and tokens/sec across concurrency levels.

Usage:
    python3 bench_vllm.py --model Qwen/Qwen2.5-3B-Instruct --concurrency-levels 1,8,32,64
"""
import argparse
import asyncio
import csv
import statistics
import time

from openai import AsyncOpenAI

DEFAULT_PROMPT = (
    "Explain how a hash table resolves collisions, in about 150 words."
)


async def run_single_request(client, model, prompt, max_tokens):
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


async def run_batch(client, model, prompt, max_tokens, concurrency):
    batch_start = time.perf_counter()
    results = await asyncio.gather(
        *[run_single_request(client, model, prompt, max_tokens) for _ in range(concurrency)]
    )
    batch_time = time.perf_counter() - batch_start
    return results, batch_time


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return s[idx]


async def bench_concurrency(client, model, prompt, max_tokens, concurrency, rounds, warmup_rounds):
    for _ in range(warmup_rounds):
        await run_batch(client, model, prompt, max_tokens, concurrency)

    all_results = []
    total_tokens = 0
    total_wall_time = 0.0
    for _ in range(rounds):
        results, batch_time = await run_batch(client, model, prompt, max_tokens, concurrency)
        all_results.extend(results)
        total_tokens += sum(r["completion_tokens"] for r in results)
        total_wall_time += batch_time

    ttfts = [r["ttft"] for r in all_results if r["ttft"] is not None]
    itls = [r["mean_itl"] for r in all_results if r["mean_itl"]]
    per_request_tps = [r["tokens_per_sec"] for r in all_results]
    aggregate_tps = total_tokens / total_wall_time if total_wall_time > 0 else 0.0

    summary = {
        "concurrency": concurrency,
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
    fieldnames = ["concurrency", "round", "ttft", "mean_itl", "completion_tokens", "total_time", "tokens_per_sec"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)


async def main():
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
    args = parser.parse_args()

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
                client, args.model, prompt, args.max_tokens, concurrency, args.rounds, args.warmup_rounds
            )
            summaries.append(summary)
            for i, r in enumerate(all_results):
                csv_rows.append({
                    "concurrency": concurrency,
                    "round": i // concurrency,
                    "ttft": r["ttft"],
                    "mean_itl": r["mean_itl"],
                    "completion_tokens": r["completion_tokens"],
                    "total_time": r["total_time"],
                    "tokens_per_sec": r["tokens_per_sec"],
                })
    finally:
        # Without an explicit close, asyncio.run() tears the loop down while the
        # client's pooled httpcore connections are still open, producing noisy
        # (harmless) GeneratorExit tracebacks on interpreter shutdown.
        await client.close()

    print()
    print_summary_table(summaries)
    write_csv(args.output_csv, csv_rows)
    print(f"\nRaw per-request results written to {args.output_csv}")


if __name__ == "__main__":
    asyncio.run(main())
