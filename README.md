# Task: Observe TTFT and ITL change with continuous batching at different concurrency levels

# 1. Lock GPU clocks first (on the box)
```
sudo nvidia-smi -pm 1
nvidia-smi -q -d SUPPORTED_CLOCKS   # find the max graphics clock
sudo nvidia-smi -lgc <clock>,<clock>
```

# 2. Run the harness

```
# Launch server on terminal 1
conda activate vllm
vllm serve Qwen/Qwen2.5-0.5B-Instruct-AWQ     --quantization awq     --gpu-memory-utilization 0.2     --max-model-len 4096     --enforce-eager

# Run harness on terminal 2
python3 bench_vllm.py --model <served-model-name> --concurrency-levels 1,8,32,64 --output-csv results.csv

python3 bench_vllm.py --model Qwen/Qwen2.5-0.5B-Instruct-AWQ --concurrency-levels 1,8,32,64,128,256 --output-csv results.csv
```

# 3. When done
```
sudo nvidia-smi -rgc
```

# Results
```
 conc  TTFT p50  TTFT p90  TTFT p99  ITL p50  ITL p90  req tok/s  agg tok/s
---------------------------------------------------------------------------
    1     37.0m     37.9m     37.9m    13.2m    13.3m       75.4       75.4
    8     63.1m     70.5m     75.2m    11.9m    14.1m       84.2      651.0
   32     83.4m     89.1m     94.4m     9.7m    11.8m       93.7     2937.1
   64    121.5m    153.4m    245.8m    10.6m    11.0m       87.7     5397.5
  128    198.1m    477.8m   1430.1m    11.4m    12.0m       73.6     7800.0
  256    403.3m   2415.6m   3098.2m    13.8m    17.8m       41.0     8227.5

Raw per-request results written to results.csv

```
But this uses prefix caching as we are using the same default prompt every time. So after the first request, each batch is getting near-free near-free prefill as vllm caches the prefix. To turn this off, 

# Observation
These are symptoms and not the actual problem.
- [agg tok/s] or aggregated tps OR Marginal gain per doubling.
Going from conc=1 to conc=8, we see a throughput increase of (75.4 -> 651=) 763%. Similarly
32 -> 64 = 84%
64 -> 128 = 45%
128 -> 256 = 5% only

So its clear how, inceasing the num of concurrent requests isn't translating to increased throughput. Especially, after 128, we are just queueing requests instead of adding throughput.


- [req tok/s] or Per-request tps peaks at conc=32 (93.7) and then collapses to 41.0 at conc=256 (nearly half of that at conc=1 baseline). This marks overload, each request is taking dramatically longer to complete, mainly because total_time which includes ballooning TTFT/queue wait is dominating the tok/s calculation.


- TTFT tail, on looking at p99/p50 ratio, 
| conc | p99/p50 |
|-|-|
| 1 | ~1|
| 8 | 1.19 |
| 32 | 1.13 |
| 64 | 2.02 | 
| 128 | 7.2 | 
| 256 | 7.6 | 

p50 is also the median - the typical experience as half the requests were faster than this and the other half were slower.

p99 only the slowest 1% were worst than this.

After conc=32, the ratio doubles, meaning that the median ttft and the ttft for the worst request has this big gap (2x). This hints at our problem: so when the median and p99 TTFT grow apart like this, it generally signals aat "scheduler preemption from KV-Cache pressure". i.e with growing concurrent requests, our pre-defined KV-cache VRAM allowance, is no more enough for the entire KV cache to be stored, so vllm preempts some i.e evicts and forces a prefill compute later. This is adding to the increased p90 or p99 latencies. 

We need to either (a) aim for a cheaper per-sequence KV-cache footprint (lower `--max-model-len`), which would push the preemption threshold higher. or (b) increase the allowance (the other lever) (`--gpu-memory-utilization`) so there is more VRAM set for KV-Cache.


- itl(inter token latency) - unlike TTFT, this measures the time for the next token generation, once the first token has already generated (mid-stream). ITL p50 reduces from conc=1 to conc=32 and goes back up again. 

Another diff b/w ttft and itl is that ttft captures admission/queuing delay - a single point-in-point time measurement with nothing to dilute it. Whereas ITL is an average for all the remaining (non-first) tokens; So the preemption-led increased delay is distributed among all these token timings. So ITL is definitely a bad metric for measuring preemptiveness.

