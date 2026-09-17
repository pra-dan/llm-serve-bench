# Continuous batching on one GPU: 
As concurrency levels rise, where does throughput stop scaling and what happens to latency and why. To observe this, lets start with  TTFT and ITL measurements.

### Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti, 12282 MiB, compute capability 8.9 (Ada) |
| Driver | 580.173.02 |
| Max clocks | 3105 MHz graphics / 10501 MHz memory |
| vLLM | 0.28.0 |
| PyTorch | 2.13.0+cu132 (CUDA 13.2) |
| Python | 3.14.7 (conda env `vllm`) |
| Model | Qwen/Qwen2.5-0.5B-Instruct-AWQ (24 layers, 14 heads, 2 KV heads, head_dim 64, fp16) |
| Client | openai 3.8.0 / httpx 0.28.1 (pool max_connections=1000) |

Resolved scheduler defaults for this GPU + OpenAI API server context:
`max_num_batched_tokens=2048`, `max_num_seqs=256`.


# Ch 1 - Baseline: what does load do?
```sh
sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 2700,2700   # see Ch 2c

vllm serve Qwen/Qwen2.5-0.5B-Instruct-AWQ \
  --quantization awq --gpu-memory-utilization 0.2 \
  --max-model-len 4096 --enforce-eager --no-enable-prefix-caching

python3 bench_vllm.py --model Qwen/Qwen2.5-0.5B-Instruct-AWQ \
  --concurrency-levels 1,8,32,64,128,256 --output-csv results.csv
```

After 3 runs:

| conc | TTFT p50 | TTFT p90 | TTFT p99 | ITL p50 | ITL p90 | req tok/s | agg tok/s |
|---|---|---|---|---|---|---|---|
| 1 | 24.2 ± 0.9 | 31.7 ± 6.3 | 31.7 ± 6.3 | 7.9 ± 0.1 | 8.7 ± 1.1 | 123.1 ± 3.2 | 122.8 ± 3.7 |
| 8 | 49.7 ± 6.0 | 59.6 ± 2.9 | 62.1 ± 2.3 | 9.1 ± 1.2 | 10.7 ± 2.0 | 106.4 ± 8.6 | 835.5 ± 81.0 |
| 32 | 75.1 ± 1.5 | 89.3 ± 1.9 | 93.4 ± 1.9 | 8.6 ± 0.2 | 8.8 ± 0.1 | 108.8 ± 1.5 | 3430.7 ± 45.3 |
| 64 | 114.8 ± 4.9 | 140.6 ± 4.2 | 154.2 ± 0.6 | 9.2 ± 0.0 | 9.5 ± 0.2 | 99.1 ± 0.6 | 6210.0 ± 42.1 |
| 128 | 192.1 ± 4.5 | 245.9 ± 23.1 | 273.9 ± 26.0 | 10.9 ± 0.4 | 11.8 ± 0.4 | 79.5 ± 2.5 | 9844.3 ± 266.2 |
| 256 | 353.1 ± 4.4 | 464.7 ± 22.8 | 586.8 ± 57.8 | 18.2 ± 0.9 | 19.5 ± 0.5 | 44.3 ± 0.4 | 10940.7 ± 82.9 |

**Note:** p99 and p50 have too few samples for conc=1 and 8. So the numbers aren't fully usable here.

Three things were observed here. 
1. **Throughput stops scaling.** Marginal gain per doubling:

| step | 1→8 | 8→32 | 32→64 | 64→128 | 128→256 |
|---|--:|--:|--:|--:|--:|
| agg tok/s | +580% | +311% | +81% | +59% | +11% |

The gain diminishes steadily from conc=32 and collapses at 256; No sharp knee and rather a smooth approach to saturation.

2. **TTFT grows 10x across the sweep** and past conc=32, every double in conc changes TTFT by a factor of [1.5, 2.0]. 

3. The p99/p50 ratio worsens with conc but let's ignore it for a while.

| conc | 1 | 8 | 32 | 64 | 128 | 256 |
|---|--:|--:|--:|--:|--:|--:|
| p99/p50 | 1.13 | 1.24 | 1.24 | 1.34 | 1.42 | 1.66 |

> BTW, p50 is also the median - the typical experience as half the requests were faster than this and the other half were slower. Also, p99 only the slowest 1% were worst than this.

We still don't know why throughput capped or why did TTFT go up? Or is the tail pulling away from the median? Before trying to answer this, I had to find out whether the numebrs are real.

# Ch 2 - Are these numbers reliable/usable?
Not fully. By default, vLLM enables/allows prefix caching. The server logs mentioned `Prefix cache hit rate: 68.3%`. So most of my prefill was being skipped! So I re-ran with `--no-enable-prefix-caching` but the numbers barely moved. 

But why?

By disabling the caching, we were reading the prefix more often, but we still don't see much change. The only explanation could be, the prefix being negligible. My prompt is 17 tokens raw but **46 after the chat template**, which injects a system prompt I never wrote:
```sh
 <|im_start|>system
 You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
 <|im_start|>user
 Explain how a hash table resolves collisions, in about 150 words.<|im_end|>
 <|im_start|>assistant
```

At default granularity: `block_size=16`, 46 tokens is 2 full cacheable blocks (32 tokens) plus a 14-token remainder that is recomputed every time. So the cache is only storing 32 tokens of prefill -> single-digit ms on a 0.5B model. This predicts a hit rate of 32/46 = **69.6**. Observed in server logs: 69.6%. 

> The arithmetic of prefix caching:
> Prefix caching can only reuse *complete blocks*. So if prompt lenth=N and block size=B, cacheable=`floor(N/B) x B` and wasted=`N % B`. This waste is a fixed cost.
>
> But this is minute on longer (general) prompt lengths:
> | prompt length | full blocks | cacheable | ceiling |
> |-|-|-|-|
> | 46 (ours) | 2 | 32 | 69.6% |
> | 512 | 32 | 512 | 100% |
> | 1000 | 62 | 992 | 99.2% |

The only change was observed in throughput at higher conc levels (128 and 256). Look at the agg. TPS and TTFT p50 numbers:

### aggregate_tps (tok/s)

| conc | on | off | change | diff / pooled sd |
|---|---|---|--:|--:|
| 1 | 129.6 ± 2.0 | 131.3 ± 1.3 | +1.3% | +1.0 |
| 8 | 968.9 ± 18.3 | 975.1 ± 18.0 | +0.6% | +0.3 |
| 32 | 3492.6 ± 326.3 | 3619.6 ± 58.9 | +3.6% | +0.5 |
| 64 | 6443.8 ± 355.6 | 6480.4 ± 232.8 | +0.6% | +0.1 |
| 128 | 10244.4 ± 295.9 | 9796.4 ± 116.5 | -4.4% | -2.0 |
| 256 | 10989.0 ± 100.1 | 10631.3 ± 105.0 | -3.3% | -3.5 |

### ttft_p50 (ms)

| conc | on | off | change | diff / pooled sd |
|---|---|---|--:|--:|
| 1 | 23.3 ± 1.1 | 23.3 ± 0.1 | -0.2% | -0.1 |
| 8 | 42.1 ± 0.9 | 43.6 ± 2.7 | +3.6% | +0.7 |
| 32 | 75.5 ± 7.2 | 69.3 ± 2.8 | -8.1% | -1.1 |
| 64 | 112.1 ± 4.2 | 107.9 ± 1.8 | -3.8% | -1.3 |
| 128 | 184.0 ± 2.9 | 191.9 ± 3.0 | +4.3% | +2.6 |
| 256 | 344.5 ± 7.0 | 358.7 ± 7.1 | +4.1% | +2.0 |

![Effect of prefix-caching](plots/03_prefix_caching.png)

The effect of prefix caching can be seen for conc=128 & 256. It wins on both throughput (+4.3%, +3.3%) and TTFT (+4.3%, +4.1%). For the remaining smaller conc levels, the changes are still within the noise/std. This brings us to `max_num_batched_tokens`.

> `max_num_batched_tokens=2048` is the per-step token budget i.e., the max number of tokens vLLM will push thorugh a single forward paass of the model. 

Cached blocks are skipped in the forward pass. More caching -> more budget saved for requests -> more requests admitted.

| | cached tokens per req | budget per req | requests admitted per step (2048 / budget_per_req) | steps to admit 256 requests |
|-|-|-|-|-|
| caching ON | 32 | 14 | ~146 | 2 |
| caching OFF | 0 | 46 | ~44 | 6 |

So caching gave 3.3x higher admission rate! But this is worth nothing until admission is the bottleneck - we are still untouched by this.

## Takeaway 1:
 to actually measure prefix caching's effect, I need a longer shared prefix (a 500 to 2k token prompt), not a 46 one.

## Takeaway 2:
Montoring uncovers something. 

> ## Monitoring
> vLLM already exports the metrics. So I set up Prometheus+Grafana. I start with the [official boilerplate](https://github.com/vllm-project/vllm/blob/main/examples/observability/prometheus_grafana/grafana.json) dashboard of vLLM. 
> 
> For inter-conc levels, I use the csv. For analysing a single conc level, I use vLLM metrics, i.e via Grafana.
> 
> All the metrics can be found at `http://localhost:9090/api/v1/label/__name__/values`. 

Once Prometheus was scraping, I could compare the harness against vLLM's own numbers on the same conc=256 run:

| | vLLM reports | harness measures | gap |
|---|--:|--:|--:|
| TTFT | 222.2 ms | 620.4 ms | **2.8x** |
| ITL | 18.6 ms | 29.5 ms | 1.6x |
| end-to-end | 2578.0 ms | 4753.8 ms | 1.8x |

> [to be moved to appendix] How did we get this table?
> All six came from a single conc=256 run (label promql-check, --rounds 4 --warmup-rounds 1, 1024 requests, prefix caching off).
>
> vLLM column — Prometheus queries of the form rate(..._sum[w]) / rate(..._count[w]), which gives an exact mean independent of bucket boundaries (the reason for not using histogram_quantile here is that vLLM's timing histograms start at le="0.3", far too coarse for millisecond latencies):
>
>value	query
>222.2 ms	rate(vllm:time_to_first_token_seconds_sum[3m]) / rate(vllm:time_to_first_token_seconds_count[3m])
>18.6 ms	rate(vllm:inter_token_latency_seconds_sum[3m]) / rate(vllm:inter_token_latency_seconds_count[3m])
>2578.0 ms	rate(vllm:e2e_request_latency_seconds_sum[2m]) / rate(vllm:e2e_request_latency_seconds_count[2m])
>
>Harness column — computed from that run's raw per-request CSV and summary row:
>value	source
>620.4 ms	statistics.mean(ttft) over the 1024 raw rows
>4753.8 ms	statistics.mean(total_time) over the same rows
>29.5 ms	itl_p50 from the summary row
>
---

About 400ms of TTFT per request is spent outside vLLM. The potential suspect is the harness itself. I thought this: vLLM parses 256 concurrent streams in one asyncio event loop, which is single-threaded, so ~8700 token-events/sec go through one core. This PC has 40 cores and load average sat at 2.66 - consistent with one core pinned, not forty busy. (I need to confirm this again by explicitly sampling the python process's CPU% during a conc=256 run) 

In other words, the numbers in chapter 1 are what an API consumer over HTTP would experience and not what the GPU does. I think its safe to generalise and say that **at high concurrency, such a workload is partly client-bound.** (Worth knowing before blaming the GPU for everything!)

# Ch 3 - So did preemption happen or not?

> Preemption is vLLM kicking out a sequence that's already generating, to free KV cache space for someone else. This unlucky sequence isn't cancelled - we still get the answer but it is rescheduled later. Its KV cache is gone so vLLM has to recompute entire thing from scratch (the original prompt + every token generated so far). vLLM has two recovery modes: recompute (default) and swapping the blocks out to CPU memory and back. Either way, the sequence stalls.

Looking at the (p99/p50) tail again, we notice that the ratio stayed in [1.14, 1.5] while the absolute throughput grew 10x. 

| conc | 1 | 8 | 32 | 64 | 128 | 256 |
|---|--:|--:|--:|--:|--:|--:|
| p99/p50 | 1.14 | 1.26 | 1.31 | 1.22 | 1.43 | 1.50 |

Visually, this is clearer.

![TTFT percentiles](plots/02_ttft_tail.png)

This indicates the entire distribution is sliding towards right (the p50 and p99 staying comparable). This flatness hints at absence of preemption but we can even figure it ourselves using KV arithmetic and the `config.json` from the model's HF page/repo. 

The KV arithmetic says the same thing independently. 

Per-token KV cost here:
```
2 (i.e K and V heads) x 24 layers x 2 kv_heads x 64 head_dim x 2 bytes (bcoz fp16) = 12,299 B = 12 KiB/token
```

Note that this model uses grouped-query attention (GQA) (2 KV heads, not 14) and not multi-head attention (MHA). Also, the model is 4-bit quantizied (AWQ) but the KV cache is not - it stays fp16 by default - interestingly, quantising weights does nothing for KV.

Even though our prompt led to 46 tokens, the average generation length is capped by min(46, `--max-tokens=128`). The KV cache stored is prompt+generated = 46+128 = 174 tokens. This is rounded off to 176 by the 16-token blocks. 

These 176 tokens cost ~2MiB. At conc=256, KV Cache would cost ~512MiB, against ~1.7GiB of max allowed KV space - over 3x headroom. vLLM reports its own capacity as `kv_cache_size_tokens=135200`, so:

```
256 seqs x 176 tokens = 45,056 tokens needed  =  33.3% of capacity
```

Even the server logs were used to confirm no preemption. We get these numbers by polling the server's Prometheus metrics endpoint, `http://localhost:8000/metrics`. `vllm:num_preemptions_total` was **0** at every sample. KV usage peaks at **32.5%**. Preemption is ruled out by measurement, not by argument.

So now we know answer to one of the question left at the end of chapter 1 (Is the tail drifing?). The tail is not pulling away, and nothing is being evicted. At higher conc level, all requests are waiting longer. The `running` topping out at exactly 256, which is `max_num_seqs` and requests queueing with `reason="capacity"` pointed somewhere else.

> **How vLLM tracks waiting requests.** The scheduler keeps two sets: requests *running*
> (being computed this step) and requests *waiting*. Each step it tries to promote waiting
> requests into running, and when it can't, it records why. `vllm:num_requests_waiting`
> is the total; `vllm:num_requests_waiting_by_reason` splits it by label, and this build
> exposes two values - `capacity` and `deferred`.
>
> `capacity` means the scheduler tried to admit the request and had no room this step -
> either no free sequence slot (`max_num_seqs`) or no token budget
> (`max_num_batched_tokens`). `deferred` covers requests held back for other scheduler
> reasons.
>
> The distinction that matters for the next section: **waiting at the door is not the same
> as being thrown out of the room.** A waiting request has never started and has no KV
> cache to lose. A preempted request was already generating and gets its KV cache
> discarded. Both inflate latency; only one is preemption, and they have different
> signatures.


# Ch 4 - Is it admission queueing then?

In Chapter, I mentioned "caching gave 3.3x higher admission rate". 

> Also note `throughput = conc / ITL`, 

# Ch 5 - 




