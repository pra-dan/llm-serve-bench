Let's make the benchmarking more realistic by 
- using a bigger version of the model (0.5B -> 3B)
- increasing the VRAM allowance 0.2 -> 0.8
- disabling eager-mode (so that cuda-graph fusion doesn't hide the latency)
- using longer prompts. 

This should lead to more realistic preemption.

> As a reminder, this benchmark is on a 4070Ti (12GiB) but this time, the driver has been updated from 580.173 to 595.99.

## Eliminating first suspect - overhead:
Each decode step has two costs:
1. GPU work: At conc=1, he GPU reads all the weights once per token, so the time depends on weight's bytes. **This is what AWQ makes smaller.**
2. CPU launch work (overhead): In eager mode (used in the benchmark till now), Python launches each kernel one at a time, multiple times per layer, for each token. Each launch costs CPU time.

CPU and GPU run at the same time so the step takes about `max(CPU_time, GPU_time)`.

```js
(EAGER) CPU slower than GPU:
CPU  |launch|launch|launch|launch|launch|launch|
GPU  |k|    |k|    |k|    |k|    |k|    |k|      

(Without EAGER) CUDA GRAPHS:
CPU  |1 launch|
GPU  |k|k|k|k|k|k|                                
```

To fit the 3B variant, I will be using AWQ quant (again) but first I need to eliminate the possibility of overhead overshadowing the shorter GPU cost. Hence, I will be running this bench without eager mode. This overhead is only considerable for lower conc levels (1 to 8-ish) because at lower levels, GPU and CPU work is kinda comparable, while at higher levels, GPU work is higher.

```sh
# with eager
 conc  TTFT p50  TTFT p90  TTFT p99  ITL p50  ITL p90  req tok/s  agg tok/s
---------------------------------------------------------------------------
    1     24.3m     27.7m     27.7m     8.0m     9.3m      121.4      120.6
    8     45.7m     60.0m     63.0m     8.2m     8.4m      118.2      941.3

# w/o eager 
 conc  TTFT p50  TTFT p90  TTFT p99  ITL p50  ITL p90  req tok/s  agg tok/s
---------------------------------------------------------------------------
    1     11.2m     12.4m     12.4m     1.9m     1.9m      512.0      511.5
    8     29.5m     31.9m     32.1m     2.0m     2.0m      452.7     3576.0
```

Its clear that eager affects TTFT more than ITL (~4.2x vs ~2x) because:
- ITL is one decode step, i.e., predicting next token. With eager, vLLM launches all the kernels for each step :0.
- TTFT contains parts that graphs don't touch: tokenisation, scheduling, etc. (again CPU work but different from kernel launches which is handled by graphs.)

**Note:** to toggle CUDA graphs, I used `--enforce-eager` during server launch but this also disables `torch.compile`. So the numbers w/o eager could have been better if I brought back `torch.compile`.

So when would I use eager mode?
1. Quick use: Creating the CUDA graph can take minutes. 
2. For debugging: Early version of such inference systems can break and debugging each kernel is only possible when all of them are loaded. 
3. Space was between KV Cache and Graphs: Skipping graphs saves more space for KV Cache.

## Moving to a bigger model, testing out quants
Let's create a baseline for the new (bigger) model. 
```sh
sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc 3105,3105

vllm serve Qwen/Qwen2.5-3B-Instruct --dtype bfloat16 \
 --max-model-len 4096 --gpu-memory-utilization 0.80 --no-enable-prefix-caching

python3 bench_vllm.py --model Qwen/Qwen2.5-3B-Instruct \
  --concurrency-levels 1,8,32,64,128,256 --output-csv csv/3B_baseline.csv
```

```
conc  TTFT p50  TTFT p90  TTFT p99  ITL p50  ITL p90  req tok/s  agg tok/s
---------------------------------------------------------------------------
    1     30.7m     31.0m     31.0m    13.9m    13.9m       71.1       71.1
    8     74.4m     77.5m     79.0m    14.3m    14.4m       67.7      539.9
   32    175.9m    203.0m    218.5m    16.1m    16.3m       57.9     1833.8
   64    339.0m    380.2m    408.3m    17.0m    18.2m       50.8     3199.5
  128    459.9m    670.3m    730.3m    21.6m    23.0m       39.6     4967.2
  256    821.8m   1273.6m   1323.6m    34.5m    37.4m       24.6     6135.9
```

**Note:** At conc=1 the GPU must read every weight from VRAM once for each new token. The ITL p50 from above table (13.9ms) is the min time required for a new token for our 3B model. But the floor is decided by the GPU bandwidth and varies with weights size. 
For 3B weights, 6.17 / 504 GPps = 12.2ms
So no token can come out faster than 12.2ms - the. This indicates that the extra 1.7ms is other work (preparing inputs, sampling, streaming to client, etc).
For the previous, lighter 0.5B model, it was 2.07GiB -> ~4.1ms 
i.e ~2.4x faster

---

## Predicting KV capacity
We saw in previous part, how the KV Cache has a **Capacity** and the requests lead to **Usage**.
I am going to do the arithmetic to predict and compare with vLLM logs.
vLLM 0.28 computes the Capacity or `kv_cache_size_tokens` as:
```py
total_KV_bytes = util * avail_VRAM - weights - misc
    = (0.8 * 11.589) - 6.17 - misc
    = 3.1012 - misc
    or under 3.1012 GiB

    # avail RAM was found using `python -c "import torch; print(torch.cuda.mem_get_info()[1]/2**30)"`  (2**30 is num bytes in a GiB)
    # misc is profiler overhead, cuda_graph mem, etc

KV_bytes_per_token = 2 * hidden_layers * 2 KV_heads * 128 head_dim * 2
    = 2 * 36 * 2 * 128 * 2
    = 36,864 bytes
    = 36 Kib

    # this formula was computed in prev part (for 0.5B)
    # head_dim = hidden_size / num_heads = 2048/16 = 128 but this is mostly specific to Qwen

theoritical_max_tokens = total_KV_bytes / KV_bytes_per_token
    = under 3.1012 GiB / 36 Kib per token
    = 90329 tokens

actual_max_token = max_blocks_allowed * 16
    = floor(90329 / 16) * 16
    = 90320
```

The GT is printed by vLLM when it starts:
```sh
(EngineCore pid=2304689) INFO 10-15 17:41:14 [kv_cache_utils.py:1869] GPU KV cache size: 87,552 tokens, ...
```
So we were close!

## 4-bit weights: bf16 vs AWQ vs GPTQ

I want to try three variants of this model and test against my quant!

| Run | Weights | KV cache |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | bf16 — **not quantised. (baseline)** | bf16 |
| `Qwen/Qwen2.5-3B-Instruct-AWQ` | 4-bit (AWQ) | 16-bit |
| `Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4` | 4-bit (GPTQ) | 16-bit |
| baseline + `--kv-cache-dtype fp8` | bf16 | **FP8** |
| FP8-dynamic checkpoint you make | FP8 weights and FP8 activations | bf16 |

> **GPTQ vs. AWQ**
> | Aspect | GPTQ | AWQ | 
> |---|---|---|
> | Core | Focuses on a layer-by-layer optimization process | Uses an activation-aware approach, targeting weights based on sensitivity |
> | Memory savings | high | high |
> | Accuracy | good for standard tasks | good for complex tasks |
> | Calibration | Moderate | Extensive |
> | Processing Time | Lower | Higher | 
> I picked this from [the newline blog](https://www.newline.co/@zaoyang/gptq-vs-awq-quantization--d792476e).

---