Let's make the benchmarking more realistic by 
- using a bigger version of the model (0.5B -> 3B)
- increasing the VRAM allowance 0.2 -> 0.9
- disabling eager-mode (so that cuda-graph fusion doesn't hide the latency)
- using longer prompts. 

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

## Moving to a bigger model
Let's create a baseline for the new (bigger) model. I wanted to calculat the KV cache size this time:
```
KV bytes  = util × total_memory − (weights + profiling overhead) − CUDA-graph estimate
blocks    = floor(KV bytes / (block_size × KV bytes per token))
kv_cache_size_tokens = blocks × 16          (when max_model_len is a multiple of 16)
```

>  In week 1, we calculated the KV cost per token, which makes it clear that different models can have different cost per token. But isn't that different from what we are trying here?
