# Non-Monotonic Latency in Apple MPS Decoding: KV Cache Interactions and Execution Regimes

> **Note:** This is research benchmark code released for reproducibility purposes. It is not production software. Primary characterization is on Apple M3 Max with PyTorch 2.8.0 / macOS 14.8.1; the qualitative anomaly was validated across PyTorch 2.7.0/2.8.0/2.11.0 and reproduced on a second device (Apple M3 Pro). Exact latency values and anomaly thresholds are hardware- and version-dependent, but the phenomenon itself is not specific to a single configuration.

## Objective

This repository contains the benchmark harness used to identify and characterize **non-monotonic latency scaling** in autoregressive decoding on Apple's MPS backend: latency spikes of up to 21x within specific decoding-budget intervals, absent on CPU and NVIDIA CUDA under identical conditions. KV cache interacts strongly with these regimes — amplifying the effect rather than causing it, since the anomaly persists (in weaker form) with KV cache disabled. Text generation is served via [LitServe](https://github.com/Lightning-AI/LitServe) so that measurements reflect realistic serving conditions rather than bare model calls.

## Scope

- Hardware: Apple M3 Max / M3 Pro (CPU and MPS backends), NVIDIA T4 (CUDA)
- Models: DistilGPT-2, GPT-2, GPT-2 Medium, GPT-2 Large, BLOOM-560M, OPT-350M
- Execution modes: `cpu`, `cpu-fp16`, `mps`, `mps-fp16`, `cuda`
- Metrics: latency (avg/p50/p95/std/min/max), tokens/sec, prefill/decode split, per-step timing, peak memory

## Quickstart

Install dependencies:

```bash
uv venv && uv sync
```

Run one of the paper's experiment configs (starts/stops the server automatically):

```bash
APP_CONFIG_PATH=configs/paper/kv_cache_on.yaml uv run python -m scripts.benchmark
```

API smoke test — start server, then query:

```bash
uv run python -m app.server
```
```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "insert your prompt here", "max_tokens": 20}' | python3 -m json.tool
```

## Experiment Configs

Configs used to produce the paper's results are in `configs/paper/`.

## Benchmark Method

- Config is selected via the `APP_CONFIG_PATH` environment variable.
- The benchmark script starts and stops an isolated server process per case on a randomly chosen free port.
- All paper configs use deterministic decoding (`benchmark_deterministic: true`, `benchmark_seed: 42`); this is opt-in per config, not a code-level default.
- Memory collection (`benchmark_collect_memory`) is kept separate from latency runs to avoid timing artifacts.
- Results are written to CSV and JSON under `benchmark_results_dir` (gitignored; not included in this repo).
