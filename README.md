# LitServe SLM Benchmark on Apple Silicon

> **Note:** This is research benchmark code released for reproducibility purposes. It is not production software. Results are tied to specific hardware (Apple M3 Max) and software versions (PyTorch 2.8.0, macOS 14.8.1); behavior may differ on other configurations.

## Objective

This project benchmarks autoregressive text-generation serving on Apple Silicon (M3), using GPT-2-family models deployed via a LitServe inference harness.

The primary research focus is characterizing non-monotonic latency scaling and KV cache pathology on the PyTorch MPS backend. See [EXPERIMENTS.md](EXPERIMENTS.md) for the full experiment execution plan.

## Scope

- Hardware: Apple M3 Max (CPU and MPS backends)
- Models: DistilGPT-2, GPT-2, GPT-2 Medium, GPT-2 Large
- Execution modes: `cpu` (FP32), `cpu-fp16`, `mps` (FP32), `mps-fp16`
- Workload: configurable prompt lengths, generation lengths (128–768 tokens), concurrency levels
- Metrics: latency (avg/p50/p95/std/min/max), tokens/sec, prefill time, decode time, per-step timing, peak memory

## Quickstart

1. Install dependencies:
   ```bash
   uv venv && uv sync
   ```

2. Run a sanity benchmark (starts/stops server automatically):
   ```bash
   APP_CONFIG_PATH=configs/mini_sanity.yaml uv run python -m scripts.benchmark
   ```

3. API smoke test — start server, then query:
   ```bash
   uv run python -m app.server
   ```
   ```bash
   curl -s -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Once upon a time", "max_tokens": 20}' | python3 -m json.tool
   ```

## Benchmark Method

- Config is selected via `APP_CONFIG_PATH` environment variable (defaults to `configs/config.yaml`).
- The benchmark script starts and stops an isolated server process per case on a randomly chosen free port.
- Deterministic decoding is enabled by default (`benchmark_deterministic: true`, `benchmark_seed: 42`).
- Memory collection (`benchmark_collect_memory`) is kept separate from latency runs to avoid timing artifacts.
- Per-step token timing (`benchmark_debug: true`) inserts a device sync barrier per decode step — results are not directly comparable to uninstrumented runs.
- Results are written to CSV and JSON under `benchmark_results_dir`.
- Qualitative samples (generated text) are saved to a separate JSON when `benchmark_capture_qualitative_samples: true`.

## Config Keys Reference

| Key | Description |
|-----|-------------|
| `benchmark_model_names` | List of HuggingFace model identifiers |
| `benchmark_execution_modes` | `cpu`, `cpu-fp16`, `mps`, `mps-fp16` |
| `benchmark_offered_load_levels` | Concurrency levels per run |
| `benchmark_max_tokens` | Generation length budget(s) |
| `benchmark_prompts` | Named prompt strings |
| `benchmark_runs` | Number of measured runs per case |
| `benchmark_warmup_runs` | Warmup runs before measurement |
| `benchmark_debug` | Enable per-step token timing |
| `benchmark_use_cache` | Enable/disable KV cache |
| `benchmark_collect_memory` | Enable peak memory sampling |
| `benchmark_memory_sample_interval_ms` | Memory sampler interval (default 20ms) |
| `benchmark_capture_qualitative_samples` | Save generated text to JSON |
| `benchmark_deterministic` | Fixed-seed greedy decoding |
| `benchmark_seed` | Random seed (default 42) |
| `benchmark_results_dir` | Output directory for CSV/JSON |

## API Contract

### POST /predict

Request:

```json
{
  "prompt": "Your prompt text here",
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 1.0,
  "seed": 42,
  "deterministic": true,
  "use_cache": true,
  "debug": false
}
```

Response:

```json
{
  "text": "Generated text...",
  "latency": 1.23,
  "tokens_generated": 45,
  "model_tokens_per_sec": 36.59,
  "prefill_time": 0.02,
  "decode_time": 1.21,
  "cpu_rss_mb": 450.0,
  "mps_allocated_mb": 1400.0,
  "debug": null
}
```