"""Benchmark runner (clean + consistent pipeline)."""

import os
import sys
import time
import json
import csv
import socket
import signal
import subprocess
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from datetime import datetime, timezone

import psutil
import numpy as np
import requests

from app.metrics import p95_latency, calculate_tokens_per_sec
from app.utils import load_config


def now():
    return time.perf_counter()


def pick_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def replace_port(base_url, port):
    p = urlparse(base_url)
    return f"{p.scheme}://{p.hostname}:{port}"


def start_server(
    model,
    mode,
    port,
    root,
    debug=False,
    collect_memory=True,
    memory_sample_interval_ms=20,
):
    env = os.environ.copy()
    env["MODEL_NAME"] = model
    env["EXECUTION_MODE"] = mode
    env["PORT"] = str(port)
    env["DEBUG"] = "1" if debug else "0"
    env["COLLECT_MEMORY"] = "1" if collect_memory else "0"
    env["MEMORY_SAMPLE_INTERVAL_MS"] = str(memory_sample_interval_ms)

    return subprocess.Popen(
        [sys.executable, "-m", "app.server"],
        env=env,
        cwd=str(root),
        stdout=None,
        stderr=None,
        text=True,
        start_new_session=True,
    )


def wait_server(url, proc):
    start = now()
    while now() - start < 60:
        if proc.poll() is not None:
            return False, None
        try:
            r = requests.post(url + "/predict", json={"prompt": "hi", "max_tokens": 1})
            if r.status_code in (200, 422):
                return True, now() - start
        except:
            pass
        time.sleep(0.5)
    return False, None


def stop(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except:
        pass


def send(url, payload, timeout):
    t0 = now()
    try:
        r = requests.post(url + "/predict", json=payload, timeout=timeout)
    except:
        return False, None, None, None

    t1 = now()

    if r.status_code != 200:
        return False, None, None, None

    d = r.json()
    return True, t1 - t0, d.get("tokens_generated"), d


def aggregate(debug_list):

    valid = [d for d in debug_list if d]

    steps, seqs = [], []

    for d in valid:
        s = d.get("token_step_times", [])
        q = d.get("sequence_lengths", [])

        steps.extend(s)
        seqs.extend(q)

    return {
        "avg_step":      float(np.mean(steps))            if steps else None,
        "p95_step_time": float(np.percentile(steps, 95))  if steps else None,
        "max_step_time": float(np.max(steps))              if steps else None,
        "avg_seq":       float(np.mean(seqs))              if seqs  else None,
    }


def run_case(url, payload, warmup, runs, concurrency, timeout):

    for _ in range(warmup):
        send(url, payload, timeout)

    lat, tok, mtps, e2e_mtps, dbg = [], [], [], [], []
    cpu_rss, mps_alloc, system_memory = [], [], []
    prefill_times, decode_times = [], []
    first_request_latency = None
    sample_text = None

    for _ in range(runs):
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(send, url, payload, timeout) for _ in range(concurrency)]

            for f in as_completed(futs):
                ok, l, t, d = f.result()
                if not ok:
                    continue
                lat.append(l)
                tok.append(t)
                mtps.append(d.get("model_tokens_per_sec"))
                e2e_mtps.append(d.get("end_to_end_tokens_per_sec"))
                dbg.append(d.get("debug"))

                if d.get("cpu_rss_mb") is not None:
                    cpu_rss.append(d["cpu_rss_mb"])
                if d.get("mps_allocated_mb") is not None:
                    mps_alloc.append(d["mps_allocated_mb"])
                if d.get("prefill_time") is not None:
                    prefill_times.append(d["prefill_time"])
                if d.get("decode_time") is not None:
                    decode_times.append(d["decode_time"])
                if d.get("system_memory_mb") is not None:
                    system_memory.append(d["system_memory_mb"])

                if first_request_latency is None:
                    first_request_latency = l
                if sample_text is None:
                    sample_text = d.get("text")

    return lat, tok, mtps, e2e_mtps, dbg, cpu_rss, mps_alloc, system_memory, prefill_times, decode_times, first_request_latency, sample_text


def run_benchmark():

    cfg = load_config()
    root = pathlib.Path(__file__).parent.parent

    out = pathlib.Path(cfg["benchmark_results_dir"])
    out.mkdir(exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path  = out / f"benchmark_results_{ts}.csv"
    json_path = out / f"benchmark_results_{ts}.json"

    capture_qualitative = cfg.get("benchmark_capture_qualitative_samples", False)
    qual_path = out / f"qualitative_samples_{ts}.json" if capture_qualitative else None
    qual_samples = [] if capture_qualitative else None

    use_cache = cfg.get("benchmark_use_cache", True)
    if not isinstance(use_cache, bool):
        raise ValueError(
            f"benchmark_use_cache must be a boolean (true/false), got: {use_cache!r}"
        )
    debug = cfg.get("benchmark_debug", True)
    collect_memory = cfg.get("benchmark_collect_memory", True)
    memory_sample_interval_ms = int(cfg.get("benchmark_memory_sample_interval_ms", 20))

    all_rows = []
    first_csv = True

    for model in cfg["benchmark_model_names"]:
        for mode in cfg["benchmark_execution_modes"]:
            for load in cfg["benchmark_offered_load_levels"]:
                for name, prompt in cfg["benchmark_prompts"].items():
                    for mt in cfg["benchmark_max_tokens"]:

                        port = pick_free_port()
                        url = replace_port(cfg["benchmark_base_url"], port)

                        proc = start_server(
                            model,
                            mode,
                            port,
                            root,
                            debug=debug,
                            collect_memory=collect_memory,
                            memory_sample_interval_ms=memory_sample_interval_ms,
                        )
                        ready, startup_latency = wait_server(url, proc)

                        if not ready:
                            stop(proc)
                            continue

                        payload = {
                            "prompt": prompt,
                            "max_tokens": mt,
                            "use_cache": use_cache,
                            "debug": debug,
                        }

                        if cfg.get("benchmark_deterministic"):
                            payload["deterministic"] = True
                            payload["seed"] = cfg.get("benchmark_seed", 42)

                        print(
                            f"Running: model={model} mode={mode} load={load} "
                            f"prompt={name} max_tokens={mt} use_cache={use_cache}"
                        )

                        (
                            lat, tok, mtps, e2e_mtps, dbg,
                            cpu_rss, mps_alloc, system_memory,
                            prefill_times, decode_times,
                            first_req_lat, sample_text,
                        ) = run_case(
                            url,
                            payload,
                            cfg["benchmark_warmup_runs"],
                            cfg["benchmark_runs"],
                            load,
                            cfg["timeout_seconds"],
                        )

                        stop(proc)

                        samples = len(lat)
                        total_attempts = cfg["benchmark_runs"] * load
                        success_rate = 100.0 * samples / total_attempts if total_attempts > 0 else 0.0

                        tps_list = calculate_tokens_per_sec(lat, tok) if lat else []

                        peak_cpu_rss_mb   = float(np.max(cpu_rss))   if cpu_rss   else None
                        peak_mps_alloc_mb = float(np.max(mps_alloc)) if mps_alloc else None
                        peak_system_memory_mb = float(np.max(system_memory)) if system_memory else None
                        peak_memory_mb = (
                            max(v for v in [peak_cpu_rss_mb, peak_mps_alloc_mb, peak_system_memory_mb] if v is not None)
                            if any(v is not None for v in [peak_cpu_rss_mb, peak_mps_alloc_mb, peak_system_memory_mb])
                            else None
                        )

                        dbg_agg = aggregate(dbg)

                        row = {
                            "model":                 model,
                            "execution_mode":        mode,
                            "prompt_name":           name,
                            "max_tokens":            mt,
                            "offered_load":          load,
                            "use_cache":             use_cache,
                            # run reliability
                            "samples":               samples,
                            "success_rate":          round(success_rate, 2),
                            "startup_latency":       round(startup_latency, 6) if startup_latency is not None else None,
                            "first_request_latency": round(first_req_lat, 6)   if first_req_lat  is not None else None,
                            # latency distribution
                            "avg_latency":           float(np.mean(lat))    if lat else None,
                            "p50_latency":           float(np.median(lat))  if lat else None,
                            "p95_latency":           p95_latency(lat)       if lat else None,
                            "std_latency":           float(np.std(lat))     if lat else None,
                            "min_latency":           float(np.min(lat))     if lat else None,
                            "max_latency":           float(np.max(lat))     if lat else None,
                            # throughput distribution
                            "avg_tokens_per_sec":    float(np.mean(tps_list))   if tps_list else None,
                            "p50_tokens_per_sec":    float(np.median(tps_list)) if tps_list else None,
                            "std_tokens_per_sec":    float(np.std(tps_list))    if tps_list else None,
                            "min_tokens_per_sec":    float(np.min(tps_list))    if tps_list else None,
                            "max_tokens_per_sec":    float(np.max(tps_list))    if tps_list else None,
                            # model-reported throughput
                            "avg_decode_tokens_per_sec": float(np.mean([x for x in mtps if x is not None])) if any(x is not None for x in mtps) else None,
                            "p50_decode_tokens_per_sec": float(np.median([x for x in mtps if x is not None])) if any(x is not None for x in mtps) else None,
                            "avg_end_to_end_model_tokens_per_sec": float(np.mean([x for x in e2e_mtps if x is not None])) if any(x is not None for x in e2e_mtps) else None,
                            "p50_end_to_end_model_tokens_per_sec": float(np.median([x for x in e2e_mtps if x is not None])) if any(x is not None for x in e2e_mtps) else None,
                            # prefill / decode split (for paper Section 4)
                            "avg_prefill_time":      float(np.mean(prefill_times)) if prefill_times else None,
                            "avg_decode_time":       float(np.mean(decode_times))  if decode_times  else None,
                            # memory
                            "peak_cpu_rss_mb":       peak_cpu_rss_mb,
                            "peak_mps_allocated_mb": peak_mps_alloc_mb,
                            "peak_system_memory_mb": peak_system_memory_mb,
                            "peak_memory_mb":        peak_memory_mb,
                            # per-token step timing (non-monotonic instability analysis)
                            "avg_step_time":         dbg_agg["avg_step"],
                            "p95_step_time":         dbg_agg["p95_step_time"],
                            "max_step_time":         dbg_agg["max_step_time"],
                            "avg_seq_length":        dbg_agg["avg_seq"],
                        }

                        print("RESULT:", {k: v for k, v in row.items() if k != "prompt_name"})

                        if capture_qualitative and sample_text is not None:
                            qual_samples.append({
                                "model": model,
                                "execution_mode": mode,
                                "prompt_name": name,
                                "max_tokens": mt,
                                "use_cache": use_cache,
                                "prompt": prompt,
                                "generated_text": sample_text,
                            })

                        all_rows.append(row)

                        with open(csv_path, "a", newline="") as f:
                            w = csv.DictWriter(f, fieldnames=row.keys())
                            if first_csv:
                                w.writeheader()
                                first_csv = False
                            w.writerow(row)

    # JSON export for qualitative / per-run analysis
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)

    if capture_qualitative and qual_samples is not None:
        with open(qual_path, "w") as f:
            json.dump(qual_samples, f, indent=2, default=str)
        print("DONE QUAL:", qual_path)

    print("DONE CSV: ", csv_path)
    print("DONE JSON:", json_path)


if __name__ == "__main__":
    run_benchmark()