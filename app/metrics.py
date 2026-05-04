"""Metric helpers used by benchmark scripts."""

import numpy as np


def calculate_tokens_per_sec(latencies, tokens_generated):
    if len(latencies) != len(tokens_generated):
        raise ValueError("latencies and tokens_generated length mismatch")

    return [t / l if l > 0 else 0 for t, l in zip(tokens_generated, latencies)]


def average_latency(latencies):
    return float(np.mean(latencies)) if latencies else 0.0


def p95_latency(latencies):
    return float(np.percentile(latencies, 95)) if latencies else 0.0