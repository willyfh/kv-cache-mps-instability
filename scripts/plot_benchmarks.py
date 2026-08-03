"""
Paper figure generator — produces 4 publication-ready PDFs in media/.

  fig1_latency_scaling.pdf   — §4.1+4.2: latency vs tokens, all models × modes
  fig2_instability_probe.pdf — §4.3:     instability probe, GPT-2 Medium MPS
  fig3_kv_cache_ablation.pdf — §4.4:     KV cache on vs off, MPS
  fig4_prompt_ablation.pdf   — §4.5:     prompt-length ablation, short vs long

All CSV paths point to the canonical final-run files.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MAIN_CSV          = "benchmark_results/main_crossover/benchmark_results_20260502T150901Z.csv"
CUDA_CSV          = "benchmark_results/cuda_baseline/benchmark_results_colab_t4.csv"
INSTABILITY_CSV   = "benchmark_results/instability_probe_final/benchmark_results_20260503T115603Z.csv"
KV_ON_CSV         = "benchmark_results/kv_cache_on/benchmark_results_paper_merged.csv"
KV_OFF_CSV        = "benchmark_results/kv_cache_off_rerun_v2/benchmark_results_20260504T011007Z.csv"
KV_OFF_FALLBACK_CSV = "benchmark_results/kv_cache_off/benchmark_results_20260502T230455Z.csv"
PROMPT_ABL_CSV    = "benchmark_results/prompt_length_ablation_final/benchmark_results_20260503T101508Z.csv"
OUT_DIR           = "media"

MODEL_ORDER = ["distilgpt2", "gpt2", "gpt2-medium", "gpt2-large"]
MODEL_LABELS = {
    "distilgpt2":  "DistilGPT-2",
    "gpt2":        "GPT-2",
    "gpt2-medium": "GPT-2 Medium",
    "gpt2-large":  "GPT-2 Large",
}
MODE_ORDER  = ["cpu", "mps", "cuda"]
MODE_LABELS = {
    "cpu": "CPU",
    "mps": "MPS",
    "cuda": "CUDA (T4)",
}

def _style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    sns.set_palette("colorblind")


# ---------------------------------------------------------------------------
# Fig 1 — Latency scaling: 4 subplots (one per model), all modes × tokens
# ---------------------------------------------------------------------------
def fig1_latency_scaling(out_path):
    df_main = pd.read_csv(MAIN_CSV)
    df_cuda = pd.read_csv(CUDA_CSV)
    df = pd.concat([df_main, df_cuda], ignore_index=True)
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["mode_label"]  = df["execution_mode"].map(MODE_LABELS)

    _style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()

    palette = sns.color_palette("colorblind", n_colors=len(MODE_ORDER))
    mode_colors = {MODE_LABELS[m]: c for m, c in zip(MODE_ORDER, palette)}

    for ax, model_key in zip(axes, MODEL_ORDER):
        sub = df[df["model"] == model_key].copy()
        for mode_key in MODE_ORDER:
            mode_label = MODE_LABELS[mode_key]
            msub = sub[sub["execution_mode"] == mode_key].sort_values("max_tokens")
            if msub.empty:
                continue
            ax.plot(
                msub["max_tokens"], msub["avg_latency"],
                marker="o", label=mode_label,
                color=mode_colors[mode_label], linewidth=1.8, markersize=5
            )
        ax.set_title(MODEL_LABELS[model_key], fontweight="bold")
        ax.set_xlabel("Decoding Budget (max_tokens)")
        ax.set_ylabel("Avg latency (s)")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(128))

    # Shared legend below
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Device",
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Latency Scaling by Model and Device", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Fig 5 — CUDA T4 baseline: latency scaling, smooth monotonic reference
# ---------------------------------------------------------------------------
def fig5_cuda_baseline(out_path):
    df = pd.read_csv(CUDA_CSV)
    df["model_label"] = df["model"].map(MODEL_LABELS)

    _style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()

    color = sns.color_palette("colorblind")[2]  # distinct from CPU/MPS

    for ax, model_key in zip(axes, MODEL_ORDER):
        sub = df[df["model"] == model_key].sort_values("max_tokens")
        if sub.empty:
            continue
        ax.plot(sub["max_tokens"], sub["avg_latency"],
                marker="o", color=color, linewidth=1.8, markersize=5)
        ax.set_title(MODEL_LABELS[model_key], fontweight="bold")
        ax.set_xlabel("Generated tokens")
        ax.set_ylabel("Avg latency (s)")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(128))

    fig.suptitle("CUDA T4 Latency Scaling (Reference Baseline)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Fig 2 — Instability probe: GPT-2 Medium MPS, 480–656 step 16
# ---------------------------------------------------------------------------
def fig2_instability_probe(out_path):
    df = pd.read_csv(INSTABILITY_CSV)
    df = df.sort_values("max_tokens")

    _style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["max_tokens"], df["avg_latency"],
            marker="o", color=sns.color_palette("colorblind")[0],
            linewidth=2, markersize=6, zorder=3)

    # Shade the pathological region (512–624)
    ax.axvspan(512, 624, alpha=0.12, color="red", label="Pathological regime")
    ax.axvline(512, color="red",  linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(624, color="red",  linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(640, color="green", linestyle="--", linewidth=1, alpha=0.7, label="Recovery (640 tok)")

    ax.set_xlabel("Decoding Budget (max_tokens)")
    ax.set_ylabel("Avg latency (s)")
    ax.set_title("Instability Probe — GPT-2 Medium / MPS", fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(16))
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Fig 3 — KV cache ablation: cache-on vs cache-off, 2 panels
# ---------------------------------------------------------------------------
def fig3_kv_cache_ablation(out_path):
    on = pd.read_csv(KV_ON_CSV)
    on["cache"] = "KV cache ON"

    # Primary cache-off points come from rerun_v2; fill missing token points (e.g., 128/256)
    # from the older run so the figure spans the same token range as cache-on.
    off_primary = pd.read_csv(KV_OFF_CSV)
    off_fallback = pd.read_csv(KV_OFF_FALLBACK_CSV)
    off = pd.concat([off_primary, off_fallback], ignore_index=True)
    off = off.sort_values("max_tokens").drop_duplicates(
        subset=["model", "execution_mode", "prompt_name", "max_tokens", "offered_load", "use_cache"],
        keep="first"
    )
    off["cache"] = "KV cache OFF"

    df  = pd.concat([on, off], ignore_index=True)
    df  = df[df["execution_mode"] == "mps"]

    _style()
    palette = {"KV cache ON": sns.color_palette("colorblind")[0],
               "KV cache OFF": sns.color_palette("colorblind")[1]}

    fig, ax = plt.subplots(figsize=(7, 5))
    for cache_label in ["KV cache ON", "KV cache OFF"]:
        csub = df[df["cache"] == cache_label].sort_values("max_tokens")
        ax.plot(csub["max_tokens"], csub["avg_latency"],
                marker="o", label=cache_label,
                color=palette[cache_label], linewidth=2, markersize=5)
    ax.set_xlabel("Decoding Budget (max_tokens)")
    ax.set_ylabel("Avg latency (s)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(128))
    ax.legend()

    fig.suptitle("KV Cache Ablation — GPT-2 Medium / MPS", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Fig 4 — Prompt-length ablation: short vs long, grouped bars
# ---------------------------------------------------------------------------
def fig4_prompt_ablation(out_path):
    df = pd.read_csv(PROMPT_ABL_CSV)
    df = df.sort_values(["max_tokens", "prompt_name"])
    df["prompt_label"] = df["prompt_name"].map({"short": "Short (3 tok)", "long": "Long (65 tok)"})

    _style()
    fig, ax = plt.subplots(figsize=(8, 5))

    palette = sns.color_palette("colorblind", n_colors=2)
    prompt_labels = ["Short (3 tok)", "Long (65 tok)"]
    prompt_colors = {p: c for p, c in zip(prompt_labels, palette)}
    token_vals = sorted(df["max_tokens"].unique())
    x = range(len(token_vals))
    width = 0.35

    for i, prompt_label in enumerate(prompt_labels):
        sub = df[df["prompt_label"] == prompt_label]
        heights = [sub[sub["max_tokens"] == t]["avg_latency"].values[0]
                   if len(sub[sub["max_tokens"] == t]) else 0
                   for t in token_vals]
        offset = (i - 0.5) * width
        bars = ax.bar([xi + offset for xi in x], heights, width,
                      label=prompt_label, color=prompt_colors[prompt_label],
                      edgecolor="black", linewidth=0.7)

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(t) for t in token_vals])
    ax.set_xlabel("Decoding Budget (max_tokens)")
    ax.set_ylabel("Avg latency (s)")
    ax.set_title("Prompt-Length Ablation — GPT-2 Medium / MPS", fontweight="bold")
    ax.legend(title="Prompt length")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Generating paper figures...")
    fig1_latency_scaling(f"{OUT_DIR}/fig1_latency_scaling.pdf")
    fig2_instability_probe(f"{OUT_DIR}/fig2_instability_probe.pdf")
    fig3_kv_cache_ablation(f"{OUT_DIR}/fig3_kv_cache_ablation.pdf")
    fig4_prompt_ablation(f"{OUT_DIR}/fig4_prompt_ablation.pdf")
    print("Done. All figures saved to media/")
