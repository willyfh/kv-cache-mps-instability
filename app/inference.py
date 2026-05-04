"""Inference helpers for token generation + profiling (+ optional FLOPs)."""

import time
import torch


# =========================================================
# SYNC
# =========================================================
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# =========================================================
# PROFILER
# =========================================================
class ProfilerState:
    def __init__(self):
        self.layer_times = {}
        self.token_step_times = []
        self.sequence_lengths = []

    def reset(self):
        self.layer_times = {}
        self.token_step_times = []
        self.sequence_lengths = []


# =========================================================
# FLOP ESTIMATOR (LIGHTWEIGHT APPROX)
# =========================================================
def _estimate_linear_flops(hidden, seq_len, vocab_size):
    """
    Very standard transformer approximation:
    - attention + MLP simplified proxy
    """
    # MLP: ~2 * hidden^2
    mlp = 2 * hidden * hidden

    # attention projection approx: 4 * hidden^2
    attn = 4 * hidden * hidden

    # logits projection: hidden * vocab
    lm_head = hidden * vocab_size

    return mlp + attn + lm_head


# =========================================================
# GENERATION
# =========================================================
def generate_text(
    prompt,
    model,
    tokenizer,
    max_tokens=50,
    temperature=0.7,
    top_p=1.0,
    seed=None,
    deterministic=False,
    use_cache=True,
    debug=True,
    compute_flops=False,   # 👈 NEW FLAG
):

    prof = ProfilerState()
    prof.reset()

    t0 = time.perf_counter()
    device = next(model.parameters()).device

    raw = tokenizer(prompt, return_tensors="pt")
    t1 = time.perf_counter()

    inputs = {k: v.to(device) for k, v in raw.items()}
    input_len = inputs["input_ids"].shape[1]
    t2 = time.perf_counter()

    model.eval()

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    _sync(device)

    # =====================================================
    # PREFILL
    # =====================================================
    t_prefill_start = time.perf_counter()

    with torch.no_grad():
        prefill = model(**inputs, use_cache=use_cache)

    _sync(device)
    t_prefill_end = time.perf_counter()
    prefill_time = t_prefill_end - t_prefill_start if debug else None

    generated = []

    # FLOPs accumulator
    total_flops = 0

    hidden = model.config.hidden_size
    vocab = model.config.vocab_size

    # Use prefill logits to produce the first generated token for both cache modes.
    logits = prefill.logits[:, -1]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    generated.append(next_token.item())

    first_seq_len = input_len + 1
    if debug:
        prof.sequence_lengths.append(first_seq_len)

    if compute_flops:
        total_flops += _estimate_linear_flops(hidden, first_seq_len, vocab)

    if use_cache:
        past = prefill.past_key_values
        input_ids = next_token
    else:
        past = None
        input_ids = torch.cat([inputs["input_ids"], next_token], dim=1)

    for step in range(1, max_tokens):
        if debug:
            t_step0 = time.perf_counter()

        with torch.no_grad():
            if use_cache:
                out = model(
                    input_ids=input_ids,
                    past_key_values=past,
                    use_cache=True,
                )
                past = out.past_key_values
            else:
                out = model(
                    input_ids=input_ids,
                    use_cache=False,
                )

        logits = out.logits[:, -1]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

        if use_cache:
            input_ids = next_token
        else:
            input_ids = torch.cat([input_ids, next_token], dim=1)

        generated.append(next_token.item())

        seq_len = input_len + step + 1
        if debug:
            prof.sequence_lengths.append(seq_len)

        # =====================================================
        # FLOPs ESTIMATION (ONLY IF ENABLED)
        # =====================================================
        if compute_flops:
            total_flops += _estimate_linear_flops(hidden, seq_len, vocab)

        _sync(device)
        if debug:
            t_step1 = time.perf_counter()
            prof.token_step_times.append(t_step1 - t_step0)

    t_end = time.perf_counter()
    decode_time = t_end - t_prefill_end

    tokens = torch.tensor(generated, device=device, dtype=torch.long).unsqueeze(0)
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)

    decode_tokens_per_sec = len(generated) / decode_time if decode_time > 0 else None
    end_to_end_tokens_per_sec = len(generated) / (t_end - t0)

    result = {
        "text": text,
        "latency": t_end - t0,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "tokens_generated": len(generated),
        "model_tokens_per_sec": decode_tokens_per_sec,
        "end_to_end_tokens_per_sec": end_to_end_tokens_per_sec,
    }

    if debug:
        result["debug"] = {
            "token_step_times": prof.token_step_times,
            "sequence_lengths": prof.sequence_lengths,
        }
    else:
        result["debug"] = None

    # =====================================================
    # OPTIONAL FLOPS OUTPUT
    # =====================================================
    if compute_flops:
        result["flops"] = total_flops
        result["flops_per_token"] = total_flops / max(1, len(generated))

    return result