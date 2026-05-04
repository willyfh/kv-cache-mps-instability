"""Model and tokenizer loading utilities."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_NAME = "distilgpt2"


def load_model(model_name=DEFAULT_MODEL_NAME, execution_mode="cpu"):
    """
    Supported modes:
    - cpu
    - cpu-fp16   (FOR BENCHMARKING ONLY, may be slower on CPU)
    - cpu-int8
    - mps
    - mps-fp16
    """

    supported_modes = {"cpu", "cpu-fp16", "cpu-int8", "mps", "mps-fp16"}

    if execution_mode not in supported_modes:
        raise ValueError(f"Unsupported mode: {execution_mode}")

    # -------------------------
    # device
    # -------------------------
    if execution_mode.startswith("mps"):
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available")
        device = "mps"
    else:
        device = "cpu"

    # -------------------------
    # dtype
    # -------------------------
    if execution_mode in {"cpu-fp16", "mps-fp16"}:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # -------------------------
    # WARNING (IMPORTANT FOR PAPER)
    # -------------------------
    if execution_mode == "cpu-fp16":
        print("[WARN] cpu-fp16 is experimental and may be SLOWER than fp32 due to lack of native support")

    # -------------------------
    # load model
    # -------------------------
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model.eval()

    # -------------------------
    # quantization (CPU INT8)
    # -------------------------
    if execution_mode == "cpu-int8":
        model = torch.ao.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )

    model = model.to(device)

    # -------------------------
    # enforce dtype consistency
    # -------------------------
    for p in model.parameters():
        if p.dtype != torch_dtype:
            p.data = p.data.to(torch_dtype)

    return model


def load_tokenizer(model_name=DEFAULT_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    return tokenizer