"""Typed request/response models for the generation API."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict, Field


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=8000)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=4096)
    temperature: Optional[float] = Field(default=None, gt=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    seed: Optional[int] = Field(default=None, ge=0)
    deterministic: Optional[bool] = None
    use_cache: Optional[bool] = None
    debug: Optional[bool] = None


class GenerationResponse(BaseModel):
    """
    Fully aligned with inference.py output + debug-safe.
    """

    model_config = ConfigDict(extra="allow")  # IMPORTANT: allow passthrough

    text: str

    latency: float
    model_time: float

    tokens_generated: int
    tokens_per_sec: float
    model_tokens_per_sec: float

    prefill_time: float
    decode_time: float

    pre_tokenize_time: float
    input_to_device_time: float
    postprocess_time: float

    eos_token_id: Optional[int] = None
    eos_seen: bool
    eos_position: Optional[int] = None
    stop_reason: str

    cpu_rss_mb: Optional[float] = None
    mps_allocated_mb: Optional[float] = None

    # ✅ DEBUG BLOCK (THIS IS THE KEY FIX)
    debug: Optional[Dict[str, Any]] = None