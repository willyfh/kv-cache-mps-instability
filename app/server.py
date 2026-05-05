"""LitServe API entrypoint for text generation requests (paper-safe version)."""

import os
import logging
import psutil
import torch
import litserve as ls
from threading import Event, Thread

from fastapi import HTTPException
from pydantic import ValidationError

from app.inference import generate_text
from app.model import DEFAULT_MODEL_NAME, load_model, load_tokenizer
from app.schemas import GenerationRequest
from app.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_cpu_rss_mb():
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def _read_mps_allocated_mb(execution_mode):
    if execution_mode not in ["mps", "mps-fp16"]:
        return None
    try:
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    except Exception:
        return None


def _read_cuda_allocated_mb(execution_mode):
    if execution_mode != "cuda":
        return None
    try:
        return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        return None

def _read_system_memory_mb():
    try:
        vm = psutil.virtual_memory()
        return (vm.total - vm.available) / (1024 * 1024)
    except Exception:
        return None

class TextGenerationAPI(ls.LitAPI):

    def setup(self, device):
        self.config = load_config()

        self.model_name = os.environ.get(
            "MODEL_NAME",
            self.config.get("model_name", DEFAULT_MODEL_NAME),
        )

        self.execution_mode = os.environ.get(
            "EXECUTION_MODE",
            self.config.get("execution_mode", "cpu"),
        )

        self.model = load_model(
            model_name=self.model_name,
            execution_mode=self.execution_mode,
        )

        self.tokenizer = load_tokenizer(self.model_name)

        self.global_debug = os.environ.get("DEBUG", "0") == "1"
        self.collect_memory = os.environ.get("COLLECT_MEMORY", "1") == "1"
        interval_ms = int(os.environ.get("MEMORY_SAMPLE_INTERVAL_MS", "20"))
        self.memory_sample_interval_s = max(0.001, interval_ms / 1000.0)

        logger.info(
            f"[Setup] model={self.model_name}, mode={self.execution_mode}, debug={self.global_debug}, collect_memory={self.collect_memory}, memory_sample_interval_ms={int(self.memory_sample_interval_s * 1000)}"
        )

    def decode_request(self, request):
        try:
            return GenerationRequest.model_validate(request)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors())

    def predict(self, request):

        req = request.model_copy(
            update={
                "max_tokens": request.max_tokens or self.config.get("default_max_tokens", 50),
                "temperature": request.temperature if request.temperature is not None else 0.7,
                "top_p": request.top_p if request.top_p is not None else 1.0,
                "seed": request.seed,
                "deterministic": request.deterministic if request.deterministic is not None else False,
                "use_cache": request.use_cache if request.use_cache is not None else True,
                "debug": request.debug if request.debug is not None else self.global_debug,
            }
        )

        if req.deterministic:
            torch.manual_seed(req.seed or 42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(req.seed or 42)

        if self.execution_mode.startswith("mps"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass
        elif self.execution_mode == "cuda":
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        peak_cpu_rss_mb = None
        peak_mps_allocated_mb = None
        peak_system_memory_mb = None
        sample_stop = Event()

        def sample_memory_once():
            nonlocal peak_cpu_rss_mb, peak_mps_allocated_mb, peak_system_memory_mb
            cpu = _read_cpu_rss_mb()
            mps = _read_mps_allocated_mb(self.execution_mode)
            cuda_mb = _read_cuda_allocated_mb(self.execution_mode)
            system = _read_system_memory_mb()

            if cpu is not None:
                peak_cpu_rss_mb = cpu if peak_cpu_rss_mb is None else max(peak_cpu_rss_mb, cpu)
            if mps is not None:
                peak_mps_allocated_mb = mps if peak_mps_allocated_mb is None else max(peak_mps_allocated_mb, mps)
            if cuda_mb is not None:
                peak_mps_allocated_mb = cuda_mb if peak_mps_allocated_mb is None else max(peak_mps_allocated_mb, cuda_mb)    
            if system is not None:
                peak_system_memory_mb = system if peak_system_memory_mb is None else max(peak_system_memory_mb, system)

        def memory_sampler_loop():
            while not sample_stop.wait(self.memory_sample_interval_s):
                sample_memory_once()

        sampler_thread = None
        if self.collect_memory:
            sample_memory_once()
            sampler_thread = Thread(target=memory_sampler_loop, daemon=True)
            sampler_thread.start()

        try:
            result = generate_text(
                prompt=req.prompt,
                model=self.model,
                tokenizer=self.tokenizer,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                seed=req.seed,
                deterministic=req.deterministic,
                use_cache=req.use_cache,
                debug=req.debug,
                compute_flops=req.debug,
            )
        finally:
            if self.collect_memory:
                sample_memory_once()
                sample_stop.set()
                if sampler_thread is not None:
                    sampler_thread.join(timeout=0.2)

        # system metrics (per-request observed peak)
        if self.collect_memory:
            result["cpu_rss_mb"] = peak_cpu_rss_mb
            result["mps_allocated_mb"] = peak_mps_allocated_mb
            result["system_memory_mb"] = peak_system_memory_mb
        else:
            result["cpu_rss_mb"] = None
            result["mps_allocated_mb"] = None
            result["system_memory_mb"] = None

        return result

    def encode_response(self, output):
        return output


if __name__ == "__main__":

    api = TextGenerationAPI()
    config = load_config()

    _mode = config.get("execution_mode", "cpu")
    accelerator = "cuda" if _mode == "cuda" else ("mps" if _mode.startswith("mps") else "cpu")

    server = ls.LitServer(
        api,
        accelerator=accelerator,
        workers_per_device=int(config.get("workers_per_device", 1)),
        timeout=float(config.get("timeout_seconds", 30)),
    )

    port = int(os.environ.get("PORT", "8000"))
    logger.info(f"[LitServe] Running on port {port}")

    server.run(port=port)