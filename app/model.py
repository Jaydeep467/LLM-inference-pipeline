"""
LLM Inference Pipeline
Core model serving with batching, mixed-precision, and async GPU scheduling.
"""

import time
import threading
import queue
import logging
from dataclasses import dataclass, field
from typing import List, Optional
from contextlib import nullcontext

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from app.observability import metrics

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
MODEL_NAME       = "gpt2"
MAX_NEW_TOKENS   = 100
MAX_BATCH_SIZE   = 8
BATCH_TIMEOUT_MS = 50        # ms to wait before flushing a partial batch
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────
@dataclass
class InferenceRequest:
    prompt:      str
    max_tokens:  int = MAX_NEW_TOKENS
    temperature: float = 1.0
    result:      Optional[threading.Event] = field(default_factory=threading.Event)
    response:    Optional[str] = None
    error:       Optional[str] = None


# ──────────────────────────────────────────────
# Model loader
# ──────────────────────────────────────────────
def load_model():
    logger.info(f"Loading {MODEL_NAME} on {DEVICE} ...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(DEVICE)

    # Warm-up pass so first real request isn't penalised
    with torch.no_grad():
        dummy = tokenizer("warmup", return_tensors="pt").to(DEVICE)
        model.generate(**dummy, max_new_tokens=1)

    logger.info("Model ready.")
    return model, tokenizer


# ──────────────────────────────────────────────
# Batch processor
# ──────────────────────────────────────────────
class BatchProcessor:
    """
    Async batching queue: collects requests for BATCH_TIMEOUT_MS,
    then runs a single batched forward pass with mixed-precision.
    """

    def __init__(self):
        self.model, self.tokenizer = load_model()
        self._queue: queue.Queue[InferenceRequest] = queue.Queue()
        self._stop  = threading.Event()

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        logger.info("BatchProcessor worker started.")

    # ── public API ──────────────────────────────
    def submit(self, request: InferenceRequest) -> None:
        self._queue.put(request)

    def shutdown(self):
        self._stop.set()
        self._worker.join()

    # ── internal loop ───────────────────────────
    def _run(self):
        while not self._stop.is_set():
            batch: List[InferenceRequest] = []

            # Block until at least one request arrives
            try:
                batch.append(self._queue.get(timeout=0.1))
            except queue.Empty:
                continue

            # Drain up to MAX_BATCH_SIZE within BATCH_TIMEOUT_MS
            deadline = time.monotonic() + BATCH_TIMEOUT_MS / 1000
            while len(batch) < MAX_BATCH_SIZE and time.monotonic() < deadline:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    time.sleep(0.002)

            self._process_batch(batch)

    def _process_batch(self, batch: List[InferenceRequest]):
        prompts = [r.prompt for r in batch]
        logger.info(f"Processing batch of {len(batch)} request(s).")

        t0 = time.perf_counter()
        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(DEVICE)

            # Mixed-precision: use autocast on CUDA, plain context on CPU
            amp_ctx = (
                torch.cuda.amp.autocast(dtype=torch.float16)
                if DEVICE == "cuda"
                else nullcontext()
            )

            with torch.no_grad(), amp_ctx:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max(r.max_tokens for r in batch),
                    do_sample=True,
                    temperature=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            latency_ms = (time.perf_counter() - t0) * 1000
            metrics.record(batch_size=len(batch), latency_ms=latency_ms)

            for req, out_ids, inp_ids in zip(
                batch, outputs, inputs["input_ids"]
            ):
                new_ids = out_ids[len(inp_ids):]
                req.response = self.tokenizer.decode(
                    new_ids, skip_special_tokens=True
                )
                req.result.set()

        except Exception as exc:
            logger.exception("Batch inference failed.")
            for req in batch:
                req.error = str(exc)
                req.result.set()


# ──────────────────────────────────────────────
# Module-level singleton (lazy-initialised)
# ──────────────────────────────────────────────
_processor: Optional[BatchProcessor] = None
_lock = threading.Lock()


def get_processor() -> BatchProcessor:
    global _processor
    if _processor is None:
        with _lock:
            if _processor is None:
                _processor = BatchProcessor()
    return _processor
