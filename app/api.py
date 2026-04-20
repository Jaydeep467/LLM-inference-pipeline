"""
Flask REST API
Exposes LLM inference endpoints with observability middleware.
"""

import time
import logging
from flask import Flask, request, jsonify, Response

from app.model import get_processor, InferenceRequest, MAX_NEW_TOKENS
from app.prompt import apply_prompt_template, list_templates
from app.observability import metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


# ──────────────────────────────────────────────
# Middleware — request timing
# ──────────────────────────────────────────────
@app.before_request
def _start_timer():
    request._start = time.perf_counter()


@app.after_request
def _log_request(response: Response) -> Response:
    elapsed = (time.perf_counter() - request._start) * 1000
    logger.info(
        f"{request.method} {request.path} → {response.status_code} "
        f"({elapsed:.1f} ms)"
    )
    return response


# ──────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ──────────────────────────────────────────────
# Templates
# ──────────────────────────────────────────────
@app.route("/templates", methods=["GET"])
def get_templates():
    return jsonify({"templates": list_templates()})


# ──────────────────────────────────────────────
# Single inference  POST /predict
# ──────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Body (JSON):
      prompt        str   required
      template      str   optional  default "default"
      system_prompt str   optional
      max_tokens    int   optional  default 100
      temperature   float optional  default 1.0
    """
    body = request.get_json(silent=True) or {}
    raw_prompt = body.get("prompt", "").strip()

    if not raw_prompt:
        return jsonify({"error": "prompt is required"}), 400

    prompt = apply_prompt_template(
        raw_prompt,
        template=body.get("template", "default"),
        system_prompt=body.get("system_prompt"),
    )

    req = InferenceRequest(
        prompt=prompt,
        max_tokens=int(body.get("max_tokens", MAX_NEW_TOKENS)),
        temperature=float(body.get("temperature", 1.0)),
    )

    processor = get_processor()
    processor.submit(req)
    req.result.wait(timeout=30)

    if req.error:
        metrics.record_error()
        return jsonify({"error": req.error}), 500

    return jsonify({
        "prompt":   raw_prompt,
        "response": req.response,
        "template": body.get("template", "default"),
    })


# ──────────────────────────────────────────────
# Batch inference  POST /batch
# ──────────────────────────────────────────────
@app.route("/batch", methods=["POST"])
def batch_predict():
    """
    Body (JSON):
      prompts    list[str]  required
      template   str        optional
      max_tokens int        optional
    """
    body = request.get_json(silent=True) or {}
    raw_prompts = body.get("prompts", [])

    if not raw_prompts or not isinstance(raw_prompts, list):
        return jsonify({"error": "prompts must be a non-empty list"}), 400

    if len(raw_prompts) > 16:
        return jsonify({"error": "max 16 prompts per batch request"}), 400

    template   = body.get("template", "default")
    max_tokens = int(body.get("max_tokens", MAX_NEW_TOKENS))

    processor = get_processor()
    requests  = []

    for raw in raw_prompts:
        prompt = apply_prompt_template(raw.strip(), template=template)
        req = InferenceRequest(prompt=prompt, max_tokens=max_tokens)
        processor.submit(req)
        requests.append((raw, req))

    results = []
    for raw, req in requests:
        req.result.wait(timeout=30)
        if req.error:
            metrics.record_error()
            results.append({"prompt": raw, "error": req.error})
        else:
            results.append({"prompt": raw, "response": req.response})

    return jsonify({"results": results, "count": len(results)})


# ──────────────────────────────────────────────
# Metrics  GET /metrics
# ──────────────────────────────────────────────
@app.route("/metrics", methods=["GET"])
def get_metrics():
    return jsonify(metrics.snapshot())


# ──────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
