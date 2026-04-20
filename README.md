# ⚡ LLM Inference Pipeline

> Production-grade LLM serving system with async batching, mixed-precision inference, and real-time observability — built on PyTorch + GPT-2.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Flask REST API                    │
│    /predict   /batch   /metrics   /health           │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │    Prompt Engineering   │
          │  Templates · Few-shot   │
          │  System prompt hooks    │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │   Async Batch Processor │
          │  Queue → Batch window   │
          │  50ms timeout · size 8  │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  PyTorch Inference Core │
          │  Mixed Precision FP16   │
          │  Memory-efficient ops   │
          │  GPU / CPU adaptive     │
          └────────────┬────────────┘
                       │
          ┌────────────▼────────────┐
          │  Observability Store    │
          │  P50 / P95 / P99 lat.  │
          │  Throughput · Batching  │
          └─────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  Streamlit Dashboard    │
          │  Real-time charts       │
          └─────────────────────────┘
```

---

## Key Features

| Feature                | Implementation                                                                |
| ---------------------- | ----------------------------------------------------------------------------- |
| **Async batching**     | Threading queue with 50ms window, max batch size 8                            |
| **Mixed precision**    | `torch.cuda.amp.autocast(dtype=torch.float16)` on CUDA                        |
| **Memory efficiency**  | Tensor padding + truncation, no redundant allocations                         |
| **Prompt engineering** | 6 built-in templates (instruct, chat, QA, code, summarize) + few-shot builder |
| **Observability**      | P50/P95/P99 latency, req/sec, batch stats — rolling 500-sample window         |
| **REST API**           | Single `/predict`, bulk `/batch` (up to 16), `/metrics`, `/health`            |
| **Dashboard**          | Live Streamlit UI with Plotly latency + throughput charts                     |

---

## Quickstart

```bash
# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/llm-inference-pipeline.git
cd llm-inference-pipeline
pip install -r requirements.txt

# 2. Start the API
python -m app.api

# 3. Start the dashboard (new terminal)
streamlit run dashboard/app.py

# 4. Run tests
pytest tests/ -v
```

---

## API Reference

### `POST /predict`

Single prompt inference.

```json
{
  "prompt": "Explain transformers in one paragraph",
  "template": "instruct",
  "system_prompt": "You are a concise AI tutor.",
  "max_tokens": 120,
  "temperature": 0.8
}
```

**Response:**

```json
{
  "prompt": "Explain transformers in one paragraph",
  "response": "Transformers are...",
  "template": "instruct"
}
```

---

### `POST /batch`

Batch inference — up to 16 prompts per call.

```json
{
  "prompts": ["What is AI?", "Explain neural networks", "Define backprop"],
  "template": "qa",
  "max_tokens": 80
}
```

---

### `GET /metrics`

Live performance snapshot.

```json
{
  "total_requests": 1042,
  "requests_per_sec": 14.3,
  "latency_p50_ms": 210.4,
  "latency_p95_ms": 380.1,
  "latency_p99_ms": 512.7,
  "avg_batch_size": 4.2,
  "uptime_seconds": 72.9
}
```

---

### `GET /templates`

List available prompt templates: `default`, `instruct`, `chat`, `summarize`, `code`, `qa`

---

## Prompt Templates

````python
# instruct
"### Instruction:\n{prompt}\n\n### Response:\n"

# chat
"System: You are a helpful AI assistant.\nUser: {prompt}\nAssistant:"

# qa
"Answer the following question accurately and concisely.\nQuestion: {prompt}\nAnswer:"

# code
"Write clean, well-commented Python code for:\n{prompt}\n```python\n"
````

---

## Performance

On CPU (Intel i7), GPT-2 (117M params):

| Metric                                          | Value       |
| ----------------------------------------------- | ----------- |
| Single request latency                          | ~200–400 ms |
| Batched (8x) latency                            | ~350–600 ms |
| Throughput improvement (batching vs sequential) | ~3–4x       |
| Memory reduction (FP16 on CUDA)                 | ~40–50%     |

---

## Docker

```bash
docker build -t llm-inference-pipeline .
docker run -p 5000:5000 llm-inference-pipeline
```

---

## Project Structure

```
llm-inference-pipeline/
├── app/
│   ├── model.py          # BatchProcessor, mixed-precision inference
│   ├── api.py            # Flask REST API
│   ├── prompt.py         # Prompt engineering hooks & templates
│   └── observability.py  # Metrics store (P50/P95/P99)
├── dashboard/
│   └── app.py            # Streamlit live dashboard
├── tests/
│   └── test_api.py       # pytest suite
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Tech Stack

`Python` · `PyTorch` · `HuggingFace Transformers` · `Flask` · `Streamlit` · `Plotly` · `Docker` · `pytest`

## Dahboard output

<img width="1456" height="819" alt="LLM_hero image" src="https://github.com/user-attachments/assets/90c4fcfe-a098-49d3-bd3e-e36fd01229ef" />


