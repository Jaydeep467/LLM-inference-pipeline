"""
Observability & Metrics
Tracks latency (P50/P99), throughput, batch sizes, and error rates.
Thread-safe rolling window — no external dependency needed.
"""

import time
import threading
from collections import deque
from typing import Dict, Any

# Rolling window size (number of requests to keep in memory)
WINDOW = 500


class MetricsStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._latencies: deque[float] = deque(maxlen=WINDOW)   # ms
        self._batch_sizes: deque[int] = deque(maxlen=WINDOW)
        self._total_requests = 0
        self._total_errors   = 0
        self._start_time     = time.time()

    # ── write ────────────────────────────────────
    def record(self, batch_size: int, latency_ms: float):
        with self._lock:
            self._latencies.append(latency_ms)
            self._batch_sizes.append(batch_size)
            self._total_requests += batch_size

    def record_error(self):
        with self._lock:
            self._total_errors += 1

    # ── read ─────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            lats = sorted(self._latencies)
            n    = len(lats)
            elapsed = max(time.time() - self._start_time, 1)

            def pct(p: float) -> float:
                if not lats:
                    return 0.0
                idx = max(0, int(p / 100 * n) - 1)
                return round(lats[idx], 2)

            avg_batch = (
                round(sum(self._batch_sizes) / len(self._batch_sizes), 2)
                if self._batch_sizes else 0
            )

            return {
                "total_requests":    self._total_requests,
                "total_errors":      self._total_errors,
                "requests_per_sec":  round(self._total_requests / elapsed, 2),
                "avg_batch_size":    avg_batch,
                "latency_p50_ms":    pct(50),
                "latency_p95_ms":    pct(95),
                "latency_p99_ms":    pct(99),
                "latency_avg_ms":    round(sum(lats) / n, 2) if lats else 0,
                "latency_min_ms":    round(lats[0],  2) if lats else 0,
                "latency_max_ms":    round(lats[-1], 2) if lats else 0,
                "window_samples":    n,
                "uptime_seconds":    round(elapsed, 1),
            }


# Module-level singleton
metrics = MetricsStore()
