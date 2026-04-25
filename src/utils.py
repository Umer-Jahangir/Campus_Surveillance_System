# High precision timing utilities

import time
import statistics


def now():
    return time.perf_counter()


def latency_ms(start, end):
    return (end - start) * 1000.0


def compute_fps(frames, duration):
    if duration <= 0:
        return 0.0
    return frames / duration


# ---------- Advanced Metrics ----------

def percentile(data, p):
    """
    Compute percentile (e.g., p=95 for P95 latency).
    Optimized: use approximate if data < 20 samples.
    """
    if not data or len(data) < 2:
        return 0.0
    
    # For small samples, use simple sorted method (faster than statistics.quantiles)
    if len(data) < 20:
        sorted_data = sorted(data)
        idx = max(0, int(len(sorted_data) * p / 100.0) - 1)
        return float(sorted_data[idx])
    
    return statistics.quantiles(data, n=100)[p - 1]


def summarize_latency(latencies):
    """
    Returns avg, p95, max latency.
    Optimized: early exit for empty/small data; reduced calculations.
    """
    if not latencies:
        return 0.0, 0.0, 0.0
    
    n = len(latencies)
    if n == 1:
        val = latencies[0]
        return val, val, val

    avg = sum(latencies) / n
    p95 = percentile(latencies, 95)
    maximum = max(latencies)

    return avg, p95, maximum