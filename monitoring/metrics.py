"""
Prometheus Metrics Integration for XAI API
Tracks prediction latency, explanation time, and model performance
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# Define metrics
prediction_counter = Counter(
    "xai_predictions_total",
    "Total number of predictions made",
    ["model_name", "status"],
)

prediction_latency = Histogram(
    "xai_prediction_latency_seconds",
    "Prediction latency in seconds",
    ["model_name"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

explanation_counter = Counter(
    "xai_explanations_total",
    "Total number of explanations generated",
    ["model_name", "method", "status"],
)

explanation_latency = Histogram(
    "xai_explanation_latency_seconds",
    "Explanation generation latency in seconds",
    ["model_name", "method"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

models_loaded = Gauge("xai_models_loaded", "Number of models currently loaded")

cache_hits = Counter("xai_cache_hits_total", "Number of cache hits", ["cache_type"])

cache_misses = Counter(
    "xai_cache_misses_total", "Number of cache misses", ["cache_type"]
)

errors_counter = Counter(
    "xai_errors_total", "Total number of errors", ["error_type", "endpoint"]
)


def track_prediction_metrics(model_name: str, success: bool, latency: float):
    """Track prediction metrics"""
    status = "success" if success else "failure"
    prediction_counter.labels(model_name=model_name, status=status).inc()
    if success:
        prediction_latency.labels(model_name=model_name).observe(latency)


def track_explanation_metrics(
    model_name: str, method: str, success: bool, latency: float
):
    """Track explanation metrics"""
    status = "success" if success else "failure"
    explanation_counter.labels(
        model_name=model_name, method=method, status=status
    ).inc()
    if success:
        explanation_latency.labels(model_name=model_name, method=method).observe(
            latency
        )


def track_error(error_type: str, endpoint: str):
    """Track error occurrences"""
    errors_counter.labels(error_type=error_type, endpoint=endpoint).inc()


def update_models_loaded_count(count: int):
    """Update number of loaded models"""
    models_loaded.set(count)


def track_cache_access(cache_type: str, hit: bool):
    """Track cache hits and misses"""
    if hit:
        cache_hits.labels(cache_type=cache_type).inc()
    else:
        cache_misses.labels(cache_type=cache_type).inc()


def get_metrics() -> Response:
    """Generate Prometheus metrics"""
    return Response(content=generate_latest(), media_type="text/plain")
