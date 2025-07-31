"""Metrics collection for observability."""

import time
from typing import Dict, Any, Optional
from collections import defaultdict
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)

class MetricsCollector:
    """Collects and tracks metrics for monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled = config.get('metrics_enabled', True)
        
        # Metrics storage
        self.counters = defaultdict(int)
        self.histograms = defaultdict(list)
        self.gauges = {}
        self.timers = {}
        
        # Start time for uptime calculation
        self.start_time = time.time()
        
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        if not self.enabled:
            return
            
        key = self._make_key(name, labels)
        self.counters[key] += value
        
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a value in a histogram."""
        if not self.enabled:
            return
            
        key = self._make_key(name, labels)
        self.histograms[key].append(value)
        
        # Keep only last 1000 values to prevent memory issues
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
            
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge value."""
        if not self.enabled:
            return
            
        key = self._make_key(name, labels)
        self.gauges[key] = value
        
    def start_timer(self, name: str) -> str:
        """Start a timer and return timer ID."""
        if not self.enabled:
            return ""
            
        timer_id = f"{name}_{time.time()}"
        self.timers[timer_id] = time.time()
        return timer_id
        
    def stop_timer(self, timer_id: str, labels: Optional[Dict[str, str]] = None):
        """Stop a timer and record the duration."""
        if not self.enabled or timer_id not in self.timers:
            return
            
        duration = time.time() - self.timers[timer_id]
        del self.timers[timer_id]
        
        # Extract metric name from timer_id
        name = "_".join(timer_id.split("_")[:-1])
        self.record_histogram(f"{name}_duration_seconds", duration, labels)
        
    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a unique key for a metric."""
        if not labels:
            return name
            
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics in Prometheus-compatible format."""
        metrics = {}
        
        # Add counters
        for key, value in self.counters.items():
            metrics[f"{key}_total"] = value
            
        # Add histograms (simplified - just avg, min, max)
        for key, values in self.histograms.items():
            if values:
                metrics[f"{key}_avg"] = sum(values) / len(values)
                metrics[f"{key}_min"] = min(values)
                metrics[f"{key}_max"] = max(values)
                metrics[f"{key}_count"] = len(values)
                
        # Add gauges
        for key, value in self.gauges.items():
            metrics[key] = value
            
        # Add system metrics
        metrics["uptime_seconds"] = time.time() - self.start_time
        
        return metrics
        
    def record_query_metrics(
        self,
        query_id: str,
        success: bool,
        latency_ms: int,
        tokens_used: int,
        model_name: str
    ):
        """Record metrics for a query."""
        # Record counters
        self.increment_counter("queries", labels={"status": "success" if success else "error"})
        
        if success:
            # Record latency
            self.record_histogram("query_latency_ms", latency_ms, labels={"model": model_name})
            
            # Record tokens
            self.increment_counter("tokens_processed", tokens_used, labels={"model": model_name})
            
            # Update gauges
            self.set_gauge("last_query_latency_ms", latency_ms)
            self.set_gauge("last_query_tokens", tokens_used)
            
        # Log metrics
        logger.info(
            f"Query metrics recorded",
            extra={
                "query_id": query_id,
                "success": success,
                "latency_ms": latency_ms,
                "tokens_used": tokens_used,
                "model_name": model_name
            }
        )