"""
Monitoring and observability utilities for E-Commerce AI Agent
"""

import logging
import time
import asyncio
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from functools import wraps
from contextlib import asynccontextmanager

import psutil
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

from config import settings

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'ecommerce_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status_code']
)

REQUEST_DURATION = Histogram(
    'ecommerce_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

AGENT_OPERATION_COUNT = Counter(
    'ecommerce_agent_operations_total',
    'Total number of agent operations',
    ['agent_type', 'operation', 'status']
)

AGENT_OPERATION_DURATION = Histogram(
    'ecommerce_agent_operation_duration_seconds',
    'Agent operation duration in seconds',
    ['agent_type', 'operation']
)

ACTIVE_USERS = Gauge(
    'ecommerce_active_users',
    'Number of active users'
)

SYSTEM_CPU_USAGE = Gauge(
    'ecommerce_system_cpu_usage_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_USAGE = Gauge(
    'ecommerce_system_memory_usage_percent',
    'System memory usage percentage'
)

FIRESTORE_OPERATIONS = Counter(
    'ecommerce_firestore_operations_total',
    'Total Firestore operations',
    ['operation_type', 'collection']
)

VERTEX_AI_REQUESTS = Counter(
    'ecommerce_vertex_ai_requests_total',
    'Total Vertex AI requests',
    ['model', 'status']
)

class MetricsCollector:
    """Prometheus metrics collector"""
    
    def __init__(self):
        self.active_users_set = set()
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_agent_operation(self, agent_type: str, operation: str, status: str, duration: float):
        """Record agent operation metrics"""
        AGENT_OPERATION_COUNT.labels(agent_type=agent_type, operation=operation, status=status).inc()
        AGENT_OPERATION_DURATION.labels(agent_type=agent_type, operation=operation).observe(duration)
    
    def record_active_user(self, user_id: str):
        """Record active user"""
        self.active_users_set.add(user_id)
        ACTIVE_USERS.set(len(self.active_users_set))
    
    def record_firestore_operation(self, operation_type: str, collection: str):
        """Record Firestore operation"""
        FIRESTORE_OPERATIONS.labels(operation_type=operation_type, collection=collection).inc()
    
    def record_vertex_ai_request(self, model: str, status: str):
        """Record Vertex AI request"""
        VERTEX_AI_REQUESTS.labels(model=model, status=status).inc()
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            SYSTEM_CPU_USAGE.set(cpu_percent)
            SYSTEM_MEMORY_USAGE.set(memory_percent)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

# Global metrics collector instance
metrics_collector = MetricsCollector()

def init_monitoring():
    """Initialize monitoring and observability"""
    if not settings.enable_monitoring:
        logger.info("Monitoring disabled")
        return
    
    # Initialize OpenTelemetry tracing
    trace.set_tracer_provider(TracerProvider())
    tracer_provider = trace.get_tracer_provider()
    
    # Configure OTLP span exporter
    otlp_exporter = OTLPSpanExporter(endpoint=settings.otel_endpoint)
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)
    
    # Initialize OpenTelemetry metrics
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=settings.otel_endpoint),
        export_interval_millis=5000
    )
    metrics.set_meter_provider(MeterProvider(metric_readers=[metric_reader]))
    
    # Instrument FastAPI and requests
    FastAPIInstrumentor.instrument()
    RequestsInstrumentor.instrument()
    
    # Start Prometheus metrics server
    try:
        start_http_server(settings.prometheus_port)
        logger.info(f"Prometheus metrics server started on port {settings.prometheus_port}")
    except Exception as e:
        logger.error(f"Failed to start Prometheus server: {e}")
    
    # Start system metrics collection
    asyncio.create_task(collect_system_metrics())
    
    logger.info("Monitoring initialized successfully")

async def collect_system_metrics():
    """Collect system metrics periodically"""
    while True:
        try:
            metrics_collector.update_system_metrics()
            await asyncio.sleep(30)  # Collect every 30 seconds
        except Exception as e:
            logger.error(f"Error in system metrics collection: {e}")
            await asyncio.sleep(60)  # Wait longer on error

def monitor_agent_operation(agent_type: str, operation: str):
    """Decorator to monitor agent operations"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            # Create OpenTelemetry span
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"{agent_type}.{operation}") as span:
                span.set_attribute("agent.type", agent_type)
                span.set_attribute("agent.operation", operation)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("agent.status", "success")
                    return result
                    
                except Exception as e:
                    status = "error"
                    span.set_attribute("agent.status", "error")
                    span.set_attribute("agent.error", str(e))
                    logger.error(f"Agent operation failed: {agent_type}.{operation} - {str(e)}")
                    raise
                    
                finally:
                    duration = time.time() - start_time
                    metrics_collector.record_agent_operation(agent_type, operation, status, duration)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = "success"
            
            # Create OpenTelemetry span
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"{agent_type}.{operation}") as span:
                span.set_attribute("agent.type", agent_type)
                span.set_attribute("agent.operation", operation)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("agent.status", "success")
                    return result
                    
                except Exception as e:
                    status = "error"
                    span.set_attribute("agent.status", "error")
                    span.set_attribute("agent.error", str(e))
                    logger.error(f"Agent operation failed: {agent_type}.{operation} - {str(e)}")
                    raise
                    
                finally:
                    duration = time.time() - start_time
                    metrics_collector.record_agent_operation(agent_type, operation, status, duration)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

@asynccontextmanager
async def trace_operation(operation_name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager for tracing operations"""
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span(operation_name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        start_time = time.time()
        try:
            yield span
        except Exception as e:
            span.set_attribute("error", str(e))
            span.set_attribute("error.type", type(e).__name__)
            raise
        finally:
            duration = time.time() - start_time
            span.set_attribute("operation.duration", duration)

class PerformanceMonitor:
    """Monitor performance metrics for operations"""
    
    def __init__(self):
        self.operation_times = {}
        self.operation_counts = {}
    
    def start_operation(self, operation_id: str, operation_type: str):
        """Start timing an operation"""
        self.operation_times[operation_id] = {
            'start_time': time.time(),
            'type': operation_type
        }
    
    def end_operation(self, operation_id: str, success: bool = True):
        """End timing an operation"""
        if operation_id not in self.operation_times:
            logger.warning(f"Operation {operation_id} not found in performance monitor")
            return
        
        operation = self.operation_times[operation_id]
        duration = time.time() - operation['start_time']
        operation_type = operation['type']
        
        # Update counts
        if operation_type not in self.operation_counts:
            self.operation_counts[operation_type] = {'total': 0, 'success': 0, 'failed': 0}
        
        self.operation_counts[operation_type]['total'] += 1
        if success:
            self.operation_counts[operation_type]['success'] += 1
        else:
            self.operation_counts[operation_type]['failed'] += 1
        
        # Log performance
        logger.info(f"Operation {operation_type} completed in {duration:.3f}s (success: {success})")
        
        # Clean up
        del self.operation_times[operation_id]
        
        return duration
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        for operation_type, counts in self.operation_counts.items():
            total = counts['total']
            if total > 0:
                stats[operation_type] = {
                    'total_operations': total,
                    'success_rate': counts['success'] / total * 100,
                    'failure_rate': counts['failed'] / total * 100
                }
        return stats

# Global performance monitor
performance_monitor = PerformanceMonitor()

class HealthChecker:
    """Health check utilities"""
    
    def __init__(self):
        self.checks = {}
    
    def register_check(self, name: str, check_func: Callable, timeout: int = 30):
        """Register a health check function"""
        self.checks[name] = {
            'func': check_func,
            'timeout': timeout
        }
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'overall_status': 'healthy'
        }
        
        for name, check_config in self.checks.items():
            try:
                # Run check with timeout
                result = await asyncio.wait_for(
                    check_config['func'](),
                    timeout=check_config['timeout']
                )
                
                results['checks'][name] = {
                    'status': 'healthy',
                    'result': result
                }
                
            except asyncio.TimeoutError:
                results['checks'][name] = {
                    'status': 'timeout',
                    'error': f"Check timed out after {check_config['timeout']}s"
                }
                results['overall_status'] = 'degraded'
                
            except Exception as e:
                results['checks'][name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                results['overall_status'] = 'unhealthy'
        
        return results

# Global health checker
health_checker = HealthChecker()

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self):
        self.alert_handlers = []
        self.alert_history = []
    
    def add_handler(self, handler: Callable):
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    async def send_alert(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Send alert through all handlers"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'details': details or {}
        }
        
        self.alert_history.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Send through handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def get_recent_alerts(self, limit: int = 100) -> list:
        """Get recent alerts"""
        return self.alert_history[-limit:]

# Global alert manager
alert_manager = AlertManager()

# Default alert handlers
async def log_alert_handler(alert: Dict[str, Any]):
    """Log alert to application logs"""
    level = alert['level'].upper()
    message = alert['message']
    details = alert['details']
    
    log_message = f"ALERT [{level}]: {message}"
    if details:
        log_message += f" - Details: {details}"
    
    if level == "CRITICAL":
        logger.critical(log_message)
    elif level == "ERROR":
        logger.error(log_message)
    elif level == "WARNING":
        logger.warning(log_message)
    else:
        logger.info(log_message)

# Register default alert handler
alert_manager.add_handler(log_alert_handler)

# Health check functions
async def firestore_health_check():
    """Check Firestore connectivity"""
    try:
        from google.cloud import firestore
        client = firestore.Client(project=settings.google_cloud_project)
        # Try to read from a collection
        collection_ref = client.collection('health_check')
        # This will test connectivity without creating documents
        list(collection_ref.limit(1).stream())
        return {"status": "connected"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def vertex_ai_health_check():
    """Check Vertex AI connectivity"""
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        
        vertexai.init(project=settings.google_cloud_project, location=settings.location)
        model = GenerativeModel(settings.vertex_ai_model)
        
        # Simple test request
        response = model.generate_content(["Hello"])
        return {"status": "connected", "response_length": len(response.text)}
    except Exception as e:
        return {"status": "error", "error": str(e)}

async def redis_health_check():
    """Check Redis connectivity"""
    try:
        import redis
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password
        )
        client.ping()
        return {"status": "connected"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# Register health checks
health_checker.register_check('firestore', firestore_health_check)
health_checker.register_check('vertex_ai', vertex_ai_health_check)
health_checker.register_check('redis', redis_health_check)