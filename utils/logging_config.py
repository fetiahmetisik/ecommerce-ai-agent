"""
Logging configuration for E-Commerce AI Agent
"""

import logging
import logging.config
import sys
import json
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

from config import settings

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

class RequestFormatter(logging.Formatter):
    """Formatter for HTTP request logs"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'type': 'request',
            'method': getattr(record, 'method', ''),
            'path': getattr(record, 'path', ''),
            'status_code': getattr(record, 'status_code', ''),
            'duration_ms': getattr(record, 'duration_ms', ''),
            'user_id': getattr(record, 'user_id', ''),
            'ip_address': getattr(record, 'ip_address', ''),
            'user_agent': getattr(record, 'user_agent', ''),
            'message': record.getMessage()
        }
        
        return json.dumps(log_entry)

class AgentFormatter(logging.Formatter):
    """Formatter for agent operation logs"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'type': 'agent_operation',
            'agent_type': getattr(record, 'agent_type', ''),
            'operation': getattr(record, 'operation', ''),
            'user_id': getattr(record, 'user_id', ''),
            'duration_ms': getattr(record, 'duration_ms', ''),
            'status': getattr(record, 'status', ''),
            'message': record.getMessage()
        }
        
        if hasattr(record, 'metadata'):
            log_entry['metadata'] = record.metadata
        
        return json.dumps(log_entry)

def setup_logging():
    """Configure logging for the application"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Determine log level
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'default': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'json': {
                '()': JSONFormatter,
            },
            'request': {
                '()': RequestFormatter,
            },
            'agent': {
                '()': AgentFormatter,
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'json' if settings.debug else 'default',
                'stream': sys.stdout
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': log_level,
                'formatter': 'json',
                'filename': log_dir / 'app.log',
                'maxBytes': 10 * 1024 * 1024,  # 10MB
                'backupCount': 5
            },
            'error_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': logging.ERROR,
                'formatter': 'json',
                'filename': log_dir / 'error.log',
                'maxBytes': 10 * 1024 * 1024,  # 10MB
                'backupCount': 5
            },
            'request_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': logging.INFO,
                'formatter': 'request',
                'filename': log_dir / 'requests.log',
                'maxBytes': 50 * 1024 * 1024,  # 50MB
                'backupCount': 10
            },
            'agent_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': logging.INFO,
                'formatter': 'agent',
                'filename': log_dir / 'agents.log',
                'maxBytes': 25 * 1024 * 1024,  # 25MB
                'backupCount': 5
            }
        },
        'loggers': {
            '': {  # Root logger
                'level': log_level,
                'handlers': ['console', 'file', 'error_file'],
                'propagate': False
            },
            'requests': {
                'level': logging.INFO,
                'handlers': ['request_file'],
                'propagate': False
            },
            'agents': {
                'level': logging.INFO,
                'handlers': ['agent_file'],
                'propagate': False
            },
            'uvicorn': {
                'level': logging.INFO,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'uvicorn.access': {
                'level': logging.INFO,
                'handlers': ['request_file'],
                'propagate': False
            },
            'fastapi': {
                'level': logging.INFO,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'google.cloud': {
                'level': logging.WARNING,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'vertexai': {
                'level': logging.INFO,
                'handlers': ['console', 'file'],
                'propagate': False
            },
            'crewai': {
                'level': logging.INFO,
                'handlers': ['agent_file'],
                'propagate': False
            }
        }
    }
    
    # Apply logging configuration
    logging.config.dictConfig(config)
    
    # Set up request logger
    setup_request_logger()
    
    # Set up agent logger
    setup_agent_logger()
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {settings.log_level}, Debug: {settings.debug}")

def setup_request_logger():
    """Set up specialized request logger"""
    request_logger = logging.getLogger('requests')
    
    def log_request(method: str, path: str, status_code: int, duration_ms: float, 
                   user_id: str = "", ip_address: str = "", user_agent: str = ""):
        """Log HTTP request"""
        request_logger.info(
            f"{method} {path} {status_code} {duration_ms}ms",
            extra={
                'method': method,
                'path': path,
                'status_code': status_code,
                'duration_ms': duration_ms,
                'user_id': user_id,
                'ip_address': ip_address,
                'user_agent': user_agent
            }
        )
    
    # Add method to logger
    request_logger.log_request = log_request
    return request_logger

def setup_agent_logger():
    """Set up specialized agent operation logger"""
    agent_logger = logging.getLogger('agents')
    
    def log_operation(agent_type: str, operation: str, user_id: str = "", 
                     duration_ms: float = 0, status: str = "success", 
                     metadata: Dict[str, Any] = None):
        """Log agent operation"""
        message = f"{agent_type}.{operation} - {status}"
        if duration_ms:
            message += f" ({duration_ms}ms)"
        
        agent_logger.info(
            message,
            extra={
                'agent_type': agent_type,
                'operation': operation,
                'user_id': user_id,
                'duration_ms': duration_ms,
                'status': status,
                'metadata': metadata or {}
            }
        )
    
    # Add method to logger
    agent_logger.log_operation = log_operation
    return agent_logger

def get_request_logger():
    """Get the request logger"""
    return logging.getLogger('requests')

def get_agent_logger():
    """Get the agent logger"""
    return logging.getLogger('agents')

class LogContext:
    """Context manager for adding context to logs"""
    
    def __init__(self, **context):
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        self.old_factory = old_factory
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.old_factory:
            logging.setLogRecordFactory(self.old_factory)

def log_with_context(**context):
    """Decorator to add context to all logs in a function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with LogContext(**context):
                return func(*args, **kwargs)
        return wrapper
    return decorator

class StructuredLogger:
    """Wrapper for structured logging"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, message: str, **kwargs):
        """Log info with structured data"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning with structured data"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error with structured data"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical with structured data"""
        self.logger.critical(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug with structured data"""
        self.logger.debug(message, extra=kwargs)

# Convenience functions
def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger"""
    return StructuredLogger(name)

def log_exception(logger: logging.Logger, message: str, **kwargs):
    """Log exception with context"""
    logger.exception(message, extra=kwargs)

def log_performance(logger: logging.Logger, operation: str, duration: float, **kwargs):
    """Log performance metrics"""
    logger.info(
        f"Performance: {operation} completed in {duration:.3f}s",
        extra={
            'operation': operation,
            'duration_seconds': duration,
            **kwargs
        }
    )

# Security and sensitive data handling
class SensitiveDataFilter(logging.Filter):
    """Filter to remove sensitive data from logs"""
    
    SENSITIVE_KEYS = {
        'password', 'secret', 'key', 'token', 'credential', 
        'authorization', 'cookie', 'session'
    }
    
    def filter(self, record):
        # Filter message
        if hasattr(record, 'msg'):
            record.msg = self._sanitize_text(str(record.msg))
        
        # Filter args
        if hasattr(record, 'args') and record.args:
            record.args = tuple(
                self._sanitize_text(str(arg)) if isinstance(arg, str) else arg 
                for arg in record.args
            )
        
        # Filter extra fields
        for key, value in record.__dict__.items():
            if isinstance(value, str) and any(sensitive in key.lower() for sensitive in self.SENSITIVE_KEYS):
                setattr(record, key, '[REDACTED]')
        
        return True
    
    def _sanitize_text(self, text: str) -> str:
        """Remove sensitive patterns from text"""
        import re
        
        # Patterns to redact
        patterns = [
            r'(password|secret|key|token)=[\w\-\.\+/]+'
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, r'\1=[REDACTED]', text, flags=re.IGNORECASE)
        
        return text

# Add sensitive data filter to all handlers
def add_security_filters():
    """Add security filters to all loggers"""
    sensitive_filter = SensitiveDataFilter()
    
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        handler.addFilter(sensitive_filter)