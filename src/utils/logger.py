"""Logging configuration for SignalCLI."""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler

from .config import get_config_value

class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'query_id'):
            log_data['query_id'] = record.query_id
        if hasattr(record, 'latency_ms'):
            log_data['latency_ms'] = record.latency_ms
        if hasattr(record, 'tokens_used'):
            log_data['tokens_used'] = record.tokens_used
        if hasattr(record, 'success'):
            log_data['success'] = record.success
            
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def setup_logging() -> None:
    """Configure logging based on settings."""
    # Get logging config
    log_level = get_config_value('logging.level', 'INFO')
    log_format = get_config_value('logging.format', 'json')
    log_file = get_config_value('logging.file', 'logs/signalcli.log')
    max_size = get_config_value('logging.max_size', '100MB')
    backup_count = get_config_value('logging.backup_count', 5)
    
    # Convert max_size to bytes
    size_units = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    for unit, multiplier in size_units.items():
        if max_size.endswith(unit):
            max_bytes = int(max_size[:-2]) * multiplier
            break
    else:
        max_bytes = 100 * 1024 * 1024  # Default 100MB
    
    # Create log directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Set formatters
    if log_format == 'json':
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Adjust third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    # Ensure logging is set up
    if not logging.getLogger().handlers:
        setup_logging()
    
    return logging.getLogger(name)

class LogContext:
    """Context manager for adding extra fields to logs."""
    
    def __init__(self, logger: logging.Logger, **kwargs):
        self.logger = logger
        self.extras = kwargs
        self.old_extras = {}
        
    def __enter__(self):
        # Save old extras
        for key, value in self.extras.items():
            if hasattr(self.logger, key):
                self.old_extras[key] = getattr(self.logger, key)
            setattr(self.logger, key, value)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old extras
        for key in self.extras:
            if key in self.old_extras:
                setattr(self.logger, key, self.old_extras[key])
            else:
                delattr(self.logger, key)