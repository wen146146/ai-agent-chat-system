import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger():
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger('agent')
    logger.setLevel(logging.ERROR)

    handler = RotatingFileHandler(
        os.path.join(log_dir, 'agent_error.log'),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )

    formatter = logging.Formatter('[%(asctime)s] [%(name)s] %(message)s | traceback: %(exc_info)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

logger = setup_logger()
