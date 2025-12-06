import logging
import sys

from fastapi import FastAPI

from app.lifespan import lifespan
from app.routes.object_clear_routes import router as object_clear_router
from app.routes.sam3_routes import router as sam3_router

# Configure root logger
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    stream=sys.stdout,
    force=True,  # Override any existing configuration
)


class CleanLoggerNameFormatter(logging.Formatter):
    """Formatter that maps uvicorn.error to 'uvicorn' for cleaner logs"""

    def format(self, record: logging.LogRecord) -> str:
        # Map logger names for cleaner display
        original_name = record.name
        if record.name == "uvicorn.error":
            # Temporarily change name for display
            record.name = "uvicorn"
            result = super().format(record)
            record.name = original_name  # Restore original
            return result
        return super().format(record)


# Configure uvicorn loggers to use the same format
uvicorn_loggers = [
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
]

formatter = CleanLoggerNameFormatter(log_format)

for logger_name in uvicorn_loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan)
app.include_router(object_clear_router)
app.include_router(sam3_router)
