import logging
import sys

from fastapi import FastAPI

from app.lifespan import lifespan
from app.routes.object_clear_routes import router as object_clear_router
from app.routes.sam3_routes import router as sam3_router

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    stream=sys.stdout,
    force=True,
)

app = FastAPI(lifespan=lifespan)
app.include_router(object_clear_router)
app.include_router(sam3_router)
