import logging
import sys

from fastapi import FastAPI

from app.lifespan import lifespan
from app.routes.health_routes import router as health_router
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


@app.get("/")
async def root():
    return {"message": "MIC2E Inference API", "status": "running"}


app.include_router(health_router)
app.include_router(object_clear_router)
app.include_router(sam3_router)
