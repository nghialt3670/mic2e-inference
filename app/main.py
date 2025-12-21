import logging
import sys

from fastapi import FastAPI

from app.lifespan import lifespan
from app.routes.box_diff_routes import router as box_diff_router
from app.routes.flux_routes import router as flux_router
from app.routes.gligen_routes import router as gligen_router
from app.routes.health_routes import router as health_router
from app.routes.object_clear_routes import router as object_clear_router
from app.routes.sam3_routes import router as sam3_router
from app.routes.sd_inpaint_routes import router as sd_inpaint_router

log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    stream=sys.stdout,
    force=True,
)

app = FastAPI(lifespan=lifespan)

app.include_router(box_diff_router)
app.include_router(flux_router)
app.include_router(gligen_router)
app.include_router(health_router)
app.include_router(object_clear_router)
app.include_router(sam3_router)
app.include_router(sd_inpaint_router)