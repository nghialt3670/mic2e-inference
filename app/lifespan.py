"""Application lifespan management."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from huggingface_hub import login

from app.env import (
    HUGGINGFACE_TOKEN,
    LOAD_SAM3,
    LOAD_OBJECT_CLEAR,
    LOAD_BOX_DIFF,
    LOAD_GLIGEN,
    LOAD_FLUX,
)
from app.models import (
    load_sam3,
    load_object_clear,
    load_box_diff,
    load_gligen,
    load_flux,
)
from app.utils.device_utils import get_device

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    logger.info("Starting application...")

    # Login to Hugging Face
    login(token=HUGGINGFACE_TOKEN)
    logger.info("Logged in to Hugging Face")

    # Detect device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Store device for service initialization
    app.state.device = device

    # Load models based on environment configuration
    if LOAD_SAM3:
        try:
            load_sam3(app, device)
        except Exception as e:
            logger.error(f"Failed to load SAM3: {e}", exc_info=True)
    
    if LOAD_OBJECT_CLEAR:
        try:
            load_object_clear(app, device)
        except Exception as e:
            logger.error(f"Failed to load ObjectClear: {e}", exc_info=True)
    
    if LOAD_BOX_DIFF:
        try:
            load_box_diff(app, device)
        except Exception as e:
            logger.error(f"Failed to load BoxDiff: {e}", exc_info=True)
    
    if LOAD_GLIGEN:
        try:
            load_gligen(app, device)
        except Exception as e:
            logger.error(f"Failed to load GLIGEN: {e}", exc_info=True)
    
    if LOAD_FLUX:
        try:
            load_flux(app, device)
        except Exception as e:
            logger.error(f"Failed to load Flux: {e}", exc_info=True)

    logger.info("Application startup complete")
    
    yield
    
    logger.info("Application shutdown")
