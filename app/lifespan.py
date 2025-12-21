"""Application lifespan management."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from huggingface_hub import login

from app.env import HUGGINGFACE_TOKEN
from app.utils.device_utils import get_device
from app.utils.model_loaders import (
    load_box_diff_standalone,
    load_flux_standalone,
    load_gligen_standalone,
    load_object_clear_standalone,
    load_sam3_standalone,
    load_sd_inpaint_standalone,
)
from app.utils.model_manager import get_model_manager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    logger.info("Starting application...")

    # Login to Hugging Face
    login(token=HUGGINGFACE_TOKEN)
    logger.info("Logged in to Hugging Face")

    # Detect device (models will get best device dynamically at load time)
    device = get_device()
    logger.info(f"Initial device detection: {device} (models will select best device at load time)")

    # Register model loaders for on-demand loading
    logger.info("Registering model loaders for on-demand loading...")
    manager = get_model_manager()
    
    # Register all model loaders
    manager.register_loader("sam3", load_sam3_standalone)
    manager.register_loader("object_clear", load_object_clear_standalone)
    manager.register_loader("box_diff", load_box_diff_standalone)
    manager.register_loader("gligen", load_gligen_standalone)
    manager.register_loader("sd_inpaint", load_sd_inpaint_standalone)
    manager.register_loader("flux", load_flux_standalone)
    
    logger.info("Model loaders registered. Models will be loaded on-demand.")

    logger.info("Application startup complete")

    yield
    
    # Cleanup all loaded models on shutdown
    logger.info("Cleaning up loaded models...")
    manager.cleanup_all()
    
    logger.info("Application shutdown")
