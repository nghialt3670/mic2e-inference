import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from huggingface_hub import login
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)

from app.env import HUGGINGFACE_TOKEN
from app.external.ObjectClear.objectclear.pipelines import ObjectClearPipeline
from app.utils.device_utils import get_device

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")

    login(token=HUGGINGFACE_TOKEN)
    logger.info("Logged in to Hugging Face")

    device = get_device()
    logger.info(f"Using device: {device}")

    logger.info("Loading SAM3 model...")
    app.state.sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    logger.info("SAM3 model loaded")

    logger.info("Loading SAM3 processor...")
    app.state.sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
    logger.info("SAM3 processor loaded")

    logger.info("Loading SAM3 tracker model...")
    app.state.sam3_tracker_model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(
        device
    )
    logger.info("SAM3 tracker model loaded")

    logger.info("Loading SAM3 tracker processor...")
    app.state.sam3_tracker_processor = Sam3TrackerProcessor.from_pretrained(
        "facebook/sam3"
    )
    logger.info("SAM3 tracker processor loaded")

    logger.info("Loading ObjectClear pipeline...")
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if device == "cuda" else None
    app.state.object_clear_pipeline = (
        ObjectClearPipeline.from_pretrained_with_custom_modules(
            "jixin0101/ObjectClear",
            torch_dtype=torch_dtype,
            apply_attention_guided_fusion=True,
            variant=variant,
        )
    )
    app.state.object_clear_pipeline.to(device)
    logger.info("ObjectClear pipeline loaded")

    yield
