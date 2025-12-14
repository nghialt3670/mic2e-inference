import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from huggingface_hub import login
from diffusers import FluxPipeline
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
    # Use fp16 for speed/memory optimization
    sam3_dtype = torch.float16 if device == "cuda" else torch.float32
    app.state.sam3_model = Sam3Model.from_pretrained(
        "facebook/sam3",
        torch_dtype=sam3_dtype,
        low_cpu_mem_usage=True
    ).to(device)
    logger.info("SAM3 model loaded")

    logger.info("Loading SAM3 processor...")
    app.state.sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
    logger.info("SAM3 processor loaded")

    logger.info("Loading SAM3 tracker model...")
    app.state.sam3_tracker_model = Sam3TrackerModel.from_pretrained(
        "facebook/sam3",
        torch_dtype=sam3_dtype,
        low_cpu_mem_usage=True
    ).to(device)
    logger.info("SAM3 tracker model loaded")

    logger.info("Loading SAM3 tracker processor...")
    app.state.sam3_tracker_processor = Sam3TrackerProcessor.from_pretrained(
        "facebook/sam3"
    )
    logger.info("SAM3 tracker processor loaded")

    logger.info("Loading ObjectClear pipeline...")
    # Aggressive optimizations: fp16, attention slicing, memory efficient attention
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if device == "cuda" else None
    app.state.object_clear_pipeline = (
        ObjectClearPipeline.from_pretrained_with_custom_modules(
            "jixin0101/ObjectClear",
            torch_dtype=torch_dtype,
            apply_attention_guided_fusion=True,
            variant=variant,
            low_cpu_mem_usage=True,
        )
    )
    app.state.object_clear_pipeline.to(device)
    
    # Enable memory optimizations
    if device == "cuda":
        app.state.object_clear_pipeline.enable_attention_slicing(slice_size="auto")
        try:
            app.state.object_clear_pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
            app.state.object_clear_pipeline.enable_attention_slicing(1)
    
    logger.info("ObjectClear pipeline loaded with optimizations")

    # logger.info("Loading Flux pipeline...")
    # # Optimized Flux loading with memory efficient attention
    # app.state.flux_pipeline = FluxPipeline.from_pretrained(
    #     "black-forest-labs/FLUX.1-schnell",
    #     torch_dtype=torch.float16,
    #     use_safetensors=True,
    #     low_cpu_mem_usage=True,
    # ).to(device)
    # 
    # if device == "cuda":
    #     app.state.flux_pipeline.enable_attention_slicing(slice_size="auto")
    #     try:
    #         app.state.flux_pipeline.enable_xformers_memory_efficient_attention()
    #         logger.info("Enabled xformers for Flux")
    #     except Exception as e:
    #         logger.warning(f"Could not enable xformers for Flux: {e}")
    # 
    # logger.info("Flux pipeline loaded with optimizations")

    yield
