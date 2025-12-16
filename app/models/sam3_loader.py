"""SAM3 model loader."""

import logging

import torch
from fastapi import FastAPI
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)

logger = logging.getLogger(__name__)


def load_sam3(app: FastAPI, device: str) -> None:
    """Load SAM3 models and processors.
    
    Args:
        app: FastAPI application instance
        device: Device to load models on (cpu, cuda, mps)
    """
    logger.info("Loading SAM3 model...")
    
    # Use fp16 for CUDA, fp32 for CPU/MPS
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
