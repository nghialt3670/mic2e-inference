"""ObjectClear model loader."""

import logging

import torch
from fastapi import FastAPI

from app.external.ObjectClear.objectclear.pipelines import ObjectClearPipeline

logger = logging.getLogger(__name__)


def load_object_clear(app: FastAPI, device: str) -> None:
    """Load ObjectClear pipeline.
    
    Args:
        app: FastAPI application instance
        device: Device to load model on (cpu, cuda, mps)
    """
    logger.info("Loading ObjectClear pipeline...")
    
    # Use fp16 for CUDA, fp32 for CPU/MPS
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
