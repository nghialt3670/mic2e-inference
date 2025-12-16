"""Flux model loader."""

import logging

import torch
from diffusers import FluxPipeline
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def load_flux(app: FastAPI, device: str) -> None:
    """Load Flux pipeline.
    
    Args:
        app: FastAPI application instance
        device: Device to load model on (cpu, cuda, mps)
    """
    logger.info("Loading Flux pipeline...")
    
    # Optimized Flux loading with memory efficient attention
    app.state.flux_pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    ).to(device)
    
    if device == "cuda":
        app.state.flux_pipeline.enable_attention_slicing(slice_size="auto")
        try:
            app.state.flux_pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers for Flux")
        except Exception as e:
            logger.warning(f"Could not enable xformers for Flux: {e}")
    
    logger.info("Flux pipeline loaded with optimizations")
