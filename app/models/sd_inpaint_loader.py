"""Stable Diffusion Inpainting model loader."""

import logging

import torch
from diffusers import StableDiffusionInpaintPipeline
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def load_sd_inpaint(app: FastAPI, device: str) -> None:
    """Load Stable Diffusion Inpainting pipeline.
    
    Args:
        app: FastAPI application instance
        device: Device to load model on (cpu, cuda, mps)
    """
    logger.info("Loading Stable Diffusion Inpainting pipeline...")
    
    # Use fp16 for CUDA, fp32 for CPU/MPS
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if device == "cuda" else None
    
    # Load the inpainting model
    app.state.sd_inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch_dtype,
        variant=variant,
        low_cpu_mem_usage=True,
    ).to(device)
    
    # Enable memory optimizations
    if device == "cuda":
        app.state.sd_inpaint_pipeline.enable_attention_slicing(slice_size="auto")
        try:
            app.state.sd_inpaint_pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
            app.state.sd_inpaint_pipeline.enable_attention_slicing(1)
    
    logger.info("Stable Diffusion Inpainting pipeline loaded with optimizations")
