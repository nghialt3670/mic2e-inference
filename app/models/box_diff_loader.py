"""BoxDiff model loader."""

import logging
import sys
from pathlib import Path

import torch
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def load_box_diff(app: FastAPI, device: str) -> None:
    """Load BoxDiff pipeline.
    
    Args:
        app: FastAPI application instance
        device: Device to load model on (cpu, cuda, mps)
    """
    # Add BoxDiff to Python path
    boxdiff_path = Path(__file__).parent.parent / "external" / "BoxDiff"
    if str(boxdiff_path) not in sys.path:
        sys.path.insert(0, str(boxdiff_path))
    
    from pipeline.sd_pipeline_boxdiff import BoxDiffPipeline
    
    logger.info("Loading BoxDiff pipeline...")
    
    # Use fp16 for CUDA, fp32 for CPU/MPS
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load BoxDiff with Stable Diffusion v1.4
    app.state.box_diff_pipeline = BoxDiffPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    
    # Enable memory optimizations for BoxDiff
    if device == "cuda":
        app.state.box_diff_pipeline.enable_attention_slicing(slice_size="auto")
        try:
            app.state.box_diff_pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers for BoxDiff")
        except Exception as e:
            logger.warning(f"Could not enable xformers for BoxDiff: {e}")
            app.state.box_diff_pipeline.enable_attention_slicing(1)
    
    logger.info("BoxDiff pipeline loaded with optimizations")
