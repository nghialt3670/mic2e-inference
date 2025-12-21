"""Dependencies for Stable Diffusion Inpainting."""

from diffusers import StableDiffusionInpaintPipeline
from fastapi import Request

from app.services.impl.sd_inpaint_service_impl import SDInpaintServiceImpl
from app.services.sd_inpaint_service import SDInpaintService
from app.utils.device_utils import get_device
from app.utils.model_manager import get_model_manager
from app.utils.model_loaders import load_sd_inpaint_standalone


def get_sd_inpaint_pipeline(request: Request) -> StableDiffusionInpaintPipeline:
    """Get Stable Diffusion inpainting pipeline, loading on-demand.
    
    Args:
        request: FastAPI request
        
    Returns:
        StableDiffusionInpaintPipeline instance
    """
    manager = get_model_manager()
    return manager.get_model("sd_inpaint", request)


async def get_sd_inpaint_service(request: Request):
    """Get Stable Diffusion inpainting service with automatic cleanup after request.
    
    Args:
        request: FastAPI request
        
    Yields:
        SDInpaintService instance
    """
    manager = get_model_manager()
    try:
        pipeline = manager.get_model("sd_inpaint", request)
        # Get the best available device dynamically
        device = get_device()
        service = SDInpaintServiceImpl(pipeline=pipeline, device=device)
        yield service
    finally:
        manager.cleanup_model("sd_inpaint")
