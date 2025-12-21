from fastapi import Request
from diffusers import FluxPipeline

from app.services.flux_service import FluxService
from app.services.impl.flux_service_impl import FluxServiceImpl
from app.utils.model_manager import get_model_manager
from app.utils.model_loaders import load_flux_standalone


def get_flux_pipeline(request: Request) -> FluxPipeline:
    """Get Flux pipeline, loading on-demand."""
    manager = get_model_manager()
    return manager.get_model("flux", request)


async def get_flux_service(request: Request):
    """Get Flux service with automatic cleanup after request."""
    manager = get_model_manager()
    try:
        pipeline = manager.get_model("flux", request)
        service = FluxServiceImpl(pipeline)
        yield service
    finally:
        manager.cleanup_model("flux")