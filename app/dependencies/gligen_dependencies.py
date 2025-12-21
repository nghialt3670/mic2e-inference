from fastapi import Request

from app.services.gligen_service import GligenService
from app.services.impl.gligen_service_impl import GligenServiceImpl
from app.utils.device_utils import get_device
from app.utils.model_manager import get_model_manager
from app.utils.model_loaders import load_gligen_standalone


def get_gligen_generation_models(request: Request):
    """Get GLIGEN generation models, loading on-demand."""
    manager = get_model_manager()
    models = manager.get_model("gligen", request)
    return models[:5]  # gen_model, gen_autoencoder, gen_text_encoder, gen_diffusion, gen_config


def get_gligen_inpainting_models(request: Request):
    """Get GLIGEN inpainting models, loading on-demand."""
    manager = get_model_manager()
    models = manager.get_model("gligen", request)
    return models[5:]  # inp_model, inp_autoencoder, inp_text_encoder, inp_diffusion, inp_config


async def get_gligen_service(request: Request):
    """Get GLIGEN service with automatic cleanup after request."""
    manager = get_model_manager()
    try:
        models = manager.get_model("gligen", request)
        gen_models = models[:5]
        inp_models = models[5:]
        # Get the best available device dynamically
        device = get_device()
        
        service = GligenServiceImpl(
            *gen_models,
            *inp_models,
            device=device,
        )
        yield service
    finally:
        manager.cleanup_model("gligen")
