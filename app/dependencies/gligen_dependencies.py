import sys
from pathlib import Path

from fastapi import Request

# Add GLIGEN to Python path
gligen_path = Path(__file__).parent.parent / "external" / "GLIGEN"
if str(gligen_path) not in sys.path:
    sys.path.insert(0, str(gligen_path))

from app.services.gligen_service import GligenService
from app.services.impl.gligen_service_impl import GligenServiceImpl


def get_gligen_generation_models(request: Request):
    return (
        request.app.state.gligen_generation_model,
        request.app.state.gligen_generation_autoencoder,
        request.app.state.gligen_generation_text_encoder,
        request.app.state.gligen_generation_diffusion,
        request.app.state.gligen_generation_config,
    )


def get_gligen_inpainting_models(request: Request):
    return (
        request.app.state.gligen_inpainting_model,
        request.app.state.gligen_inpainting_autoencoder,
        request.app.state.gligen_inpainting_text_encoder,
        request.app.state.gligen_inpainting_diffusion,
        request.app.state.gligen_inpainting_config,
    )


def get_gligen_service(request: Request) -> GligenService:
    gen_models = get_gligen_generation_models(request)
    inp_models = get_gligen_inpainting_models(request)
    device = request.app.state.device
    
    return GligenServiceImpl(
        *gen_models,
        *inp_models,
        device=device,
    )
