from diffusers import FluxPipeline
from PIL import Image

from app.services.flux_service import FluxService

class FluxServiceImpl(FluxService):
    def __init__(
        self,
        pipeline: FluxPipeline,
        num_inference_steps: int = 4,  # FLUX.1-schnell optimized for 4 steps
        guidance_scale: float = 0.0,  # schnell doesn't need guidance
        height: int = 512,  # Reduced from default 1024 for speed
        width: int = 512,  # Reduced from default 1024 for speed
    ):
        self._pipeline = pipeline
        self._num_inference_steps = num_inference_steps
        self._guidance_scale = guidance_scale
        self._height = height
        self._width = width

    async def generate(self, prompt: str) -> Image.Image:
        return self._pipeline(
            prompt=prompt,
            num_inference_steps=self._num_inference_steps,
            guidance_scale=self._guidance_scale,
            height=self._height,
            width=self._width,
        ).images[0]