"""Stable Diffusion Inpainting service implementation."""

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

from app.services.sd_inpaint_service import SDInpaintService


class SDInpaintServiceImpl(SDInpaintService):
    """Implementation of Stable Diffusion inpainting service."""

    def __init__(
        self,
        pipeline: StableDiffusionInpaintPipeline,
        device: str = "cpu",
    ):
        """Initialize the service.
        
        Args:
            pipeline: Stable Diffusion inpainting pipeline
            device: Device to run inference on
        """
        self._pipeline = pipeline
        self._device = device

    async def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        seed: int = 42,
    ) -> Image.Image:
        """Inpaint an image using a mask.
        
        Args:
            image: Input image to inpaint
            mask: Binary mask (white = inpaint, black = keep)
            prompt: Text prompt describing what to generate in masked area
            negative_prompt: Text prompt describing what to avoid
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducibility
            
        Returns:
            Inpainted image resized to original dimensions
        """
        # Store original size to resize result back
        original_size = image.size
        
        # Set random seed for reproducibility
        generator = torch.Generator(device=self._device).manual_seed(seed)
        
        # Convert images to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        if mask.mode != "RGB":
            mask = mask.convert("RGB")
        
        # Run inpainting
        result = self._pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        # Resize result back to original image size
        result_image = result.images[0]
        return result_image.resize(original_size, resample=Image.BICUBIC)
