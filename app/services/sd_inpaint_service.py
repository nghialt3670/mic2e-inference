"""Stable Diffusion Inpainting service interface."""

from abc import ABC, abstractmethod

from PIL import Image


class SDInpaintService(ABC):
    """Abstract service for Stable Diffusion inpainting."""

    @abstractmethod
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
            Inpainted image
        """
        pass
