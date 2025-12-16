from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image


class GligenService(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        phrases: List[str],
        locations: List[List[float]],
        seed: int = 42,
        alpha_type: Optional[List[float]] = None,
    ) -> Image.Image:
        """
        Generate an image using GLIGEN with text-box grounding.
        
        Args:
            prompt: The text prompt for image generation
            phrases: List of text phrases to ground (e.g., ['a cat', 'a dog'])
            locations: List of bounding boxes in normalized coordinates [[x1,y1,x2,y2], ...]
                      where coordinates are in range [0, 1]
            seed: Random seed for reproducibility
            alpha_type: Alpha schedule [stage0, stage1, stage2] that sum to 1.0
                       Controls grounding strength over timesteps (default: [0.3, 0.0, 0.7])
            
        Returns:
            Generated PIL Image
        """
        pass

    @abstractmethod
    async def inpaint(
        self,
        input_image: Image.Image,
        prompt: str,
        phrases: List[str],
        locations: List[List[float]],
        seed: int = 42,
    ) -> Image.Image:
        """
        Inpaint an image using GLIGEN with text-box grounding.
        
        Args:
            input_image: Input image to inpaint
            prompt: The text prompt for inpainting
            phrases: List of text phrases to ground in the masked regions
            locations: List of bounding boxes (also used as inpainting masks) [[x1,y1,x2,y2], ...]
            seed: Random seed for reproducibility
            
        Returns:
            Inpainted PIL Image
        """
        pass
