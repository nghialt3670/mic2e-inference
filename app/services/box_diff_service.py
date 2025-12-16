from abc import ABC, abstractmethod
from typing import List

from PIL import Image


class BoxDiffService(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        token_indices: List[int],
        bbox: List[List[int]],
        seed: int = 42,
    ) -> Image.Image:
        """
        Generate an image using BoxDiff with spatial control.
        
        Args:
            prompt: The text prompt for image generation
            token_indices: Indices of tokens to spatially control
            bbox: List of bounding boxes [[x1, y1, x2, y2], ...] in pixel coordinates
            seed: Random seed for reproducibility
            
        Returns:
            Generated PIL Image
        """
        pass
