"""Aesthetic Regressor service interface."""

from abc import ABC, abstractmethod
from typing import Dict

from PIL import Image


class AestheticRegressorService(ABC):
    """Abstract service for aesthetic regression."""

    @abstractmethod
    async def score(
        self,
        image: Image.Image,
    ) -> Dict[str, float]:
        """Score an image for aesthetic factors.
        
        Args:
            image: Input image to score
            
        Returns:
            Dictionary with aesthetic factor scores (saturation, brightness, tint, temperature, contrast)
        """
        pass
