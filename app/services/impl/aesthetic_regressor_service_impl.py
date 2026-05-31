"""Aesthetic Regressor service implementation."""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add AestheticEnhancer to Python path
aesthetic_path = Path(__file__).parent.parent.parent / "external" / "AestheticEnhancer"
if str(aesthetic_path) not in sys.path:
    sys.path.insert(0, str(aesthetic_path))

from model.aesthetic_regressor import AestheticRegressor

from app.services.aesthetic_regressor_service import AestheticRegressorService


class AestheticRegressorServiceImpl(AestheticRegressorService):
    """Implementation of Aesthetic Regressor service."""

    def __init__(
        self,
        model: AestheticRegressor,
        device: str = "cpu",
    ):
        """Initialize the service.
        
        Args:
            model: Aesthetic Regressor model
            device: Device to run inference on
        """
        self._model = model
        self._device = device
        self._model.eval()
        
        # Factors and their normalization coefficients (from test notebook)
        self._factors = ['saturation', 'brightness', 'tint', 'temperature', 'contrast']
        self._factors_coefs = np.array([43, 43, 30, 30, 43], dtype=np.float32)
        
        # Image transform (from test notebook)
        self._transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    async def score(
        self,
        image: Image.Image,
    ) -> Dict[str, float]:
        """Score an image for aesthetic factors.
        
        Args:
            image: Input image to score
            
        Returns:
            Dictionary with aesthetic factor scores
        """
        # Convert image to RGB
        image = image.convert("RGB")
        
        # Transform image
        image_tensor = self._transform(image).unsqueeze(0).to(self._device)
        
        # Run inference
        with torch.no_grad():
            preds = self._model(image_tensor).cpu().numpy().flatten()
        
        # Denormalize predictions
        preds_denorm = preds * self._factors_coefs
        
        # Create result dictionary
        result = {self._factors[i]: float(preds_denorm[i]) for i in range(len(self._factors))}
        
        return result
