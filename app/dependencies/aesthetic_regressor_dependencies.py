"""Dependencies for Aesthetic Regressor."""

import logging
import sys
from pathlib import Path

import torch
from fastapi import Request

from app.env import (
    AESTHETIC_REGRESSOR_WEIGHT_PATH,
    AESTHETIC_REGRESSOR_BACKBONE,
    AESTHETIC_REGRESSOR_N_FACTORS,
    AESTHETIC_REGRESSOR_ACTIVATION,
)
from app.services.impl.aesthetic_regressor_service_impl import AestheticRegressorServiceImpl
from app.services.aesthetic_regressor_service import AestheticRegressorService
from app.utils.device_utils import get_device

logger = logging.getLogger(__name__)

# Add AestheticEnhancer to Python path
aesthetic_path = Path(__file__).parent.parent.parent / "external" / "AestheticEnhancer"
if str(aesthetic_path) not in sys.path:
    sys.path.insert(0, str(aesthetic_path))

from model.aesthetic_regressor import AestheticRegressor


def load_aesthetic_regressor_model(device: str) -> AestheticRegressor:
    """Load Aesthetic Regressor model.
    
    Args:
        device: Device to load model on
        
    Returns:
        AestheticRegressor model instance
        
    Raises:
        FileNotFoundError: If weight path is provided but file doesn't exist
        ValueError: If weight path is required but not provided
    """
    # Default values
    n_factors = AESTHETIC_REGRESSOR_N_FACTORS if AESTHETIC_REGRESSOR_N_FACTORS is not None else 5
    activation = AESTHETIC_REGRESSOR_ACTIVATION or 'tanh'
    backbone = AESTHETIC_REGRESSOR_BACKBONE or 'resnet18'
    
    # Create model
    model = AestheticRegressor(
        n_factors=n_factors,
        activation=activation,
        backbone=backbone,
    ).to(device)
    
    # Load weights - required for the model to work properly
    if not AESTHETIC_REGRESSOR_WEIGHT_PATH:
        raise ValueError(
            "AESTHETIC_REGRESSOR_WEIGHT_PATH environment variable is required. "
            "Please set it to the path of the trained model weights."
        )
    
    weight_path = Path(AESTHETIC_REGRESSOR_WEIGHT_PATH)
    if not weight_path.exists():
        raise FileNotFoundError(f"Aesthetic Regressor weight file not found: {weight_path}")
    
    logger.info(f"Loading Aesthetic Regressor weights from: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    
    return model


async def get_aesthetic_regressor_service(request: Request) -> AestheticRegressorService:
    """Get Aesthetic Regressor service.
    
    Args:
        request: FastAPI request
        
    Returns:
        AestheticRegressorService instance
    """
    device = get_device()
    model = load_aesthetic_regressor_model(device)
    service = AestheticRegressorServiceImpl(model=model, device=device)
    return service
