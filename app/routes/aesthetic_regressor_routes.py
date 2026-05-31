"""Aesthetic Regressor API routes."""

import io
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image

from app.dependencies.aesthetic_regressor_dependencies import get_aesthetic_regressor_service
from app.services.aesthetic_regressor_service import AestheticRegressorService

router = APIRouter(prefix="/aesthetic-regressor", tags=["aesthetic-regressor"])
logger = logging.getLogger(__name__)


@router.post("/score", summary="Score image for aesthetic factors")
async def score(
    image: UploadFile = File(..., description="Input image to score"),
    service: AestheticRegressorService = Depends(get_aesthetic_regressor_service),
):
    """Score an image for aesthetic factors (saturation, brightness, tint, temperature, contrast).
    
    Args:
        image: Input image file (JPEG, PNG, etc.)
        service: Injected Aesthetic Regressor service
        
    Returns:
        JSON dictionary with aesthetic factor scores
        
    Raises:
        HTTPException: If image loading fails or processing error occurs
    """
    try:
        # Load input image
        try:
            image_bytes = await image.read()
            input_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Failed to load input image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        logger.info(f"Scoring image {input_image.size} for aesthetic factors")
        
        # Score the image
        scores = await service.score(image=input_image)
        
        return scores
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during scoring: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
