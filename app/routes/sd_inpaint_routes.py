"""Stable Diffusion Inpainting API routes."""

import io
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image

from app.dependencies.sd_inpaint_dependencies import get_sd_inpaint_service
from app.services.sd_inpaint_service import SDInpaintService

router = APIRouter(prefix="/sd-inpaint", tags=["Stable Diffusion Inpainting"])
logger = logging.getLogger(__name__)


def save_image_to_bytes(image: Image.Image, format: str = "PNG") -> io.BytesIO:
    """Save PIL Image to BytesIO buffer.
    
    Args:
        image: PIL Image to save
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        BytesIO buffer containing the image
    """
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    return img_byte_arr


@router.post("/inpaint", summary="Inpaint image with Stable Diffusion")
async def inpaint(
    image: UploadFile = File(..., description="Input image to inpaint"),
    mask: UploadFile = File(..., description="Binary mask (white = inpaint, black = keep)"),
    prompt: str = Form(..., description="Text prompt for inpainting"),
    negative_prompt: str = Form("", description="Negative prompt (what to avoid)"),
    num_inference_steps: int = Form(50, ge=1, le=150, description="Number of inference steps"),
    guidance_scale: float = Form(7.5, ge=1.0, le=20.0, description="Guidance scale"),
    seed: int = Form(42, description="Random seed for reproducibility"),
    service: SDInpaintService = Depends(get_sd_inpaint_service),
):
    """Inpaint an image using Stable Diffusion with a binary mask.
    
    The mask should be a grayscale or RGB image where:
    - White (255) pixels = areas to inpaint
    - Black (0) pixels = areas to keep unchanged
    
    Args:
        image: Input image file (JPEG, PNG, etc.)
        mask: Mask image file (JPEG, PNG, etc.)
        prompt: Description of what to generate in masked areas
        negative_prompt: Description of what to avoid
        num_inference_steps: Number of denoising steps (higher = better quality, slower)
        guidance_scale: How closely to follow the prompt (higher = more strict)
        seed: Random seed for reproducible results
        service: Injected SD inpaint service
        
    Returns:
        Inpainted image as PNG
        
    Raises:
        HTTPException: If image/mask loading fails or processing error occurs
    """
    try:
        # Load input image
        try:
            image_bytes = await image.read()
            input_image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Failed to load input image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Load mask image
        try:
            mask_bytes = await mask.read()
            mask_image = Image.open(io.BytesIO(mask_bytes))
        except Exception as e:
            logger.error(f"Failed to load mask image: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid mask file: {str(e)}")
        
        # Validate images have same dimensions
        if input_image.size != mask_image.size:
            raise HTTPException(
                status_code=400,
                detail=f"Image and mask must have same dimensions. "
                       f"Image: {input_image.size}, Mask: {mask_image.size}"
            )
        
        logger.info(
            f"Inpainting image {input_image.size} with prompt: '{prompt}', "
            f"steps: {num_inference_steps}, guidance: {guidance_scale}, seed: {seed}"
        )
        
        # Perform inpainting
        result_image = await service.inpaint(
            image=input_image,
            mask=mask_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        
        # Convert result to bytes
        image_bytes = save_image_to_bytes(result_image)
        
        return StreamingResponse(
            image_bytes,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=sd_inpaint_result.png"},
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during inpainting: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inpainting failed: {str(e)}")


@router.get("/health", summary="Health check for SD Inpaint service")
async def health_check():
    """Check if the Stable Diffusion Inpaint service is available.
    
    Returns:
        Service status
    """
    return {
        "status": "healthy",
        "service": "stable-diffusion-inpaint",
        "model": "runwayml/stable-diffusion-inpainting",
    }
