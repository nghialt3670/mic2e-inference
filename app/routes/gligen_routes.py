import json
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.dependencies.gligen_dependencies import get_gligen_service
from app.services.gligen_service import GligenService
from app.utils.image_utils import read_upload_file_as_image, save_image_to_bytes

router = APIRouter(prefix="/gligen", tags=["gligen"])


@router.post("/generate")
async def generate(
    prompt: str = Form(...),
    phrases: str = Form(...),
    locations: str = Form(...),
    seed: int = Form(42),
    alpha_type: Optional[str] = Form(None),
    service: GligenService = Depends(get_gligen_service),
):
    """
    Generate an image using GLIGEN with text-box grounding.
    
    Args:
        prompt: Text prompt for image generation
        phrases: JSON array of text phrases, e.g., '["a cat", "a dog"]'
        locations: JSON array of bounding boxes in normalized coordinates (0-1), 
                  e.g., '[[0.0, 0.1, 0.5, 0.8], [0.5, 0.1, 1.0, 0.8]]'
        seed: Random seed for reproducibility (default: 42)
        alpha_type: Optional JSON array [stage0, stage1, stage2] summing to 1.0,
                   controls grounding strength over timesteps (default: [0.3, 0.0, 0.7])
        
    Returns:
        Generated image as PNG
        
    Example:
        prompt: "a teddy bear sitting next to a bird"
        phrases: '["a teddy bear", "a bird"]'
        locations: '[[0.0, 0.09, 0.33, 0.76], [0.55, 0.11, 1.0, 0.8]]'
    """
    # Parse phrases
    try:
        phrases_list: List[str] = json.loads(phrases)
        if not isinstance(phrases_list, list) or not all(isinstance(p, str) for p in phrases_list):
            raise ValueError("phrases must be a list of strings")
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid phrases JSON: {str(e)}"
        )
    
    # Parse locations
    try:
        locations_list: List[List[float]] = json.loads(locations)
        if not isinstance(locations_list, list):
            raise ValueError("locations must be a list")
        for loc in locations_list:
            if not isinstance(loc, list) or len(loc) != 4:
                raise ValueError("Each location must be a list of 4 floats [x1, y1, x2, y2]")
            if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in loc):
                raise ValueError("Location coordinates must be floats between 0 and 1")
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid locations JSON: {str(e)}"
        )
    
    # Validate that phrases and locations have the same length
    if len(phrases_list) != len(locations_list):
        raise HTTPException(
            status_code=400,
            detail=f"phrases and locations must have the same length. Got {len(phrases_list)} phrases and {len(locations_list)} locations"
        )
    
    # Parse alpha_type if provided
    alpha_type_list = None
    if alpha_type:
        try:
            alpha_type_list: List[float] = json.loads(alpha_type)
            if not isinstance(alpha_type_list, list) or len(alpha_type_list) != 3:
                raise ValueError("alpha_type must be a list of 3 floats")
            if not all(isinstance(v, (int, float)) for v in alpha_type_list):
                raise ValueError("alpha_type values must be floats")
            if abs(sum(alpha_type_list) - 1.0) > 0.001:
                raise ValueError("alpha_type values must sum to 1.0")
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid alpha_type JSON: {str(e)}"
            )
    
    # Generate image
    image = await service.generate(
        prompt=prompt,
        phrases=phrases_list,
        locations=locations_list,
        seed=seed,
        alpha_type=alpha_type_list,
    )
    
    # Convert to bytes and return
    image_bytes = save_image_to_bytes(image)
    return StreamingResponse(
        image_bytes,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=gligen_generated.png"},
    )


@router.post("/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    phrases: str = Form(...),
    locations: str = Form(...),
    seed: int = Form(42),
    service: GligenService = Depends(get_gligen_service),
):
    """
    Inpaint an image using GLIGEN with text-box grounding.
    
    Args:
        image: Input image file to inpaint
        prompt: Text prompt for inpainting
        phrases: JSON array of text phrases for the masked regions
        locations: JSON array of bounding boxes (also used as masks) in normalized coordinates (0-1)
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Inpainted image as PNG
        
    Example:
        image: <upload file>
        prompt: "a corgi and a cake"
        phrases: '["corgi", "cake"]'
        locations: '[[0.25, 0.28, 0.42, 0.52], [0.14, 0.58, 0.58, 0.92]]'
    """
    # Parse phrases
    try:
        phrases_list: List[str] = json.loads(phrases)
        if not isinstance(phrases_list, list) or not all(isinstance(p, str) for p in phrases_list):
            raise ValueError("phrases must be a list of strings")
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid phrases JSON: {str(e)}"
        )
    
    # Parse locations
    try:
        locations_list: List[List[float]] = json.loads(locations)
        if not isinstance(locations_list, list):
            raise ValueError("locations must be a list")
        for loc in locations_list:
            if not isinstance(loc, list) or len(loc) != 4:
                raise ValueError("Each location must be a list of 4 floats [x1, y1, x2, y2]")
            if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in loc):
                raise ValueError("Location coordinates must be floats between 0 and 1")
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid locations JSON: {str(e)}"
        )
    
    # Validate that phrases and locations have the same length
    if len(phrases_list) != len(locations_list):
        raise HTTPException(
            status_code=400,
            detail=f"phrases and locations must have the same length. Got {len(phrases_list)} phrases and {len(locations_list)} locations"
        )
    
    # Read input image
    input_image = await read_upload_file_as_image(image)
    
    # Inpaint image
    result_image = await service.inpaint(
        input_image=input_image,
        prompt=prompt,
        phrases=phrases_list,
        locations=locations_list,
        seed=seed,
    )
    
    # Convert to bytes and return
    image_bytes = save_image_to_bytes(result_image)
    return StreamingResponse(
        image_bytes,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=gligen_inpainted.png"},
    )
