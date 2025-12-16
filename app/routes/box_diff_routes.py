import json
from typing import List

from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.responses import StreamingResponse

from app.dependencies.box_diff_dependencies import get_box_diff_service
from app.services.box_diff_service import BoxDiffService
from app.utils.image_utils import save_image_to_bytes

router = APIRouter(prefix="/box-diff", tags=["box-diff"])


@router.post("/generate")
async def generate(
    prompt: str = Form(...),
    token_indices: str = Form(...),
    bbox: str = Form(...),
    seed: int = Form(42),
    service: BoxDiffService = Depends(get_box_diff_service),
):
    """
    Generate an image using BoxDiff with spatial control.
    
    Args:
        prompt: Text prompt for image generation
        token_indices: JSON array of token indices to control, e.g., "[2, 4]"
        bbox: JSON array of bounding boxes in format [[x1,y1,x2,y2], ...], e.g., "[[67,87,366,512],[66,130,364,262]]"
        seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Generated image as PNG
        
    Example:
        prompt: "A rabbit wearing sunglasses looks very proud"
        token_indices: "[2, 4]"  # indices for "rabbit" and "sunglasses"
        bbox: "[[67,87,366,512],[66,130,364,262]]"
    """
    # Parse token_indices
    try:
        token_indices_list: List[int] = json.loads(token_indices)
        if not isinstance(token_indices_list, list) or not all(isinstance(i, int) for i in token_indices_list):
            raise ValueError("token_indices must be a list of integers")
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid token_indices JSON: {str(e)}"
        )
    
    # Parse bbox
    try:
        bbox_list: List[List[int]] = json.loads(bbox)
        if not isinstance(bbox_list, list):
            raise ValueError("bbox must be a list")
        for box in bbox_list:
            if not isinstance(box, list) or len(box) != 4 or not all(isinstance(i, int) for i in box):
                raise ValueError("Each bbox must be a list of 4 integers [x1, y1, x2, y2]")
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid bbox JSON: {str(e)}"
        )
    
    # Validate that token_indices and bbox have the same length
    if len(token_indices_list) != len(bbox_list):
        raise HTTPException(
            status_code=400, 
            detail=f"token_indices and bbox must have the same length. Got {len(token_indices_list)} token_indices and {len(bbox_list)} bboxes"
        )
    
    # Generate image
    image = await service.generate(
        prompt=prompt,
        token_indices=token_indices_list,
        bbox=bbox_list,
        seed=seed,
    )
    
    # Convert to bytes and return
    image_bytes = save_image_to_bytes(image)
    return StreamingResponse(
        image_bytes,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=boxdiff_result.png"},
    )
