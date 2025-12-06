import json
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.dependencies.sam3_dependencies import get_sam3_service
from app.schemas.common_schemas import Box, MaskLabeledPoint
from app.services.sam3_service import Sam3Service
from app.utils.image_utils import (
    read_upload_file_as_image,
    save_image_to_bytes,
    zip_images_to_bytes,
)

router = APIRouter(prefix="/sam3", tags=["sam3"])


@router.post("/generate-mask")
async def generate_mask(
    image: UploadFile = File(...),
    points: str = Form(None),
    box: str = Form(None),
    service: Sam3Service = Depends(get_sam3_service),
):
    if points is None and box is None:
        raise HTTPException(
            status_code=400, detail="Either points or box must be provided"
        )

    points_list = None
    box_obj = None

    if points is not None:
        try:
            points_list = [MaskLabeledPoint(**p) for p in json.loads(points)]
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid points JSON: {str(e)}"
            )

    if box is not None:
        try:
            box_obj = Box(**json.loads(box))
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid box JSON: {str(e)}")

    pil_image = await read_upload_file_as_image(image)
    generated_mask = await service.generate_mask(pil_image, points_list, box_obj)
    mask_bytes = save_image_to_bytes(generated_mask.image)
    return StreamingResponse(
        mask_bytes,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=mask.png"},
    )


@router.post("/generate-masks")
async def generate_masks(
    image: UploadFile = File(...),
    text: str = Form(...),
    service: Sam3Service = Depends(get_sam3_service),
):
    pil_image = await read_upload_file_as_image(image)
    generated_masks = await service.generate_masks(pil_image, text)
    mask_images = [mask.image for mask in generated_masks]
    mask_scores = [mask.score for mask in generated_masks]
    filenames = [f"{score:.3f}.png" for score in mask_scores]
    zip_bytes = zip_images_to_bytes(mask_images, filenames)
    return StreamingResponse(
        zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=masks.zip"},
    )
