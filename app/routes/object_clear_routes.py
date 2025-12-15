from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse
from typing import Optional
from app.dependencies.object_clear_dependencies import get_object_clear_service
from app.services.object_clear_service import ObjectClearService
from app.utils.image_utils import read_upload_file_as_image, save_image_to_bytes

router = APIRouter(prefix="/object-clear", tags=["object-clear"])


@router.post("/inpaint")
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    service: ObjectClearService = Depends(get_object_clear_service),
):
    pil_image = await read_upload_file_as_image(image)
    pil_mask = await read_upload_file_as_image(mask)
    inpainted_image = await service.inpaint(pil_image, pil_mask, prompt)
    inpainted_image_bytes = save_image_to_bytes(inpainted_image)
    return StreamingResponse(
        inpainted_image_bytes,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=result.png"},
    )
