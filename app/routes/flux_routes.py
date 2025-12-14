from fastapi import APIRouter, Depends, Form
from fastapi.responses import StreamingResponse

from app.dependencies.flux_dependencies import get_flux_service
from app.services.flux_service import FluxService
from app.utils.image_utils import save_image_to_bytes

router = APIRouter(prefix="/flux", tags=["flux"])


@router.post("/generate")
async def generate(
    prompt: str = Form(...),
    service: FluxService = Depends(get_flux_service),
):
    image = await service.generate(prompt)
    image_bytes = save_image_to_bytes(image)
    return StreamingResponse(
        image_bytes,
        media_type="image/png",
        headers={"Content-Disposition": "attachment; filename=result.png"},
    )
