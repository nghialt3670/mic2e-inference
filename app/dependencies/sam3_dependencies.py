from fastapi import Request
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)

from app.services.impl.sam3_service_impl import Sam3ServiceImpl
from app.services.sam3_service import Sam3Service
from app.utils.model_manager import get_model_manager
from app.utils.model_loaders import load_sam3_standalone


def get_sam3_model(request: Request) -> Sam3Model:
    """Get SAM3 model, loading on-demand."""
    manager = get_model_manager()
    models = manager.get_model("sam3", request)
    return models[0]


def get_sam3_processor(request: Request) -> Sam3Processor:
    """Get SAM3 processor, loading on-demand."""
    manager = get_model_manager()
    models = manager.get_model("sam3", request)
    return models[1]


def get_sam3_tracker_model(request: Request) -> Sam3TrackerModel:
    """Get SAM3 tracker model, loading on-demand."""
    manager = get_model_manager()
    models = manager.get_model("sam3", request)
    return models[2]


def get_sam3_tracker_processor(request: Request) -> Sam3TrackerProcessor:
    """Get SAM3 tracker processor, loading on-demand."""
    manager = get_model_manager()
    models = manager.get_model("sam3", request)
    return models[3]


async def get_sam3_service(request: Request):
    """Get SAM3 service with automatic cleanup after request."""
    manager = get_model_manager()
    try:
        models = manager.get_model("sam3", request)
        sam3_model, sam3_processor, sam3_tracker_model, sam3_tracker_processor = models
        service = Sam3ServiceImpl(
            sam3_model, sam3_processor, sam3_tracker_model, sam3_tracker_processor
        )
        yield service
    finally:
        manager.cleanup_model("sam3")
