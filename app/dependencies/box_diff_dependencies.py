from fastapi import Request

from app.services.box_diff_service import BoxDiffService
from app.services.impl.box_diff_service_impl import BoxDiffServiceImpl
from app.utils.model_manager import get_model_manager
from app.utils.model_loaders import load_box_diff_standalone


def get_box_diff_pipeline(request: Request):
    """Get BoxDiff pipeline, loading on-demand."""
    manager = get_model_manager()
    return manager.get_model("box_diff", request)


async def get_box_diff_service(request: Request):
    """Get BoxDiff service with automatic cleanup after request."""
    manager = get_model_manager()
    try:
        pipeline = manager.get_model("box_diff", request)
        service = BoxDiffServiceImpl(pipeline)
        yield service
    finally:
        manager.cleanup_model("box_diff")
