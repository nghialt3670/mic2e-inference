from fastapi import Request

from app.external.ObjectClear.objectclear.pipelines import ObjectClearPipeline
from app.services.impl.object_clear_service_impl import ObjectClearServiceImpl
from app.services.object_clear_service import ObjectClearService
from app.utils.model_manager import get_model_manager
from app.utils.model_loaders import load_object_clear_standalone


def get_object_clear_pipeline(request: Request) -> ObjectClearPipeline:
    """Get ObjectClear pipeline, loading on-demand."""
    manager = get_model_manager()
    return manager.get_model("object_clear", request)


async def get_object_clear_service(request: Request):
    """Get ObjectClear service with automatic cleanup after request."""
    manager = get_model_manager()
    try:
        pipeline = manager.get_model("object_clear", request)
        service = ObjectClearServiceImpl(pipeline)
        yield service
    finally:
        manager.cleanup_model("object_clear")
