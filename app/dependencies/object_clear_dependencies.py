from fastapi import Request

from app.external.ObjectClear.objectclear.pipelines import ObjectClearPipeline
from app.services.impl.object_clear_service_impl import ObjectClearServiceImpl
from app.services.object_clear_service import ObjectClearService


def get_object_clear_pipeline(request: Request) -> ObjectClearPipeline:
    return request.app.state.object_clear_pipeline


def get_object_clear_service(request: Request) -> ObjectClearService:
    pipeline = request.app.state.object_clear_pipeline
    return ObjectClearServiceImpl(pipeline)
