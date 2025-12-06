from fastapi import Request
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)

from app.services.impl.sam3_service_impl import Sam3ServiceImpl
from app.services.sam3_service import Sam3Service


def get_sam3_model(request: Request) -> Sam3Model:
    return request.app.state.sam3_model


def get_sam3_processor(request: Request) -> Sam3Processor:
    return request.app.state.sam3_processor


def get_sam3_tracker_model(request: Request) -> Sam3TrackerModel:
    return request.app.state.sam3_tracker_model


def get_sam3_tracker_processor(request: Request) -> Sam3TrackerProcessor:
    return request.app.state.sam3_tracker_processor


def get_sam3_service(request: Request) -> Sam3Service:
    sam3_model = request.app.state.sam3_model
    sam3_processor = request.app.state.sam3_processor
    sam3_tracker_model = request.app.state.sam3_tracker_model
    sam3_tracker_processor = request.app.state.sam3_tracker_processor
    return Sam3ServiceImpl(
        sam3_model, sam3_processor, sam3_tracker_model, sam3_tracker_processor
    )
