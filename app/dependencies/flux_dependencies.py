from fastapi import Depends, Request
from diffusers import FluxPipeline

from app.services.flux_service import FluxService
from app.services.impl.flux_service_impl import FluxServiceImpl


def get_flux_pipeline(request: Request) -> FluxPipeline:
    return request.app.state.flux_pipeline


def get_flux_service(request: Request) -> FluxService:
    pipeline = request.app.state.flux_pipeline
    return FluxServiceImpl(pipeline)