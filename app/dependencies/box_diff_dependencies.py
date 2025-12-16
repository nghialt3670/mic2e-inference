import sys
from pathlib import Path

from fastapi import Request

# Add BoxDiff to Python path
boxdiff_path = Path(__file__).parent.parent / "external" / "BoxDiff"
if str(boxdiff_path) not in sys.path:
    sys.path.insert(0, str(boxdiff_path))

from pipeline.sd_pipeline_boxdiff import BoxDiffPipeline

from app.services.box_diff_service import BoxDiffService
from app.services.impl.box_diff_service_impl import BoxDiffServiceImpl


def get_box_diff_pipeline(request: Request) -> BoxDiffPipeline:
    return request.app.state.box_diff_pipeline


def get_box_diff_service(request: Request) -> BoxDiffService:
    pipeline = request.app.state.box_diff_pipeline
    return BoxDiffServiceImpl(pipeline)
