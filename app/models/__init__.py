"""Model loaders for the inference service."""

from app.models.sam3_loader import load_sam3
from app.models.object_clear_loader import load_object_clear
from app.models.box_diff_loader import load_box_diff
from app.models.gligen_loader import load_gligen
from app.models.sd_inpaint_loader import load_sd_inpaint
from app.models.flux_loader import load_flux

__all__ = [
    "load_sam3",
    "load_object_clear",
    "load_box_diff",
    "load_gligen",
    "load_sd_inpaint",
    "load_flux",
]
