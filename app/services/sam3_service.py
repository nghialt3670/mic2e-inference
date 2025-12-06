from abc import ABC, abstractmethod
from typing import List

from PIL import Image

from app.schemas.common_schemas import Box, GeneratedMask, MaskLabeledPoint


class Sam3Service(ABC):
    @abstractmethod
    def generate_mask_by_points(
        self, image: Image.Image, points: List[MaskLabeledPoint]
    ) -> GeneratedMask:
        pass

    @abstractmethod
    def generate_mask_by_box(self, image: Image.Image, box: Box) -> GeneratedMask:
        pass

    @abstractmethod
    def generate_masks_by_text(
        self, image: Image.Image, text: str
    ) -> List[GeneratedMask]:
        pass
