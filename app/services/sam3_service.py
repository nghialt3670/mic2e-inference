from abc import ABC, abstractmethod
from typing import List, Optional

from PIL import Image

from app.schemas.common_schemas import Box, GeneratedMask, MaskLabeledPoint


class Sam3Service(ABC):
    @abstractmethod
    def generate_mask(
        self,
        image: Image.Image,
        points: Optional[List[MaskLabeledPoint]] = None,
        box: Optional[Box] = None,
    ) -> GeneratedMask:
        pass

    @abstractmethod
    def generate_masks(self, image: Image.Image, text: str) -> List[GeneratedMask]:
        pass
