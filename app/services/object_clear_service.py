from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image


class ObjectClearService(ABC):
    @abstractmethod
    async def inpaint(
        self, image: Image.Image, mask: Image.Image, label: Optional[str] = None
    ) -> Image.Image:
        pass
