from abc import ABC, abstractmethod
from typing import Optional

from PIL import Image

from diffusers import FluxPipeline

class FluxService(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> Image.Image:
        pass