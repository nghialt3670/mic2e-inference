from abc import ABC, abstractmethod

from PIL import Image

class FluxService(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> Image.Image:
        pass