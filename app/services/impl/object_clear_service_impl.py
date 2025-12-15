from typing import Optional

import torch
from PIL import Image

from app.external.ObjectClear.objectclear.pipelines import ObjectClearPipeline
from app.external.ObjectClear.objectclear.utils import resize_by_short_side
from app.services.object_clear_service import ObjectClearService


class ObjectClearServiceImpl(ObjectClearService):
    def __init__(
        self,
        pipeline: ObjectClearPipeline,
        num_inference_steps: int = 8,  # Reduced from 20 to 8 for speed
        guidance_scale: float = 2.0,  # Reduced from 2.5 to 2.0 for speed
        seed: int = 42,
    ):
        self._pipeline = pipeline
        self._num_inference_steps = num_inference_steps
        self._guidance_scale = guidance_scale
        self._seed = seed

    async def inpaint(
        self, image: Image.Image, mask: Image.Image, prompt: Optional[str] = None
    ) -> Image.Image:
        image = image.convert("RGB")
        mask = mask.convert("L")

        original_size = image.size

        # Reduced resolution from 512 to 384 for speed/memory
        image_resized = resize_by_short_side(image, 384, resample=Image.BILINEAR)  # BILINEAR faster than BICUBIC
        mask_resized = resize_by_short_side(mask, 384, resample=Image.NEAREST)

        w, h = image_resized.size

        device = self._pipeline.unet.device
        generator = torch.Generator(device=device).manual_seed(self._seed)
        prompt = prompt or "remove the instance of object"
        result = self._pipeline(
            prompt=prompt,
            image=image_resized,
            mask_image=mask_resized,
            generator=generator,
            num_inference_steps=self._num_inference_steps,
            guidance_scale=self._guidance_scale,
            height=h,
            width=w,
            return_attn_map=False,
        )

        result_image = result.images[0]
        return result_image.resize(original_size, resample=Image.BICUBIC)
