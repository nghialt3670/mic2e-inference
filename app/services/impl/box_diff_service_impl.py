import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image

# Add BoxDiff to Python path
boxdiff_path = Path(__file__).parent.parent.parent / "external" / "BoxDiff"
if str(boxdiff_path) not in sys.path:
    sys.path.insert(0, str(boxdiff_path))

from pipeline.sd_pipeline_boxdiff import BoxDiffPipeline
from utils.ptp_utils import AttentionStore

from app.services.box_diff_service import BoxDiffService


class BoxDiffServiceImpl(BoxDiffService):
    def __init__(
        self,
        pipeline: BoxDiffPipeline,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        attention_res: int = 16,
        max_iter_to_alter: int = 25,
        scale_factor: int = 20,
        scale_range: tuple = (1.0, 0.5),
        smooth_attentions: bool = True,
        sigma: float = 0.5,
        kernel_size: int = 3,
        P: float = 0.2,
        L: int = 1,
        refine: bool = True,
        sd_2_1: bool = False,
    ):
        self._pipeline = pipeline
        self._height = height
        self._width = width
        self._num_inference_steps = num_inference_steps
        self._guidance_scale = guidance_scale
        self._attention_res = attention_res
        self._max_iter_to_alter = max_iter_to_alter
        self._scale_factor = scale_factor
        self._scale_range = scale_range
        self._smooth_attentions = smooth_attentions
        self._sigma = sigma
        self._kernel_size = kernel_size
        self._P = P
        self._L = L
        self._refine = refine
        self._sd_2_1 = sd_2_1
        
        # Create a simple config object to pass BoxDiff parameters
        self._config = type('Config', (), {
            'P': P,
            'L': L,
            'refine': refine,
        })()

    async def generate(
        self,
        prompt: str,
        token_indices: List[int],
        bbox: List[List[int]],
        seed: int = 42,
    ) -> Image.Image:
        """Generate image with BoxDiff spatial control."""
        # Create attention store for tracking cross-attention
        attention_store = AttentionStore()
        
        # Register attention control
        from utils import ptp_utils
        ptp_utils.register_attention_control(self._pipeline, attention_store)
        
        # Create generator for reproducibility
        device = self._pipeline.device
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Run the pipeline
        outputs = self._pipeline(
            prompt=prompt,
            attention_store=attention_store,
            indices_to_alter=token_indices,
            attention_res=self._attention_res,
            height=self._height,
            width=self._width,
            num_inference_steps=self._num_inference_steps,
            guidance_scale=self._guidance_scale,
            generator=generator,
            max_iter_to_alter=self._max_iter_to_alter,
            run_standard_sd=False,
            thresholds={0: 0.05, 10: 0.5, 20: 0.8},
            scale_factor=self._scale_factor,
            scale_range=self._scale_range,
            smooth_attentions=self._smooth_attentions,
            sigma=self._sigma,
            kernel_size=self._kernel_size,
            sd_2_1=self._sd_2_1,
            bbox=bbox,
            config=self._config,
        )
        
        return outputs.images[0]
