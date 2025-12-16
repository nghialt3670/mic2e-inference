import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

# Add GLIGEN to Python path
gligen_path = Path(__file__).parent.parent.parent / "external" / "GLIGEN"
if str(gligen_path) not in sys.path:
    sys.path.insert(0, str(gligen_path))

from app.services.gligen_service import GligenService


class GligenServiceImpl(GligenService):
    def __init__(
        self,
        generation_model,
        generation_autoencoder,
        generation_text_encoder,
        generation_diffusion,
        generation_config,
        inpainting_model,
        inpainting_autoencoder,
        inpainting_text_encoder,
        inpainting_diffusion,
        inpainting_config,
        device: str = "cpu",
        batch_size: int = 1,
        guidance_scale: float = 7.5,
        negative_prompt: str = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
    ):
        self._generation_model = generation_model
        self._generation_autoencoder = generation_autoencoder
        self._generation_text_encoder = generation_text_encoder
        self._generation_diffusion = generation_diffusion
        self._generation_config = generation_config
        
        self._inpainting_model = inpainting_model
        self._inpainting_autoencoder = inpainting_autoencoder
        self._inpainting_text_encoder = inpainting_text_encoder
        self._inpainting_diffusion = inpainting_diffusion
        self._inpainting_config = inpainting_config
        
        self._device = device
        self._batch_size = batch_size
        self._guidance_scale = guidance_scale
        self._negative_prompt = negative_prompt

    async def generate(
        self,
        prompt: str,
        phrases: List[str],
        locations: List[List[float]],
        seed: int = 42,
        alpha_type: Optional[List[float]] = None,
    ) -> Image.Image:
        """Generate image with GLIGEN text-box grounding."""
        from gligen_inference import run, alpha_generator, set_alpha_scale, prepare_batch
        from ldm.models.diffusion.plms import PLMSSampler
        from omegaconf import OmegaConf
        from functools import partial
        
        if alpha_type is None:
            alpha_type = [0.3, 0.0, 0.7]
        
        # Prepare metadata
        meta = {
            "prompt": prompt,
            "phrases": phrases,
            "locations": locations,
            "alpha_type": alpha_type,
        }
        
        # Prepare batch
        batch = prepare_batch(meta, self._batch_size)
        
        # Encode text
        context = self._generation_text_encoder.encode([prompt] * self._batch_size)
        uc = self._generation_text_encoder.encode([self._negative_prompt] * self._batch_size)
        
        # Setup sampler
        alpha_generator_func = partial(alpha_generator, type=alpha_type)
        sampler = PLMSSampler(
            self._generation_diffusion,
            self._generation_model,
            alpha_generator_func=alpha_generator_func,
            set_alpha_scale=set_alpha_scale
        )
        
        # Prepare grounding input
        grounding_tokenizer_input = self._generation_model.grounding_tokenizer_input
        grounding_input = grounding_tokenizer_input.prepare(batch)
        
        # Set random seed
        torch.manual_seed(seed)
        if self._device == "cuda":
            torch.cuda.manual_seed(seed)
        
        # Generate
        shape = (self._batch_size, self._generation_model.in_channels, 
                self._generation_model.image_size, self._generation_model.image_size)
        
        input_dict = dict(
            x=None,
            timesteps=None,
            context=context,
            grounding_input=grounding_input,
            inpainting_extra_input=None,
            grounding_extra_input=None,
        )
        
        samples = sampler.sample(
            S=50,
            shape=shape,
            input=input_dict,
            uc=uc,
            guidance_scale=self._guidance_scale,
        )
        
        samples = self._generation_autoencoder.decode(samples)
        
        # Convert to PIL image
        sample = samples[0]
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.detach().cpu().numpy().transpose(1, 2, 0) * 255
        image = Image.fromarray(sample.astype('uint8'))
        
        return image

    async def inpaint(
        self,
        input_image: Image.Image,
        prompt: str,
        phrases: List[str],
        locations: List[List[float]],
        seed: int = 42,
    ) -> Image.Image:
        """Inpaint image with GLIGEN text-box grounding."""
        from gligen_inference import prepare_batch, set_alpha_scale
        from ldm.models.diffusion.plms import PLMSSampler
        from inpaint_mask_func import draw_masks_from_boxes
        from functools import partial
        import torchvision.transforms.functional as F
        
        # Store original size for later
        original_size = input_image.size
        
        # Prepare metadata
        meta = {
            "prompt": prompt,
            "phrases": phrases,
            "locations": locations,
        }
        
        # Prepare batch
        batch = prepare_batch(meta, self._batch_size)
        
        # Encode text
        context = self._inpainting_text_encoder.encode([prompt] * self._batch_size)
        uc = self._inpainting_text_encoder.encode([self._negative_prompt] * self._batch_size)
        
        # Setup sampler
        alpha_generator_func = lambda length: [1] * length  # constant alpha for inpainting
        sampler = PLMSSampler(
            self._inpainting_diffusion,
            self._inpainting_model,
            alpha_generator_func=alpha_generator_func,
            set_alpha_scale=set_alpha_scale
        )
        
        # Prepare grounding input
        grounding_tokenizer_input = self._inpainting_model.grounding_tokenizer_input
        grounding_input = grounding_tokenizer_input.prepare(batch)
        
        # Prepare inpainting mask and input
        inpainting_mask = draw_masks_from_boxes(
            batch['boxes'], 
            self._inpainting_model.image_size
        ).to(self._device)
        
        # Process input image
        input_image_resized = input_image.convert("RGB").resize((512, 512))
        input_tensor = F.pil_to_tensor(input_image_resized)
        input_tensor = (input_tensor.float().unsqueeze(0).to(self._device) / 255 - 0.5) / 0.5
        z0 = self._inpainting_autoencoder.encode(input_tensor)
        
        masked_z = z0 * inpainting_mask
        inpainting_extra_input = torch.cat([masked_z, inpainting_mask], dim=1)
        
        # Set random seed
        torch.manual_seed(seed)
        if self._device == "cuda":
            torch.cuda.manual_seed(seed)
        
        # Generate
        shape = (self._batch_size, self._inpainting_model.in_channels,
                self._inpainting_model.image_size, self._inpainting_model.image_size)
        
        input_dict = dict(
            x=None,
            timesteps=None,
            context=context,
            grounding_input=grounding_input,
            inpainting_extra_input=inpainting_extra_input,
            grounding_extra_input=None,
        )
        
        samples = sampler.sample(
            S=50,
            shape=shape,
            input=input_dict,
            uc=uc,
            guidance_scale=self._guidance_scale,
            mask=inpainting_mask,
            x0=z0,
        )
        
        samples = self._inpainting_autoencoder.decode(samples)
        
        # Convert to PIL image
        sample = samples[0]
        sample = torch.clamp(sample, min=-1, max=1) * 0.5 + 0.5
        sample = sample.detach().cpu().numpy().transpose(1, 2, 0) * 255
        image = Image.fromarray(sample.astype('uint8'))
        
        # Resize to original input image size
        if image.size != original_size:
            image = image.resize(original_size, Image.Resampling.LANCZOS)
        
        return image
