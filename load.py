from diffusers import FluxPipeline
import torch

device = "mps"

FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to(device)