import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI
from huggingface_hub import login
from diffusers import FluxPipeline
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)

from app.env import HUGGINGFACE_TOKEN
from app.external.ObjectClear.objectclear.pipelines import ObjectClearPipeline
from app.utils.device_utils import get_device

# Add BoxDiff to Python path
boxdiff_path = Path(__file__).parent / "external" / "BoxDiff"
if str(boxdiff_path) not in sys.path:
    sys.path.insert(0, str(boxdiff_path))

from pipeline.sd_pipeline_boxdiff import BoxDiffPipeline

# Add GLIGEN to Python path
gligen_path = Path(__file__).parent / "external" / "GLIGEN"
if str(gligen_path) not in sys.path:
    sys.path.insert(0, str(gligen_path))

from gligen_inference import load_ckpt
from ldm.util import instantiate_from_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application...")

    login(token=HUGGINGFACE_TOKEN)
    logger.info("Logged in to Hugging Face")

    device = get_device()
    logger.info(f"Using device: {device}")

    # logger.info("Loading SAM3 model...")
    # # Use fp16 for speed/memory optimization
    # sam3_dtype = torch.float16 if device == "cuda" else torch.float32
    # app.state.sam3_model = Sam3Model.from_pretrained(
    #     "facebook/sam3",
    #     torch_dtype=sam3_dtype,
    #     low_cpu_mem_usage=True
    # ).to(device)
    # logger.info("SAM3 model loaded")

    # logger.info("Loading SAM3 processor...")
    # app.state.sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
    # logger.info("SAM3 processor loaded")

    # logger.info("Loading SAM3 tracker model...")
    # app.state.sam3_tracker_model = Sam3TrackerModel.from_pretrained(
    #     "facebook/sam3",
    #     torch_dtype=sam3_dtype,
    #     low_cpu_mem_usage=True
    # ).to(device)
    # logger.info("SAM3 tracker model loaded")

    # logger.info("Loading SAM3 tracker processor...")
    # app.state.sam3_tracker_processor = Sam3TrackerProcessor.from_pretrained(
    #     "facebook/sam3"
    # )
    # logger.info("SAM3 tracker processor loaded")

    # logger.info("Loading ObjectClear pipeline...")
    # # Aggressive optimizations: fp16, attention slicing, memory efficient attention
    # torch_dtype = torch.float16 if device == "cuda" else torch.float32
    # variant = "fp16" if device == "cuda" else None
    # app.state.object_clear_pipeline = (
    #     ObjectClearPipeline.from_pretrained_with_custom_modules(
    #         "jixin0101/ObjectClear",
    #         torch_dtype=torch_dtype,
    #         apply_attention_guided_fusion=True,
    #         variant=variant,
    #         low_cpu_mem_usage=True,
    #     )
    # )
    # app.state.object_clear_pipeline.to(device)
    
    # # Enable memory optimizations
    # if device == "cuda":
    #     app.state.object_clear_pipeline.enable_attention_slicing(slice_size="auto")
    #     try:
    #         app.state.object_clear_pipeline.enable_xformers_memory_efficient_attention()
    #         logger.info("Enabled xformers memory efficient attention")
    #     except Exception as e:
    #         logger.warning(f"Could not enable xformers: {e}")
    #         app.state.object_clear_pipeline.enable_attention_slicing(1)
    
    # logger.info("ObjectClear pipeline loaded with optimizations")

    # logger.info("Loading BoxDiff pipeline...")
    # # Load BoxDiff with Stable Diffusion v1.4
    # app.state.box_diff_pipeline = BoxDiffPipeline.from_pretrained(
    #     "CompVis/stable-diffusion-v1-4",
    #     torch_dtype=torch_dtype,
    #     low_cpu_mem_usage=True,
    # ).to(device)
    
    # # Enable memory optimizations for BoxDiff
    # if device == "cuda":
    #     app.state.box_diff_pipeline.enable_attention_slicing(slice_size="auto")
    #     try:
    #         app.state.box_diff_pipeline.enable_xformers_memory_efficient_attention()
    #         logger.info("Enabled xformers for BoxDiff")
    #     except Exception as e:
    #         logger.warning(f"Could not enable xformers for BoxDiff: {e}")
    #         app.state.box_diff_pipeline.enable_attention_slicing(1)
    
    # logger.info("BoxDiff pipeline loaded with optimizations")

    logger.info("Loading GLIGEN generation model...")
    # Load GLIGEN generation checkpoint (text-box)
    gligen_gen_ckpt = Path(__file__).parent / "external" / "GLIGEN" / "gligen_checkpoints" / "checkpoint_generation_text.pth"
    if gligen_gen_ckpt.exists():
        (
            app.state.gligen_generation_model,
            app.state.gligen_generation_autoencoder,
            app.state.gligen_generation_text_encoder,
            app.state.gligen_generation_diffusion,
            app.state.gligen_generation_config,
        ) = load_ckpt(str(gligen_gen_ckpt))
        
        # Initialize and attach grounding_tokenizer_input to model
        grounding_tokenizer_input = instantiate_from_config(app.state.gligen_generation_config['grounding_tokenizer_input'])
        app.state.gligen_generation_model.grounding_tokenizer_input = grounding_tokenizer_input
        
        logger.info("GLIGEN generation model loaded")
    else:
        logger.warning(f"GLIGEN generation checkpoint not found at {gligen_gen_ckpt}")
        logger.warning("Please download from: https://huggingface.co/gligen/gligen-generation-text-box")
        # Set to None to indicate not loaded
        app.state.gligen_generation_model = None
        app.state.gligen_generation_autoencoder = None
        app.state.gligen_generation_text_encoder = None
        app.state.gligen_generation_diffusion = None
        app.state.gligen_generation_config = None

    logger.info("Loading GLIGEN inpainting model...")
    # Load GLIGEN inpainting checkpoint (text-box)
    gligen_inp_ckpt = Path(__file__).parent / "external" / "GLIGEN" / "gligen_checkpoints" / "checkpoint_inpainting_text.pth"
    if gligen_inp_ckpt.exists():
        (
            app.state.gligen_inpainting_model,
            app.state.gligen_inpainting_autoencoder,
            app.state.gligen_inpainting_text_encoder,
            app.state.gligen_inpainting_diffusion,
            app.state.gligen_inpainting_config,
        ) = load_ckpt(str(gligen_inp_ckpt))
        
        # Initialize and attach grounding_tokenizer_input to model
        grounding_tokenizer_input = instantiate_from_config(app.state.gligen_inpainting_config['grounding_tokenizer_input'])
        app.state.gligen_inpainting_model.grounding_tokenizer_input = grounding_tokenizer_input
        
        logger.info("GLIGEN inpainting model loaded")
    else:
        logger.warning(f"GLIGEN inpainting checkpoint not found at {gligen_inp_ckpt}")
        logger.warning("Please download from: https://huggingface.co/gligen/gligen-inpainting-text-box")
        # Set to None to indicate not loaded
        app.state.gligen_inpainting_model = None
        app.state.gligen_inpainting_autoencoder = None
        app.state.gligen_inpainting_text_encoder = None
        app.state.gligen_inpainting_diffusion = None
        app.state.gligen_inpainting_config = None
    
    # Store device for service initialization
    app.state.device = device

    # logger.info("Loading Flux pipeline...")
    # # Optimized Flux loading with memory efficient attention
    # app.state.flux_pipeline = FluxPipeline.from_pretrained(
    #     "black-forest-labs/FLUX.1-schnell",
    #     torch_dtype=torch.float16,
    #     use_safetensors=True,
    #     low_cpu_mem_usage=True,
    # ).to(device)
    # 
    # if device == "cuda":
    #     app.state.flux_pipeline.enable_attention_slicing(slice_size="auto")
    #     try:
    #         app.state.flux_pipeline.enable_xformers_memory_efficient_attention()
    #         logger.info("Enabled xformers for Flux")
    #     except Exception as e:
    #         logger.warning(f"Could not enable xformers for Flux: {e}")
    # 
    # logger.info("Flux pipeline loaded with optimizations")

    yield
