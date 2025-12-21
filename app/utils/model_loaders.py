"""Standalone model loader functions for on-demand loading."""

import logging
import sys
from pathlib import Path
from typing import Tuple

import torch
from diffusers import FluxPipeline, StableDiffusionInpaintPipeline
from transformers import (
    Sam3Model,
    Sam3Processor,
    Sam3TrackerModel,
    Sam3TrackerProcessor,
)

from app.env import (
    GLIGEN_AUTO_DOWNLOAD_GENERATION,
    GLIGEN_AUTO_DOWNLOAD_INPAINTING,
    GLIGEN_CHECKPOINT_DIR,
)
from app.external.ObjectClear.objectclear.pipelines import ObjectClearPipeline
from app.utils.checkpoint_downloader import (
    ensure_gligen_generation_checkpoint,
    ensure_gligen_inpainting_checkpoint,
)

logger = logging.getLogger(__name__)


def load_sam3_standalone(device: str) -> Tuple[Sam3Model, Sam3Processor, Sam3TrackerModel, Sam3TrackerProcessor]:
    """Load SAM3 models and processors.
    
    Args:
        device: Device to load models on (cpu, cuda, mps)
        
    Returns:
        Tuple of (model, processor, tracker_model, tracker_processor)
    """
    logger.info("Loading SAM3 model...")
    
    # Use fp16 for CUDA, fp32 for CPU/MPS
    sam3_dtype = torch.float16 if device == "cuda" else torch.float32
    
    model = Sam3Model.from_pretrained(
        "facebook/sam3",
        torch_dtype=sam3_dtype,
        low_cpu_mem_usage=True
    ).to(device)
    logger.info("SAM3 model loaded")

    logger.info("Loading SAM3 processor...")
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    logger.info("SAM3 processor loaded")

    logger.info("Loading SAM3 tracker model...")
    tracker_model = Sam3TrackerModel.from_pretrained(
        "facebook/sam3",
        torch_dtype=sam3_dtype,
        low_cpu_mem_usage=True
    ).to(device)
    logger.info("SAM3 tracker model loaded")

    logger.info("Loading SAM3 tracker processor...")
    tracker_processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    logger.info("SAM3 tracker processor loaded")
    
    return model, processor, tracker_model, tracker_processor


def load_object_clear_standalone(device: str) -> ObjectClearPipeline:
    """Load ObjectClear pipeline.
    
    Args:
        device: Device to load model on (cpu, cuda, mps)
        
    Returns:
        ObjectClearPipeline instance
    """
    logger.info("Loading ObjectClear pipeline...")
    
    # Use fp16 for CUDA, fp32 for CPU/MPS
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if device == "cuda" else None
    
    pipeline = (
        ObjectClearPipeline.from_pretrained_with_custom_modules(
            "jixin0101/ObjectClear",
            torch_dtype=torch_dtype,
            apply_attention_guided_fusion=True,
            variant=variant,
            low_cpu_mem_usage=True,
        )
    )
    pipeline.to(device)
    
    # Enable memory optimizations
    if device == "cuda":
        pipeline.enable_attention_slicing(slice_size="auto")
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
            pipeline.enable_attention_slicing(1)
    
    logger.info("ObjectClear pipeline loaded with optimizations")
    return pipeline


def load_box_diff_standalone(device: str):
    """Load BoxDiff pipeline.
    
    Args:
        device: Device to load model on (cpu, cuda, mps)
        
    Returns:
        BoxDiffPipeline instance
    """
    # Add BoxDiff to Python path
    boxdiff_path = Path(__file__).parent.parent / "external" / "BoxDiff"
    if str(boxdiff_path) not in sys.path:
        sys.path.insert(0, str(boxdiff_path))
    
    from pipeline.sd_pipeline_boxdiff import BoxDiffPipeline
    
    logger.info("Loading BoxDiff pipeline...")
    
    # Use fp16 for CUDA, fp32 for CPU/MPS
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load BoxDiff with Stable Diffusion v1.4
    pipeline = BoxDiffPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    
    # Enable memory optimizations for BoxDiff
    if device == "cuda":
        pipeline.enable_attention_slicing(slice_size="auto")
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers for BoxDiff")
        except Exception as e:
            logger.warning(f"Could not enable xformers for BoxDiff: {e}")
            pipeline.enable_attention_slicing(1)
    
    logger.info("BoxDiff pipeline loaded with optimizations")
    return pipeline


def load_gligen_standalone(device: str) -> Tuple:
    """Load GLIGEN generation and inpainting models.
    
    Args:
        device: Device to load models on (cpu, cuda, mps)
        
    Returns:
        Tuple of (gen_model, gen_autoencoder, gen_text_encoder, gen_diffusion, gen_config,
                 inp_model, inp_autoencoder, inp_text_encoder, inp_diffusion, inp_config)
    """
    # Add GLIGEN to Python path
    gligen_path = Path(__file__).parent.parent / "external" / "GLIGEN"
    if str(gligen_path) not in sys.path:
        sys.path.insert(0, str(gligen_path))
    
    from gligen_inference import load_ckpt
    from ldm.util import instantiate_from_config
    
    # Determine checkpoint directory
    if GLIGEN_CHECKPOINT_DIR:
        checkpoint_dir = Path(GLIGEN_CHECKPOINT_DIR)
    else:
        checkpoint_dir = gligen_path / "gligen_checkpoints"
    
    # Load GLIGEN generation model
    logger.info("Loading GLIGEN generation model...")
    gligen_gen_ckpt = checkpoint_dir / "checkpoint_generation_text.pth"
    
    # Auto-download generation checkpoint if enabled
    if GLIGEN_AUTO_DOWNLOAD_GENERATION and not gligen_gen_ckpt.exists():
        logger.info("Downloading GLIGEN generation checkpoint...")
        gen_available = ensure_gligen_generation_checkpoint(checkpoint_dir)
        if not gen_available:
            logger.warning("Failed to download GLIGEN generation checkpoint")
    
    if gligen_gen_ckpt.exists():
        (
            gen_model,
            gen_autoencoder,
            gen_text_encoder,
            gen_diffusion,
            gen_config,
        ) = load_ckpt(str(gligen_gen_ckpt))
        
        # Initialize and attach grounding_tokenizer_input to model
        grounding_tokenizer_input = instantiate_from_config(
            gen_config['grounding_tokenizer_input']
        )
        gen_model.grounding_tokenizer_input = grounding_tokenizer_input
        
        logger.info("GLIGEN generation model loaded")
    else:
        logger.warning(f"GLIGEN generation checkpoint not found at {gligen_gen_ckpt}")
        logger.warning("Please download from: https://huggingface.co/gligen/gligen-generation-text-box")
        gen_model = None
        gen_autoencoder = None
        gen_text_encoder = None
        gen_diffusion = None
        gen_config = None
    
    # Load GLIGEN inpainting model
    logger.info("Loading GLIGEN inpainting model...")
    gligen_inp_ckpt = checkpoint_dir / "checkpoint_inpainting_text.pth"
    
    # Auto-download inpainting checkpoint if enabled
    if GLIGEN_AUTO_DOWNLOAD_INPAINTING and not gligen_inp_ckpt.exists():
        logger.info("Downloading GLIGEN inpainting checkpoint...")
        inp_available = ensure_gligen_inpainting_checkpoint(checkpoint_dir)
        if not inp_available:
            logger.warning("Failed to download GLIGEN inpainting checkpoint")
    
    if gligen_inp_ckpt.exists():
        (
            inp_model,
            inp_autoencoder,
            inp_text_encoder,
            inp_diffusion,
            inp_config,
        ) = load_ckpt(str(gligen_inp_ckpt))
        
        # Initialize and attach grounding_tokenizer_input to model
        grounding_tokenizer_input = instantiate_from_config(
            inp_config['grounding_tokenizer_input']
        )
        inp_model.grounding_tokenizer_input = grounding_tokenizer_input
        
        logger.info("GLIGEN inpainting model loaded")
    else:
        logger.warning(f"GLIGEN inpainting checkpoint not found at {gligen_inp_ckpt}")
        logger.warning("Please download from: https://huggingface.co/gligen/gligen-inpainting-text-box")
        inp_model = None
        inp_autoencoder = None
        inp_text_encoder = None
        inp_diffusion = None
        inp_config = None
    
    return (
        gen_model, gen_autoencoder, gen_text_encoder, gen_diffusion, gen_config,
        inp_model, inp_autoencoder, inp_text_encoder, inp_diffusion, inp_config,
    )


def load_sd_inpaint_standalone(device: str) -> StableDiffusionInpaintPipeline:
    """Load Stable Diffusion Inpainting pipeline.
    
    Args:
        device: Device to load model on (cpu, cuda, mps)
        
    Returns:
        StableDiffusionInpaintPipeline instance
    """
    logger.info("Loading Stable Diffusion Inpainting pipeline...")
    
    # Use fp16 for CUDA, fp32 for CPU/MPS
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    variant = "fp16" if device == "cuda" else None
    
    # Load the inpainting model
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch_dtype,
        variant=variant,
        low_cpu_mem_usage=True,
    ).to(device)
    
    # Enable memory optimizations
    if device == "cuda":
        pipeline.enable_attention_slicing(slice_size="auto")
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
            pipeline.enable_attention_slicing(1)
    
    logger.info("Stable Diffusion Inpainting pipeline loaded with optimizations")
    return pipeline


def load_flux_standalone(device: str) -> FluxPipeline:
    """Load Flux pipeline.
    
    Args:
        device: Device to load model on (cpu, cuda, mps)
        
    Returns:
        FluxPipeline instance
    """
    logger.info("Loading Flux pipeline...")
    
    # Optimized Flux loading with memory efficient attention
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-schnell",
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    ).to(device)
    
    if device == "cuda":
        pipeline.enable_attention_slicing(slice_size="auto")
        try:
            pipeline.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers for Flux")
        except Exception as e:
            logger.warning(f"Could not enable xformers for Flux: {e}")
    
    logger.info("Flux pipeline loaded with optimizations")
    return pipeline

