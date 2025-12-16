"""GLIGEN model loader."""

import logging
import sys
from pathlib import Path

from fastapi import FastAPI

from app.env import (
    GLIGEN_AUTO_DOWNLOAD_GENERATION,
    GLIGEN_AUTO_DOWNLOAD_INPAINTING,
    GLIGEN_CHECKPOINT_DIR,
)
from app.utils.checkpoint_downloader import (
    ensure_gligen_generation_checkpoint,
    ensure_gligen_inpainting_checkpoint,
)

logger = logging.getLogger(__name__)


def load_gligen(app: FastAPI, device: str) -> None:
    """Load GLIGEN generation and inpainting models.
    
    Args:
        app: FastAPI application instance
        device: Device to load models on (cpu, cuda, mps)
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
            app.state.gligen_generation_model,
            app.state.gligen_generation_autoencoder,
            app.state.gligen_generation_text_encoder,
            app.state.gligen_generation_diffusion,
            app.state.gligen_generation_config,
        ) = load_ckpt(str(gligen_gen_ckpt))
        
        # Initialize and attach grounding_tokenizer_input to model
        grounding_tokenizer_input = instantiate_from_config(
            app.state.gligen_generation_config['grounding_tokenizer_input']
        )
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
            app.state.gligen_inpainting_model,
            app.state.gligen_inpainting_autoencoder,
            app.state.gligen_inpainting_text_encoder,
            app.state.gligen_inpainting_diffusion,
            app.state.gligen_inpainting_config,
        ) = load_ckpt(str(gligen_inp_ckpt))
        
        # Initialize and attach grounding_tokenizer_input to model
        grounding_tokenizer_input = instantiate_from_config(
            app.state.gligen_inpainting_config['grounding_tokenizer_input']
        )
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
