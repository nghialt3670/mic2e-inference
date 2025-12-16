"""Utilities for downloading and caching model checkpoints."""

import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


def download_gligen_checkpoint(
    checkpoint_name: str,
    checkpoint_dir: Path,
    repo_id: str,
    filename: str = "diffusion_pytorch_model.bin",
    force_download: bool = False,
) -> Optional[Path]:
    """Download a GLIGEN checkpoint from HuggingFace Hub if it doesn't exist.
    
    Args:
        checkpoint_name: Local name for the checkpoint (e.g., 'checkpoint_generation_text.pth')
        checkpoint_dir: Directory to store the checkpoint
        repo_id: HuggingFace repository ID (e.g., 'gligen/gligen-generation-text-box')
        filename: Filename in the HF repo
        force_download: Force re-download even if file exists
        
    Returns:
        Path to the checkpoint file, or None if download failed
    """
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    # Check if checkpoint already exists
    if checkpoint_path.exists() and not force_download:
        logger.info(f"Checkpoint already exists: {checkpoint_path}")
        return checkpoint_path
    
    # Ensure checkpoint directory exists
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Downloading {checkpoint_name} from {repo_id}...")
        
        # Download from HuggingFace Hub
        # This uses HF's caching mechanism automatically
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            force_download=force_download,
            resume_download=True,
        )
        
        # Create symlink or copy to the expected location
        import shutil
        shutil.copy2(downloaded_path, checkpoint_path)
        
        logger.info(f"Checkpoint downloaded successfully: {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        logger.error(f"Failed to download checkpoint {checkpoint_name}: {e}", exc_info=True)
        return None


def ensure_gligen_generation_checkpoint(checkpoint_dir: Path) -> bool:
    """Ensure GLIGEN generation checkpoint is available, downloading if necessary.
    
    Args:
        checkpoint_dir: Directory where checkpoints should be stored
        
    Returns:
        True if checkpoint is available, False otherwise
    """
    gen_path = download_gligen_checkpoint(
        checkpoint_name="checkpoint_generation_text.pth",
        checkpoint_dir=checkpoint_dir,
        repo_id="gligen/gligen-generation-text-box",
        filename="diffusion_pytorch_model.bin",
    )
    return gen_path is not None


def ensure_gligen_inpainting_checkpoint(checkpoint_dir: Path) -> bool:
    """Ensure GLIGEN inpainting checkpoint is available, downloading if necessary.
    
    Args:
        checkpoint_dir: Directory where checkpoints should be stored
        
    Returns:
        True if checkpoint is available, False otherwise
    """
    inp_path = download_gligen_checkpoint(
        checkpoint_name="checkpoint_inpainting_text.pth",
        checkpoint_dir=checkpoint_dir,
        repo_id="gligen/gligen-inpainting-text-box",
        filename="diffusion_pytorch_model.bin",
    )
    return inp_path is not None
