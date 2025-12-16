# Model Loaders

This directory contains individual model loader modules for the inference service. Each loader is responsible for loading and configuring a specific model.

## Available Loaders

### SAM3 (`sam3_loader.py`)
Loads the Segment Anything Model 3 (SAM3) for image segmentation tasks.

**Models loaded:**
- SAM3 base model
- SAM3 processor
- SAM3 tracker model
- SAM3 tracker processor

**Environment variable:** `LOAD_SAM3=true`

### ObjectClear (`object_clear_loader.py`)
Loads the ObjectClear pipeline for object removal from images.

**Models loaded:**
- ObjectClear pipeline with attention-guided fusion

**Environment variable:** `LOAD_OBJECT_CLEAR=true`

### BoxDiff (`box_diff_loader.py`)
Loads the BoxDiff pipeline for text-to-image synthesis with box-constrained diffusion.

**Models loaded:**
- BoxDiff pipeline based on Stable Diffusion v1.4

**Environment variable:** `LOAD_BOX_DIFF=true`

### GLIGEN (`gligen_loader.py`)
Loads GLIGEN models for grounded text-to-image generation and inpainting.

**Models loaded:**
- GLIGEN generation model (text-box)
- GLIGEN inpainting model (text-box)
- Associated autoencoders, text encoders, and diffusion models

**Environment variable:** `LOAD_GLIGEN=true` (default)

### Flux (`flux_loader.py`)
Loads the Flux pipeline for high-quality image generation.

**Models loaded:**
- FLUX.1-schnell pipeline

**Environment variable:** `LOAD_FLUX=true`

## Usage

### Environment Configuration

Set environment variables in your `.env` file to control which models are loaded:

```bash
# Model loading flags (set to "true" to enable)
LOAD_SAM3=false
LOAD_OBJECT_CLEAR=false
LOAD_BOX_DIFF=false
LOAD_GLIGEN=true
LOAD_FLUX=false
```

### Adding a New Model Loader

1. Create a new file `{model_name}_loader.py` in this directory
2. Implement a `load_{model_name}(app: FastAPI, device: str)` function
3. Add the loader to `__init__.py`
4. Add a corresponding environment variable in `app/env.py`
5. Call the loader in `app/lifespan.py` with the environment flag check

Example structure:

```python
"""My Model loader."""

import logging
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def load_my_model(app: FastAPI, device: str) -> None:
    """Load My Model.
    
    Args:
        app: FastAPI application instance
        device: Device to load model on (cpu, cuda, mps)
    """
    logger.info("Loading My Model...")
    
    # Your model loading logic here
    app.state.my_model = load_model()
    
    logger.info("My Model loaded")
```

## Device Support

All loaders support the following devices:
- **CUDA** (NVIDIA GPUs)
- **MPS** (Apple Silicon GPUs)
- **CPU** (fallback)

Loaders automatically select the appropriate data type:
- CUDA: `torch.float16` for memory efficiency
- MPS/CPU: `torch.float32` for compatibility

## Error Handling

If a model fails to load, the error is logged but the application continues to start. This allows partial functionality even if some models are unavailable.
