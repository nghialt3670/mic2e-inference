# Configuration Guide

## Environment Variables

All configuration is managed through environment variables, which can be set in a `.env` file at the root of the project.

### Required Variables

```bash
# Server port
PORT=8000

# Hugging Face authentication token (required for model downloads)
HUGGINGFACE_TOKEN=your_token_here
```

### Model Loading Configuration

Control which models are loaded at startup to optimize memory usage and startup time. Set to `true` to enable, `false` to disable.

```bash
# SAM3 - Segment Anything Model 3
# Used for: Image segmentation, object detection
# Memory: ~2-3GB
LOAD_SAM3=false

# ObjectClear - Object Removal Pipeline
# Used for: Removing objects from images
# Memory: ~2-4GB
LOAD_OBJECT_CLEAR=false

# BoxDiff - Box-Constrained Diffusion
# Used for: Text-to-image with spatial control
# Memory: ~4-6GB
LOAD_BOX_DIFF=false

# GLIGEN - Grounded Text-to-Image Generation
# Used for: Text-box grounded generation and inpainting
# Memory: ~4-6GB
# Default: true (currently active service)
LOAD_GLIGEN=true

# Flux - High-Quality Image Generation
# Used for: State-of-the-art text-to-image generation
# Memory: ~8-12GB
LOAD_FLUX=false
```

## Example Configurations

### Minimal Setup (GLIGEN only)
```bash
PORT=8000
HUGGINGFACE_TOKEN=your_token_here
LOAD_SAM3=false
LOAD_OBJECT_CLEAR=false
LOAD_BOX_DIFF=false
LOAD_GLIGEN=true
LOAD_FLUX=false
```

### Full Setup (All Models)
```bash
PORT=8000
HUGGINGFACE_TOKEN=your_token_here
LOAD_SAM3=true
LOAD_OBJECT_CLEAR=true
LOAD_BOX_DIFF=true
LOAD_GLIGEN=true
LOAD_FLUX=true
```

### Development Setup (Fast Startup)
```bash
PORT=8000
HUGGINGFACE_TOKEN=your_token_here
# Disable all models for fast iteration
LOAD_SAM3=false
LOAD_OBJECT_CLEAR=false
LOAD_BOX_DIFF=false
LOAD_GLIGEN=false
LOAD_FLUX=false
```

## Memory Requirements

Approximate GPU/RAM requirements based on enabled models:

| Configuration | VRAM (CUDA) | RAM (CPU/MPS) |
|--------------|-------------|---------------|
| GLIGEN only | 4-6 GB | 8-12 GB |
| + SAM3 | 6-9 GB | 12-18 GB |
| + ObjectClear | 8-13 GB | 16-24 GB |
| + BoxDiff | 12-19 GB | 20-32 GB |
| + Flux | 20-31 GB | 32-48 GB |
| All models | 24-40 GB | 40-64 GB |

## Device Support

The application automatically detects and uses the best available device:

1. **CUDA** (NVIDIA GPUs) - Preferred for best performance
2. **MPS** (Apple Silicon) - Good performance on M1/M2/M3 Macs
3. **CPU** - Fallback option (slower)

Models automatically use appropriate precision:
- CUDA: FP16 (faster, less memory)
- MPS/CPU: FP32 (better compatibility)

## Getting Started

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and set your `HUGGINGFACE_TOKEN`

3. Enable only the models you need

4. Start the application:
   ```bash
   python run.py
   ```

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:
1. Disable unused models
2. Use a GPU with more VRAM
3. Run on CPU (slower but uses system RAM)

### Slow Startup

If startup is taking too long:
1. Disable models you don't need
2. Check your internet connection (models download on first run)
3. Models are cached after first download

### Model Not Loading

If a model fails to load:
1. Check the logs for specific error messages
2. Verify your `HUGGINGFACE_TOKEN` is valid
3. Ensure you have enough disk space for model files
4. Check if model checkpoints exist (for GLIGEN)
