# Models Setup Guide

This document provides instructions for setting up all the models used in the MIC2E Inference API.

## Overview

The inference API supports the following models:
1. **SAM3** - Segment Anything Model 3 (auto-loaded from HuggingFace)
2. **ObjectClear** - Object removal and inpainting (auto-loaded from HuggingFace)
3. **BoxDiff** - Box-constrained diffusion (auto-loaded from HuggingFace)
4. **GLIGEN** - Grounded text-to-image generation (requires manual download)

## Auto-loaded Models

These models are automatically downloaded from HuggingFace on first startup:

### SAM3
- Model: `facebook/sam3`
- Auto-loaded: ✅
- No action required

### ObjectClear
- Model: `jixin0101/ObjectClear`
- Auto-loaded: ✅
- No action required

### BoxDiff
- Model: `CompVis/stable-diffusion-v1-4`
- Auto-loaded: ✅
- No action required

## Manual Download Required

### GLIGEN

GLIGEN requires manual checkpoint download due to its custom format.

#### 1. Create checkpoints directory

```bash
mkdir -p app/external/GLIGEN/gligen_checkpoints
```

#### 2. Download Generation Model (Text-Box)

```bash
# Option 1: Using wget
wget https://huggingface.co/gligen/gligen-generation-text-box/resolve/main/diffusion_pytorch_model.bin \
  -O app/external/GLIGEN/gligen_checkpoints/checkpoint_generation_text.pth

# Option 2: Using curl
curl -L https://huggingface.co/gligen/gligen-generation-text-box/resolve/main/diffusion_pytorch_model.bin \
  -o app/external/GLIGEN/gligen_checkpoints/checkpoint_generation_text.pth
```

#### 3. Download Inpainting Model (Text-Box)

```bash
# Option 1: Using wget
wget https://huggingface.co/gligen/gligen-inpainting-text-box/resolve/main/diffusion_pytorch_model.bin \
  -O app/external/GLIGEN/gligen_checkpoints/checkpoint_inpainting_text.pth

# Option 2: Using curl
curl -L https://huggingface.co/gligen/gligen-inpainting-text-box/resolve/main/diffusion_pytorch_model.bin \
  -o app/external/GLIGEN/gligen_checkpoints/checkpoint_inpainting_text.pth
```

#### 4. Verify Download

```bash
ls -lh app/external/GLIGEN/gligen_checkpoints/

# You should see:
# checkpoint_generation_text.pth (size: ~4GB)
# checkpoint_inpainting_text.pth (size: ~4GB)
```

## Storage Requirements

| Model | Size | Auto-loaded | Storage Location |
|-------|------|-------------|------------------|
| SAM3 | ~2GB | Yes | HuggingFace cache |
| ObjectClear | ~5GB | Yes | HuggingFace cache |
| BoxDiff | ~4GB | Yes | HuggingFace cache |
| GLIGEN Generation | ~4GB | No | `app/external/GLIGEN/gligen_checkpoints/` |
| GLIGEN Inpainting | ~4GB | No | `app/external/GLIGEN/gligen_checkpoints/` |

**Total**: ~19GB (including both GLIGEN models)

## Startup Behavior

### With GLIGEN Checkpoints Present

The application will load all models and all endpoints will be available:
- ✅ `/sam3/*`
- ✅ `/object-clear/*`
- ✅ `/box-diff/*`
- ✅ `/gligen/*`

### Without GLIGEN Checkpoints

The application will start successfully but with warnings:
- ✅ `/sam3/*` - Available
- ✅ `/object-clear/*` - Available
- ✅ `/box-diff/*` - Available
- ⚠️ `/gligen/*` - Will return errors

You'll see warnings in the logs:
```
WARNING - GLIGEN generation checkpoint not found at ...
WARNING - Please download from: https://huggingface.co/gligen/gligen-generation-text-box
```

## HuggingFace Cache Location

Auto-loaded models are stored in the HuggingFace cache:
- Linux/Mac: `~/.cache/huggingface/`
- Windows: `%USERPROFILE%\.cache\huggingface\`

To clear the cache:
```bash
rm -rf ~/.cache/huggingface/hub/
```

## Troubleshooting

### GLIGEN Models Not Loading

1. **Check file names**:
   ```bash
   ls app/external/GLIGEN/gligen_checkpoints/
   ```
   Files must be named exactly:
   - `checkpoint_generation_text.pth`
   - `checkpoint_inpainting_text.pth`

2. **Verify file integrity**:
   ```bash
   # Check file sizes (should be ~4GB each)
   du -sh app/external/GLIGEN/gligen_checkpoints/*.pth
   ```

3. **Re-download if corrupted**:
   ```bash
   rm app/external/GLIGEN/gligen_checkpoints/*.pth
   # Then re-download using commands above
   ```

### Out of Memory

If you encounter OOM errors:

1. **Reduce batch size** (already set to 1 by default)
2. **Use CPU instead of GPU** for testing
3. **Don't load all models** - comment out models you don't need in `app/lifespan.py`
4. **Enable model offloading** (requires code modification)

### Disk Space Issues

To save space, you can:
1. Only download GLIGEN models you need (generation OR inpainting, not both)
2. Clear HuggingFace cache for unused models
3. Use lower precision (fp16) models where available

## Additional Model Variants

GLIGEN supports other modalities that can be added:
- Image grounding: `gligen-generation-text-image-box`
- Keypoint grounding: `gligen-generation-keypoint`
- Edge maps: `gligen-generation-canny`, `gligen-generation-hed`
- Depth maps: `gligen-generation-depth`
- Normal maps: `gligen-generation-normal`
- Semantic maps: `gligen-generation-sem`

See [GLIGEN_USAGE.md](GLIGEN_USAGE.md) for implementation details.

## References

- [GLIGEN HuggingFace](https://huggingface.co/gligen)
- [GLIGEN GitHub](https://github.com/gligen/GLIGEN)
- [GLIGEN Paper](https://arxiv.org/abs/2301.07093)
