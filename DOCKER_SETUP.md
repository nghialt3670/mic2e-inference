# Docker Setup Guide

## Quick Start

1. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and set your HUGGINGFACE_TOKEN
   ```

2. **Run with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

3. **Check logs:**
   ```bash
   docker-compose logs -f
   ```

The service will automatically:
- Download required models from HuggingFace
- Download GLIGEN checkpoints if not present
- Cache everything for subsequent runs

## Configuration

### Environment Variables

Set these in your `.env` file or `docker-compose.yml`:

```bash
# Required
HUGGINGFACE_TOKEN=your_token_here

# Model Selection
LOAD_GLIGEN=true
LOAD_SAM3=false
LOAD_OBJECT_CLEAR=false
LOAD_BOX_DIFF=false
LOAD_FLUX=false

# GLIGEN Checkpoint Configuration
GLIGEN_AUTO_DOWNLOAD_GENERATION=true   # Auto-download generation model
GLIGEN_AUTO_DOWNLOAD_INPAINTING=true   # Auto-download inpainting model
GLIGEN_CHECKPOINT_DIR=/custom/path     # Optional custom location
```

### Volume Mounting

The docker-compose configuration includes two persistent volumes:

#### 1. HuggingFace Cache (`huggingface_cache`)
```yaml
volumes:
  - huggingface_cache:/root/.cache/huggingface
```
- Stores all HuggingFace models (SAM3, ObjectClear, BoxDiff, Flux, etc.)
- Persists between container restarts
- Prevents re-downloading models

#### 2. GLIGEN Checkpoints (`gligen_checkpoints`)
```yaml
volumes:
  - gligen_checkpoints:/app/app/external/GLIGEN/gligen_checkpoints
```
- Stores GLIGEN model checkpoints
- Auto-downloaded on first run (generation and/or inpainting)
- Each model downloads separately, only if enabled
- Persists between deployments

### Custom Checkpoint Location

To use a custom directory for GLIGEN checkpoints:

```yaml
services:
  mic2e-inference:
    environment:
      - GLIGEN_CHECKPOINT_DIR=/app/checkpoints/gligen
    volumes:
      - ./local/checkpoints:/app/checkpoints/gligen
```

## Automatic Checkpoint Download

### How It Works

1. **On first startup**:
   - If `GLIGEN_AUTO_DOWNLOAD_GENERATION=true` and generation checkpoint not found:
     - Downloads from HuggingFace Hub: `gligen/gligen-generation-text-box` (~2GB)
   - If `GLIGEN_AUTO_DOWNLOAD_INPAINTING=true` and inpainting checkpoint not found:
     - Downloads from HuggingFace Hub: `gligen/gligen-inpainting-text-box` (~2GB)
   - Each model downloads independently, only if needed

2. **On subsequent runs**:
   - Checkpoints are already in the mounted volume
   - No download occurs (instant startup)

**Tip:** If you only need generation OR inpainting, disable the other to save ~2GB and download time!

### Manual Download (Alternative)

If you prefer to download manually:

1. Disable auto-download in your environment:
   ```bash
   GLIGEN_AUTO_DOWNLOAD_GENERATION=false
   GLIGEN_AUTO_DOWNLOAD_INPAINTING=false
   ```

2. Download checkpoints:
   ```bash
   # Create checkpoint directory
   mkdir -p ./gligen_checkpoints

   # Download generation checkpoint
   wget https://huggingface.co/gligen/gligen-generation-text-box/resolve/main/diffusion_pytorch_model.bin \
     -O ./gligen_checkpoints/checkpoint_generation_text.pth

   # Download inpainting checkpoint
   wget https://huggingface.co/gligen/gligen-inpainting-text-box/resolve/main/diffusion_pytorch_model.bin \
     -O ./gligen_checkpoints/checkpoint_inpainting_text.pth
   ```

3. Mount the directory:
   ```yaml
   volumes:
     - ./gligen_checkpoints:/app/app/external/GLIGEN/gligen_checkpoints
   ```

## GPU Support

### NVIDIA GPU (CUDA)

Uncomment the GPU configuration in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Requirements:
- NVIDIA Docker runtime installed
- NVIDIA GPU drivers

### Apple Silicon (MPS)

MPS is not directly supported in Docker on macOS. Use native installation instead.

## Production Deployment

### Using Named Volumes (Recommended)

```yaml
volumes:
  huggingface_cache:
    driver: local
  gligen_checkpoints:
    driver: local
```

Benefits:
- Managed by Docker
- Persist across deployments
- Easy backup with `docker volume`

### Using Bind Mounts (Alternative)

```yaml
volumes:
  - /srv/model-cache/huggingface:/root/.cache/huggingface
  - /srv/model-cache/gligen:/app/app/external/GLIGEN/gligen_checkpoints
```

Benefits:
- Easy to access from host
- Can be shared across services
- Direct file system access

## Troubleshooting

### Checkpoints Not Downloading

1. **Check HuggingFace token:**
   ```bash
   docker-compose exec mic2e-inference env | grep HUGGINGFACE_TOKEN
   ```

2. **Check logs:**
   ```bash
   docker-compose logs mic2e-inference | grep -i gligen
   ```

3. **Verify volume mounts:**
   ```bash
   docker-compose exec mic2e-inference ls -la /app/app/external/GLIGEN/gligen_checkpoints
   ```

### Out of Disk Space

Check volume sizes:
```bash
docker system df -v
```

Clean up if needed:
```bash
docker volume prune
```

### Slow Download

The first run downloads several GB of data:
- HuggingFace models: ~2-4 GB per model
- GLIGEN checkpoints: ~2-4 GB

Be patient on first startup (10-30 minutes depending on connection).

## Volume Management

### Backup Volumes

```bash
# Backup HuggingFace cache
docker run --rm -v mic2e-inference_huggingface_cache:/data \
  -v $(pwd):/backup alpine tar czf /backup/hf_cache.tar.gz /data

# Backup GLIGEN checkpoints
docker run --rm -v mic2e-inference_gligen_checkpoints:/data \
  -v $(pwd):/backup alpine tar czf /backup/gligen_checkpoints.tar.gz /data
```

### Restore Volumes

```bash
# Restore HuggingFace cache
docker run --rm -v mic2e-inference_huggingface_cache:/data \
  -v $(pwd):/backup alpine tar xzf /backup/hf_cache.tar.gz -C /

# Restore GLIGEN checkpoints
docker run --rm -v mic2e-inference_gligen_checkpoints:/data \
  -v $(pwd):/backup alpine tar xzf /backup/gligen_checkpoints.tar.gz -C /
```

### Clean Volumes

```bash
# Remove all volumes (will re-download on next start)
docker-compose down -v
```

## Performance Tips

1. **Use SSD for volumes** - Significantly faster model loading
2. **Allocate enough memory** - At least 8GB for GLIGEN
3. **Use GPU** - 10-100x faster inference
4. **Pre-download models** - Run once with all models, then disable unused ones
