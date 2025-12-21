# Stable Diffusion Inpainting API Usage

This guide covers the Stable Diffusion Inpainting API endpoints for inpainting images with masks.

## Overview

The SD Inpainting service uses the official `runwayml/stable-diffusion-inpainting` model to fill in or modify specific areas of an image based on a text prompt and a binary mask.

## Setup

### Environment Configuration

Enable the SD Inpaint model in your `.env` file:

```bash
LOAD_SD_INPAINT=true
```

### Docker Configuration

```yaml
environment:
  - LOAD_SD_INPAINT=true
```

## API Endpoints

### POST `/sd-inpaint/inpaint`

Inpaint an image using a binary mask.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `image` | File | Yes | - | Input image to inpaint (JPEG, PNG, etc.) |
| `mask` | File | Yes | - | Binary mask image (white = inpaint, black = keep) |
| `prompt` | String | Yes | - | Text description for the inpainted area |
| `negative_prompt` | String | No | "" | What to avoid in generation |
| `num_inference_steps` | Integer | No | 50 | Number of denoising steps (1-150) |
| `guidance_scale` | Float | No | 7.5 | How closely to follow prompt (1.0-20.0) |
| `seed` | Integer | No | 42 | Random seed for reproducibility |

**Mask Format:**
- White pixels (255) = areas to inpaint/regenerate
- Black pixels (0) = areas to keep unchanged
- Gray pixels (1-254) = partial inpainting (blending)

**Response:**
- Content-Type: `image/png`
- The inpainted image

**Example using cURL:**

```bash
curl -X POST "http://localhost:8000/sd-inpaint/inpaint" \
  -F "image=@input.jpg" \
  -F "mask=@mask.png" \
  -F "prompt=a beautiful red rose" \
  -F "negative_prompt=blurry, low quality" \
  -F "num_inference_steps=50" \
  -F "guidance_scale=7.5" \
  -F "seed=42" \
  --output result.png
```

**Example using Python:**

```python
import requests

url = "http://localhost:8000/sd-inpaint/inpaint"

with open("input.jpg", "rb") as img_file, open("mask.png", "rb") as mask_file:
    files = {
        "image": img_file,
        "mask": mask_file,
    }
    data = {
        "prompt": "a beautiful red rose",
        "negative_prompt": "blurry, low quality",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "seed": 42,
    }
    
    response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        with open("result.png", "wb") as f:
            f.write(response.content)
        print("Inpainting successful!")
    else:
        print(f"Error: {response.json()}")
```

**Example using JavaScript/TypeScript:**

```typescript
const formData = new FormData();
formData.append('image', imageFile);
formData.append('mask', maskFile);
formData.append('prompt', 'a beautiful red rose');
formData.append('negative_prompt', 'blurry, low quality');
formData.append('num_inference_steps', '50');
formData.append('guidance_scale', '7.5');
formData.append('seed', '42');

const response = await fetch('http://localhost:8000/sd-inpaint/inpaint', {
  method: 'POST',
  body: formData,
});

if (response.ok) {
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  // Use the URL to display or download the image
}
```

### GET `/sd-inpaint/health`

Check if the SD Inpaint service is available.

**Response:**
```json
{
  "status": "healthy",
  "service": "stable-diffusion-inpaint",
  "model": "runwayml/stable-diffusion-inpainting"
}
```

## Creating Masks

### Using Python/PIL

```python
from PIL import Image, ImageDraw
import numpy as np

# Create a blank white mask (inpaint everything)
mask = Image.new('RGB', (512, 512), color='white')

# Or create a specific region mask
mask = Image.new('RGB', (512, 512), color='black')
draw = ImageDraw.Draw(mask)

# Draw white rectangle (area to inpaint)
draw.rectangle([100, 100, 400, 400], fill='white')

mask.save('mask.png')
```

### Using GIMP/Photoshop

1. Open your image
2. Create a new layer
3. Paint white on areas you want to inpaint
4. Paint black on areas you want to keep
5. Export as PNG (flatten if needed)

### Using OpenCV

```python
import cv2
import numpy as np

# Load image
image = cv2.imread('input.jpg')
height, width = image.shape[:2]

# Create black mask
mask = np.zeros((height, width, 3), dtype=np.uint8)

# Define region to inpaint (white)
cv2.rectangle(mask, (100, 100), (400, 400), (255, 255, 255), -1)

# Save mask
cv2.imwrite('mask.png', mask)
```

## Use Cases

### 1. Object Removal
Remove unwanted objects from photos:

```bash
# Mask the object in white, prompt for background
prompt: "empty background, natural scene"
negative_prompt: "object, person, watermark"
```

### 2. Object Replacement
Replace one object with another:

```bash
# Mask the object, describe replacement
prompt: "a red sports car"
negative_prompt: "old car, damaged"
```

### 3. Image Extension
Extend images beyond their borders:

```bash
# Mask the edge areas
prompt: "continuation of the scene, seamless"
negative_prompt: "cut off, border, edge"
```

### 4. Face/Detail Modification
Modify specific details:

```bash
# Mask the face area
prompt: "smiling face, happy expression"
negative_prompt: "sad, angry, distorted"
```

## Parameters Guide

### `num_inference_steps`
- **Lower (20-30)**: Faster, less detailed
- **Medium (40-60)**: Balanced quality and speed
- **Higher (70-150)**: Better quality, slower

### `guidance_scale`
- **Lower (3-5)**: More creative, less prompt adherence
- **Medium (7-9)**: Balanced creativity and control
- **Higher (10-15)**: Strict prompt following, may be less natural

### `seed`
- Use the same seed for reproducible results
- Change seed to get different variations with the same prompt

## Tips for Best Results

1. **High-Quality Masks**
   - Clean edges for better blending
   - Expand mask slightly beyond object boundaries
   - Use feathered edges for natural transitions

2. **Effective Prompts**
   - Be specific about desired content
   - Include style descriptors (e.g., "photorealistic", "artistic")
   - Describe lighting and colors

3. **Negative Prompts**
   - List unwanted artifacts: "blurry, distorted, low quality"
   - Mention objects to avoid
   - Include style to avoid: "cartoon, painting, sketch"

4. **Image Resolution**
   - Works best with 512x512 images (model's native resolution)
   - Larger images will be processed but may take longer
   - Consider downscaling very large images

## Error Handling

### Common Errors

**400 Bad Request - Invalid image file**
- Ensure image is a valid format (JPEG, PNG, etc.)
- Check file isn't corrupted

**400 Bad Request - Image and mask size mismatch**
- Mask must be same dimensions as input image
- Resize mask to match image size

**500 Internal Server Error**
- Check server logs for details
- May indicate out of memory (reduce image size)

## Performance

### Approximate Processing Times

| Image Size | Device | Inference Steps | Time |
|------------|--------|-----------------|------|
| 512x512 | CPU | 50 | ~2-5 min |
| 512x512 | GPU (CUDA) | 50 | ~5-10 sec |
| 512x512 | MPS (Apple Silicon) | 50 | ~15-30 sec |
| 1024x1024 | GPU (CUDA) | 50 | ~15-30 sec |

### Memory Requirements

- **CPU/MPS**: ~8-12 GB RAM
- **CUDA**: ~4-6 GB VRAM

## Model Information

- **Model**: `runwayml/stable-diffusion-inpainting`
- **Base**: Stable Diffusion v1.5
- **Trained**: Specifically fine-tuned for inpainting tasks
- **License**: CreativeML Open RAIL-M License

## Additional Resources

- [Stable Diffusion Inpainting Documentation](https://huggingface.co/runwayml/stable-diffusion-inpainting)
- [Diffusers Library](https://huggingface.co/docs/diffusers/index)
- [Prompt Engineering Guide](https://huggingface.co/docs/diffusers/using-diffusers/write_prompts)
