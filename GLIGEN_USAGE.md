# GLIGEN API Usage

## Overview
GLIGEN (Grounded-Language-to-Image GENeration) is an open-set grounded text-to-image generation model that allows you to control the spatial layout of generated objects using bounding boxes with text descriptions.

## Setup

### Download Checkpoints

Before using GLIGEN, you need to download the model checkpoints:

```bash
# Create checkpoints directory
mkdir -p app/external/GLIGEN/gligen_checkpoints

# Download generation model (text-box)
wget https://huggingface.co/gligen/gligen-generation-text-box/resolve/main/diffusion_pytorch_model.bin \
  -O app/external/GLIGEN/gligen_checkpoints/checkpoint_generation_text.pth

# Download inpainting model (text-box)
wget https://huggingface.co/gligen/gligen-inpainting-text-box/resolve/main/diffusion_pytorch_model.bin \
  -O app/external/GLIGEN/gligen_checkpoints/checkpoint_inpainting_text.pth
```

## Endpoints

### 1. Generate (`POST /gligen/generate`)

Generate an image with text-box grounding.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Overall text prompt for image generation |
| `phrases` | JSON string | Yes | Array of text phrases to ground (e.g., `'["a cat", "a dog"]'`) |
| `locations` | JSON string | Yes | Array of bounding boxes in normalized coordinates `[[x1,y1,x2,y2], ...]` where values are in range [0, 1] |
| `seed` | integer | No | Random seed for reproducibility (default: 42) |
| `alpha_type` | JSON string | No | Array `[stage0, stage1, stage2]` summing to 1.0, controls grounding strength (default: `[0.3, 0.0, 0.7]`) |

#### Alpha Type Explanation

The `alpha_type` parameter controls how the grounding strength changes during the diffusion process:
- **stage0**: Percentage of steps with full grounding (alpha=1)
- **stage1**: Percentage of steps with linear decay
- **stage2**: Percentage of steps with no grounding (alpha=0)

For example, `[0.3, 0.0, 0.7]` means:
- First 30% of steps: full grounding
- No decay phase
- Last 70% of steps: no grounding (allows natural image completion)

### 2. Inpaint (`POST /gligen/inpaint`)

Inpaint an image with text-box grounding.

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `image` | File | Yes | Input image file to inpaint |
| `prompt` | string | Yes | Text prompt for inpainting |
| `phrases` | JSON string | Yes | Array of text phrases for the masked regions |
| `locations` | JSON string | Yes | Array of bounding boxes (also used as inpainting masks) `[[x1,y1,x2,y2], ...]` |
| `seed` | integer | No | Random seed (default: 42) |

## Examples

### Example 1: Generate with Text-Box Grounding

```bash
curl -X POST "http://localhost:8000/gligen/generate" \
  -F "prompt=a teddy bear sitting next to a bird" \
  -F 'phrases=["a teddy bear", "a bird"]' \
  -F 'locations=[[0.0, 0.09, 0.33, 0.76], [0.55, 0.11, 1.0, 0.8]]' \
  -F "seed=42" \
  -F 'alpha_type=[0.3, 0.0, 0.7]' \
  --output teddy_bear_bird.png
```

### Example 2: Generate Simple Scene

```bash
curl -X POST "http://localhost:8000/gligen/generate" \
  -F "prompt=a cat and a dog in a garden" \
  -F 'phrases=["a cat", "a dog"]' \
  -F 'locations=[[0.1, 0.2, 0.4, 0.8], [0.6, 0.3, 0.9, 0.85]]' \
  -F "seed=123" \
  --output cat_dog_garden.png
```

### Example 3: Inpaint with Grounding

```bash
curl -X POST "http://localhost:8000/gligen/inpaint" \
  -F "image=@input_image.jpg" \
  -F "prompt=a corgi and a cake" \
  -F 'phrases=["corgi", "cake"]' \
  -F 'locations=[[0.25, 0.28, 0.42, 0.52], [0.14, 0.58, 0.58, 0.92]]' \
  -F "seed=42" \
  --output inpainted.png
```

### Python Example

```python
import requests

url = "http://localhost:8000/gligen/generate"

# Generate image with text-box grounding
data = {
    "prompt": "a beach scene with a surfboard and a umbrella",
    "phrases": '["a surfboard", "a umbrella"]',
    "locations": '[[0.1, 0.5, 0.4, 0.95], [0.6, 0.1, 0.95, 0.6]]',
    "seed": 42,
    "alpha_type": '[0.3, 0.0, 0.7]'
}

response = requests.post(url, data=data)
with open("beach_scene.png", "wb") as f:
    f.write(response.content)

print("Image generated successfully!")
```

### Python Inpainting Example

```python
import requests

url = "http://localhost:8000/gligen/inpaint"

# Inpaint with grounding
files = {
    "image": open("input.jpg", "rb")
}

data = {
    "prompt": "add a modern laptop and coffee mug on the desk",
    "phrases": '["laptop", "coffee mug"]',
    "locations": '[[0.2, 0.3, 0.6, 0.7], [0.65, 0.5, 0.85, 0.8]]',
    "seed": 42
}

response = requests.post(url, files=files, data=data)
with open("inpainted.png", "wb") as f:
    f.write(response.content)

print("Inpainting completed!")
```

## Understanding Bounding Box Coordinates

GLIGEN uses normalized coordinates where all values are between 0 and 1:

- `[0.0, 0.0]` = top-left corner
- `[1.0, 1.0]` = bottom-right corner
- Format: `[x1, y1, x2, y2]`
  - `x1`: left edge (0 = far left, 1 = far right)
  - `y1`: top edge (0 = top, 1 = bottom)
  - `x2`: right edge
  - `y2`: bottom edge

### Visual Example

```
(0.0, 0.0) ------------------- (1.0, 0.0)
     |                              |
     |    Box: [0.2, 0.3, 0.6, 0.7] |
     |         ┌──────────┐          |
     |         │          │          |
     |         │  Object  │          |
     |         └──────────┘          |
     |                              |
(0.0, 1.0) ------------------- (1.0, 1.0)
```

## Tips for Best Results

1. **Phrase Specificity**: Use specific, descriptive phrases (e.g., "a fluffy white cat" instead of just "cat")

2. **Box Placement**: Ensure bounding boxes don't overlap too much and have reasonable sizes

3. **Alpha Type Tuning**:
   - For stronger grounding: `[0.5, 0.0, 0.5]` or `[0.7, 0.0, 0.3]`
   - For more natural results: `[0.3, 0.0, 0.7]` (default)
   - For very subtle grounding: `[0.1, 0.2, 0.7]`

4. **Seed Control**: Use the same seed for reproducible results

5. **Inpainting**: The bounding boxes serve dual purpose - they define both the mask region and where to place the new content

## Response

Both endpoints return a PNG image (512x512 pixels) with the generated/inpainted content.

## Model Information

- **Base Model**: Stable Diffusion v1.4
- **Generation Steps**: 50 (PLMS sampler)
- **Guidance Scale**: 7.5
- **Output Resolution**: 512x512

## Notes

- Generation takes approximately 20-60 seconds depending on hardware
- GPU is highly recommended for acceptable performance
- The first request may take longer due to model initialization
- GLIGEN provides better control over object placement compared to text-only models

## Citation

```bibtex
@article{li2023gligen,
  title={GLIGEN: Open-Set Grounded Text-to-Image Generation},
  author={Li, Yuheng and Liu, Haotian and Wu, Qingyang and Mu, Fangzhou and Yang, Jianwei and Gao, Jianfeng and Li, Chunyuan and Lee, Yong Jae},
  journal={CVPR},
  year={2023}
}
```
