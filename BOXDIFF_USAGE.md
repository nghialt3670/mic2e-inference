# BoxDiff API Usage

## Overview
BoxDiff is a text-to-image synthesis model with training-free box-constrained diffusion. It allows you to control the spatial layout of generated objects using bounding boxes.

## Endpoint
`POST /box-diff/generate`

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt` | string | Yes | Text prompt for image generation |
| `token_indices` | JSON string | Yes | Array of token indices to spatially control (e.g., `"[2, 4]"`) |
| `bbox` | JSON string | Yes | Array of bounding boxes in format `[[x1,y1,x2,y2], ...]` |
| `seed` | integer | No | Random seed for reproducibility (default: 42) |

## How to Find Token Indices

Token indices correspond to the position of words in your prompt that you want to control spatially. For example:

```
Prompt: "A rabbit wearing sunglasses looks very proud"
Tokens: [<start>, "a", "rabbit", "wearing", "sunglasses", "looks", "very", "proud", <end>]
Indices: [0,       1,   2,        3,        4,             5,       6,     7,       8]
```

To control "rabbit" (index 2) and "sunglasses" (index 4), use `token_indices: "[2, 4]"`.

## Bounding Box Format

Bounding boxes are specified in pixel coordinates `[x1, y1, x2, y2]` where:
- `(x1, y1)` is the top-left corner
- `(x2, y2)` is the bottom-right corner
- Image size is 512x512 pixels by default

## Example Request

### Using cURL

```bash
curl -X POST "http://localhost:8000/box-diff/generate" \
  -F "prompt=A rabbit wearing sunglasses looks very proud" \
  -F "token_indices=[2, 4]" \
  -F "bbox=[[67,87,366,512],[66,130,364,262]]" \
  -F "seed=42" \
  --output result.png
```

### Using Python

```python
import requests

url = "http://localhost:8000/box-diff/generate"

# Example 1: Rabbit with sunglasses
data = {
    "prompt": "A rabbit wearing sunglasses looks very proud",
    "token_indices": "[2, 4]",  # rabbit, sunglasses
    "bbox": "[[67,87,366,512],[66,130,364,262]]",
    "seed": 42
}

response = requests.post(url, data=data)
with open("rabbit.png", "wb") as f:
    f.write(response.content)

# Example 2: Complex scene with multiple objects
data = {
    "prompt": "as the aurora lights up the sky, a herd of reindeer leisurely wanders on the grassy meadow, admiring the breathtaking view, a serene lake quietly reflects the magnificent display, and in the distance, a snow-capped mountain stands majestically, fantasy, 8k, highly detailed",
    "token_indices": "[3, 12, 21, 30, 46]",  # aurora, reindeer, meadow, lake, mountain
    "bbox": "[[1,3,512,202],[75,344,421,495],[1,327,508,507],[2,217,507,341],[1,135,509,242]]",
    "seed": 1
}

response = requests.post(url, data=data)
with open("landscape.png", "wb") as f:
    f.write(response.content)
```

## Response

Returns a PNG image (512x512 pixels) with the generated content.

## Notes

- Each token index must have a corresponding bounding box
- The number of token_indices must equal the number of bboxes
- Generation takes approximately 20-50 seconds depending on hardware
- GPU is highly recommended for acceptable performance
- The pipeline uses Stable Diffusion v1.4 as the base model

## Configuration

Default generation parameters:
- Image size: 512x512
- Inference steps: 50
- Guidance scale: 7.5
- P (attention control): 0.2
- L (corner constraint): 1
- Refinement: enabled

These parameters are optimized for the BoxDiff algorithm and generally don't need adjustment.
