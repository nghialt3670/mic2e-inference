# Speed & Memory Optimization Guide

This document describes the aggressive optimizations applied to prioritize speed and memory efficiency over result quality.

## Applied Optimizations

### 1. ✅ Reduced Precision (FP16)
- **Change**: All models use `torch.float16` on CUDA
- **Impact**: 2x memory reduction, 1.5-2x speed improvement
- **Quality Loss**: Minimal to moderate

### 2. ✅ Reduced Inference Steps
- **Change**: 20 → 8 steps for ObjectClear
- **Impact**: 2.5x faster generation
- **Quality Loss**: Moderate - less refined results

### 3. ✅ Lower Resolution
- **Change**: 512 → 384 pixels (short side)
- **Impact**: ~44% memory reduction, ~30% faster processing
- **Quality Loss**: Moderate - less detail

### 4. ✅ Lower Guidance Scale
- **Change**: 2.5 → 2.0
- **Impact**: Slightly faster
- **Quality Loss**: Minimal - less prompt adherence

### 5. ✅ Faster Resampling
- **Change**: BICUBIC → BILINEAR for image resizing
- **Impact**: Faster preprocessing
- **Quality Loss**: Minimal

### 6. ✅ Memory Efficient Attention
- **Feature**: Enabled xformers memory efficient attention
- **Impact**: ~20% memory reduction with minimal speed impact
- **Quality Loss**: None
- **Requirement**: Install with `uv pip install xformers` (added to dependencies)

### 7. ✅ Attention Slicing
- **Feature**: Enabled as fallback if xformers unavailable
- **Impact**: ~30% memory reduction, slight speed decrease
- **Quality Loss**: None

### 8. ✅ Low CPU Memory Usage During Loading
- **Feature**: Enabled for all models
- **Impact**: Faster loading, less CPU RAM during model loading
- **Quality Loss**: None

## Further Optimization Options

### Ultra-Aggressive (Not Yet Implemented)

If you want even more speed/memory savings, you can add:

#### 1. **4-Step Inference**
```python
# In object_clear_service_impl.py
num_inference_steps: int = 4  # Currently 8
```
- Impact: 2x faster than current
- Quality: Significant loss, very rough results

#### 2. **Lower Resolution (256px)**
```python
# In object_clear_service_impl.py
image_resized = resize_by_short_side(image, 256, resample=Image.BILINEAR)
mask_resized = resize_by_short_side(mask, 256, resample=Image.NEAREST)
```
- Impact: Another 2x memory reduction
- Quality: Significant loss, blurry results

#### 3. **8-bit Quantization** (Requires bitsandbytes)
```python
# In lifespan.py, add to pipeline loading:
load_in_8bit=True
```
- Impact: Another 2x memory reduction
- Quality: Moderate loss
- Requirement: `uv pip install bitsandbytes`

#### 4. **Model CPU Offloading**
```python
# In lifespan.py, after pipeline creation:
app.state.object_clear_pipeline.enable_model_cpu_offload()
```
- Impact: 3-4x memory reduction (GPU), much slower inference
- Quality: None (just slower)

#### 5. **Sequential CPU Offload** (Most Aggressive)
```python
# In lifespan.py, after pipeline creation:
app.state.object_clear_pipeline.enable_sequential_cpu_offload()
```
- Impact: Minimal GPU memory (~2GB), very slow
- Quality: None (just much slower)

#### 6. **Disable Guidance (CFG)**
```python
# In object_clear_service_impl.py
guidance_scale: float = 1.0  # Currently 2.0
```
- Impact: 2x faster (no double forward pass)
- Quality: Significant - poor prompt following

## Current Performance Expectations

With the applied optimizations:
- **Speed**: ~2.5-3x faster than original
- **Memory**: ~60-70% of original usage
- **Quality**: Acceptable for most use cases, noticeably lower than original

## Installation

To use xformers (recommended):
```bash
uv pip install xformers
```

## Monitoring

Check logs for optimization status:
- `Enabled xformers memory efficient attention` - xformers is active
- `Could not enable xformers` - falling back to attention slicing

## Rollback

To restore quality, reverse these changes in:
1. `app/lifespan.py` - Remove optimization flags
2. `app/services/impl/object_clear_service_impl.py`:
   - `num_inference_steps: int = 20`
   - `guidance_scale: float = 2.5`
   - Resolution: `512`
   - Resampling: `Image.BICUBIC`
