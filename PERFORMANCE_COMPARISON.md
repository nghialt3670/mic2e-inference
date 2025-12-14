# Performance Comparison

## Before vs After Optimizations

### ObjectClear Pipeline

| Metric | Original | Current (Balanced) | Ultra-Fast | Low Memory |
|--------|----------|-------------------|------------|------------|
| **Inference Steps** | 20 | 8 | 4 | 6 |
| **Resolution** | 512px | 384px | 256px | 320px |
| **Guidance Scale** | 2.5 | 2.0 | 1.5 | 1.8 |
| **Precision** | FP32/FP16 | FP16 | FP16 | FP16 |
| **Est. Speed** | 1.0x (baseline) | 2.5-3.0x | 4-5x | 2.0-2.5x |
| **Est. VRAM** | 100% (baseline) | ~50% | ~35% | ~25% |
| **Quality** | 100% | ~75-80% | ~50-60% | ~65-70% |

### Flux Pipeline (when enabled)

| Metric | Original | Current | Ultra-Fast |
|--------|----------|---------|------------|
| **Inference Steps** | 4 (schnell) | 4 | 4 |
| **Resolution** | 1024x1024 | 512x512 | 512x512 |
| **Precision** | FP16 | FP16 | FP16 |
| **Est. Speed** | 1.0x | 4x | 4x |
| **Est. VRAM** | 100% | ~25% | ~25% |
| **Quality** | 100% | ~60-70% | ~60-70% |

### SAM3 Models

| Optimization | Applied | Impact |
|--------------|---------|--------|
| **FP16 Precision** | ✅ Yes | 2x memory reduction |
| **Low CPU Memory** | ✅ Yes | Faster loading |
| **No other changes** | - | Quality maintained |

## Detailed Performance Metrics

### Expected Inference Times (on RTX 3090)

**ObjectClear:**
- Original: ~8-10 seconds per image
- Current (Balanced): ~3-4 seconds per image ⚡
- Ultra-Fast: ~2-2.5 seconds per image ⚡⚡
- Low Memory: ~4-5 seconds per image

**Flux (512x512):**
- Original (1024x1024): ~6-8 seconds
- Current (512x512): ~1.5-2 seconds ⚡⚡⚡

### VRAM Usage (Approximate)

**Single Model Loaded:**
- ObjectClear Original: ~6-8 GB
- ObjectClear Current: ~3-4 GB 💾
- ObjectClear Ultra-Fast: ~2-3 GB 💾💾
- ObjectClear Low Memory: ~1.5-2.5 GB 💾💾💾

**All Models Loaded:**
- Original: ~12-15 GB
- Current: ~6-8 GB 💾💾
- With CPU Offload: ~3-4 GB 💾💾💾

### CPU RAM During Model Loading

- Original: ~16-20 GB peak
- Current (with low_cpu_mem_usage): ~8-10 GB peak 💾

## Quality Impact Examples

### ObjectClear Inpainting

**High Inference Steps (20)**
- Smooth transitions
- Fine details preserved
- Natural looking results

**Medium Inference Steps (8-12)** ⭐ Current
- Good transitions
- Most details preserved
- Acceptable for production

**Low Inference Steps (4-6)**
- Rougher transitions
- Some artifacts
- Fast prototyping only

### Resolution Impact

**512px** (Original)
- Sharp details
- Good for high-res outputs

**384px** ⭐ Current
- Balanced quality/speed
- Good for most use cases

**256px**
- Softer results
- Fast iteration/testing

## Recommendations

### For Development/Testing
Use **ULTRA_FAST** preset:
```python
from app.config.optimization_config import ULTRA_FAST
# Apply settings from ULTRA_FAST
```

### For Production (Balanced)
Use **BALANCED** preset (current default):
```python
from app.config.optimization_config import BALANCED
# Already applied in codebase
```

### For Quality-Focused Production
Use **QUALITY_FOCUSED** preset:
```python
from app.config.optimization_config import QUALITY_FOCUSED
# Adjust values in service implementations
```

### For Low VRAM GPUs (<6GB)
Use **LOW_MEMORY** preset:
```python
from app.config.optimization_config import LOW_MEMORY
# Enable CPU offloading in lifespan.py
```

## Additional Optimization Opportunities

### 1. Batch Processing
Process multiple images in parallel on different GPU streams (not yet implemented)
- **Impact**: ~1.5-2x throughput
- **Complexity**: Medium

### 2. Model Quantization (bitsandbytes)
```python
load_in_8bit=True  # or load_in_4bit=True
```
- **Impact**: 2-4x memory reduction, 10-20% slower
- **Complexity**: Easy (add parameter)

### 3. Torch Compile (PyTorch 2.0+)
```python
torch.compile(pipeline.unet, mode="reduce-overhead")
```
- **Impact**: 10-30% faster after warmup
- **Complexity**: Easy (one line)

### 4. TensorRT Conversion
Convert models to TensorRT format
- **Impact**: 2-3x faster inference
- **Complexity**: High (requires conversion pipeline)

### 5. ONNX Runtime
Export models to ONNX format
- **Impact**: 1.5-2x faster on CPU, similar on GPU
- **Complexity**: Medium

## Monitoring Performance

### Log Analysis
Check logs for optimization status:
```
INFO: Enabled xformers memory efficient attention
INFO: ObjectClear pipeline loaded with optimizations
```

### Memory Profiling
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

### Speed Profiling
```python
import time
start = time.time()
# ... inference ...
print(f"Inference took: {time.time() - start:.2f}s")
```

## Troubleshooting

### Out of Memory Errors
1. Reduce resolution further (256px)
2. Reduce inference steps (4-6)
3. Enable CPU offloading
4. Enable 8-bit quantization

### Slow Inference
1. Ensure xformers is installed and enabled
2. Check CUDA is available
3. Increase resolution/steps if too aggressive
4. Consider torch.compile for repeated inference

### Quality Issues
1. Increase inference steps (12-15)
2. Increase resolution (448-512)
3. Increase guidance scale (2.5-3.0)
4. Use QUALITY_FOCUSED preset
