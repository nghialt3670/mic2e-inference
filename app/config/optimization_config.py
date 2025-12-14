"""
Optimization configuration for speed/memory tradeoffs.
Adjust these values based on your performance requirements.
"""

from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    """
    Configuration for model optimization settings.
    
    Lower values = faster but lower quality
    Higher values = slower but higher quality
    """
    
    # ObjectClear optimization settings
    object_clear_num_inference_steps: int = 8  # Default: 20, Min: 4, Aggressive: 4-6, Balanced: 8-12
    object_clear_guidance_scale: float = 2.0  # Default: 2.5, Min: 1.0, Aggressive: 1.5-2.0
    object_clear_resolution: int = 384  # Default: 512, Aggressive: 256-384, Balanced: 384-448
    
    # Flux optimization settings
    flux_num_inference_steps: int = 4  # FLUX.1-schnell optimized for 4 steps
    flux_guidance_scale: float = 0.0  # schnell doesn't need guidance
    flux_height: int = 512  # Default: 1024, Aggressive: 512, Balanced: 768
    flux_width: int = 512  # Default: 1024, Aggressive: 512, Balanced: 768
    
    # Memory optimization flags
    enable_attention_slicing: bool = True  # Reduces memory, slight speed cost
    enable_xformers: bool = True  # Reduces memory, requires xformers package
    enable_model_cpu_offload: bool = False  # Huge memory savings, much slower
    enable_sequential_cpu_offload: bool = False  # Maximum memory savings, extremely slow
    
    # Model precision
    use_fp16: bool = True  # Use float16 for 2x memory reduction
    use_8bit_quantization: bool = False  # Requires bitsandbytes, 2x memory reduction
    use_4bit_quantization: bool = False  # Requires bitsandbytes, 4x memory reduction
    
    # Loading optimization
    low_cpu_mem_usage: bool = True  # Faster loading, less CPU RAM


# Preset configurations

ULTRA_FAST = OptimizationConfig(
    object_clear_num_inference_steps=4,
    object_clear_guidance_scale=1.5,
    object_clear_resolution=256,
    flux_height=512,
    flux_width=512,
    enable_attention_slicing=True,
    enable_xformers=True,
    use_fp16=True,
)

BALANCED = OptimizationConfig(
    object_clear_num_inference_steps=8,
    object_clear_guidance_scale=2.0,
    object_clear_resolution=384,
    flux_height=512,
    flux_width=512,
    enable_attention_slicing=True,
    enable_xformers=True,
    use_fp16=True,
)

QUALITY_FOCUSED = OptimizationConfig(
    object_clear_num_inference_steps=12,
    object_clear_guidance_scale=2.5,
    object_clear_resolution=448,
    flux_height=768,
    flux_width=768,
    enable_attention_slicing=True,
    enable_xformers=True,
    use_fp16=True,
)

LOW_MEMORY = OptimizationConfig(
    object_clear_num_inference_steps=6,
    object_clear_guidance_scale=1.8,
    object_clear_resolution=320,
    flux_height=512,
    flux_width=512,
    enable_attention_slicing=True,
    enable_xformers=True,
    enable_model_cpu_offload=True,  # Trade speed for memory
    use_fp16=True,
)

# Current active configuration (BALANCED is applied in the codebase)
ACTIVE_CONFIG = BALANCED
