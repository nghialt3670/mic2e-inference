"""Model manager for lazy loading and cleanup."""

import gc
import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, Optional

import torch
from fastapi import Request

from app.utils.device_utils import get_device

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages lazy loading and cleanup of models."""
    
    def __init__(self):
        self._loaded_models: Dict[str, Any] = {}
        self._loaders: Dict[str, Callable] = {}
        self._cleanup_functions: Dict[str, Callable] = {}
    
    def register_loader(
        self,
        model_name: str,
        loader: Callable,
        cleanup: Optional[Callable[[Any], None]] = None,
    ):
        """Register a model loader.
        
        Args:
            model_name: Name identifier for the model
            loader: Function that loads and returns the model
            cleanup: Optional function to clean up the model (takes model as arg)
        """
        self._loaders[model_name] = loader
        if cleanup:
            self._cleanup_functions[model_name] = cleanup
    
    def get_model(self, model_name: str, request: Request) -> Any:
        """Get a model, loading it if necessary.
        
        Args:
            model_name: Name identifier for the model
            request: FastAPI request object
            
        Returns:
            The loaded model
        """
        if model_name not in self._loaded_models:
            logger.info(f"Loading model on-demand: {model_name}")
            if model_name not in self._loaders:
                raise ValueError(f"No loader registered for model: {model_name}")
            
            loader = self._loaders[model_name]
            # Get the best available device at load time (not cached from startup)
            device = get_device()
            logger.info(f"Selected device for {model_name}: {device}")
            model = loader(device)
            self._loaded_models[model_name] = model
            logger.info(f"Model loaded: {model_name} on {device}")
        
        return self._loaded_models[model_name]
    
    def _cleanup_pipeline(self, pipeline):
        """Aggressively clean up a diffusers pipeline."""
        try:
            # List of common pipeline components to clean up
            components_to_clean = [
                'unet', 'vae', 'text_encoder', 'text_encoder_2', 
                'tokenizer', 'tokenizer_2', 'scheduler', 'safety_checker',
                'feature_extractor', 'image_encoder', 'controlnet',
                'transformer', 'prior', 'decoder', 'encoder'
            ]
            
            # Move all components to CPU and delete
            for component_name in components_to_clean:
                if hasattr(pipeline, component_name):
                    component = getattr(pipeline, component_name)
                    if component is not None:
                        try:
                            # Recursively clean up if it's a model
                            if hasattr(component, 'parameters') or hasattr(component, 'forward'):
                                self._cleanup_pytorch_model(component)
                            else:
                                # Move to CPU for other components (offload from GPU/MPS)
                                if hasattr(component, 'to'):
                                    component.to('cpu')
                                    # Sync to ensure move is complete
                                    if torch.cuda.is_available():
                                        torch.cuda.synchronize()
                                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                        try:
                                            torch.mps.synchronize()
                                        except:
                                            pass
                                elif hasattr(component, 'cpu'):
                                    component.cpu()
                            
                            # Delete the component
                            del component
                            setattr(pipeline, component_name, None)
                        except Exception as e:
                            logger.debug(f"Error cleaning up {component_name}: {e}")
            
            # Clear pipeline's internal state - iterate through all attributes
            if hasattr(pipeline, '__dict__'):
                attrs_to_delete = list(pipeline.__dict__.keys())
                for attr in attrs_to_delete:
                    try:
                        obj = getattr(pipeline, attr)
                        if obj is not None:
                            # If it's a model, clean it up properly
                            if hasattr(obj, 'parameters') or hasattr(obj, 'forward'):
                                self._cleanup_pytorch_model(obj)
                            elif hasattr(obj, 'to'):
                                try:
                                    obj.to('cpu')
                                    # Sync to ensure move is complete
                                    if torch.cuda.is_available():
                                        torch.cuda.synchronize()
                                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                        try:
                                            torch.mps.synchronize()
                                        except:
                                            pass
                                except:
                                    pass
                        delattr(pipeline, attr)
                    except Exception as e:
                        logger.debug(f"Error deleting pipeline attr {attr}: {e}")
            
            # Move entire pipeline to CPU as fallback
            try:
                if hasattr(pipeline, 'to'):
                    pipeline.to('cpu')
                    # Sync to ensure move is complete
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        try:
                            torch.mps.synchronize()
                        except:
                            pass
            except:
                pass
            
        except Exception as e:
            logger.debug(f"Error in pipeline cleanup: {e}")
    
    def _cleanup_pytorch_model(self, model):
        """Aggressively clean up a PyTorch model."""
        try:
            # Move to CPU first (offload from GPU/MPS)
            try:
                if hasattr(model, 'to'):
                    # Explicitly move to CPU to free GPU/MPS memory
                    model.to('cpu')
                    # Ensure all operations are complete
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        try:
                            torch.mps.synchronize()
                        except:
                            pass
                elif hasattr(model, 'cpu'):
                    model.cpu()
            except Exception as e:
                logger.debug(f"Error moving model to CPU: {e}")
            
            # Set to eval mode to disable gradients and detach from computation graph
            try:
                if hasattr(model, 'eval'):
                    model.eval()
                if hasattr(model, 'requires_grad_'):
                    model.requires_grad_(False)
                # Disable autograd tracking
                if hasattr(model, 'train'):
                    model.train(False)
            except:
                pass
            
            # Try to detach all parameters from computation graph
            try:
                if hasattr(model, 'parameters'):
                    for param in model.parameters(recurse=True):
                        if param is not None and hasattr(param, 'detach'):
                            try:
                                param.detach_()
                            except:
                                pass
            except:
                pass
            
            # Delete all parameters and buffers - this is critical for memory release
            try:
                # Synchronize device operations before cleanup
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    try:
                        torch.mps.synchronize()
                    except:
                        pass
                
                if hasattr(model, 'parameters'):
                    params = list(model.parameters(recurse=True))
                    for param in params:
                        if param is not None:
                            try:
                                # Clear gradient if exists (this frees memory)
                                if hasattr(param, 'grad') and param.grad is not None:
                                    param.grad = None
                                # Set data to None to release tensor memory
                                # Note: We can't directly delete param.data as it's a property
                                # but setting to None and deleting the param object should work
                                if hasattr(param, 'data'):
                                    try:
                                        # Try to clear the tensor by moving to CPU and setting to None
                                        if param.data is not None:
                                            param.data = param.data.cpu()
                                            param.data = None
                                    except:
                                        pass
                            except:
                                pass
                            finally:
                                # Delete the parameter object itself
                                del param
                
                if hasattr(model, 'buffers'):
                    buffers = list(model.buffers(recurse=True))
                    for buffer in buffers:
                        if buffer is not None:
                            try:
                                # Move buffer to CPU and clear
                                if hasattr(buffer, 'cpu'):
                                    buffer = buffer.cpu()
                                buffer = None
                            except:
                                pass
                            finally:
                                del buffer
            except Exception as e:
                logger.debug(f"Error deleting parameters/buffers: {e}")
            
            # Clear model's internal state
            try:
                if hasattr(model, '__dict__'):
                    attrs_to_delete = list(model.__dict__.keys())
                    for attr in attrs_to_delete:
                        try:
                            delattr(model, attr)
                        except:
                            pass
            except Exception as e:
                logger.debug(f"Error clearing model state: {e}")
            
        except Exception as e:
            logger.debug(f"Error in PyTorch model cleanup: {e}")
    
    def _get_memory_usage(self) -> str:
        """Get current memory usage for logging."""
        memory_info = []
        
        # CUDA memory (NVIDIA GPUs)
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                memory_info.append(f"CUDA: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            except:
                memory_info.append("CUDA: Unable to get memory info")
        
        # MPS memory (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                # MPS doesn't have direct memory query, but we can note it
                memory_info.append("MPS: Active")
            except:
                pass
        
        # CPU memory (using psutil if available, otherwise just note)
        try:
            import psutil
            process = psutil.Process()
            cpu_mem = process.memory_info().rss / 1024**3  # GB
            memory_info.append(f"CPU RAM: {cpu_mem:.2f}GB")
        except ImportError:
            # psutil not available, skip CPU memory info
            pass
        except Exception:
            pass
        
        return ", ".join(memory_info) if memory_info else "Memory info unavailable"
    
    def cleanup_model(self, model_name: str):
        """Clean up a model and free memory aggressively.
        
        Args:
            model_name: Name identifier for the model
        """
        if model_name not in self._loaded_models:
            return
        
        memory_before = self._get_memory_usage()
        logger.info(f"Cleaning up model: {model_name} ({memory_before})")
        model = self._loaded_models.pop(model_name, None)
        
        if model is None:
            return
        
        try:
            # Call custom cleanup if provided
            if model_name in self._cleanup_functions:
                try:
                    self._cleanup_functions[model_name](model)
                except Exception as e:
                    logger.debug(f"Error in custom cleanup for {model_name}: {e}")
            
            # Handle tuple/list of models (like GLIGEN, SAM3)
            if isinstance(model, (tuple, list)):
                for m in model:
                    if m is not None:
                        try:
                            # Check if it's a diffusers pipeline
                            if hasattr(m, 'unet') or hasattr(m, 'vae') or hasattr(m, 'text_encoder'):
                                self._cleanup_pipeline(m)
                            # Check if it's a PyTorch model
                            elif hasattr(m, 'parameters') or hasattr(m, 'forward'):
                                self._cleanup_pytorch_model(m)
                            # Generic cleanup - move to CPU
                            else:
                                if hasattr(m, 'to'):
                                    m.to('cpu')
                                    # Sync to ensure move is complete
                                    if torch.cuda.is_available():
                                        torch.cuda.synchronize()
                                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                        try:
                                            torch.mps.synchronize()
                                        except:
                                            pass
                                elif hasattr(m, 'cpu'):
                                    m.cpu()
                        except Exception as e:
                            logger.debug(f"Error cleaning up model component: {e}")
                        finally:
                            del m
            else:
                # Single model object
                try:
                    # Check if it's a diffusers pipeline
                    if hasattr(model, 'unet') or hasattr(model, 'vae') or hasattr(model, 'text_encoder'):
                        self._cleanup_pipeline(model)
                    # Check if it's a PyTorch model
                    elif hasattr(model, 'parameters') or hasattr(model, 'forward'):
                        self._cleanup_pytorch_model(model)
                    # Generic cleanup - move to CPU
                    else:
                        if hasattr(model, 'to'):
                            model.to('cpu')
                            # Sync to ensure move is complete
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                try:
                                    torch.mps.synchronize()
                                except:
                                    pass
                        elif hasattr(model, 'cpu'):
                            model.cpu()
                except Exception as e:
                    logger.debug(f"Error cleaning up model: {e}")
            
            # Delete the model reference
            del model
            
            # Synchronize all device operations before cleanup
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS synchronization
                try:
                    torch.mps.synchronize()
                except:
                    pass
            
            # Force garbage collection multiple times to ensure cleanup
            # This is critical for CPU memory release
            # Collect all generations (0, 1, 2) to ensure everything is cleaned
            for generation in range(3):
                collected = gc.collect(generation)
                if collected > 0:
                    logger.debug(f"GC generation {generation}: collected {collected} objects")
            
            # Additional full collection passes
            for _ in range(2):
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"GC full pass: collected {collected} objects")
            
            # Clear GPU cache if using CUDA - do this multiple times
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # One more round to be sure
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
            
            # Clear MPS cache if using Apple Silicon
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    # MPS doesn't have empty_cache, but we can try to clear via synchronization
                    torch.mps.synchronize()
                    # Force garbage collection again after MPS sync
                    for _ in range(2):
                        gc.collect()
                except Exception as e:
                    logger.debug(f"Error clearing MPS cache: {e}")
            
            # Final aggressive CPU memory cleanup
            # Clear Python's internal caches and try to force allocator to release memory
            import sys
            try:
                # Clear any cached imports that might hold references
                # This is more aggressive but helps ensure memory is freed
                for _ in range(2):
                    gc.collect()
                
                # Try to force Python's memory allocator to release memory back to OS
                # On macOS, this is limited but we can try
                try:
                    # Force collection of all generations one more time
                    for generation in [2, 1, 0]:
                        gc.collect(generation)
                except:
                    pass
                
                # Clear any weak references
                gc.collect()
            except:
                pass
            
            # Additional attempt to free memory on macOS
            # Note: Python's allocator may still hold memory pools, but OS has reclaimed it
            # The memory pressure graph is the accurate indicator, not process memory count
            try:
                # One final comprehensive collection
                final_collected = gc.collect()
                if final_collected > 0:
                    logger.debug(f"Final GC: collected {final_collected} objects")
            except:
                pass
            
            memory_after = self._get_memory_usage()
            logger.info(f"Model cleaned up and memory freed: {model_name} ({memory_after})")
            logger.debug(
                "Note: Process memory may remain high in Activity Monitor due to Python's "
                "memory allocator pools. Memory pressure graph is the accurate indicator."
            )
        except Exception as e:
            logger.warning(f"Error during cleanup of {model_name}: {e}", exc_info=True)
            # Still try to clear cache even if cleanup failed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    torch.mps.synchronize()
                except:
                    pass
            # Force garbage collection
            for _ in range(2):
                gc.collect()
    
    def cleanup_all(self):
        """Clean up all loaded models."""
        model_names = list(self._loaded_models.keys())
        for model_name in model_names:
            self.cleanup_model(model_name)


# Global model manager instance
_model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    return _model_manager


@asynccontextmanager
async def model_context(model_name: str, request: Request):
    """Context manager for model loading and cleanup.
    
    Usage:
        async with model_context("box_diff", request) as model:
            # Use model here
            pass
        # Model is automatically cleaned up after context exits
    """
    manager = get_model_manager()
    try:
        model = manager.get_model(model_name, request)
        yield model
    finally:
        manager.cleanup_model(model_name)

