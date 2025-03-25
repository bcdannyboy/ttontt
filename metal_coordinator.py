"""
PyTorch Device Coordinator
--------------------------
Centralizes device selection logic for PyTorch ensuring proper GPU usage.
Follows priority order: CUDA > MPS > CPU
"""
import os
import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# Track initialization status
_initialized = False
_device = None
_dtype = None
_device_info = {}

def get_device_info() -> Dict[str, Any]:
    """Return detailed information about the selected device"""
    if not _initialized:
        initialize()
    return _device_info

def initialize() -> None:
    """Initialize PyTorch and select the best available device"""
    global _initialized, _device, _dtype, _device_info
    
    if _initialized:
        return
    
    import torch
    _device_info["torch_version"] = torch.__version__
    
    # Check for CUDA first (highest priority)
    if torch.cuda.is_available():
        _device = torch.device("cuda")
        _dtype = torch.float32  # Most stable for CUDA
        
        # Gather CUDA device information
        cuda_id = torch.cuda.current_device()
        _device_info.update({
            "device_type": "cuda",
            "device_name": torch.cuda.get_device_name(cuda_id),
            "compute_capability": torch.cuda.get_device_capability(cuda_id),
            "device_count": torch.cuda.device_count(),
            "memory_allocated": f"{torch.cuda.memory_allocated(cuda_id) / (1024**2):.2f} MB",
            "memory_reserved": f"{torch.cuda.memory_reserved(cuda_id) / (1024**2):.2f} MB",
        })
        
        # Set CUDA optimizations
        torch.backends.cudnn.benchmark = True
        _device_info["cudnn_benchmark"] = True
        
        logger.info(f"PyTorch using CUDA device: {_device_info['device_name']}")
    
    # Check for MPS (Apple Metal) next
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = torch.device("mps")
        _dtype = torch.float32  # MPS only supports float32
        
        # Gather MPS device information
        _device_info.update({
            "device_type": "mps",
            "device_name": "Apple Metal",
            "is_available": torch.backends.mps.is_available(),
            "is_built": torch.backends.mps.is_built(),
        })
        
        # Empty MPS cache to ensure clean start
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        
        logger.info("PyTorch using Apple Metal (MPS) device")
    
    # Fallback to CPU
    else:
        _device = torch.device("cpu")
        _dtype = torch.float64  # Higher precision on CPU
        
        # Gather CPU information
        _device_info.update({
            "device_type": "cpu",
            "device_name": "CPU",
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads()
        })
        
        # Set optimal thread count for CPU
        import multiprocessing
        num_physical_cores = os.cpu_count() or multiprocessing.cpu_count()
        recommended_threads = max(1, num_physical_cores - 1)  # Leave one core free
        torch.set_num_threads(recommended_threads)
        _device_info["num_threads_set"] = recommended_threads
        
        logger.info(f"PyTorch using CPU with {recommended_threads} threads")
    
    # Mark as initialized
    _initialized = True

def get_device():
    """Get the optimal PyTorch device (CUDA > MPS > CPU)"""
    if not _initialized:
        initialize()
    return _device

def get_dtype():
    """Get the optimal data type for the selected device"""
    if not _initialized:
        initialize()
    return _dtype

def move_to_device(tensor):
    """Helper to move a tensor to the selected device"""
    if not _initialized:
        initialize()
    return tensor.to(_device)

def optimize_for_inference():
    """Apply optimizations for inference workloads"""
    if not _initialized:
        initialize()
    
    import torch
    
    # Apply optimizations based on device
    if _device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    elif _device.type == "mps":
        # MPS-specific optimizations (limited options)
        pass
    elif _device.type == "cpu":
        # Enable inference optimizations for CPU
        try:
            torch.set_num_threads(os.cpu_count() or 4)
        except:
            pass
    
    logger.info(f"Optimized {_device.type} device for inference")

def optimize_for_training(mixed_precision=True):
    """Apply optimizations for training workloads"""
    if not _initialized:
        initialize()
    
    import torch
    
    # Apply optimizations based on device
    if _device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        
        # Enable mixed precision if requested
        if mixed_precision:
            try:
                from torch.cuda.amp import autocast
                _device_info["amp_enabled"] = True
                logger.info("CUDA Automatic Mixed Precision enabled")
            except ImportError:
                _device_info["amp_enabled"] = False
    
    elif _device.type == "mps":
        # Limited optimization options for MPS
        pass
    
    logger.info(f"Optimized {_device.type} device for training")

def clear_memory_cache():
    """Clear GPU memory cache based on device type"""
    if not _initialized:
        initialize()
    
    import torch
    
    if _device.type == "cuda":
        torch.cuda.empty_cache()
        logger.info("CUDA memory cache cleared")
    elif _device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
        logger.info("MPS memory cache cleared")
    
    return True