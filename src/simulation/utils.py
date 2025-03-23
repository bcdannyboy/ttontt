"""
Utilities for Monte Carlo Simulation
===================================

This module provides utility functions for the Monte Carlo simulation package.
"""

import os
import numpy as np
import logging
import multiprocessing
from typing import Dict, Any, List, Tuple, Optional
from functools import lru_cache

# Configure logging
logger = logging.getLogger(__name__)

# Set up global constants - reduce CPU_COUNT to avoid system saturation
CPU_COUNT = min(
    multiprocessing.cpu_count() if os.cpu_count() <= multiprocessing.cpu_count() else os.cpu_count(),
    os.cpu_count() // 2  # Use at most half of logical CPUs for better overall system performance
)

DEFAULT_SIMULATION_COUNT = 1000
DEFAULT_TIME_HORIZONS = [1, 5, 10, 21, 63, 126, 252]

# Configure device for potential GPU acceleration
import torch

if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    LOGGER_DEVICE = "MPS"
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    LOGGER_DEVICE = "CUDA"
else:
    DEVICE = torch.device("cpu")
    LOGGER_DEVICE = "CPU"

# Set PyTorch configuration for maximum performance
if DEVICE.type == 'cuda':
    # Set CUDA flags for better performance
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.deterministic = False
    logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    # Set memory allocation to be more aggressive
    torch.cuda.empty_cache()
elif DEVICE.type == 'mps':
    torch.mps.empty_cache()

logger.info(f"Using device: {LOGGER_DEVICE}")

# Initialize global OpenBB client to avoid recreating it across threads/processes
obb_client = None

def initialize_openbb():
    """Initialize the OpenBB client once per process"""
    global obb_client
    if obb_client is None:
        try:
            from openbb import obb as openbb_client
            obb_client = openbb_client
        except ImportError:
            logger.error("Failed to import OpenBB client")
            raise
    return obb_client

def calculate_statistics(data: np.ndarray) -> Dict[str, Any]:
    """
    Calculate descriptive statistics for the simulation results.
    
    Args:
        data: Array of data to analyze
        
    Returns:
        Dictionary of statistics
    """
    # Move computation to GPU if large dataset and GPU available
    if DEVICE.type != 'cpu' and len(data) > 10000:
        # Create tensor on GPU
        data_tensor = torch.tensor(data, device=DEVICE)
        
        # Calculate statistics
        stats = {
            "mean": float(torch.mean(data_tensor).cpu().item()),
            "median": float(torch.median(data_tensor).cpu().item()),
            "std": float(torch.std(data_tensor).cpu().item()),
            "min": float(torch.min(data_tensor).cpu().item()),
            "max": float(torch.max(data_tensor).cpu().item()),
            "q1": float(torch.quantile(data_tensor, 0.25).cpu().item()),
            "q3": float(torch.quantile(data_tensor, 0.75).cpu().item()),
        }
        
        # Clean up to avoid memory leaks
        del data_tensor
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        elif DEVICE.type == 'mps':
            torch.mps.empty_cache()
    else:
        # For CPU or smaller datasets, use numpy
        from scipy.stats import skew, kurtosis
        
        stats = {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "q1": float(np.percentile(data, 25)),
            "q3": float(np.percentile(data, 75)),
            "skew": float(skew(data)),
            "kurtosis": float(kurtosis(data))
        }
    return stats

def optimal_chunk_size(total_size: int, num_workers: int) -> int:
    """
    Calculate optimal chunk size for parallel processing.
    
    Args:
        total_size: Total number of items to process
        num_workers: Number of available workers
        
    Returns:
        Optimal chunk size
    """
    # Minimum of 200 items per chunk to reduce overhead
    min_chunk_size = 200
    
    # Aim for each worker to get at least 2-4 chunks for better load balancing
    target_chunks_per_worker = min(4, max(2, num_workers // 2))
    total_chunks = num_workers * target_chunks_per_worker
    
    # Base chunk size
    base_chunk_size = max(min_chunk_size, total_size // min(total_size, total_chunks))
    
    # For very large workloads, increase chunk size to reduce overhead
    if total_size > 10000 * num_workers:
        base_chunk_size *= 2
    
    return base_chunk_size

def batch_items(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split items into batches of specified size.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

@lru_cache(maxsize=32)
def _format_time_horizon(days: int) -> str:
    """
    Format a time horizon in days to a human-readable string.
    
    Args:
        days: Number of days
        
    Returns:
        Formatted string (e.g., "1D", "1W", "1M", etc.)
    """
    if days == 1:
        return "1D"
    elif days <= 5:
        return "1W"
    elif days <= 10:
        return "2W"
    elif days <= 21:
        return "1M"
    elif days <= 63:
        return "3M"
    elif days <= 126:
        return "6M"
    elif days <= 252:
        return "1Y"
    else:
        return f"{days}D"