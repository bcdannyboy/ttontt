"""
Utilities for Monte Carlo Simulation
===================================

This module provides utility functions for the Monte Carlo simulation package.
"""

import torch
from metal_coordinator import get_device, get_dtype
DEVICE = get_device()
LOGGER_DEVICE = DEVICE.type.upper()

import os
import numpy as np
import logging
import multiprocessing
from typing import Dict, Any, List, Tuple, Optional
from functools import lru_cache
import importlib

# Configure logging
logger = logging.getLogger(__name__)

# Set up global constants - reduce CPU_COUNT to avoid system saturation
CPU_COUNT = os.cpu_count()

DEFAULT_SIMULATION_COUNT = 1000
DEFAULT_TIME_HORIZONS = [1, 5, 10, 21, 63, 126, 252]


logger.info(f"Using device: {LOGGER_DEVICE}")

# Global variables for OpenBB client
_obb_client = None
_obb_version = None

def initialize_openbb():
    """
    Initialize the OpenBB client once per process and properly set up 
    all required modules.
    
    This function handles different versions of the OpenBB SDK and ensures
    the technical module is properly loaded.
    
    Returns:
        OpenBB client with proper module initialization
    """
    global _obb_client, _obb_version
    
    if _obb_client is None:
        try:
            # First try the modern OpenBB import
            from openbb import obb as openbb_sdk
            _obb_client = openbb_sdk
            _obb_version = "modern"
            
            # Check if the technical module exists
            if not hasattr(_obb_client, 'technical'):
                # Check if we need to load extensions
                if hasattr(_obb_client, 'extensions') and callable(getattr(_obb_client.extensions, 'load', None)):
                    logger.info("Loading OpenBB extensions")
                    _obb_client.extensions.load("technical")
                    
                # If still not available, try alternative loading
                if not hasattr(_obb_client, 'technical'):
                    logger.warning("Technical module not found in OpenBB SDK. Attempting to load as a submodule.")
                    try:
                        # Try using importlib to dynamically load the module
                        technical_module = importlib.import_module('openbb.technical')
                        setattr(_obb_client, 'technical', technical_module)
                    except ImportError:
                        logger.warning("Could not load technical module directly. Using fallback implementations.")
            
            logger.info("OpenBB SDK initialized successfully (modern version).")
            
        except ImportError:
            try:
                # Try legacy OpenBB import
                import openbb
                _obb_client = openbb
                _obb_version = "legacy"
                
                logger.info("OpenBB SDK initialized successfully (legacy version).")
                
            except ImportError:
                logger.error("Failed to import OpenBB client. Ensure it's installed properly.")
                raise
    
    return _obb_client

def get_openbb_client():
    """
    Returns the initialized OpenBB client.
    If not initialized, initializes it first.
    
    Returns:
        OpenBB client instance
    """
    global _obb_client
    if _obb_client is None:
        _obb_client = initialize_openbb()
    return _obb_client

def openbb_has_technical():
    """
    Check if the technical module is available in the OpenBB client.
    
    Returns:
        bool: True if available, False otherwise
    """
    client = get_openbb_client()
    return hasattr(client, 'technical')

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