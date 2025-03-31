"""
Utilities for Monte Carlo Simulation
===================================

This module provides utility functions for the Monte Carlo simulation package.
"""

import torch
import os
import numpy as np
import logging
import importlib
from typing import Dict, Any, List, Tuple, Optional
from functools import lru_cache
from metal_coordinator import get_device, get_dtype

# Set up device
DEVICE = get_device()
LOGGER_DEVICE = DEVICE.type.upper()

# Configure logging
logger = logging.getLogger(__name__)
logger.info(f"Using device: {LOGGER_DEVICE}")

# Set up global constants
CPU_COUNT = max(1, (os.cpu_count() or 4) // 2)  # Use half the CPUs to avoid system overload

DEFAULT_SIMULATION_COUNT = 1000
DEFAULT_TIME_HORIZONS = [1, 5, 10, 21, 63, 126, 252]

# Global variables for OpenBB
_obb_client = None
_obb_version = None
_extensions_loaded = False

def initialize_openbb():
    """
    Initialize the OpenBB client with proper handling of different versions
    and extension management.
    
    This handles OpenBB Platform v4.x (modern) and older SDK versions (legacy),
    ensuring technical extension is properly loaded.
    
    Returns:
        OpenBB client instance with properly initialized modules
    """
    global _obb_client, _obb_version, _extensions_loaded
    
    if _obb_client is not None and _extensions_loaded:
        return _obb_client
    
    try:
        # First try modern OpenBB Platform import (v4+)
        from openbb import obb
        _obb_client = obb
        _obb_version = "modern"
        logger.info("Detected modern OpenBB Platform (v4+)")
        
        # Check if technical extension is needed and not already loaded
        if not hasattr(_obb_client, 'technical'):
            # First check if we can install the extension
            try:
                import subprocess
                import sys
                
                logger.info("Installing technical extension for OpenBB Platform")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "openbb-technical", "--quiet"
                ])
                
                # After installation, reload the client to pick up the extension
                importlib.reload(obb)
                _obb_client = obb
                
                # If still not available, try manual loading
                if not hasattr(_obb_client, 'technical'):
                    try:
                        # Try explicit import and attachment
                        import openbb_technical
                        _obb_client.technical = openbb_technical
                        logger.info("Manually attached technical module")
                    except ImportError:
                        logger.warning("Could not import openbb_technical module")
            except Exception as e:
                logger.warning(f"Failed to install technical extension: {str(e)}")
                
    except ImportError:
        # Fall back to legacy OpenBB SDK
        try:
            # Try to import legacy SDK
            from openbb_terminal.sdk import openbb
            _obb_client = openbb
            _obb_version = "legacy"
            logger.info("Using legacy OpenBB SDK")
            
            # Check if technical module exists in legacy SDK
            if not hasattr(_obb_client, 'ta') and not hasattr(_obb_client, 'technical'):
                logger.warning("Technical analysis module not found in legacy SDK")
        except ImportError:
            logger.error("Failed to import any version of OpenBB. Please install it first.")
            raise ImportError("OpenBB not installed or not accessible")
    
    _extensions_loaded = True
    return _obb_client

def get_openbb_client():
    """
    Get the initialized OpenBB client, ensuring it has the technical module.
    
    Returns:
        OpenBB client instance
    """
    global _obb_client
    if _obb_client is None:
        _obb_client = initialize_openbb()
    return _obb_client

def openbb_has_technical():
    """
    Check if the OpenBB client has technical analysis capabilities.
    
    Returns:
        bool: True if available through any module path, False otherwise
    """
    client = get_openbb_client()
    
    # Check all possible technical module locations based on version
    if hasattr(client, 'technical'):  # Modern platform
        return True
    elif hasattr(client, 'ta'):       # Legacy SDK
        return True
    elif _obb_version == "legacy" and hasattr(client, 'stocks') and hasattr(client.stocks, 'ta'):
        return True  # Alternative legacy path
    
    return False

def get_technical_module():
    """
    Get the technical analysis module from OpenBB in a version-agnostic way.
    
    Returns:
        The technical analysis module or None if not available
    """
    client = get_openbb_client()
    
    # Handle different versions' module structures
    if hasattr(client, 'technical'):  # Modern platform
        return client.technical
    elif hasattr(client, 'ta'):       # Legacy SDK direct path
        return client.ta
    elif _obb_version == "legacy" and hasattr(client, 'stocks') and hasattr(client.stocks, 'ta'):
        return client.stocks.ta        # Legacy nested path
    
    logger.warning("No technical analysis module found in OpenBB")
    return None

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

def format_time_horizon(days: int) -> str:
    """
    Format a time horizon in days to a human-readable string.

    Args:
        days (int): Number of days.

    Returns:
        str: A formatted string representing the time horizon, for example:
             "1D" for 1 day,
             "1W" for up to 5 days,
             "2W" for up to 10 days,
             "1M" for up to 21 days,
             "3M" for up to 63 days,
             "6M" for up to 126 days,
             "1Y" for up to 252 days,
             or simply the number of days with "D" suffix otherwise.
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