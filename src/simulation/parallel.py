"""
Parallel Execution Utilities for Monte Carlo Simulation
=====================================================

This module provides optimized parallel execution functions for running
Monte Carlo simulations across multiple threads and processes.
"""

import os
import numpy as np
import logging
import traceback
import time
import asyncio
import concurrent.futures
import multiprocessing
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from tqdm import tqdm
from functools import partial
import torch

from src.simulation.utils import (
    initialize_openbb, batch_items, CPU_COUNT, optimal_chunk_size, DEVICE
)

# Configure logging
logger = logging.getLogger(__name__)

# Thread pool executor that can be reused
_THREAD_POOL = None

def get_thread_pool(max_workers=None):
    """
    Get or create a thread pool executor.
    Reusing pools avoids the overhead of creating new ones.
    """
    global _THREAD_POOL
    if _THREAD_POOL is None or _THREAD_POOL._max_workers != max_workers:
        if _THREAD_POOL is not None:
            _THREAD_POOL.shutdown(wait=False)
        _THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    return _THREAD_POOL

def run_parallel_simulation(simulation_func: Callable, time_horizon: int, 
                         num_paths: int, max_workers: int = None, **kwargs) -> np.ndarray:
    """
    Run a simulation in parallel using a thread pool for optimal CPU utilization
    with GPU acceleration where appropriate.
    
    Args:
        simulation_func: Function that implements the stochastic model
        time_horizon: Number of days to simulate
        num_paths: Total number of paths to simulate
        max_workers: Maximum number of worker threads
        **kwargs: Additional keyword arguments for the simulation function
        
    Returns:
        Array of simulated final prices
    """
    # For small simulations or GPU use, don't use parallelism
    if (num_paths <= 1000) or (DEVICE.type != 'cpu' and num_paths * time_horizon > 5000):
        return simulation_func(time_horizon, num_paths, **kwargs)
    
    # Use default number of workers if not specified
    if max_workers is None:
        max_workers = min(16, CPU_COUNT)
    
    # Determine optimal chunk size for CPU parallelism
    # Larger chunks reduce overhead but decrease parallelism
    chunk_size = max(200, min(2000, num_paths // max_workers))
    if chunk_size * max_workers >= num_paths:
        max_workers = max(1, num_paths // chunk_size)
    
    chunks = []
    paths_remaining = num_paths
    start = 0
    
    while paths_remaining > 0:
        paths_in_chunk = min(chunk_size, paths_remaining)
        chunks.append((start, start + paths_in_chunk))
        start += paths_in_chunk
        paths_remaining -= paths_in_chunk
    
    # Get thread pool (creating if needed)
    executor = get_thread_pool(max_workers)
    
    # Submit jobs to the thread pool
    futures = []
    for start, end in chunks:
        future = executor.submit(
            _simulate_chunk, simulation_func, time_horizon, end - start, **kwargs
        )
        futures.append(future)
    
    # Gather results as they complete
    results = []
    for future in concurrent.futures.as_completed(futures):
        try:
            chunk_result = future.result()
            results.append(chunk_result)
        except Exception as e:
            logger.error(f"Error in simulation chunk: {str(e)}")
            logger.error(traceback.format_exc())
            # Re-raise to propagate the error
            raise
    
    # Combine results
    if len(results) == 1:
        return results[0]
    else:
        return np.concatenate(results)

def _simulate_chunk(simulation_func: Callable, time_horizon: int, 
                  num_paths: int, **kwargs) -> np.ndarray:
    """
    Helper function to simulate a chunk of paths.
    
    Args:
        simulation_func: Simulation function to call
        time_horizon: Number of days to simulate
        num_paths: Number of paths in this chunk
        **kwargs: Additional arguments for the simulation function
        
    Returns:
        Array of final prices for this chunk
    """
    return simulation_func(time_horizon, num_paths, **kwargs)

# Process pool for multi-ticker simulations
_PROCESS_POOL = None

def get_process_pool(max_workers=None):
    """Get or create a process pool executor"""
    global _PROCESS_POOL
    if _PROCESS_POOL is None or _PROCESS_POOL._max_workers != max_workers:
        if _PROCESS_POOL is not None:
            _PROCESS_POOL.shutdown(wait=False)
        ctx = multiprocessing.get_context('spawn')
        _PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx
        )
    return _PROCESS_POOL

# ============== Multi-Ticker Simulation Functions ==============

def simulate_ticker(ticker: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Simulate a single ticker with the provided configuration.
    This function is intended to run in its own process.
    
    Args:
        ticker: Stock ticker symbol
        config: Optional configuration parameters
        
    Returns:
        Simulation results for the ticker
    """
    # Initialize OpenBB client in this process
    initialize_openbb()
    
    try:
        # Import here to avoid circular imports
        from src.simulation.montecarlo import MonteCarloSimulator
        
        # Set default configuration if not provided
        if config is None:
            config = {}
            
        # Create simulator with specified or default parameters
        simulator = MonteCarloSimulator(
            ticker=ticker,
            num_simulations=config.get('num_simulations', 500),
            time_horizons=config.get('time_horizons', [1, 5, 10, 21, 63, 126, 252]),
            use_heston=config.get('use_heston', True),
            use_multiple_volatility_models=config.get('use_multiple_volatility_models', True),
            max_threads=config.get('max_threads', min(CPU_COUNT, 8))  # Reduced from 24 to avoid oversaturation
        )
        
        # Run simulation using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if not loop.run_until_complete(simulator.calibrate_model_parameters()):
                return {"ticker": ticker, "error": "Calibration failed"}
                
            loop.run_until_complete(simulator.run_simulation())
        finally:
            loop.close()
        
        # Return summary results
        return simulator.get_results_summary()
    
    except Exception as e:
        logger.error(f"Error simulating {ticker}: {e}")
        logger.error(traceback.format_exc())
        return {"ticker": ticker, "error": str(e)}

def simulate_ticker_batch(tickers: List[str], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Run simulations for a batch of tickers using an optimized thread pool.
    Each ticker runs in its own thread within the current process with
    improved concurrency management.
    
    Args:
        tickers: List of ticker symbols to simulate
        config: Optional configuration parameters
        
    Returns:
        Dictionary of simulation results by ticker
    """
    # Initialize OpenBB client in this process
    initialize_openbb()
    
    results = {}
    
    # Determine optimal concurrency based on CPU cores and batch size
    # Avoid oversaturating the system
    if config is None:
        config = {}
    
    max_concurrent = min(
        len(tickers), 
        config.get('max_concurrent', min(8, CPU_COUNT))  # Reduced from 12
    )
    
    # Using thread pool with semaphore control
    executor = get_thread_pool(max_concurrent)
    
    # Submit all tasks
    futures = {
        executor.submit(simulate_ticker, ticker, config): ticker 
        for ticker in tickers
    }
    
    # Process results as they complete
    for future in tqdm(concurrent.futures.as_completed(futures), 
                    total=len(futures), 
                    desc=f"Processing ticker batch ({max_concurrent} threads)"):
        
        ticker = futures[future]
        try:
            results[ticker] = future.result()
        except Exception as e:
            logger.error(f"Error in thread for {ticker}: {e}")
            logger.error(traceback.format_exc())
            results[ticker] = {"ticker": ticker, "error": str(e)}
    
    return results

def run_monte_carlo_analysis(tickers: List[str], config: Dict[str, Any] = None) -> Dict[str, Dict[str, Any]]:
    """
    Run Monte Carlo simulations for multiple tickers using optimized process-based parallelism
    with dynamic workload balancing.
    
    Args:
        tickers: List of ticker symbols to simulate
        config: Optional configuration parameters
        
    Returns:
        Dictionary of simulation results by ticker
    """
    if config is None:
        config = {}
        
    # Calculate optimal process count - reduce from previous to avoid oversaturation
    # Use no more than CPU_COUNT/2 processes for better overall performance
    available_processes = config.get('max_processes', min(CPU_COUNT // 2, 6))
    
    # For small workloads, reduce process count further
    total_tickers = len(tickers)
    if total_tickers <= 4:
        process_count = min(total_tickers, 2)
    elif total_tickers <= 10:
        process_count = min(total_tickers, 4)
    else:
        process_count = available_processes
    
    # Calculate optimal batch size - larger batches for fewer processes
    # This improves efficiency by reducing process creation overhead
    if total_tickers <= process_count:
        batch_size = 1
    else:
        batch_size = max(2, (total_tickers + process_count - 1) // process_count)
    
    logger.info(f"Running Monte Carlo analysis for {total_tickers} tickers using {process_count} processes")
    logger.info(f"Batch size: {batch_size} tickers per process")
    
    # Create batches of tickers
    ticker_batches = batch_items(tickers, batch_size)
    
    # Track results
    results = {}
    
    # Set up process pool with optimized settings
    executor = get_process_pool(process_count)
    
    # Submit batches to process pool
    futures = {
        executor.submit(simulate_ticker_batch, batch, config): batch 
        for batch in ticker_batches
    }
    
    # Use tqdm to track progress
    completed = 0
    with tqdm(total=len(tickers), desc="Simulating tickers") as pbar:
        for future in concurrent.futures.as_completed(futures):
            batch = futures[future]
            try:
                # Collect batch results
                batch_results = future.result()
                results.update(batch_results)
                
                # Update progress
                batch_size = len(batch)
                completed += batch_size
                pbar.update(batch_size)
                
            except Exception as e:
                # Handle errors at batch level
                logger.error(f"Error processing batch: {e}")
                logger.error(traceback.format_exc())
                for ticker in batch:
                    results[ticker] = {"ticker": ticker, "error": f"Batch processing error: {str(e)}"}
                pbar.update(len(batch))
    
    return results