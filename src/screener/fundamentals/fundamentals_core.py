import os
import threading
import time
import asyncio
import logging
import numpy as np
import torch

# Set seed for reproducibility with both numpy and torch
np.random.seed(42)
torch.manual_seed(42)

# Set device to MPS if available; otherwise use CPU
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    logger_device = "MPS"
else:
    device = torch.device("cpu")
    logger_device = "CPU"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.info(f"Using device: {logger_device}")

# Thread-local storage for OpenBB API session
thread_local = threading.local()

# Define weights for different categories of financial metrics
CATEGORY_WEIGHTS = {
    'profitability': 0.20,
    'growth': 0.20,
    'financial_health': 0.20,
    'valuation': 0.15,
    'efficiency': 0.10,
    'analyst_estimates': 0.15
}

# Profitability metrics weights
PROFITABILITY_WEIGHTS = {
    'gross_profit_margin': 0.20,
    'operating_income_margin': 0.20,
    'net_income_margin': 0.25,
    'ebitda_margin': 0.15,
    'return_on_equity': 0.10,
    'return_on_assets': 0.10
}

# Growth metrics weights
GROWTH_WEIGHTS = {
    'growth_revenue': 0.20,
    'growth_gross_profit': 0.15,
    'growth_ebitda': 0.15,
    'growth_net_income': 0.20,
    'growth_eps': 0.15,
    'growth_total_assets': 0.05,
    'growth_total_shareholders_equity': 0.10
}

# Financial health metrics weights
FINANCIAL_HEALTH_WEIGHTS = {
    'current_ratio': 0.15,
    'debt_to_equity': -0.20,
    'debt_to_assets': -0.15,
    'growth_total_debt': -0.15,
    'growth_net_debt': -0.15,
    'interest_coverage': 0.10,
    'cash_to_debt': 0.10
}

# Valuation metrics weights
VALUATION_WEIGHTS = {
    'pe_ratio': -0.25,
    'price_to_book': -0.15,
    'price_to_sales': -0.15,
    'ev_to_ebitda': -0.20,
    'dividend_yield': 0.15,
    'peg_ratio': -0.10
}

# Efficiency metrics weights
EFFICIENCY_WEIGHTS = {
    'asset_turnover': 0.25,
    'inventory_turnover': 0.20,
    'receivables_turnover': 0.20,
    'cash_conversion_cycle': -0.20,
    'capex_to_revenue': -0.15
}

# Analyst estimates metrics weights
ANALYST_ESTIMATES_WEIGHTS = {
    'estimate_eps_accuracy': 0.35,
    'estimate_revenue_accuracy': 0.35,
    'estimate_consensus_deviation': -0.10,
    'estimate_revision_momentum': 0.20
}

# API rate limiting
API_CALLS_PER_MINUTE = 240
api_semaphore = asyncio.Semaphore(40)  # Allow 40 concurrent API calls
api_call_timestamps = []
api_lock = threading.RLock()

# Cache for API results with proper locking to avoid race conditions
CACHE_SIZE = 1000
api_cache = {}
api_cache_lock = threading.RLock()
metrics_cache = {}
metrics_cache_lock = threading.RLock()

def get_openbb_client():
    """Returns a thread-local OpenBB client instance."""
    from openbb import obb
    if not hasattr(thread_local, "openbb_client"):
        thread_local.openbb_client = obb
    return thread_local.openbb_client

async def rate_limited_api_call(func, *args, **kwargs):
    """
    Rate limiting for API calls with async support and caching.
    """
    cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
    
    # Check cache with proper locking
    with api_cache_lock:
        if cache_key in api_cache:
            return api_cache[cache_key]
    
    async with api_semaphore:
        with api_lock:
            current_time = time.time()
            global api_call_timestamps
            api_call_timestamps = [ts for ts in api_call_timestamps if current_time - ts < 60]
            if len(api_call_timestamps) >= API_CALLS_PER_MINUTE:
                sleep_time = 60 - (current_time - api_call_timestamps[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    await asyncio.sleep(sleep_time)
            api_call_timestamps.append(time.time())
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            
            # Update cache with proper locking
            with api_cache_lock:
                api_cache[cache_key] = result
                if len(api_cache) > CACHE_SIZE:
                    oldest_key = next(iter(api_cache))
                    del api_cache[oldest_key]
            
            return result
        except Exception as e:
            logger.error(f"API call error: {e}")
            raise

def select_valid_record(records, key, min_value=0.001):
    """
    Returns the first record with a nonzero value for `key` above min_value.
    If none is found, returns the first record.
    """
    import pandas as pd
    if not records:
        return None
        
    for rec in records:
        value = getattr(rec, key, None)
        try:
            if value is not None and not pd.isna(value) and float(value) > min_value:
                return rec
        except Exception:
            continue
    
    return records[0]
