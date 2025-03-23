import os
import threading
import time
import asyncio
import logging
import numpy as np
import torch
import json
from datetime import datetime, timedelta

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

# API rate limiting with proper per-provider limits
API_CALLS_PER_MINUTE = {
    'default': 240,
    'fmp': 300,
    'intrinio': 100,
    'yfinance': 2000,
    'polygon': 50,
    'alphavantage': 75
}

# Create separate semaphores for each provider for better control
api_semaphores = {
    'default': asyncio.Semaphore(40),
    'fmp': asyncio.Semaphore(50),
    'intrinio': asyncio.Semaphore(20),
    'yfinance': asyncio.Semaphore(100),
    'polygon': asyncio.Semaphore(10),
    'alphavantage': asyncio.Semaphore(15)
}

# Track API calls per provider
api_call_timestamps = {
    'default': [],
    'fmp': [],
    'intrinio': [],
    'yfinance': [],
    'polygon': [],
    'alphavantage': []
}
api_locks = {
    provider: threading.RLock() for provider in api_call_timestamps.keys()
}

# Cache for API results with proper locking to avoid race conditions
CACHE_SIZE = 5000  # Increased cache size
api_cache = {}
api_cache_lock = threading.RLock()
metrics_cache = {}
metrics_cache_lock = threading.RLock()

# Create a cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(key_type, key):
    """Get path for cached data file"""
    safe_key = ''.join(c if c.isalnum() else '_' for c in key)
    return os.path.join(CACHE_DIR, f"{key_type}_{safe_key}_cache.json")

def get_openbb_client():
    """Returns a thread-local OpenBB client instance."""
    from openbb import obb
    if not hasattr(thread_local, "openbb_client"):
        thread_local.openbb_client = obb
    return thread_local.openbb_client

async def rate_limited_api_call(func, *args, provider='default', cache_ttl=86400, **kwargs):
    """
    Rate limiting for API calls with async support, caching, and provider-specific limits.
    
    Args:
        func: The API function to call
        provider: The data provider to use (default, fmp, intrinio, etc.)
        cache_ttl: Cache time-to-live in seconds (default 24 hours)
        *args, **kwargs: Arguments to pass to the API function
    
    Returns:
        The API response or cached response
    """
    # Create a cache key from function name and arguments
    cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
    
    # Check memory cache first
    with api_cache_lock:
        if cache_key in api_cache:
            cache_entry = api_cache[cache_key]
            # Check if cache entry is still valid
            if datetime.now().timestamp() - cache_entry['timestamp'] < cache_ttl:
                return cache_entry['data']
    
    # Check disk cache
    cache_file = get_cache_path('api', cache_key.replace('/', '_').replace('\\', '_')[:100])
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                if datetime.now().timestamp() - cache_data.get('timestamp', 0) < cache_ttl:
                    # Update memory cache with disk data
                    with api_cache_lock:
                        api_cache[cache_key] = {
                            'data': cache_data['data'],
                            'timestamp': cache_data['timestamp']
                        }
                        return cache_data['data']
        except Exception as e:
            logger.warning(f"Error reading API cache: {e}")
    
    # Use the appropriate semaphore based on provider
    semaphore = api_semaphores.get(provider, api_semaphores['default'])
    
    async with semaphore:
        # Apply rate limiting based on provider
        provider_key = provider if provider in api_locks else 'default'
        rate_limit = API_CALLS_PER_MINUTE.get(provider_key, API_CALLS_PER_MINUTE['default'])
        
        with api_locks[provider_key]:
            current_time = time.time()
            # Filter out timestamps older than 60 seconds
            api_call_timestamps[provider_key] = [
                ts for ts in api_call_timestamps[provider_key] 
                if current_time - ts < 60
            ]
            
            # Check if we've reached the rate limit
            if len(api_call_timestamps[provider_key]) >= rate_limit:
                wait_time = 60 - (current_time - api_call_timestamps[provider_key][0])
                if wait_time > 0:
                    logger.debug(f"Rate limit reached for {provider_key}, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            # Add current timestamp to the list
            api_call_timestamps[provider_key].append(time.time())
        
        # Execute the API call
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            
            # Update caches with the result
            with api_cache_lock:
                api_cache[cache_key] = {
                    'data': result,
                    'timestamp': datetime.now().timestamp()
                }
                
                # Prune memory cache if too large
                if len(api_cache) > CACHE_SIZE:
                    # Find oldest entries
                    oldest_keys = sorted(
                        api_cache.keys(),
                        key=lambda k: api_cache[k]['timestamp']
                    )[:len(api_cache) // 10]  # Remove 10% of oldest entries
                    
                    # Remove oldest entries
                    for key in oldest_keys:
                        del api_cache[key]
            
            # Save to disk cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'data': result if not hasattr(result, '__dict__') else result.__dict__,
                        'timestamp': datetime.now().timestamp()
                    }, f, default=lambda o: str(o) if hasattr(o, '__dict__') else o.__dict__ if hasattr(o, '__dict__') else str(o))
            except Exception as e:
                logger.warning(f"Error writing API cache to disk: {e}")
            
            return result
        
        except Exception as e:
            # Check if error is about subscription/access
            error_str = str(e).lower()
            if any(term in error_str for term in ['subscription', 'access', 'api key', 'unauthorized']):
                logger.error(f"API authentication/subscription error for {provider}: {e}")
                raise Exception(f"API call error: \n[Error] -> Error in {provider} request -> An active subscription is required to view this data.")
            elif 'no results' in error_str or 'not found' in error_str:
                logger.error(f"No results found for API call: {e}")
                raise Exception(f"API call error: \n[Empty] -> No results found. Try adjusting the query parameters.")
            else:
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