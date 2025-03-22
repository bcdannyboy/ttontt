"""
Technical Stock Screener
=======================

A comprehensive stock screening tool that evaluates companies based on their technical indicators
and assigns composite scores that reflect their momentum, trends, and price patterns.

Overall Approach:
----------------
1. Data Collection: Retrieves extensive technical indicators data using the OpenBB API
   for a list of ticker symbols, including moving averages, oscillators, volatility measures,
   and volume indicators.

2. Indicator Extraction: Processes raw technical data to extract relevant metrics across key
   categories: trend indicators, momentum indicators, volatility indicators, and volume indicators.

3. Standardization: Transforms raw metrics into standardized z-scores that represent how many
   standard deviations a company's metric is from the mean of all companies being screened.
   This enables direct comparison across different metrics with different scales and units.

4. Weighted Scoring System: Applies carefully calibrated weights to each metric within its
   category, with positive weights for metrics where higher values are better (e.g., bullish MACD)
   and negative weights for metrics where lower values are better (e.g., high volatility).

5. Composite Scoring: Computes category scores and an overall composite score that reflects
   a company's overall technical strength, allowing for ranking and comparison across companies.

6. Handling Outliers: Uses statistical methods to handle extreme values without arbitrary caps,
   preserving the nuance of real-world data while still maintaining fair comparisons.

7. Trend Analysis: Evaluates price trends, momentum patterns, and potential reversal points
   using a blend of traditional technical indicators.

Statistical Methods:
-------------------
- Z-score calculation to standardize metrics across companies (implemented via PyTorch to leverage MPS)
- Weighted averaging to combine multiple factors
- Statistical detection of outliers using quartile-based methods

Key Categories:
--------------
- Trend Indicators: Measures the direction and strength of price trends
- Momentum Indicators: Evaluates the speed or rate of price changes
- Volatility Indicators: Assesses market stability and potential price swings
- Volume Indicators: Examines the strength of price moves based on volume patterns

This approach provides a robust, data-driven method for evaluating stocks based on 
technical indicators to identify stocks with favorable price patterns and momentum.
"""

import os
from openbb import obb
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Set
import logging
from datetime import datetime, timedelta
import concurrent.futures
import threading
import traceback
import time
from tqdm import tqdm
import asyncio
import aiohttp
import functools

# Import PyTorch to leverage MPS on Apple Silicon
import torch

# Set device to mps if available; otherwise use cpu
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

# Define weights for different categories of technical metrics
CATEGORY_WEIGHTS = {
    'trend': 0.30,
    'momentum': 0.30,
    'volatility': 0.20,
    'volume': 0.20
}

# Trend indicators weights
TREND_WEIGHTS = {
    'sma_cross_signal': 0.15,  # SMA crossover signal (50/200)
    'ema_cross_signal': 0.15,  # EMA crossover signal (12/26)
    'price_rel_sma': 0.10,     # Price relative to SMA (200-day)
    'price_rel_ema': 0.10,     # Price relative to EMA (50-day)
    'adx_trend': 0.15,         # ADX trend strength
    'ichimoku_signal': 0.10,   # Ichimoku Cloud signal
    'macd_signal': 0.15,       # MACD signal
    'bb_position': 0.10        # Position in Bollinger Bands
}

# Momentum indicators weights
MOMENTUM_WEIGHTS = {
    'rsi_signal': 0.20,             # RSI signal
    'stoch_signal': 0.15,           # Stochastic signal
    'cci_signal': 0.15,             # CCI signal
    'clenow_momentum': 0.15,        # Clenow Volatility Adjusted Momentum
    'fisher_transform': 0.10,       # Fisher Transform
    'price_performance_1m': 0.10,   # 1-month price performance
    'price_performance_3m': 0.15    # 3-month price performance
}

# Volatility indicators weights
VOLATILITY_WEIGHTS = {
    'atr_percent': -0.20,        # ATR as percentage of price (lower is better)
    'bb_width': -0.15,           # Bollinger Band width (narrower often better)
    'keltner_width': -0.15,      # Keltner Channel width
    'volatility_cones': -0.20,   # Volatility cones position
    'donchian_width': -0.15,     # Donchian channel width
    'price_target_upside': 0.15  # Price target upside potential
}

# Volume indicators weights
VOLUME_WEIGHTS = {
    'obv_trend': 0.25,               # OBV trend
    'adl_trend': 0.20,               # Accumulation/Distribution Line trend
    'adosc_signal': 0.20,            # Accumulation/Distribution Oscillator
    'vwap_position': 0.20,           # Position relative to VWAP
    'volume_trend': 0.15             # Volume trend
}

# API rate limiting
API_CALLS_PER_MINUTE = 240
api_semaphore = asyncio.Semaphore(40)  # Allow 40 concurrent API calls
api_call_timestamps = []
api_lock = threading.RLock()

# Cache for API results
CACHE_SIZE = 1000
api_cache = {}
indicators_cache = {}

def get_openbb_client():
    """Returns a thread-local OpenBB client instance."""
    if not hasattr(thread_local, "openbb_client"):
        thread_local.openbb_client = obb
    return thread_local.openbb_client

async def rate_limited_api_call(func, *args, **kwargs):
    """
    Rate limiting for API calls with async support and caching.
    """
    cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
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
            api_cache[cache_key] = result
            if len(api_cache) > CACHE_SIZE:
                oldest_key = next(iter(api_cache))
                del api_cache[oldest_key]
            return result
        except Exception as e:
            logger.error(f"API call error: {e}")
            raise

def get_attribute_value(obj: Any, attr_name: str, default=0) -> Any:
    """Safely get attribute value from an object."""
    if hasattr(obj, attr_name):
        value = getattr(obj, attr_name)
        if value is not None and not pd.isna(value):
            return value
    return default

async def get_technical_data_async(ticker: str) -> Tuple[str, Dict[str, Any]]:
    """
    Async version of get_technical_data using OpenBB API.
    Fetches all required technical indicator data for a stock, handling provider fallbacks properly.
    """
    obb_client = get_openbb_client()
    technical_data = {}
    
    try:
        # Get historical price data (1 year)
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        price_history_task = rate_limited_api_call(
            obb_client.equity.price.historical,
            symbol=ticker, start_date=one_year_ago, provider='fmp'
        )
        
        # Get price performance data
        price_perf_task = rate_limited_api_call(
            obb_client.equity.price.performance,
            symbol=ticker, provider='fmp'
        )
        
        # Get consensus price target
        price_target_task = rate_limited_api_call(
            obb_client.equity.estimates.consensus,
            symbol=ticker, provider='fmp'
        )
        
        # Execute first batch of API calls
        price_history_response, price_perf_response, price_target_response = await asyncio.gather(
            price_history_task, price_perf_task, price_target_task,
            return_exceptions=True
        )
        
        if isinstance(price_history_response, Exception):
            logger.error(f"Essential price history data fetch failed for {ticker}: {price_history_response}")
            return (ticker, technical_data)
        
        # Store the results
        technical_data['price_history'] = price_history_response.results if not isinstance(price_history_response, Exception) else []
        technical_data['price_performance'] = price_perf_response.results if not isinstance(price_perf_response, Exception) else []
        technical_data['price_target'] = price_target_response.results if not isinstance(price_target_response, Exception) else []
        
        # If we have price history, calculate technical indicators
        if technical_data['price_history']:
            # Calculate SMA indicators
            sma_50_task = rate_limited_api_call(
                obb_client.technical.sma,
                data=technical_data['price_history'], target='close', length=50
            )
            
            sma_200_task = rate_limited_api_call(
                obb_client.technical.sma,
                data=technical_data['price_history'], target='close', length=200
            )
            
            # Calculate EMA indicators
            ema_12_task = rate_limited_api_call(
                obb_client.technical.ema,
                data=technical_data['price_history'], target='close', length=12
            )
            
            ema_26_task = rate_limited_api_call(
                obb_client.technical.ema,
                data=technical_data['price_history'], target='close', length=26
            )
            
            ema_50_task = rate_limited_api_call(
                obb_client.technical.ema,
                data=technical_data['price_history'], target='close', length=50
            )
            
            # Calculate Bollinger Bands
            bbands_task = rate_limited_api_call(
                obb_client.technical.bbands,
                data=technical_data['price_history'], target='close', length=20, std=2
            )
            
            # Calculate Keltner Channels
            keltner_task = rate_limited_api_call(
                obb_client.technical.kc,
                data=technical_data['price_history'], length=20, scalar=2
            )
            
            # Execute second batch of API calls
            (
                sma_50_response, sma_200_response, 
                ema_12_response, ema_26_response, ema_50_response,
                bbands_response, keltner_response
            ) = await asyncio.gather(
                sma_50_task, sma_200_task, 
                ema_12_task, ema_26_task, ema_50_task,
                bbands_task, keltner_task,
                return_exceptions=True
            )
            
            # Store the results for moving averages and bands
            technical_data['sma_50'] = sma_50_response.results if not isinstance(sma_50_response, Exception) else []
            technical_data['sma_200'] = sma_200_response.results if not isinstance(sma_200_response, Exception) else []
            technical_data['ema_12'] = ema_12_response.results if not isinstance(ema_12_response, Exception) else []
            technical_data['ema_26'] = ema_26_response.results if not isinstance(ema_26_response, Exception) else []
            technical_data['ema_50'] = ema_50_response.results if not isinstance(ema_50_response, Exception) else []
            technical_data['bbands'] = bbands_response.results if not isinstance(bbands_response, Exception) else []
            technical_data['keltner'] = keltner_response.results if not isinstance(keltner_response, Exception) else []
            
            # Calculate more advanced indicators
            macd_task = rate_limited_api_call(
                obb_client.technical.macd,
                data=technical_data['price_history'], target='close', fast=12, slow=26, signal=9
            )
            
            rsi_task = rate_limited_api_call(
                obb_client.technical.rsi,
                data=technical_data['price_history'], target='close', length=14
            )
            
            stoch_task = rate_limited_api_call(
                obb_client.technical.stoch,
                data=technical_data['price_history'], fast_k_period=14, slow_d_period=3
            )
            
            cci_task = rate_limited_api_call(
                obb_client.technical.cci,
                data=technical_data['price_history'], length=20
            )
            
            adx_task = rate_limited_api_call(
                obb_client.technical.adx,
                data=technical_data['price_history'], length=14
            )
            
            obv_task = rate_limited_api_call(
                obb_client.technical.obv,
                data=technical_data['price_history']
            )
            
            ad_task = rate_limited_api_call(
                obb_client.technical.ad,
                data=technical_data['price_history']
            )
            
            # Execute third batch of API calls
            (
                macd_response, rsi_response, stoch_response, 
                cci_response, adx_response, obv_response, ad_response
            ) = await asyncio.gather(
                macd_task, rsi_task, stoch_task, 
                cci_task, adx_task, obv_task, ad_task,
                return_exceptions=True
            )
            
            # Store the results for oscillators and other indicators
            technical_data['macd'] = macd_response.results if not isinstance(macd_response, Exception) else []
            technical_data['rsi'] = rsi_response.results if not isinstance(rsi_response, Exception) else []
            technical_data['stoch'] = stoch_response.results if not isinstance(stoch_response, Exception) else []
            technical_data['cci'] = cci_response.results if not isinstance(cci_response, Exception) else []
            technical_data['adx'] = adx_response.results if not isinstance(adx_response, Exception) else []
            technical_data['obv'] = obv_response.results if not isinstance(obv_response, Exception) else []
            technical_data['ad'] = ad_response.results if not isinstance(ad_response, Exception) else []
            
            # Calculate additional indicators
            atr_task = rate_limited_api_call(
                obb_client.technical.atr,
                data=technical_data['price_history'], length=14
            )
            
            donchian_task = rate_limited_api_call(
                obb_client.technical.donchian,
                data=technical_data['price_history'], lower_length=20, upper_length=20
            )
            
            fisher_task = rate_limited_api_call(
                obb_client.technical.fisher,
                data=technical_data['price_history'], length=14
            )
            
            ichimoku_task = rate_limited_api_call(
                obb_client.technical.ichimoku,
                data=technical_data['price_history'], conversion=9, base=26
            )
            
            adosc_task = rate_limited_api_call(
                obb_client.technical.adosc,
                data=technical_data['price_history'], fast=3, slow=10
            )
            
            vwap_task = rate_limited_api_call(
                obb_client.technical.vwap,
                data=technical_data['price_history'], anchor='D'
            )
            
            clenow_task = rate_limited_api_call(
                obb_client.technical.clenow,
                data=technical_data['price_history'], period=90
            )
            
            # Execute fourth batch of API calls
            (
                atr_response, donchian_response, fisher_response, 
                ichimoku_response, adosc_response, vwap_response, clenow_response
            ) = await asyncio.gather(
                atr_task, donchian_task, fisher_task,
                ichimoku_task, adosc_task, vwap_task, clenow_task,
                return_exceptions=True
            )
            
            # Store the results for additional indicators
            technical_data['atr'] = atr_response.results if not isinstance(atr_response, Exception) else []
            technical_data['donchian'] = donchian_response.results if not isinstance(donchian_response, Exception) else []
            technical_data['fisher'] = fisher_response.results if not isinstance(fisher_response, Exception) else []
            technical_data['ichimoku'] = ichimoku_response.results if not isinstance(ichimoku_response, Exception) else []
            technical_data['adosc'] = adosc_response.results if not isinstance(adosc_response, Exception) else []
            technical_data['vwap'] = vwap_response.results if not isinstance(vwap_response, Exception) else []
            technical_data['clenow'] = clenow_response.results if not isinstance(clenow_response, Exception) else []
            
            # Calculate additional technical indicators from OpenBB documentation
            try:
                aroon_task = rate_limited_api_call(
                    obb_client.technical.aroon,
                    data=technical_data['price_history'], length=25
                )
                aroon_response = await aroon_task
                technical_data['aroon'] = aroon_response.results if not isinstance(aroon_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching Aroon indicator for {ticker}: {e}")
                technical_data['aroon'] = []
            
            try:
                fib_task = rate_limited_api_call(
                    obb_client.technical.fib,
                    data=technical_data['price_history'], period=120
                )
                fib_response = await fib_task
                technical_data['fib'] = fib_response.results if not isinstance(fib_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching Fibonacci levels for {ticker}: {e}")
                technical_data['fib'] = []
            
            try:
                hma_task = rate_limited_api_call(
                    obb_client.technical.hma,
                    data=technical_data['price_history'], target='close', length=50
                )
                hma_response = await hma_task
                technical_data['hma'] = hma_response.results if not isinstance(hma_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching HMA for {ticker}: {e}")
                technical_data['hma'] = []
            
            try:
                wma_task = rate_limited_api_call(
                    obb_client.technical.wma,
                    data=technical_data['price_history'], target='close', length=50
                )
                wma_response = await wma_task
                technical_data['wma'] = wma_response.results if not isinstance(wma_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching WMA for {ticker}: {e}")
                technical_data['wma'] = []
            
            try:
                zlma_task = rate_limited_api_call(
                    obb_client.technical.zlma,
                    data=technical_data['price_history'], target='close', length=50
                )
                zlma_response = await zlma_task
                technical_data['zlma'] = zlma_response.results if not isinstance(zlma_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching ZLMA for {ticker}: {e}")
                technical_data['zlma'] = []
            
            try:
                demark_task = rate_limited_api_call(
                    obb_client.technical.demark,
                    data=technical_data['price_history']
                )
                demark_response = await demark_task
                technical_data['demark'] = demark_response.results if not isinstance(demark_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching Demark Sequential for {ticker}: {e}")
                technical_data['demark'] = []
            
            # Calculate volatility cones
            try:
                cones_task = rate_limited_api_call(
                    obb_client.technical.cones,
                    data=technical_data['price_history'], lower_q=0.25, upper_q=0.75, model='std'
                )
                cones_response = await cones_task
                technical_data['cones'] = cones_response.results if not isinstance(cones_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching volatility cones for {ticker}: {e}")
                technical_data['cones'] = []
            
            # Try to get price targets
            try:
                price_targets_task = rate_limited_api_call(
                    obb_client.equity.estimates.price_target,
                    symbol=ticker, provider='fmp', limit=10
                )
                price_targets_response = await price_targets_task
                technical_data['price_targets'] = price_targets_response.results if not isinstance(price_targets_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching price targets for {ticker}: {e}")
                technical_data['price_targets'] = []
    
    except Exception as e:
        logger.error(f"Error fetching technical data for {ticker}: {e}")
        logger.debug(traceback.format_exc())
    
    return (ticker, technical_data)

def extract_indicators_from_technical_data(technical_data):
    """
    Extract all relevant technical indicators from the data.
    Handles missing data gracefully with fallbacks and derived indicators.
    """
    indicators = {}
    
    if not technical_data or not isinstance(technical_data, dict):
        return indicators
    
    # First, check what data we have available
    has_price_history = bool(technical_data.get('price_history'))
    has_price_performance = bool(technical_data.get('price_performance'))
    has_sma_data = bool(technical_data.get('sma_50')) or bool(technical_data.get('sma_200'))
    has_ema_data = bool(technical_data.get('ema_12')) or bool(technical_data.get('ema_26'))
    has_macd_data = bool(technical_data.get('macd'))
    has_rsi_data = bool(technical_data.get('rsi'))
    has_stoch_data = bool(technical_data.get('stoch'))
    has_volume_data = has_price_history  # We can derive volume data from price history
    
    # Helper function to safely extract numerical values from Data objects
    def extract_value(data_obj):
        if data_obj is None:
            return None
        # If it's already a number, return it directly
        if isinstance(data_obj, (int, float)):
            return data_obj
        # If it's a Data object, try to get its value attribute
        if hasattr(data_obj, 'value'):
            return data_obj.value
        # If it's a Data object with a float representation, use that
        try:
            return float(data_obj)
        except (ValueError, TypeError):
            return None
    
    # Extract current price - this is essential
    current_price = None
    if has_price_history:
        price_history = technical_data['price_history']
        if price_history and len(price_history) > 0:
            last_price = price_history[-1]
            if hasattr(last_price, 'close'):
                current_price = extract_value(last_price.close)
            indicators['current_price'] = current_price
    
    if not current_price:
        return indicators  # Can't do much without a current price
    
    # Extract price performance data - critical for basic technical analysis
    if has_price_performance and len(technical_data['price_performance']) > 0:
        perf_data = technical_data['price_performance'][0]
        
        # Extract various time period performances
        one_day = get_attribute_value(perf_data, 'one_day', 0)
        one_week = get_attribute_value(perf_data, 'one_week', 0)
        one_month = get_attribute_value(perf_data, 'one_month', 0)
        three_month = get_attribute_value(perf_data, 'three_month', 0)
        year_to_date = get_attribute_value(perf_data, 'year_to_date', 0)
        one_year = get_attribute_value(perf_data, 'one_year', 0)
        
        indicators['price_performance_1d'] = one_day
        indicators['price_performance_1w'] = one_week
        indicators['price_performance_1m'] = one_month
        indicators['price_performance_3m'] = three_month
        indicators['price_performance_ytd'] = year_to_date
        indicators['price_performance_1y'] = one_year
        
        # Derive momentum indicators from performance data if other indicators are missing
        if not has_rsi_data and not has_stoch_data:
            # Simple momentum signal based on recent performance
            momentum_signal = 0.0
            if one_month is not None:
                momentum_signal += one_month * 0.6  # 60% weight to one month performance
            if three_month is not None:
                momentum_signal += three_month * 0.4  # 40% weight to three month performance
                
            indicators['momentum_signal'] = momentum_signal
            
            # Convert to RSI-like scale (0-100) for compatibility
            if momentum_signal is not None:
                # Scale from typical -0.2 to +0.2 range to 0-100 range
                pseudo_rsi = 50 + momentum_signal * 250
                pseudo_rsi = max(0, min(100, pseudo_rsi))  # Clamp to 0-100
                indicators['rsi'] = pseudo_rsi
                
                # Calculate RSI signal (scaled from -1 to 1)
                if pseudo_rsi >= 70:
                    indicators['rsi_signal'] = -1.0  # Overbought
                elif pseudo_rsi <= 30:
                    indicators['rsi_signal'] = 1.0   # Oversold
                else:
                    indicators['rsi_signal'] = (50 - pseudo_rsi) / 20  # Linear scale between
    
    # Extract moving averages if available
    # SMA data
    if has_sma_data:
        sma_50 = None
        sma_200 = None
        if technical_data.get('sma_50') and len(technical_data['sma_50']) > 0:
            sma_50 = extract_value(technical_data['sma_50'][-1])
            indicators['sma_50'] = sma_50
        
        if technical_data.get('sma_200') and len(technical_data['sma_200']) > 0:
            sma_200 = extract_value(technical_data['sma_200'][-1])
            indicators['sma_200'] = sma_200
        
        # Calculate SMA-based signals
        if sma_50 is not None and sma_200 is not None:
            # Golden Cross (50-day SMA crosses above 200-day SMA)
            indicators['sma_cross_signal'] = 1.0 if sma_50 > sma_200 else -1.0
        
        if current_price is not None:
            if sma_50 is not None:
                # Price relative to 50-day SMA
                indicators['price_rel_sma_50'] = (current_price / sma_50) - 1.0
            
            if sma_200 is not None:
                # Price relative to 200-day SMA
                indicators['price_rel_sma_200'] = (current_price / sma_200) - 1.0
                # Price relative to SMA (combined)
                indicators['price_rel_sma'] = indicators['price_rel_sma_200']
    # If no SMA data, derive a trend indicator from performance data
    elif has_price_performance:
        # Use 3-month performance as a proxy for trend
        if indicators.get('price_performance_3m') is not None:
            perf_3m = indicators['price_performance_3m']
            # Convert to normalized trend signal (-1 to 1)
            indicators['price_rel_sma'] = min(max(perf_3m * 5, -1), 1)  # Scale and clamp
    
    # Calculate volume trend from price history - crucial for the volume component
    if has_price_history and len(technical_data['price_history']) > 5:
        recent_volumes = []
        for ph in technical_data['price_history'][-20:]:
            if hasattr(ph, 'volume'):
                vol = extract_value(ph.volume)
                if vol is not None:
                    recent_volumes.append(vol)
                
        if len(recent_volumes) >= 5:  # Need at least 5 data points for meaningful trend
            avg_recent_volume = sum(recent_volumes[-5:]) / 5
            
            if len(recent_volumes) >= 10:  # Better if we have 10+ data points
                avg_earlier_volume = sum(recent_volumes[-10:-5]) / 5
                
                if avg_earlier_volume > 0:
                    # Volume trend as percentage change
                    vol_change = (avg_recent_volume / avg_earlier_volume) - 1.0
                    # Scale to a reasonable range (-1 to 1)
                    indicators['volume_trend'] = min(1.0, max(-1.0, vol_change))
            else:
                # Simpler calculation if we have fewer points
                avg_all_volume = sum(recent_volumes) / len(recent_volumes)
                if avg_all_volume > 0:
                    vol_change = (avg_recent_volume / avg_all_volume) - 1.0
                    indicators['volume_trend'] = min(1.0, max(-1.0, vol_change))
    
    # Extract EMA indicators
    if has_ema_data:
        ema_12 = None
        ema_26 = None
        
        if technical_data.get('ema_12') and len(technical_data['ema_12']) > 0:
            ema_12 = extract_value(technical_data['ema_12'][-1])
            indicators['ema_12'] = ema_12
        
        if technical_data.get('ema_26') and len(technical_data['ema_26']) > 0:
            ema_26 = extract_value(technical_data['ema_26'][-1])
            indicators['ema_26'] = ema_26
        
        # Calculate EMA cross signal if both are available
        if ema_12 is not None and ema_26 is not None:
            indicators['ema_cross_signal'] = 1.0 if ema_12 > ema_26 else -1.0
        
        # Price relative to EMA if available
        if current_price is not None and ema_26 is not None:
            indicators['price_rel_ema'] = (current_price / ema_26) - 1.0
    
    # Extract MACD indicators
    if has_macd_data and len(technical_data['macd']) > 0:
        macd_data = technical_data['macd'][-1]
        
        macd_line = extract_value(getattr(macd_data, 'macd', None))
        macd_signal = extract_value(getattr(macd_data, 'signal', None))
        macd_hist = extract_value(getattr(macd_data, 'histogram', None))
        
        if macd_line is not None:
            indicators['macd_line'] = macd_line
        
        if macd_signal is not None:
            indicators['macd_signal_line'] = macd_signal
        
        if macd_hist is not None:
            indicators['macd_hist'] = macd_hist
        
        # Calculate MACD signal if both line and signal are available
        if macd_line is not None and macd_signal is not None:
            indicators['macd_signal'] = 1.0 if macd_line > macd_signal else -1.0
    
    # Extract RSI data
    if has_rsi_data and len(technical_data['rsi']) > 0:
        rsi_value = extract_value(technical_data['rsi'][-1])
        
        if rsi_value is not None:
            indicators['rsi'] = rsi_value
            
            # Calculate RSI signal (scaled from -1 to 1)
            if rsi_value >= 70:
                indicators['rsi_signal'] = -1.0  # Overbought
            elif rsi_value <= 30:
                indicators['rsi_signal'] = 1.0   # Oversold
            else:
                indicators['rsi_signal'] = (50 - rsi_value) / 20  # Linear scale between
    
    # Extract Stochastic Oscillator
    if has_stoch_data and len(technical_data['stoch']) > 0:
        stoch_data = technical_data['stoch'][-1]
        
        stoch_k = extract_value(getattr(stoch_data, 'k', None))
        stoch_d = extract_value(getattr(stoch_data, 'd', None))
        
        if stoch_k is not None:
            indicators['stoch_k'] = stoch_k
        
        if stoch_d is not None:
            indicators['stoch_d'] = stoch_d
        
        # Calculate Stochastic signal if available
        if stoch_k is not None and stoch_d is not None:
            # Typical interpretation: overbought above 80, oversold below 20
            if stoch_d >= 80:
                indicators['stoch_signal'] = -1.0  # Overbought
            elif stoch_d <= 20:
                indicators['stoch_signal'] = 1.0   # Oversold
            else:
                indicators['stoch_signal'] = (50 - stoch_d) / 30  # Scale to -1 to 1
    
    # Extract Bollinger Bands if available
    if technical_data.get('bbands') and len(technical_data['bbands']) > 0:
        bb_data = technical_data['bbands'][-1]
        
        bb_upper = extract_value(getattr(bb_data, 'upper', None))
        bb_middle = extract_value(getattr(bb_data, 'middle', None))
        bb_lower = extract_value(getattr(bb_data, 'lower', None))
        
        if bb_upper is not None:
            indicators['bb_upper'] = bb_upper
        
        if bb_middle is not None:
            indicators['bb_middle'] = bb_middle
        
        if bb_lower is not None:
            indicators['bb_lower'] = bb_lower
        
        # Calculate BB position and width if all components are available
        if current_price is not None and bb_upper is not None and bb_lower is not None:
            bb_range = bb_upper - bb_lower
            
            if bb_range > 0:
                position = (current_price - bb_lower) / bb_range
                # Scale to -1 to 1 around the middle
                indicators['bb_position'] = (position - 0.5) * 2
                
                # Calculate BB width as percentage of middle
                if bb_middle is not None and bb_middle > 0:
                    indicators['bb_width'] = bb_range / bb_middle
    
    # Extract ADX if available
    if technical_data.get('adx') and len(technical_data['adx']) > 0:
        adx_data = technical_data['adx'][-1]
        adx_value = extract_value(getattr(adx_data, 'adx', None))
        
        if adx_value is not None:
            indicators['adx'] = adx_value
            
            # Calculate ADX trend strength
            if adx_value >= 25:
                # Strong trend (ADX > 25)
                trend_strength = 0.25 + (adx_value - 25) * 0.75 / 75
                indicators['adx_trend'] = min(1.0, trend_strength)
            else:
                # Weak trend (ADX < 25)
                indicators['adx_trend'] = adx_value * 0.25 / 25
    
    # Extract ATR if available
    if technical_data.get('atr') and len(technical_data['atr']) > 0:
        atr_value = extract_value(technical_data['atr'][-1])
        
        if atr_value is not None:
            indicators['atr'] = atr_value
            
            # Calculate ATR percentage if current price is available
            if current_price is not None and current_price > 0:
                indicators['atr_percent'] = atr_value / current_price
    
    # Extract OBV if available
    if technical_data.get('obv') and len(technical_data['obv']) > 1:
        obv_current = extract_value(technical_data['obv'][-1])
        obv_prev = extract_value(technical_data['obv'][-2])
        
        if obv_current is not None:
            indicators['obv'] = obv_current
            
            # Calculate OBV trend if previous value is available
            if obv_prev is not None:
                # Simple trend: 1.0 if increasing, -1.0 if decreasing
                indicators['obv_trend'] = 1.0 if obv_current > obv_prev else -1.0
                
                # If more history is available, calculate a more nuanced trend
                if len(technical_data['obv']) >= 10:
                    obv_10_periods_ago = extract_value(technical_data['obv'][-10])
                    
                    if obv_10_periods_ago is not None and obv_10_periods_ago != 0:
                        # Calculate rate of change
                        obv_roc = (obv_current - obv_10_periods_ago) / abs(obv_10_periods_ago)
                        # Scale to a reasonable range
                        indicators['obv_trend'] = min(1.0, max(-1.0, obv_roc * 5))
    
    # Extract Accumulation/Distribution Line if available
    if technical_data.get('ad') and len(technical_data['ad']) > 1:
        ad_current = extract_value(technical_data['ad'][-1])
        ad_prev = extract_value(technical_data['ad'][-2])
        
        if ad_current is not None:
            indicators['ad'] = ad_current
            
            # Calculate AD trend if previous value is available
            if ad_prev is not None:
                # Simple trend: 1.0 if increasing, -1.0 if decreasing
                indicators['adl_trend'] = 1.0 if ad_current > ad_prev else -1.0
                
                # If more history is available, calculate a more nuanced trend
                if len(technical_data['ad']) >= 10:
                    ad_10_periods_ago = extract_value(technical_data['ad'][-10])
                    
                    if ad_10_periods_ago is not None and ad_10_periods_ago != 0:
                        # Calculate rate of change
                        ad_roc = (ad_current - ad_10_periods_ago) / abs(ad_10_periods_ago)
                        # Scale to a reasonable range
                        indicators['adl_trend'] = min(1.0, max(-1.0, ad_roc * 5))
    
    # Extract VWAP if available
    if technical_data.get('vwap') and len(technical_data['vwap']) > 0:
        vwap_value = extract_value(technical_data['vwap'][-1])
        
        if vwap_value is not None:
            indicators['vwap'] = vwap_value
            
            # Calculate VWAP position if current price is available
            if current_price is not None and vwap_value > 0:
                # Calculate percentage difference
                vwap_diff = (current_price / vwap_value) - 1.0
                # Scale to a reasonable range
                indicators['vwap_position'] = min(1.0, max(-1.0, vwap_diff * 5))
    
    # Ensure we have essential category indicators through derivation
    if not any(k in ['sma_cross_signal', 'ema_cross_signal', 'price_rel_sma', 'price_rel_ema', 'adx_trend'] for k in indicators):
        # Derive basic trend indicator from 3-month performance if we have it
        if 'price_performance_3m' in indicators:
            perf_3m = indicators['price_performance_3m']
            indicators['trend_signal'] = 1.0 if perf_3m > 0 else -1.0
    
    if not any(k in ['rsi_signal', 'stoch_signal', 'cci_signal', 'price_performance_1m'] for k in indicators):
        # Derive basic momentum from 1-month performance
        if 'price_performance_1m' in indicators:
            perf_1m = indicators['price_performance_1m']
            indicators['momentum_signal'] = 1.0 if perf_1m > 0 else -1.0
    
    # Ensure volume trend is available
    if 'volume_trend' not in indicators and has_price_history:
        # Default to neutral if we can't calculate
        indicators['volume_trend'] = 0.0
    
    return indicators

def normalize_data(data_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Normalize data using vectorized operations for better performance.
    """
    normalized_data = {ticker: {} for ticker in data_dict}
    metrics_data = {}
    for ticker, metrics in data_dict.items():
        for metric, value in metrics.items():
            if metric not in metrics_data:
                metrics_data[metric] = []
            if isinstance(value, (int, float)) and not pd.isna(value) and value is not None:
                metrics_data[metric].append(value)
    for ticker, metrics in data_dict.items():
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)) or pd.isna(value) or value is None:
                continue
            normalized_data[ticker][metric] = value
    return normalized_data

def calculate_z_scores(data_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate z-scores for each metric using PyTorch and clamp extreme values to [-3, 3].
    Handles sparse data by using available metrics and implementing fallbacks.
    """
    z_scores = {ticker: {} for ticker in data_dict}
    metrics_dict = {}
    
    # First pass: collect available metrics
    for ticker, metrics in data_dict.items():
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and not pd.isna(value) and value is not None:
                metrics_dict.setdefault(metric, []).append((ticker, value))
    
    # Second pass: calculate z-scores for metrics with sufficient data
    for metric, ticker_values in metrics_dict.items():
        if len(ticker_values) < 2:
            # For metrics with only one value, set z-score to 0 (neutral)
            if len(ticker_values) == 1:
                ticker, value = ticker_values[0]
                z_scores[ticker][metric] = 0.0
            continue
        
        tickers, values = zip(*ticker_values)
        
        # Use PyTorch tensor on the appropriate device (MPS if available)
        try:
            values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
            
            # Check for data variability
            if torch.std(values_tensor).item() < 1e-8:
                # If all values are essentially the same, assign neutral z-scores
                for ticker in tickers:
                    z_scores[ticker][metric] = 0.0
                continue
            
            # Standard z-score calculation
            mean_val = torch.mean(values_tensor)
            std_val = torch.std(values_tensor)
            std_val = std_val if std_val > 1e-10 else torch.tensor(1e-10, device=device)
            
            # If enough values and high skewness, use robust standardization
            if len(values) > 4:
                skewness = torch.abs(torch.mean(((values_tensor - mean_val) / std_val) ** 3)).item()
                if skewness > 2:
                    median_val = torch.median(values_tensor)
                    q1 = torch.quantile(values_tensor, 0.25)
                    q3 = torch.quantile(values_tensor, 0.75)
                    iqr = q3 - q1
                    robust_std = max((iqr / 1.349).item(), 1e-10)
                    metric_z_scores = (values_tensor - median_val) / robust_std
                else:
                    metric_z_scores = (values_tensor - mean_val) / std_val
            else:
                metric_z_scores = (values_tensor - mean_val) / std_val
            
            # Clamp extreme values
            metric_z_scores = torch.clamp(metric_z_scores, -3, 3)
            
            # Store z-scores
            for ticker, z_score in zip(tickers, metric_z_scores):
                z_scores[ticker][metric] = z_score.item()
                
        except Exception as e:
            logger.warning(f"Error calculating z-scores for {metric}: {e}")
            # Fallback: use normalized values instead of z-scores
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                range_val = max_val - min_val
                for ticker, value in zip(tickers, values):
                    normalized_val = (value - min_val) / range_val * 2 - 1  # Scale to [-1, 1]
                    z_scores[ticker][metric] = normalized_val
            else:
                # If all values are the same, assign neutral z-scores
                for ticker in tickers:
                    z_scores[ticker][metric] = 0.0
    
    # Third pass: derive missing z-scores where possible
    # For each ticker, try to infer missing metrics from related ones
    derived_metrics = {
        'trend': ['price_rel_sma', 'price_rel_ema', 'macd_signal', 'sma_cross_signal', 'ema_cross_signal'],
        'momentum': ['price_performance_1m', 'price_performance_3m', 'rsi_signal', 'stoch_signal', 'cci_signal'],
        'volatility': ['atr_percent', 'bb_width', 'keltner_width'],
        'volume': ['volume_trend', 'obv_trend', 'adl_trend']
    }
    
    # For each ticker, ensure critical metrics exist by deriving from performance data if needed
    for ticker in data_dict:
        # Performance data fallbacks
        if 'price_performance_1m' in data_dict[ticker] and 'price_performance_1m' not in z_scores[ticker]:
            raw_value = data_dict[ticker]['price_performance_1m']
            z_scores[ticker]['price_performance_1m'] = min(max(raw_value * 5, -3), 3)  # Simple scaling
        
        if 'price_performance_3m' in data_dict[ticker] and 'price_performance_3m' not in z_scores[ticker]:
            raw_value = data_dict[ticker]['price_performance_3m']
            z_scores[ticker]['price_performance_3m'] = min(max(raw_value * 4, -3), 3)  # Simple scaling
        
        # Volume trend fallback
        if 'volume_trend' in data_dict[ticker] and 'volume_trend' not in z_scores[ticker]:
            raw_value = data_dict[ticker]['volume_trend']
            z_scores[ticker]['volume_trend'] = min(max(raw_value * 3, -3), 3)  # Simple scaling
    
    return z_scores

def calculate_weighted_score(z_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculate weighted score based on z-scores and weights.
    Handles sparse data with minimum thresholds and category fallbacks.
    """
    if not z_scores or not weights:
        return 0
    
    # Group metrics by category for fallback mechanism
    category_metrics = {
        'trend': ['sma_cross_signal', 'ema_cross_signal', 'price_rel_sma', 'price_rel_ema', 
                 'adx_trend', 'ichimoku_signal', 'macd_signal', 'bb_position', 'trend_signal'],
        'momentum': ['rsi_signal', 'stoch_signal', 'cci_signal', 'clenow_momentum', 
                    'fisher_transform', 'price_performance_1m', 'price_performance_3m', 'momentum_signal'],
        'volatility': ['atr_percent', 'bb_width', 'keltner_width', 'volatility_cones', 
                      'donchian_width', 'price_target_upside'],
        'volume': ['obv_trend', 'adl_trend', 'adosc_signal', 'vwap_position', 'volume_trend']
    }
    
    # Find which categories have data
    available_categories = set()
    for category, metrics in category_metrics.items():
        if any(metric in z_scores for metric in metrics):
            available_categories.add(category)
    
    # If we're missing critical categories, use fallbacks for basic scoring
    fallback_used = False
    if len(available_categories) < 2:
        fallback_used = True
        # Ensure we have at least momentum data from price performance
        if 'momentum' not in available_categories and ('price_performance_1m' in z_scores or 'price_performance_3m' in z_scores):
            available_categories.add('momentum')
            
        # Ensure we have volume data if volume_trend is available
        if 'volume' not in available_categories and 'volume_trend' in z_scores:
            available_categories.add('volume')
    
    score = 0
    total_weight = 0
    valid_metrics = 0
    
    # Standard weighted scoring for available metrics
    common_metrics = set(z_scores.keys()) & set(weights.keys())
    for metric in common_metrics:
        weight = weights[metric]
        score += z_scores[metric] * weight
        total_weight += abs(weight)
        valid_metrics += 1
    
    # Return early if we have sufficient data
    if valid_metrics >= 3 and total_weight > 0:
        return score / total_weight
    
    # Fallback: Calculate score based on category averages for sparse data
    if fallback_used or valid_metrics < 3:
        category_scores = {}
        category_weights = {
            'trend': 0.30,
            'momentum': 0.30,
            'volatility': 0.20,
            'volume': 0.20
        }
        
        # Calculate average z-score for each available category
        for category, metrics in category_metrics.items():
            available_metrics = [m for m in metrics if m in z_scores]
            if available_metrics:
                category_scores[category] = sum(z_scores[m] for m in available_metrics) / len(available_metrics)
        
        # Compute weighted average of available category scores
        if category_scores:
            total_cat_weight = sum(category_weights[cat] for cat in category_scores)
            if total_cat_weight > 0:
                return sum(category_scores[cat] * category_weights[cat] for cat in category_scores) / total_cat_weight
    
    # Last resort: return simple average if everything else fails
    if valid_metrics > 0:
        return sum(z_scores[metric] for metric in common_metrics) / valid_metrics
    
    return 0.0  # No valid metrics

async def process_ticker_async(ticker: str) -> Tuple[str, Optional[Dict[str, float]]]:
    """
    Process a single ticker asynchronously with improved error handling.
    """
    try:
        ticker, technical_data = await get_technical_data_async(ticker)
        if not technical_data.get('price_history'):
            logger.warning(f"Insufficient technical data for {ticker}. Skipping...")
            return (ticker, None)
        try:
            if ticker in indicators_cache:
                indicators_dict = indicators_cache[ticker]
            else:
                indicators_dict = extract_indicators_from_technical_data(technical_data)
                indicators_cache[ticker] = indicators_dict
                if len(indicators_cache) > CACHE_SIZE:
                    oldest_key = next(iter(indicators_cache))
                    del indicators_cache[oldest_key]
            logger.debug(f"Successfully processed {ticker}")
            return (ticker, indicators_dict)
        except Exception as e:
            logger.error(f"Error extracting indicators for {ticker}: {e}")
            logger.debug(traceback.format_exc())
            return (ticker, None)
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        logger.debug(traceback.format_exc())
        return (ticker, None)

def process_ticker_sync(ticker: str) -> Tuple[str, Optional[Dict[str, float]]]:
    """
    Synchronous wrapper for process_ticker_async.
    """
    try:
        return asyncio.run(process_ticker_async(ticker))
    except Exception as e:
        logger.error(f"Error in process_ticker_sync for {ticker}: {e}")
        return (ticker, None)

async def screen_stocks_async(tickers: List[str], max_concurrent: int = 20) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Asynchronously screen stocks based on technical indicators.
    """
    results = []
    all_indicators = {}
    valid_tickers = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(ticker):
        async with semaphore:
            return await process_ticker_async(ticker)
    
    tasks = [process_with_semaphore(ticker) for ticker in tickers]
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        result = await task
        ticker, indicators_dict = result
        progress = i / len(tickers) * 100
        if i % 5 == 0 or i == 1 or i == len(tickers):
            print(f"Processing stocks (technical): {progress:.0f}% completed ({i}/{len(tickers)})", end='\r')
        if indicators_dict is not None:
            all_indicators[ticker] = indicators_dict
            valid_tickers.append(ticker)
    print()
    
    if not valid_tickers:
        logger.warning("No valid tickers could be processed for technical analysis.")
        return []
    
    normalized_indicators = normalize_data(all_indicators)
    z_scores = calculate_z_scores(normalized_indicators)
    
    for ticker in valid_tickers:
        ticker_z_scores = z_scores[ticker]
        
        trend_score = calculate_weighted_score(ticker_z_scores, TREND_WEIGHTS)
        momentum_score = calculate_weighted_score(ticker_z_scores, MOMENTUM_WEIGHTS)
        volatility_score = calculate_weighted_score(ticker_z_scores, VOLATILITY_WEIGHTS)
        volume_score = calculate_weighted_score(ticker_z_scores, VOLUME_WEIGHTS)
        
        composite_score = (
            trend_score * CATEGORY_WEIGHTS['trend'] +
            momentum_score * CATEGORY_WEIGHTS['momentum'] +
            volatility_score * CATEGORY_WEIGHTS['volatility'] +
            volume_score * CATEGORY_WEIGHTS['volume']
        )
        
        detailed_results = {
            'raw_indicators': all_indicators[ticker],
            'normalized_indicators': normalized_indicators[ticker],
            'z_scores': ticker_z_scores,
            'category_scores': {
                'trend': trend_score,
                'momentum': momentum_score,
                'volatility': volatility_score,
                'volume': volume_score
            },
            'composite_score': composite_score
        }
        
        results.append((ticker, composite_score, detailed_results))
    
    results.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Successfully screened {len(results)} stocks using technical indicators.")
    return results

def screen_stocks(tickers: List[str], max_workers: int = None) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Synchronously screen stocks using technical indicators.
    """
    if max_workers is None:
        max_workers = min(32, os.cpu_count() * 4)
    
    all_indicators = {}
    valid_tickers = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for ticker, indicators in tqdm(executor.map(process_ticker_sync, tickers), total=len(tickers), desc="Processing stocks (technical)"):
            if indicators is not None:
                all_indicators[ticker] = indicators
                valid_tickers.append(ticker)
    
    if not valid_tickers:
        logger.warning("No valid tickers could be processed for technical analysis.")
        return []
    
    normalized_indicators = normalize_data(all_indicators)
    z_scores = calculate_z_scores(normalized_indicators)
    
    results = []
    for ticker in valid_tickers:
        ticker_z_scores = z_scores[ticker]
        
        category_scores = {
            'trend': calculate_weighted_score(ticker_z_scores, TREND_WEIGHTS),
            'momentum': calculate_weighted_score(ticker_z_scores, MOMENTUM_WEIGHTS),
            'volatility': calculate_weighted_score(ticker_z_scores, VOLATILITY_WEIGHTS),
            'volume': calculate_weighted_score(ticker_z_scores, VOLUME_WEIGHTS)
        }
        
        composite_score = sum(score * CATEGORY_WEIGHTS[cat] for cat, score in category_scores.items())
        
        detailed_results = {
            'raw_indicators': all_indicators[ticker],
            'normalized_indicators': normalized_indicators[ticker],
            'z_scores': ticker_z_scores,
            'category_scores': category_scores,
            'composite_score': composite_score
        }
        
        results.append((ticker, composite_score, detailed_results))
    
    results.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Successfully screened {len(results)} stocks using technical indicators.")
    return results

def get_indicator_contributions(ticker: str) -> pd.DataFrame:
    """
    Get detailed breakdown of indicator contributions to the composite technical score.
    """
    results = screen_stocks([ticker], max_workers=1)
    if not results:
        logger.warning(f"No technical data found for {ticker}")
        return pd.DataFrame()
    
    _, _, detailed_results = results[0]
    z_scores = detailed_results['z_scores']
    raw_indicators = detailed_results['raw_indicators']
    
    data = []
    for category, weights in [
        ('Trend', TREND_WEIGHTS),
        ('Momentum', MOMENTUM_WEIGHTS),
        ('Volatility', VOLATILITY_WEIGHTS),
        ('Volume', VOLUME_WEIGHTS)
    ]:
        for indicator, weight in weights.items():
            if indicator in z_scores:
                z_score = z_scores[indicator]
                raw_value = raw_indicators.get(indicator)
                contribution = z_score * weight
                data.append({
                    'Category': category,
                    'Indicator': indicator,
                    'Raw Value': raw_value,
                    'Weight': weight,
                    'Z-Score': round(z_score, 2),
                    'Contribution': round(contribution, 3),
                    'Impact': 'Positive' if contribution > 0 else 'Negative'
                })
    
    df = pd.DataFrame(data)
    df['Abs Contribution'] = df['Contribution'].abs()
    df = df.sort_values('Abs Contribution', ascending=False)
    df = df.drop('Abs Contribution', axis=1)
    
    return df

def generate_stock_report(ticker: str) -> Dict[str, Any]:
    """
    Generate a comprehensive technical analysis report for a stock.
    """
    obb_client = get_openbb_client()
    technical_data = {}
    
    try:
        # Get historical price data (1 year)
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        price_history_response = obb_client.equity.price.historical(
            symbol=ticker, start_date=one_year_ago, provider='fmp'
        )
        technical_data['price_history'] = price_history_response.results
        
        # Get price performance data
        price_perf_response = obb_client.equity.price.performance(
            symbol=ticker, provider='fmp'
        )
        technical_data['price_performance'] = price_perf_response.results
        
        # Get price target consensus
        try:
            price_target_response = obb_client.equity.estimates.consensus(
                symbol=ticker, provider='fmp'
            )
            technical_data['price_target'] = price_target_response.results
        except Exception as e:
            logger.warning(f"Error fetching price target consensus for {ticker}: {e}")
            technical_data['price_target'] = []
        
        # Calculate technical indicators if we have price history
        if technical_data['price_history']:
            price_data = technical_data['price_history']
            
            # Calculate common indicators
            technical_data['sma_50'] = obb_client.technical.sma(data=price_data, target='close', length=50).results
            technical_data['sma_200'] = obb_client.technical.sma(data=price_data, target='close', length=200).results
            technical_data['ema_12'] = obb_client.technical.ema(data=price_data, target='close', length=12).results
            technical_data['ema_26'] = obb_client.technical.ema(data=price_data, target='close', length=26).results
            technical_data['rsi'] = obb_client.technical.rsi(data=price_data, target='close', length=14).results
            technical_data['macd'] = obb_client.technical.macd(data=price_data, target='close', fast=12, slow=26, signal=9).results
            technical_data['bbands'] = obb_client.technical.bbands(data=price_data, target='close', length=20, std=2).results
            technical_data['stoch'] = obb_client.technical.stoch(data=price_data, fast_k_period=14, slow_d_period=3).results
            technical_data['atr'] = obb_client.technical.atr(data=price_data, length=14).results
            
            # Get additional indicators as needed
            try:
                technical_data['adx'] = obb_client.technical.adx(data=price_data, length=14).results
            except Exception as e:
                logger.warning(f"Error calculating ADX for {ticker}: {e}")
            
            try:
                technical_data['obv'] = obb_client.technical.obv(data=price_data).results
            except Exception as e:
                logger.warning(f"Error calculating OBV for {ticker}: {e}")
            
            try:
                technical_data['cci'] = obb_client.technical.cci(data=price_data, length=20).results
            except Exception as e:
                logger.warning(f"Error calculating CCI for {ticker}: {e}")
    
    except Exception as e:
        logger.error(f"Error fetching technical data for {ticker}: {e}")
        logger.debug(traceback.format_exc())
    
    # Extract indicators from the data
    indicators = extract_indicators_from_technical_data(technical_data)
    
    # Create the report
    report = {
        'ticker': ticker,
        'composite_score': 0.0,
        'category_scores': {
            'trend': 0.0,
            'momentum': 0.0,
            'volatility': 0.0,
            'volume': 0.0
        },
        'key_indicators': {
            'trend': {
                'sma_cross': indicators.get('sma_cross_signal'),
                'price_rel_sma': indicators.get('price_rel_sma'),
                'macd_signal': indicators.get('macd_signal'),
                'adx_trend': indicators.get('adx_trend')
            },
            'momentum': {
                'rsi': indicators.get('rsi'),
                'stoch_k': indicators.get('stoch_k'),
                'stoch_d': indicators.get('stoch_d'),
                'price_performance_1m': indicators.get('price_performance_1m'),
                'price_performance_3m': indicators.get('price_performance_3m')
            },
            'volatility': {
                'atr_percent': indicators.get('atr_percent'),
                'bb_width': indicators.get('bb_width'),
                'price_target_upside': indicators.get('price_target_upside')
            },
            'volume': {
                'obv_trend': indicators.get('obv_trend'),
                'adl_trend': indicators.get('adl_trend'),
                'volume_trend': indicators.get('volume_trend')
            }
        },
        'signals': [],
        'warnings': [],
        'raw_indicators': indicators
    }
    
    # Add signals and warnings based on indicators
    # RSI signals
    if indicators.get('rsi') is not None:
        rsi_value = indicators['rsi']
        if rsi_value >= 70:
            report['warnings'].append(f"RSI is overbought at {rsi_value:.1f}")
        elif rsi_value <= 30:
            report['signals'].append(f"RSI is oversold at {rsi_value:.1f}")
    
    # MACD signals
    if indicators.get('macd_signal') is not None:
        macd_signal = indicators['macd_signal']
        if macd_signal > 0:
            report['signals'].append("MACD indicates bullish momentum")
        elif macd_signal < 0:
            report['warnings'].append("MACD indicates bearish momentum")
    
    # SMA cross signals
    if indicators.get('sma_cross_signal') is not None:
        sma_cross = indicators['sma_cross_signal']
        if sma_cross > 0:
            report['signals'].append("Golden Cross: 50-day SMA above 200-day SMA")
        elif sma_cross < 0:
            report['warnings'].append("Death Cross: 50-day SMA below 200-day SMA")
    
    # Price relative to SMA
    if indicators.get('price_rel_sma_200') is not None:
        price_rel_sma = indicators['price_rel_sma_200']
        if price_rel_sma > 0.05:
            report['signals'].append(f"Price is {price_rel_sma*100:.1f}% above 200-day SMA")
        elif price_rel_sma < -0.05:
            report['warnings'].append(f"Price is {-price_rel_sma*100:.1f}% below 200-day SMA")
    
    # Bollinger Band position
    if indicators.get('bb_position') is not None:
        bb_pos = indicators['bb_position']
        if bb_pos > 0.8:
            report['warnings'].append("Price near upper Bollinger Band, potentially overbought")
        elif bb_pos < -0.8:
            report['signals'].append("Price near lower Bollinger Band, potentially oversold")
    
    # Price target upside
    if indicators.get('price_target_upside') is not None:
        upside = indicators['price_target_upside']
        if upside > 0.15:
            report['signals'].append(f"Analyst consensus suggests {upside*100:.1f}% upside potential")
        elif upside < -0.15:
            report['warnings'].append(f"Analyst consensus suggests {-upside*100:.1f}% downside risk")
    
    # Price performance signals
    if indicators.get('price_performance_1m') is not None:
        perf_1m = indicators['price_performance_1m']
        if perf_1m > 0.1:
            report['signals'].append(f"Strong 1-month performance: +{perf_1m*100:.1f}%")
        elif perf_1m < -0.1:
            report['warnings'].append(f"Weak 1-month performance: {perf_1m*100:.1f}%")
    
    return report