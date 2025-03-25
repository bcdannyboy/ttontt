import asyncio
import time
import threading
import logging

from .utils import get_openbb_client, openbb_has_technical
from .constants import API_CALLS_PER_MINUTE, CACHE_SIZE

logger = logging.getLogger(__name__)

# Semaphore for concurrent API calls.
api_semaphore = asyncio.Semaphore(40)
api_call_timestamps = []
api_lock = threading.RLock()

# Simple cache for API results.
api_cache = {}

async def rate_limited_api_call(func, *args, **kwargs):
    """
    Call an API function with rate limiting and caching.
    Retries up to 3 times if a rate limit error is encountered.
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
            for attempt in range(3):
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
                    if "Limit Reach" in str(e):
                        logger.warning(f"Attempt {attempt+1}: Rate limit error encountered. Retrying after delay...")
                        await asyncio.sleep(60)
                    else:
                        raise
            raise Exception("Max retry attempts reached for API call")
        except Exception as e:
            logger.error(f"API call error: {e}")
            raise

async def get_technical_data_async(ticker: str):
    """
    Fetch technical data for a ticker asynchronously using multiple API calls.
    Includes a fallback mechanism for when technical module is not available.
    """
    from datetime import datetime, timedelta
    technical_data = {}
    try:
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        obb_client = get_openbb_client()
        
        # First try to get basic price data - this should work regardless of technical module
        price_history_task = rate_limited_api_call(
            obb_client.equity.price.historical,
            symbol=ticker, start_date=one_year_ago, provider='fmp'
        )
        price_perf_task = rate_limited_api_call(
            obb_client.equity.price.performance,
            symbol=ticker, provider='fmp'
        )
        price_target_task = rate_limited_api_call(
            obb_client.equity.estimates.consensus,
            symbol=ticker, provider='fmp'
        )
        results = await asyncio.gather(
            price_history_task, price_perf_task, price_target_task,
            return_exceptions=True
        )
        price_history_response, price_perf_response, price_target_response = results
        
        if isinstance(price_history_response, Exception):
            logger.error(f"Essential price history data fetch failed for {ticker}: {price_history_response}")
            return (ticker, technical_data)
            
        technical_data['price_history'] = price_history_response.results if not isinstance(price_history_response, Exception) else []
        technical_data['price_performance'] = price_perf_response.results if not isinstance(price_perf_response, Exception) else []
        technical_data['price_target'] = price_target_response.results if not isinstance(price_target_response, Exception) else []
        
        # Check if technical module is available
        if openbb_has_technical() and technical_data['price_history']:
            try:
                # If technical module is available, use it
                await _fetch_technical_data_with_obb(obb_client, technical_data, ticker)
            except Exception as e:
                logger.error(f"Error fetching technical data with OpenBB for {ticker}: {e}")
                logger.error("Falling back to internal calculations")
                
                # If any error occurs, use our fallback implementation
                _calculate_technical_indicators_fallback(technical_data)
        else:
            # If no technical module, use our fallback implementation
            logger.warning(f"Technical module not available for {ticker}. Using fallback calculations.")
            _calculate_technical_indicators_fallback(technical_data)
    
    except Exception as e:
        logger.error(f"Error fetching technical data for {ticker}: {e}")
        logger.exception(e)
    
    return (ticker, technical_data)

async def _fetch_technical_data_with_obb(obb_client, technical_data, ticker):
    """
    Fetch technical indicator data using OpenBB technical module.
    
    Args:
        obb_client: OpenBB client instance
        technical_data: Dictionary to populate with technical data
        ticker: Stock ticker symbol
    """
    price_data = technical_data['price_history']
    
    # Define technical indicators to fetch
    indicator_tasks = []
    
    # SMA calculations
    indicator_tasks.append(("sma_50", rate_limited_api_call(
        obb_client.technical.sma,
        data=price_data, target='close', length=50
    )))
    
    indicator_tasks.append(("sma_200", rate_limited_api_call(
        obb_client.technical.sma,
        data=price_data, target='close', length=200
    )))
    
    # EMA calculations
    indicator_tasks.append(("ema_12", rate_limited_api_call(
        obb_client.technical.ema,
        data=price_data, target='close', length=12
    )))
    
    indicator_tasks.append(("ema_26", rate_limited_api_call(
        obb_client.technical.ema,
        data=price_data, target='close', length=26
    )))
    
    indicator_tasks.append(("ema_50", rate_limited_api_call(
        obb_client.technical.ema,
        data=price_data, target='close', length=50
    )))
    
    # Other indicators
    indicator_tasks.append(("bbands", rate_limited_api_call(
        obb_client.technical.bbands,
        data=price_data, target='close', length=20, std=2
    )))
    
    indicator_tasks.append(("keltner", rate_limited_api_call(
        obb_client.technical.kc,
        data=price_data, length=20, scalar=2
    )))
    
    indicator_tasks.append(("macd", rate_limited_api_call(
        obb_client.technical.macd,
        data=price_data, target='close', fast=12, slow=26, signal=9
    )))
    
    indicator_tasks.append(("rsi", rate_limited_api_call(
        obb_client.technical.rsi,
        data=price_data, target='close', length=14
    )))
    
    indicator_tasks.append(("stoch", rate_limited_api_call(
        obb_client.technical.stoch,
        data=price_data, fast_k_period=14, slow_d_period=3
    )))
    
    indicator_tasks.append(("cci", rate_limited_api_call(
        obb_client.technical.cci,
        data=price_data, length=20
    )))
    
    indicator_tasks.append(("adx", rate_limited_api_call(
        obb_client.technical.adx,
        data=price_data, length=14
    )))
    
    indicator_tasks.append(("obv", rate_limited_api_call(
        obb_client.technical.obv,
        data=price_data
    )))
    
    indicator_tasks.append(("ad", rate_limited_api_call(
        obb_client.technical.ad,
        data=price_data
    )))
    
    # Volume & volatility indicators
    indicator_tasks.append(("atr", rate_limited_api_call(
        obb_client.technical.atr,
        data=price_data, length=14
    )))
    
    indicator_tasks.append(("donchian", rate_limited_api_call(
        obb_client.technical.donchian,
        data=price_data, lower_length=20, upper_length=20
    )))
    
    indicator_tasks.append(("fisher", rate_limited_api_call(
        obb_client.technical.fisher,
        data=price_data, length=14
    )))
    
    indicator_tasks.append(("ichimoku", rate_limited_api_call(
        obb_client.technical.ichimoku,
        data=price_data, conversion=9, base=26
    )))
    
    indicator_tasks.append(("adosc", rate_limited_api_call(
        obb_client.technical.adosc,
        data=price_data, fast=3, slow=10
    )))
    
    indicator_tasks.append(("vwap", rate_limited_api_call(
        obb_client.technical.vwap,
        data=price_data, anchor='D'
    )))
    
    indicator_tasks.append(("clenow", rate_limited_api_call(
        obb_client.technical.clenow,
        data=price_data, period=90
    )))
    
    # Process results
    for name, task in indicator_tasks:
        try:
            result = await task
            technical_data[name] = result.results if hasattr(result, 'results') else []
        except Exception as e:
            logger.warning(f"Error calculating {name} for {ticker}: {e}")
            technical_data[name] = []
    
    # Try additional indicators with separate error handling
    try:
        cones_task = rate_limited_api_call(
            obb_client.technical.cones,
            data=price_data, lower_q=0.25, upper_q=0.75, model='std'
        )
        cones_result = await cones_task
        technical_data['cones'] = cones_result.results if hasattr(cones_result, 'results') else []
    except Exception as e:
        logger.warning(f"Error calculating volatility cones for {ticker}: {e}")
        technical_data['cones'] = []
    
    try:
        aroon_task = rate_limited_api_call(
            obb_client.technical.aroon,
            data=price_data, length=25
        )
        aroon_result = await aroon_task
        technical_data['aroon'] = aroon_result.results if hasattr(aroon_result, 'results') else []
    except Exception as e:
        logger.warning(f"Error calculating Aroon for {ticker}: {e}")
        technical_data['aroon'] = []

def _calculate_technical_indicators_fallback(technical_data):
    """
    Calculate technical indicators using internal implementations when
    OpenBB technical module is not available.
    
    Args:
        technical_data: Dictionary containing price history data
    """
    import numpy as np
    
    # Make sure we have price data to work with
    if not technical_data.get('price_history') or len(technical_data['price_history']) < 10:
        logger.warning("Insufficient price history for technical calculations")
        return
    
    price_data = technical_data['price_history']
    closes = []
    highs = []
    lows = []
    opens = []
    volumes = []
    dates = []
    
    # Extract OHLCV data from price history
    for p in price_data:
        if hasattr(p, 'close'):
            closes.append(getattr(p, 'close', None))
        if hasattr(p, 'high'):
            highs.append(getattr(p, 'high', None))
        if hasattr(p, 'low'):
            lows.append(getattr(p, 'low', None))
        if hasattr(p, 'open'):
            opens.append(getattr(p, 'open', None))
        if hasattr(p, 'volume'):
            volumes.append(getattr(p, 'volume', None))
        if hasattr(p, 'date'):
            dates.append(getattr(p, 'date', None))
    
    # Convert to numpy arrays for calculations
    closes = np.array(closes, dtype=float)
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    opens = np.array(opens, dtype=float)
    volumes = np.array(volumes, dtype=float)
    
    # Calculate SMA
    if len(closes) >= 50:
        sma_50 = _calculate_sma(closes, 50)
        technical_data['sma_50'] = _format_indicator_output(sma_50, dates[-len(sma_50):], 'sma_50')
    
    if len(closes) >= 200:
        sma_200 = _calculate_sma(closes, 200)
        technical_data['sma_200'] = _format_indicator_output(sma_200, dates[-len(sma_200):], 'sma_200')
    
    # Calculate EMA
    if len(closes) >= 12:
        ema_12 = _calculate_ema(closes, 12)
        technical_data['ema_12'] = _format_indicator_output(ema_12, dates[-len(ema_12):], 'ema_12')
    
    if len(closes) >= 26:
        ema_26 = _calculate_ema(closes, 26)
        technical_data['ema_26'] = _format_indicator_output(ema_26, dates[-len(ema_26):], 'ema_26')
    
    if len(closes) >= 50:
        ema_50 = _calculate_ema(closes, 50)
        technical_data['ema_50'] = _format_indicator_output(ema_50, dates[-len(ema_50):], 'ema_50')
    
    # Calculate RSI
    if len(closes) >= 14:
        rsi = _calculate_rsi(closes, 14)
        technical_data['rsi'] = _format_indicator_output(rsi, dates[-len(rsi):], 'rsi')
    
    # Calculate MACD
    if len(closes) >= 26:
        macd_line, signal_line, histogram = _calculate_macd(closes, 12, 26, 9)
        macd_results = []
        for i in range(len(macd_line)):
            macd_results.append({
                'date': dates[-len(macd_line) + i] if i < len(dates) else None,
                'macd': macd_line[i],
                'signal': signal_line[i],
                'histogram': histogram[i]
            })
        technical_data['macd'] = macd_results
    
    # Calculate Bollinger Bands
    if len(closes) >= 20:
        upper, middle, lower = _calculate_bbands(closes, 20, 2)
        bbands_results = []
        for i in range(len(upper)):
            bbands_results.append({
                'date': dates[-len(upper) + i] if i < len(dates) else None,
                'upper': upper[i],
                'middle': middle[i],
                'lower': lower[i]
            })
        technical_data['bbands'] = bbands_results
    
    # Calculate ATR
    if len(closes) >= 14 and len(highs) >= 14 and len(lows) >= 14:
        atr = _calculate_atr(highs, lows, closes, 14)
        technical_data['atr'] = _format_indicator_output(atr, dates[-len(atr):], 'atr')

def _format_indicator_output(values, dates, indicator_name):
    """Format indicator output to match OpenBB's format"""
    results = []
    for i, value in enumerate(values):
        date = dates[i] if i < len(dates) else None
        result = {'date': date, indicator_name: value}
        results.append(result)
    return results

def _calculate_sma(prices, window):
    """Simple Moving Average calculation"""
    import numpy as np
    sma = np.convolve(prices, np.ones(window)/window, mode='valid')
    return sma

def _calculate_ema(prices, window):
    """Exponential Moving Average calculation"""
    import numpy as np
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    ema = np.convolve(prices, weights, mode='valid')
    # Pad with NaN to maintain original length
    return ema

def _calculate_rsi(prices, window):
    """Relative Strength Index calculation"""
    import numpy as np
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1. + rs)
    
    for i in range(window, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
            
        up = (up*(window-1) + upval)/window
        down = (down*(window-1) + downval)/window
        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)
    
    return rsi[window-1:]

def _calculate_macd(prices, fast_length, slow_length, signal_length):
    """Moving Average Convergence Divergence calculation"""
    import numpy as np
    # Calculate EMAs
    ema_fast = _calculate_ema(prices, fast_length)
    ema_slow = _calculate_ema(prices, slow_length)
    
    # Truncate to same length
    min_len = min(len(ema_fast), len(ema_slow))
    ema_fast = ema_fast[-min_len:]
    ema_slow = ema_slow[-min_len:]
    
    # Calculate MACD line
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    signal_line = _calculate_ema(macd_line, signal_length)
    
    # Calculate histogram
    histogram = macd_line[-len(signal_line):] - signal_line
    
    # Make all arrays the same length
    macd_line = macd_line[-len(signal_line):]
    
    return macd_line, signal_line, histogram

def _calculate_bbands(prices, window, num_std_dev):
    """Bollinger Bands calculation"""
    import numpy as np
    sma = _calculate_sma(prices, window)
    
    # Calculate rolling standard deviation
    rstd = np.zeros_like(sma)
    for i in range(len(sma)):
        start_idx = len(prices) - len(sma) + i - window + 1
        end_idx = len(prices) - len(sma) + i + 1
        rstd[i] = np.std(prices[start_idx:end_idx])
    
    upper_band = sma + rstd * num_std_dev
    lower_band = sma - rstd * num_std_dev
    
    return upper_band, sma, lower_band

def _calculate_atr(high, low, close, window):
    """Average True Range calculation"""
    import numpy as np
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    
    tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
    atr = np.zeros(len(tr) - window + 1)
    
    # First ATR is simple average
    atr[0] = np.mean(tr[:window])
    
    # Rest use smoothing formula
    k = 1.0 / window
    for i in range(1, len(atr)):
        atr[i] = (1 - k) * atr[i-1] + k * tr[window + i - 1]
    
    return atr