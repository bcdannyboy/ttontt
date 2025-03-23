"""
Historical Volatility Analysis
============================

This module provides functions for calculating and analyzing historical volatility
using various methods and time windows.
"""

import numpy as np
import pandas as pd
import logging
import statistics
from typing import List, Dict, Any, Tuple, Union, Optional
import concurrent.futures
from functools import partial
from scipy.stats import skew, kurtosis

# Configure logging
logger = logging.getLogger(__name__)

def get_combined_historical_volatility(data, lower_q=0.25, upper_q=0.75, trading_periods=252, is_crypto=False, index="date"):
    """
    Calculate the realized volatility cones using all available volatility models,
    combine the results by window size, and compute summary statistics including
    average, standard deviation bounds, skew, and kurtosis for each volatility metric.
    
    The returned dictionary is keyed by the window size. For each window, the dictionary
    contains the following keys from all volatility models:
        - 'realized'
        - 'min'
        - 'lower_25%'
        - 'median'
        - 'upper_75%'
        - 'max'
    
    For each metric above, additional keys are added:
        - 'avg_<metric>': the average of the array.
        - 'min<metric>': the lower bound (average minus the standard deviation).
        - 'max<metric>': the upper bound (average plus the standard deviation).
        - 'skew_<metric>': the skew of the array.
        - 'kurtosis_<metric>': the kurtosis of the array.
    
    Additionally, this function computes a "timeframe indicator" based on the realized volatility,
    its skew, and its kurtosis. In this example the score is defined as:
    
        score = avg_realized + abs(skew_realized) + abs(kurtosis_realized)
    
    The best timeframe is the one with the lowest score.
    
    Parameters:
        data (list[dict]): Price data to use for the calculation.
        lower_q (float): Lower quantile value for calculations.
        upper_q (float): Upper quantile value for calculations.
        trading_periods (int): Number of trading periods in a year (default: 252).
        is_crypto (bool): Whether the data is crypto (True uses 365 days instead of 252).
        index (str): The index column name to use from the data (default: "date").
    
    Returns:
        tuple: (combined_vols, best_timeframe)
            - combined_vols: dict with window keys and raw/statistical values.
            - best_timeframe: dict with the selected window, its score, and details.
    """
    from openbb import obb
    
    models = [
        "std", 
        "parkinson", 
        "garman_klass", 
        "hodges_tompkins", 
        "rogers_satchell", 
        "yang_zhang"
    ]
    combined_vols = {}
    
    # Collect arrays from each volatility model for each window.
    for model in models:
        try:
            cones_data = obb.technical.cones(
                data=data,
                lower_q=lower_q,
                upper_q=upper_q,
                model=model,
                trading_periods=trading_periods,
                is_crypto=is_crypto,
                index=index
            )
            for cone in cones_data.results:
                try:
                    window = cone.window
                except AttributeError:
                    logger.warning(f"Cone object does not have attribute 'window': {cone}")
                    continue

                if window not in combined_vols:
                    combined_vols[window] = {
                        "realized": [],
                        "min": [],
                        "lower_25%": [],
                        "median": [],
                        "upper_75%": [],
                        "max": []
                    }
                combined_vols[window]["realized"].append(getattr(cone, "realized", None))
                combined_vols[window]["min"].append(getattr(cone, "min", None))
                combined_vols[window]["lower_25%"].append(getattr(cone, "lower_25%", None))
                combined_vols[window]["median"].append(getattr(cone, "median", None))
                combined_vols[window]["upper_75%"].append(getattr(cone, "upper_75%", None))
                combined_vols[window]["max"].append(getattr(cone, "max", None))
        except Exception as e:
            logger.error(f"Error calculating cones for model {model}: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # If no results were successfully calculated with obb, fall back to custom implementation
    if not combined_vols:
        logger.warning("No volatility cones calculated with OpenBB, falling back to custom implementation")
        return _get_combined_historical_volatility_fallback(
            data, 
            lower_q=lower_q, 
            upper_q=upper_q,
            trading_periods=trading_periods,
            is_crypto=is_crypto,
            index=index
        )
    
    # Define the metrics for which to compute statistics.
    metrics = ["realized", "min", "lower_25%", "median", "upper_75%", "max"]
    
    # For each window and each metric, compute average, bounds, skew, and kurtosis.
    for window, values_dict in combined_vols.items():
        for metric in metrics:
            # Filter out any None values.
            arr = [v for v in values_dict[metric] if v is not None]
            if not arr:
                continue
            try:
                avg_val = statistics.mean(arr)
                std_val = statistics.stdev(arr) if len(arr) > 1 else 0
                values_dict[f"avg_{metric}"] = avg_val
                values_dict[f"min{metric}"] = avg_val - std_val
                values_dict[f"max{metric}"] = avg_val + std_val
                values_dict[f"skew_{metric}"] = skew(arr)
                values_dict[f"kurtosis_{metric}"] = kurtosis(arr)
            except Exception as e:
                logger.error(f"Error calculating statistics for window {window}, metric {metric}: {e}")
                continue
    
    # Compute the best timeframe indicator.
    # Here we use the 'realized' metric as the key measure.
    best_window = None
    best_score = None
    for window, values in combined_vols.items():
        if "avg_realized" in values and "skew_realized" in values and "kurtosis_realized" in values:
            # Define a simple scoring function: lower is better.
            score = values["avg_realized"] + abs(values["skew_realized"]) + abs(values["kurtosis_realized"])
            if best_score is None or score < best_score:
                best_score = score
                best_window = window
    
    best_timeframe = {
        "window": best_window,
        "score": best_score,
        "details": combined_vols.get(best_window, {})
    }
    
    return combined_vols, best_timeframe

# Maintain compatibility with existing code by keeping calculation helpers

def calculate_realized_volatility(returns: np.ndarray, annualization_factor: float = 252) -> float:
    """
    Calculate realized volatility from a series of returns.
    
    Args:
        returns: Array of returns
        annualization_factor: Factor to annualize volatility
        
    Returns:
        Annualized volatility
    """
    if len(returns) < 2:
        return np.nan
    return np.std(returns, ddof=1) * np.sqrt(annualization_factor)

def calculate_parkinson_volatility(high: np.ndarray, low: np.ndarray, 
                                annualization_factor: float = 252) -> float:
    """
    Calculate Parkinson volatility using high-low price range.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        annualization_factor: Factor to annualize volatility
        
    Returns:
        Annualized Parkinson volatility
    """
    if len(high) < 2 or len(low) < 2:
        return np.nan
    
    # Calculate squared log range
    squared_log_range = (np.log(high / low) ** 2) / (4 * np.log(2))
    
    # Mean of squared log range
    mean_squared_log_range = np.nanmean(squared_log_range)
    
    # Annualize
    parkinson_vol = np.sqrt(mean_squared_log_range * annualization_factor)
    
    return parkinson_vol

def calculate_garman_klass_volatility(open_price: np.ndarray, high: np.ndarray, 
                                   low: np.ndarray, close: np.ndarray,
                                   annualization_factor: float = 252) -> float:
    """
    Calculate Garman-Klass volatility using OHLC data.
    
    Args:
        open_price: Array of opening prices
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        annualization_factor: Factor to annualize volatility
        
    Returns:
        Annualized Garman-Klass volatility
    """
    if len(high) < 2 or len(low) < 2 or len(close) < 2 or len(open_price) < 2:
        return np.nan
    
    # Calculate log ratios
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_price) ** 2
    
    # Garman-Klass estimator
    gk_estimator = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    
    # Mean of estimator
    mean_gk = np.nanmean(gk_estimator)
    
    # Annualize
    gk_vol = np.sqrt(mean_gk * annualization_factor)
    
    return gk_vol

def calculate_rogers_satchell_volatility(open_price: np.ndarray, high: np.ndarray, 
                                      low: np.ndarray, close: np.ndarray,
                                      annualization_factor: float = 252) -> float:
    """
    Calculate Rogers-Satchell volatility using OHLC data.
    
    Args:
        open_price: Array of opening prices
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        annualization_factor: Factor to annualize volatility
        
    Returns:
        Annualized Rogers-Satchell volatility
    """
    if len(high) < 2 or len(low) < 2 or len(close) < 2 or len(open_price) < 2:
        return np.nan
    
    # Calculate log ratios
    log_ho = np.log(high / open_price)
    log_lo = np.log(low / open_price)
    log_co = np.log(close / open_price)
    
    # Rogers-Satchell estimator
    rs_estimator = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    # Mean of estimator
    mean_rs = np.nanmean(rs_estimator)
    
    # Annualize
    rs_vol = np.sqrt(mean_rs * annualization_factor)
    
    return rs_vol

def calculate_yang_zhang_volatility(open_price: np.ndarray, high: np.ndarray, 
                                 low: np.ndarray, close: np.ndarray, 
                                 prev_close: np.ndarray = None,
                                 annualization_factor: float = 252,
                                 k: float = 0.34) -> float:
    """
    Calculate Yang-Zhang volatility using OHLC data.
    
    Args:
        open_price: Array of opening prices
        high: Array of high prices
        low: Array of low prices
        close: Array of closing prices
        prev_close: Array of previous closing prices (optional)
        annualization_factor: Factor to annualize volatility
        k: Yang-Zhang parameter
        
    Returns:
        Annualized Yang-Zhang volatility
    """
    if len(high) < 2 or len(low) < 2 or len(close) < 2 or len(open_price) < 2:
        return np.nan
    
    # If previous close not provided, create it by shifting close
    if prev_close is None:
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
    
    # Calculate overnight volatility (close to open)
    log_co_overnight = np.log(open_price / prev_close)
    sigma_overnight = np.nanvar(log_co_overnight)
    
    # Calculate open to close volatility
    log_oc = np.log(close / open_price)
    sigma_open_close = np.nanvar(log_oc)
    
    # Calculate Rogers-Satchell volatility
    rs_vol = calculate_rogers_satchell_volatility(
        open_price, high, low, close, annualization_factor=1)
    sigma_rs = rs_vol ** 2
    
    # Yang-Zhang volatility combining overnight, open-close and RS
    sigma_yz = sigma_overnight + k * sigma_open_close + (1 - k) * sigma_rs
    
    # Annualize
    yz_vol = np.sqrt(sigma_yz * annualization_factor)
    
    return yz_vol

def _calculate_volatility_for_window(data: List[Dict[str, Any]], 
                                  window_size: int, 
                                  trading_periods: int,
                                  index: str = "date") -> Dict[str, float]:
    """
    Calculate volatility metrics for a specific window size.
    
    Args:
        data: List of price data dictionaries
        window_size: Size of the rolling window
        trading_periods: Number of trading periods per year
        index: Name of the index column
        
    Returns:
        Dictionary with volatility metrics
    """
    # Convert data to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure required columns exist
    required_cols = ['close', 'high', 'low']
    for col in required_cols:
        if col not in df.columns:
            # For missing columns, use close price
            df[col] = df['close']
    
    # Add open column if missing
    if 'open' not in df.columns:
        df['open'] = df['close'].shift(1).fillna(df['close'])
    
    # Sort by date/index if available
    if index in df.columns:
        df = df.sort_values(index)
    
    # Calculate returns
    df['return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate realized volatility
    returns = df['return'].dropna().values
    realized_vol = calculate_realized_volatility(returns[-window_size:], trading_periods)
    
    # Only proceed with other calculations if we have sufficient OHLC data
    if window_size > 0 and len(df) >= window_size:
        window_df = df.iloc[-window_size:]
        
        # Extract arrays for calculations
        open_prices = window_df['open'].values
        high_prices = window_df['high'].values
        low_prices = window_df['low'].values
        close_prices = window_df['close'].values
        prev_close_prices = np.roll(close_prices, 1)
        prev_close_prices[0] = open_prices[0]
        
        # Calculate different volatility metrics
        parkinson_vol = calculate_parkinson_volatility(
            high_prices, low_prices, trading_periods)
        
        gk_vol = calculate_garman_klass_volatility(
            open_prices, high_prices, low_prices, close_prices, trading_periods)
        
        rs_vol = calculate_rogers_satchell_volatility(
            open_prices, high_prices, low_prices, close_prices, trading_periods)
        
        yz_vol = calculate_yang_zhang_volatility(
            open_prices, high_prices, low_prices, close_prices, 
            prev_close_prices, trading_periods)
    else:
        parkinson_vol = np.nan
        gk_vol = np.nan
        rs_vol = np.nan
        yz_vol = np.nan
    
    # Return calculated volatilities
    return {
        'window': window_size,
        'avg_realized': realized_vol,
        'parkinson': parkinson_vol,
        'garman_klass': gk_vol,
        'rogers_satchell': rs_vol,
        'yang_zhang': yz_vol
    }

def _get_combined_historical_volatility_fallback(data: List[Dict[str, Any]], 
                                             window_sizes: List[int] = None,
                                             lower_q: float = 0.25, 
                                             upper_q: float = 0.75,
                                             trading_periods: int = 252,
                                             is_crypto: bool = False,
                                             index: str = "date") -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Fallback implementation to calculate historical volatility using multiple methods and window sizes
    when OpenBB's obb.technical.cones function fails.
    
    Args:
        data: List of price data dictionaries
        window_sizes: List of window sizes to use
        lower_q: Lower quantile for filtering
        upper_q: Upper quantile for filtering
        trading_periods: Number of trading periods per year
        is_crypto: Whether the data is for a cryptocurrency
        index: Name of the index column
        
    Returns:
        Tuple of (volatility_models, best_timeframe)
    """
    # Adjust trading periods for crypto if 24/7 market
    if is_crypto:
        trading_periods = 365
    
    # Set default window sizes if not provided
    if window_sizes is None:
        window_sizes = [20, 30, 60, 90, 120, 252]
    
    # Filter out window sizes larger than available data
    max_window = min(len(data), max(window_sizes))
    valid_windows = [w for w in window_sizes if w <= max_window]
    
    if not valid_windows:
        logger.warning(f"No valid window sizes for data of length {len(data)}")
        min_window = min(max(5, len(data) // 2), len(data))
        valid_windows = [min_window]
    
    # Calculate volatility for each window size in parallel
    results = {}
    
    # Use a thread pool for parallel calculation
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(valid_windows), 8)) as executor:
        # Create a partial function with fixed parameters
        calc_func = partial(
            _calculate_volatility_for_window, 
            data=data, 
            trading_periods=trading_periods,
            index=index
        )
        
        # Submit tasks for each window size
        future_to_window = {
            executor.submit(calc_func, window_size=window): window 
            for window in valid_windows
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_window):
            window = future_to_window[future]
            try:
                result = future.result()
                results[window] = result
            except Exception as e:
                logger.error(f"Error calculating volatility for window {window}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    # If no results were successfully calculated, return empty
    if not results:
        return {}, {"error": "No valid volatility calculations"}
    
    # Convert results to format compatible with main function
    combined_vols = {}
    for window, result in results.items():
        # Create entry structure
        combined_vols[window] = {
            "realized": [result.get('avg_realized')],
            "min": [],
            "lower_25%": [],
            "median": [],
            "upper_75%": [],
            "max": []
        }
        
        # Add statistics
        for metric in ["realized"]:
            if metric == "realized" and result.get('avg_realized') is not None:
                value = result.get('avg_realized')
                combined_vols[window][f"avg_{metric}"] = value
                combined_vols[window][f"min{metric}"] = value * 0.9  # Approximation
                combined_vols[window][f"max{metric}"] = value * 1.1  # Approximation
                combined_vols[window][f"skew_{metric}"] = 0  # Default
                combined_vols[window][f"kurtosis_{metric}"] = 0  # Default
    
    # Determine best window size based on consistency
    volatilities = [(window, result.get('avg_realized', 0)) 
                   for window, result in results.items() 
                   if result.get('avg_realized') is not None]
    
    if not volatilities:
        # No valid volatility calculations
        return {}, {"error": "No valid volatility calculations"}
    
    # Sort by window size (largest window first)
    volatilities.sort(key=lambda x: x[0], reverse=True)
    
    # Select largest window with valid volatility
    best_window, best_vol = volatilities[0]
    
    # Format output
    best_timeframe = {
        "window": best_window,
        "score": 0,  # Default score
        "details": combined_vols.get(best_window, {})
    }
    
    return combined_vols, best_timeframe

def analyze_historical_volatility(ticker: str) -> Dict[str, Any]:
    """
    Analyze historical volatility patterns to enhance Monte Carlo simulations.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with volatility analysis data
    """
    try:
        from src.simulation.utils import initialize_openbb
        
        # Initialize OpenBB client
        initialize_openbb()
        from openbb import obb
        
        # Fetch historical data
        from datetime import datetime, timedelta
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        price_history_response = obb.equity.price.historical(
            symbol=ticker, start_date=one_year_ago, provider='fmp'
        )
        
        if not hasattr(price_history_response, 'results') or not price_history_response.results:
            return {"error": f"No historical price data available for {ticker}"}
        
        # Calculate volatility
        combined_vols, best_timeframe = get_combined_historical_volatility(
            data=price_history_response.results,
            lower_q=0.25,
            upper_q=0.75,
            trading_periods=252,
            is_crypto=False,
            index="date"
        )
        
        # Format results
        volatility_analysis = {
            "ticker": ticker,
            "best_timeframe": best_timeframe,
            "volatility_by_window": {}
        }
        
        for window, values in combined_vols.items():
            volatility_analysis["volatility_by_window"][window] = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) and np.isfinite(v) else v
                for k, v in values.items()
                if not isinstance(v, (list, np.ndarray))
            }
            
        return volatility_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing historical volatility for {ticker}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def analyze_normality(ticker: str) -> Dict[str, Any]:
    """
    Analyze the normality of returns to improve simulation accuracy.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with normality test results
    """
    try:
        from src.simulation.utils import initialize_openbb
        
        # Initialize OpenBB client
        initialize_openbb()
        from openbb import obb
        
        # Fetch historical data
        from datetime import datetime, timedelta
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        price_history_response = obb.equity.price.historical(
            symbol=ticker, start_date=one_year_ago, provider='fmp'
        )
        
        if not hasattr(price_history_response, 'results') or not price_history_response.results:
            return {"error": f"No historical price data available for {ticker}"}
        
        # Process data
        price_data = []
        for record in price_history_response.results:
            price_data.append({
                'date': getattr(record, 'date', None),
                'close': getattr(record, 'close', None)
            })
            
        df = pd.DataFrame(price_data)
        df['return'] = df['close'].pct_change()
        df = df.dropna()
        
        if len(df) < 30:
            return {"error": f"Insufficient historical data for {ticker}. Need at least 30 days."}
        
        # Run normality tests
        normality_results = obb.quantitative.normality(
            data=df.to_dict('records'), target='return'
        )
        
        if hasattr(normality_results, 'results'):
            results = normality_results.results
            
            # Format results
            normality_analysis = {
                "ticker": ticker,
                "jarque_bera": {
                    "statistic": float(getattr(results, 'jarque_bera_stat', 0)),
                    "p_value": float(getattr(results, 'jarque_bera_p', 0)),
                    "is_normal": bool(getattr(results, 'jarque_bera_normal', False))
                },
                "shapiro": {
                    "statistic": float(getattr(results, 'shapiro_stat', 0)),
                    "p_value": float(getattr(results, 'shapiro_p', 0)),
                    "is_normal": bool(getattr(results, 'shapiro_normal', False))
                },
                "kolmogorov": {
                    "statistic": float(getattr(results, 'kolmogorov_stat', 0)),
                    "p_value": float(getattr(results, 'kolmogorov_p', 0)),
                    "is_normal": bool(getattr(results, 'kolmogorov_normal', False))
                },
                "kurtosis": {
                    "value": float(getattr(results, 'kurtosis', 0)),
                    "is_normal": bool(getattr(results, 'kurtosis_normal', False))
                },
                "skewness": {
                    "value": float(getattr(results, 'skew', 0)),
                    "is_normal": bool(getattr(results, 'skew_normal', False))
                },
                "overall_normal": bool(getattr(results, 'is_normal', False))
            }
            
            return normality_analysis
        else:
            return {"error": "Failed to run normality tests"}
            
    except Exception as e:
        logger.error(f"Error analyzing normality for {ticker}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}