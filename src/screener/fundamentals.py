"""
Fundamental Stock Screener
==========================

A comprehensive stock screening tool that evaluates companies based on their fundamental financial metrics 
and assigns composite scores that reflect their financial strength and growth prospects.

Overall Approach:
----------------
1. Data Collection: Retrieves extensive financial data from multiple financial statements 
   (income statement, balance sheet, cash flow) and derived metrics (ratios, growth rates)
   for a list of ticker symbols using the OpenBB API.

2. Metric Extraction: Processes raw financial data to extract relevant metrics across five key
   categories: profitability, growth, financial health, valuation, and efficiency.

3. Standardization: Transforms raw metrics into standardized z-scores that represent how many
   standard deviations a company's metric is from the mean of all companies being screened.
   This enables direct comparison across different metrics with different scales and units.

4. Weighted Scoring System: Applies carefully calibrated weights to each metric within its
   category, with positive weights for metrics where higher values are better (e.g., profit margins)
   and negative weights for metrics where lower values are better (e.g., debt ratios).

5. Composite Scoring: Computes category scores and an overall composite score that reflects
   a company's overall fundamental strength, allowing for ranking and comparison across companies.

6. Handling Outliers: Uses statistical methods to handle extreme values without arbitrary caps,
   preserving the nuance of real-world data while still maintaining fair comparisons.

Statistical Methods:
-------------------
- Z-score calculation to standardize metrics across companies
- Weighted averaging to combine multiple factors
- Statistical detection of outliers using quartile-based methods

Key Categories:
--------------
- Profitability: Measures a company's ability to generate earnings
- Growth: Evaluates the company's expansion in revenue, earnings, and assets
- Financial Health: Assesses the company's debt levels, liquidity, and solvency
- Valuation: Examines how the company's stock price relates to its fundamentals
- Efficiency: Gauges how effectively the company utilizes its assets and resources

This approach provides a robust, data-driven method for evaluating stocks based on 
fundamentals rather than market sentiment or technical patterns.
"""

import os
from openbb import obb
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-local storage for OpenBB API session
thread_local = threading.local()

# Define weights for different categories of financial metrics
CATEGORY_WEIGHTS = {
    'profitability': 0.25,
    'growth': 0.30,
    'financial_health': 0.20,
    'valuation': 0.15,
    'efficiency': 0.10
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

# API rate limiting
API_CALLS_PER_MINUTE = 240  # Increased from 120 to 240
api_semaphore = asyncio.Semaphore(40)  # Allow 40 concurrent API calls
api_call_timestamps = []
api_lock = threading.RLock()

# Cache for API results
CACHE_SIZE = 1000
api_cache = {}
metrics_cache = {}

def get_openbb_client():
    """Returns a thread-local OpenBB client instance."""
    if not hasattr(thread_local, "openbb_client"):
        thread_local.openbb_client = obb
    return thread_local.openbb_client

async def rate_limited_api_call(func, *args, **kwargs):
    """
    Rate limiting for API calls with async support and caching
    """
    # Create a cache key based on function name and arguments
    cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
    
    # Check if result is in cache
    if cache_key in api_cache:
        return api_cache[cache_key]
    
    async with api_semaphore:
        with api_lock:
            current_time = time.time()
            # Remove timestamps older than 1 minute
            global api_call_timestamps
            api_call_timestamps = [ts for ts in api_call_timestamps if current_time - ts < 60]
            
            # Check if we're at the rate limit
            if len(api_call_timestamps) >= API_CALLS_PER_MINUTE:
                # Sleep until we can make another call
                sleep_time = 60 - (current_time - api_call_timestamps[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    await asyncio.sleep(sleep_time)
            
            # Add current timestamp to the list
            api_call_timestamps.append(time.time())
        
        # Execute the API call (either in thread or via asyncio)
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                # Execute sync function in a thread pool to avoid blocking event loop
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
            
            # Cache the result
            api_cache[cache_key] = result
            
            # Manage cache size by removing oldest items when needed
            if len(api_cache) > CACHE_SIZE:
                oldest_key = next(iter(api_cache))
                del api_cache[oldest_key]
                
            return result
        except Exception as e:
            logger.error(f"API call error: {e}")
            raise

async def get_financial_data_async(ticker: str) -> Tuple[str, Dict[str, List]]:
    """
    Async version of get_financial_data using OpenBB API.
    Fetches all required financial data for a stock.
    """
    obb_client = get_openbb_client()
    financial_data = {}

    # Create tasks for all the API calls needed
    try:
        # Basic data batch 1
        income_task = rate_limited_api_call(
            obb_client.equity.fundamental.income,
            symbol=ticker, period='annual', limit=5, provider='fmp'
        )
        balance_task = rate_limited_api_call(
            obb_client.equity.fundamental.balance,
            symbol=ticker, period='annual', limit=5, provider='fmp'
        )
        cash_task = rate_limited_api_call(
            obb_client.equity.fundamental.cash,
            symbol=ticker, period='annual', limit=5, provider='fmp'
        )
        
        # Execute the first batch of tasks
        income_response, balance_response, cash_response = await asyncio.gather(
            income_task, balance_task, cash_task, 
            return_exceptions=True
        )
        
        # Check if we have the essential data before proceeding
        if isinstance(income_response, Exception) or isinstance(balance_response, Exception):
            logger.error(f"Essential data fetch failed for {ticker}")
            return (ticker, financial_data)
            
        financial_data['income'] = income_response.results if not isinstance(income_response, Exception) else []
        financial_data['balance'] = balance_response.results if not isinstance(balance_response, Exception) else []
        financial_data['cash'] = cash_response.results if not isinstance(cash_response, Exception) else []
        
        # Proceed only if we have the essential data
        if financial_data['income'] and financial_data['balance']:
            # Growth and metrics batch 2
            income_growth_task = rate_limited_api_call(
                obb_client.equity.fundamental.income_growth,
                symbol=ticker, period='annual', limit=5, provider='fmp'
            )
            balance_growth_task = rate_limited_api_call(
                obb_client.equity.fundamental.balance_growth,
                symbol=ticker, period='annual', limit=5, provider='fmp'
            )
            cash_growth_task = rate_limited_api_call(
                obb_client.equity.fundamental.cash_growth,
                symbol=ticker, period='annual', limit=5, provider='fmp'
            )
            ratios_task = rate_limited_api_call(
                obb_client.equity.fundamental.ratios,
                symbol=ticker, period='annual', limit=5, provider='fmp'
            )
            metrics_task = rate_limited_api_call(
                obb_client.equity.fundamental.metrics,
                symbol=ticker, period='annual', limit=5, provider='fmp'
            )
            
            # Execute the second batch of tasks
            results = await asyncio.gather(
                income_growth_task, balance_growth_task, cash_growth_task, 
                ratios_task, metrics_task,
                return_exceptions=True
            )
            
            # Process results
            financial_data['income_growth'] = results[0].results if not isinstance(results[0], Exception) else []
            financial_data['balance_growth'] = results[1].results if not isinstance(results[1], Exception) else []
            financial_data['cash_growth'] = results[2].results if not isinstance(results[2], Exception) else []
            financial_data['ratios'] = results[3].results if not isinstance(results[3], Exception) else []
            financial_data['metrics'] = results[4].results if not isinstance(results[4], Exception) else []
            
            # Get dividend data separately as it can fail more often
            try:
                dividends_response = await rate_limited_api_call(
                    obb_client.equity.fundamental.dividends,
                    symbol=ticker, provider='fmp'
                )
                financial_data['dividends'] = dividends_response.results
            except Exception as e:
                logger.warning(f"No dividend data for {ticker}: {e}")
                financial_data['dividends'] = []
                
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
    
    return (ticker, financial_data)

def get_attribute_value(obj: Any, attr_name: str, default=0) -> Any:
    """Safely get attribute value from an object."""
    if hasattr(obj, attr_name):
        value = getattr(obj, attr_name)
        if value is not None and not pd.isna(value):
            return value
    return default

# Removed @lru_cache decorator to fix unhashable type: 'dict' error.
def extract_metrics_from_financial_data(financial_data):
    """
    Extract all relevant metrics from financial data.
    
    Args:
        financial_data: Dictionary containing financial data
        
    Returns:
        Dict[str, float]: Dictionary of extracted financial metrics
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Process income statement data (most recent year)
    if isinstance(financial_data, dict) and financial_data.get('income') and len(financial_data['income']) > 0:
        income = financial_data['income'][0]
        
        # Extract profitability metrics
        metrics['gross_profit_margin'] = get_attribute_value(income, 'gross_profit_margin')
        metrics['operating_income_margin'] = get_attribute_value(income, 'operating_income_margin')
        metrics['net_income_margin'] = get_attribute_value(income, 'net_income_margin')
        metrics['ebitda_margin'] = get_attribute_value(income, 'ebitda_margin')
        
        # Extract raw values for possible calculations
        revenue = get_attribute_value(income, 'revenue')
        metrics['revenue'] = revenue
        metrics['gross_profit'] = get_attribute_value(income, 'gross_profit')
        metrics['operating_income'] = get_attribute_value(income, 'operating_income')
        metrics['net_income'] = get_attribute_value(income, 'net_income')
        metrics['ebitda'] = get_attribute_value(income, 'ebitda')
    
    # Process balance sheet data (most recent year)
    if isinstance(financial_data, dict) and financial_data.get('balance') and len(financial_data['balance']) > 0:
        balance = financial_data['balance'][0]
        
        # Extract balance sheet metrics
        metrics['total_assets'] = get_attribute_value(balance, 'total_assets')
        metrics['total_liabilities'] = get_attribute_value(balance, 'total_liabilities')
        metrics['total_shareholders_equity'] = get_attribute_value(balance, 'total_shareholders_equity')
        metrics['cash_and_cash_equivalents'] = get_attribute_value(balance, 'cash_and_cash_equivalents')
        metrics['total_debt'] = get_attribute_value(balance, 'total_debt')
        metrics['net_debt'] = get_attribute_value(balance, 'net_debt')
        
        # Calculate financial health metrics
        if metrics.get('total_assets', 0) > 0:
            metrics['debt_to_assets'] = metrics.get('total_debt', 0) / metrics['total_assets']
        
        if metrics.get('total_shareholders_equity', 0) > 0:
            metrics['debt_to_equity'] = metrics.get('total_debt', 0) / metrics['total_shareholders_equity']
            
            # Return on equity
            if 'net_income' in metrics:
                metrics['return_on_equity'] = metrics['net_income'] / metrics['total_shareholders_equity']
        
        # Return on assets
        if metrics.get('total_assets', 0) > 0 and 'net_income' in metrics:
            metrics['return_on_assets'] = metrics['net_income'] / metrics['total_assets']
        
        # Cash to debt ratio
        if metrics.get('total_debt', 0) > 0:
            metrics['cash_to_debt'] = metrics.get('cash_and_cash_equivalents', 0) / metrics['total_debt']
        else:
            metrics['cash_to_debt'] = 10  # High value for no debt
    
    # Process cash flow data (most recent year)
    if isinstance(financial_data, dict) and financial_data.get('cash') and len(financial_data['cash']) > 0:
        cash_flow = financial_data['cash'][0]
        
        # Extract cash flow metrics
        metrics['operating_cash_flow'] = get_attribute_value(cash_flow, 'operating_cash_flow')
        metrics['capital_expenditure'] = get_attribute_value(cash_flow, 'capital_expenditure')
        metrics['free_cash_flow'] = get_attribute_value(cash_flow, 'free_cash_flow')
        
        # Capital expenditure to revenue ratio
        if 'revenue' in metrics and metrics['revenue'] > 0 and abs(metrics.get('capital_expenditure', 0)) > 0:
            metrics['capex_to_revenue'] = abs(metrics['capital_expenditure']) / metrics['revenue']
    
    # Process growth metrics
    if isinstance(financial_data, dict) and financial_data.get('income_growth') and len(financial_data['income_growth']) > 0:
        income_growth = financial_data['income_growth'][0]
        
        # Extract income growth metrics
        metrics['growth_revenue'] = get_attribute_value(income_growth, 'growth_revenue')
        metrics['growth_gross_profit'] = get_attribute_value(income_growth, 'growth_gross_profit')
        metrics['growth_ebitda'] = get_attribute_value(income_growth, 'growth_ebitda')
        metrics['growth_operating_income'] = get_attribute_value(income_growth, 'growth_operating_income')
        metrics['growth_net_income'] = get_attribute_value(income_growth, 'growth_net_income')
        metrics['growth_eps'] = get_attribute_value(income_growth, 'growth_eps')
    
    # Process balance sheet growth metrics
    if isinstance(financial_data, dict) and financial_data.get('balance_growth') and len(financial_data['balance_growth']) > 0:
        balance_growth = financial_data['balance_growth'][0]
        
        # Extract balance sheet growth metrics
        metrics['growth_total_assets'] = get_attribute_value(balance_growth, 'growth_total_assets')
        metrics['growth_total_liabilities'] = get_attribute_value(balance_growth, 'growth_total_liabilities')
        metrics['growth_total_shareholders_equity'] = get_attribute_value(balance_growth, 'growth_total_shareholders_equity')
        metrics['growth_total_debt'] = get_attribute_value(balance_growth, 'growth_total_debt')
        metrics['growth_net_debt'] = get_attribute_value(balance_growth, 'growth_net_debt')
    
    # Process ratios
    if isinstance(financial_data, dict) and financial_data.get('ratios') and len(financial_data['ratios']) > 0:
        ratios = financial_data['ratios'][0]
        
        # Extract financial ratios
        metrics['current_ratio'] = get_attribute_value(ratios, 'current_ratio')
        metrics['quick_ratio'] = get_attribute_value(ratios, 'quick_ratio')
        metrics['interest_coverage'] = get_attribute_value(ratios, 'interest_coverage')
        metrics['asset_turnover'] = get_attribute_value(ratios, 'asset_turnover')
        metrics['inventory_turnover'] = get_attribute_value(ratios, 'inventory_turnover')
        metrics['receivables_turnover'] = get_attribute_value(ratios, 'receivables_turnover')
        metrics['cash_conversion_cycle'] = get_attribute_value(ratios, 'cash_conversion_cycle')
    
    # Process market metrics
    if isinstance(financial_data, dict) and financial_data.get('metrics') and len(financial_data['metrics']) > 0:
        market_metrics = financial_data['metrics'][0]
        
        # Extract market metrics
        metrics['pe_ratio'] = get_attribute_value(market_metrics, 'pe_ratio')
        metrics['price_to_book'] = get_attribute_value(market_metrics, 'price_to_book')
        metrics['price_to_sales'] = get_attribute_value(market_metrics, 'price_to_sales')
        metrics['ev_to_ebitda'] = get_attribute_value(market_metrics, 'ev_to_ebitda')
        metrics['peg_ratio'] = get_attribute_value(market_metrics, 'peg_ratio')
        metrics['market_cap'] = get_attribute_value(market_metrics, 'market_cap')
        metrics['price'] = get_attribute_value(market_metrics, 'price')
    
    # Process dividend data
    if isinstance(financial_data, dict) and financial_data.get('dividends'):
        dividends = financial_data['dividends']
        annual_dividend = 0
        
        # Calculate annual dividend
        if dividends:
            # Convert datetime to date to avoid comparison issues between date and datetime
            one_year_ago = (datetime.now() - timedelta(days=365)).date()
            recent_dividends = []
            
            for div in dividends:
                try:
                    if hasattr(div, 'ex_dividend_date') and hasattr(div, 'amount'):
                        div_date = div.ex_dividend_date
                        
                        # Convert string date to date object if needed
                        if isinstance(div_date, str):
                            try:
                                div_date = datetime.strptime(div_date, '%Y-%m-%d').date()
                            except ValueError:
                                continue
                        
                        # Convert datetime to date if needed
                        if isinstance(div_date, datetime):
                            div_date = div_date.date()
                        
                        # Now both are date objects for proper comparison
                        if div_date >= one_year_ago:
                            recent_dividends.append(div.amount)
                except Exception as e:
                    # Skip problematic dividend entries
                    logger.warning(f"Error processing dividend entry: {e}")
                    continue
            
            annual_dividend = sum(recent_dividends)
        
        # Calculate dividend yield
        if annual_dividend > 0 and 'price' in metrics and metrics['price'] > 0:
            metrics['dividend_yield'] = annual_dividend / metrics['price']
        else:
            metrics['dividend_yield'] = 0
    
    return metrics

# We'll use NumPy vectorized operations for better performance
def normalize_data(data_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Normalize data using vectorized operations for better performance.
    """
    normalized_data = {ticker: {} for ticker in data_dict}
    metrics_data = {}
    
    # Collect all values for each metric across stocks
    for ticker, metrics in data_dict.items():
        for metric, value in metrics.items():
            if metric not in metrics_data:
                metrics_data[metric] = []
            # Only include valid numerical values
            if isinstance(value, (int, float)) and not pd.isna(value) and value is not None:
                metrics_data[metric].append(value)
    
    # Process and normalize metrics
    for ticker, metrics in data_dict.items():
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)) or pd.isna(value) or value is None:
                continue
                
            # Store the original value, preserving all nuances of the data
            normalized_data[ticker][metric] = value
    
    return normalized_data

def calculate_z_scores(data_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate z-scores with optimized performance using NumPy.
    """
    # Create output dictionary
    z_scores = {ticker: {} for ticker in data_dict}
    
    # Collect all metrics in a dictionary
    metrics_dict = {}
    for ticker, metrics in data_dict.items():
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and not pd.isna(value) and value is not None:
                if metric not in metrics_dict:
                    metrics_dict[metric] = []
                metrics_dict[metric].append((ticker, value))
    
    # Calculate z-scores for each metric using NumPy
    for metric, ticker_values in metrics_dict.items():
        # Skip if we don't have enough data points
        if len(ticker_values) < 2:
            continue
            
        # Extract tickers and values
        tickers, values = zip(*ticker_values)
        values_array = np.array(values)
        
        # Check for skewness only when we have enough data points
        if len(values) > 4:
            # Calculate skewness
            if np.std(values_array) > 0:
                skewness = np.abs(np.mean(((values_array - np.mean(values_array)) / np.std(values_array))**3))
                
                # Use robust statistics for highly skewed data
                if skewness > 2:
                    median = np.median(values_array)
                    iqr = np.percentile(values_array, 75) - np.percentile(values_array, 25)
                    robust_std = max(iqr / 1.349, 1e-10)  # Avoid division by zero
                    
                    # Calculate z-scores using median and robust standard deviation
                    metric_z_scores = (values_array - median) / robust_std
                    
                    # Assign z-scores to each ticker
                    for ticker, z_score in zip(tickers, metric_z_scores):
                        z_scores[ticker][metric] = z_score
                    
                    continue
        
        # Default case: use regular z-scores
        mean = np.mean(values_array)
        std = max(np.std(values_array), 1e-10)  # Avoid division by zero
        
        # Calculate z-scores
        metric_z_scores = (values_array - mean) / std
        
        # Assign z-scores to each ticker
        for ticker, z_score in zip(tickers, metric_z_scores):
            z_scores[ticker][metric] = z_score
    
    return z_scores

def calculate_weighted_score(z_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculate weighted score based on z-scores and weights.
    Optimized version with early returns and preconditions.
    """
    if not z_scores or not weights:
        return 0
        
    score = 0
    total_weight = 0
    valid_metrics = 0
    
    # Only process metrics that exist in both dictionaries
    common_metrics = set(z_scores.keys()) & set(weights.keys())
    
    for metric in common_metrics:
        weight = weights[metric]
        score += z_scores[metric] * weight
        total_weight += abs(weight)
        valid_metrics += 1
    
    # Return 0 if no valid metrics or weights
    if total_weight == 0 or valid_metrics == 0:
        return 0
        
    # Normalize by total weight
    return score / total_weight

async def process_ticker_async(ticker: str) -> Tuple[str, Optional[Dict[str, float]]]:
    """
    Process a single ticker: fetches financial data and extracts metrics.
    Async version for better throughput.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Tuple of (ticker, metrics_dict) or (ticker, None) if processing fails
    """
    try:
        # Get financial data
        ticker, financial_data = await get_financial_data_async(ticker)
        
        # Skip if insufficient data
        if not financial_data.get('income') or not financial_data.get('balance'):
            logger.warning(f"Insufficient financial data for {ticker}. Skipping...")
            return (ticker, None)
        
        # Extract all metrics from financial data
        try:
            # Direct processing without JSON serialization/deserialization
            if ticker in metrics_cache:
                metrics_dict = metrics_cache[ticker]
            else:
                metrics_dict = extract_metrics_from_financial_data(financial_data)
                metrics_cache[ticker] = metrics_dict
                
                # Limit cache size
                if len(metrics_cache) > CACHE_SIZE:
                    oldest_key = next(iter(metrics_cache))
                    del metrics_cache[oldest_key]
                
            logger.debug(f"Successfully processed {ticker}")
            return (ticker, metrics_dict)
        except Exception as e:
            logger.error(f"Error extracting metrics for {ticker}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return (ticker, None)
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return (ticker, None)

async def screen_stocks_async(tickers: List[str], max_concurrent: int = 20) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Async version of screen_stocks for better performance.
    """
    results = []
    all_metrics = {}
    valid_tickers = []
    
    # Process tickers concurrently but with a limit on concurrent operations
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(ticker):
        async with semaphore:
            return await process_ticker_async(ticker)
    
    # Create tasks for all tickers
    tasks = [process_with_semaphore(ticker) for ticker in tickers]
    
    # Process all tasks with a progress bar
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        result = await task
        ticker, metrics_dict = result
        
        # Update progress manually
        progress = i / len(tickers) * 100
        if i % 5 == 0 or i == 1 or i == len(tickers):  # Only update every 5 tickers to reduce overhead
            print(f"Processing stocks: {progress:.0f}% completed ({i}/{len(tickers)})", end='\r')
        
        if metrics_dict is not None:
            all_metrics[ticker] = metrics_dict
            valid_tickers.append(ticker)
    
    print()  # New line after progress bar
    
    # If no valid tickers were processed, return empty results
    if not valid_tickers:
        logger.warning("No valid tickers could be processed.")
        return []
    
    # Normalize data and calculate z-scores
    normalized_metrics = normalize_data(all_metrics)
    z_scores = calculate_z_scores(normalized_metrics)
    
    # Calculate scores for each ticker
    for ticker in valid_tickers:
        ticker_z_scores = z_scores[ticker]
        
        # Calculate category scores
        profitability_score = calculate_weighted_score(
            ticker_z_scores, PROFITABILITY_WEIGHTS
        )
        
        growth_score = calculate_weighted_score(
            ticker_z_scores, GROWTH_WEIGHTS
        )
        
        financial_health_score = calculate_weighted_score(
            ticker_z_scores, FINANCIAL_HEALTH_WEIGHTS
        )
        
        valuation_score = calculate_weighted_score(
            ticker_z_scores, VALUATION_WEIGHTS
        )
        
        efficiency_score = calculate_weighted_score(
            ticker_z_scores, EFFICIENCY_WEIGHTS
        )
        
        # Calculate composite score
        composite_score = (
            profitability_score * CATEGORY_WEIGHTS['profitability'] +
            growth_score * CATEGORY_WEIGHTS['growth'] +
            financial_health_score * CATEGORY_WEIGHTS['financial_health'] +
            valuation_score * CATEGORY_WEIGHTS['valuation'] +
            efficiency_score * CATEGORY_WEIGHTS['efficiency']
        )
        
        # Create detailed results
        detailed_results = {
            'raw_metrics': all_metrics[ticker],
            'normalized_metrics': normalized_metrics[ticker],
            'z_scores': ticker_z_scores,
            'category_scores': {
                'profitability': profitability_score,
                'growth': growth_score,
                'financial_health': financial_health_score,
                'valuation': valuation_score,
                'efficiency': efficiency_score
            },
            'composite_score': composite_score
        }
        
        results.append((ticker, composite_score, detailed_results))
    
    # Sort by composite score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Log results summary
    logger.info(f"Successfully screened {len(results)} stocks.")
    
    return results

def screen_stocks(tickers: List[str], max_workers: int = None) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Screen stocks with a synchronous approach to avoid event loop complications.
    
    Args:
        tickers: List of stock ticker symbols to screen
        max_workers: Maximum number of worker threads to use
        
    Returns:
        List of tuples containing (ticker, composite_score, detailed_data)
    """
    # Use a default that's optimized for IO-bound operations if not specified
    if max_workers is None:
        max_workers = min(32, os.cpu_count() * 4)  # 4x CPU count for IO-bound, max 32
    
    # Get OpenBB client
    obb_client = get_openbb_client()
    
    # Process results
    all_metrics = {}
    valid_tickers = []
    
    # Function to process a single ticker synchronously
    def process_ticker_sync(ticker):
        """
        Process a single ticker synchronously, fetching financial data and calculating metrics.
        Uses proper caching strategy with thread safety.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Tuple of (ticker, metrics_dict) or (ticker, None) if processing fails
        """
        try:
            # Check if we have cached metrics for this ticker
            if ticker in metrics_cache:
                return (ticker, metrics_cache[ticker])
            
            # Fetch financial data for the ticker
            financial_data = {}
            obb_client = get_openbb_client()
            
            # Get income statement
            try:
                income_response = obb_client.equity.fundamental.income(
                    symbol=ticker, period='annual', limit=5, provider='fmp'
                )
                financial_data['income'] = income_response.results
            except Exception as e:
                logger.error(f"Error fetching income data for {ticker}: {e}")
                financial_data['income'] = []
                
            # Get balance sheet
            try:
                balance_response = obb_client.equity.fundamental.balance(
                    symbol=ticker, period='annual', limit=5, provider='fmp'
                )
                financial_data['balance'] = balance_response.results
            except Exception as e:
                logger.error(f"Error fetching balance data for {ticker}: {e}")
                financial_data['balance'] = []
            
            # Check if we have essential data
            if not financial_data.get('income') or not financial_data.get('balance'):
                logger.warning(f"Insufficient financial data for {ticker}. Skipping...")
                return (ticker, None)
                
            # Fetch additional data only if essential data exists
            try:
                cash_response = obb_client.equity.fundamental.cash(
                    symbol=ticker, period='annual', limit=5, provider='fmp'
                )
                financial_data['cash'] = cash_response.results
            except Exception as e:
                logger.warning(f"Error fetching cash flow data for {ticker}: {e}")
                financial_data['cash'] = []
                
            # Fetch growth metrics
            for data_type in ['income_growth', 'balance_growth', 'cash_growth']:
                try:
                    response = getattr(obb_client.equity.fundamental, data_type)(
                        symbol=ticker, period='annual', limit=5, provider='fmp'
                    )
                    financial_data[data_type] = response.results
                except Exception as e:
                    logger.warning(f"Error fetching {data_type} for {ticker}: {e}")
                    financial_data[data_type] = []
            
            # Fetch ratios and metrics
            for data_type in ['ratios', 'metrics']:
                try:
                    response = getattr(obb_client.equity.fundamental, data_type)(
                        symbol=ticker, period='annual', limit=5, provider='fmp'
                    )
                    financial_data[data_type] = response.results
                except Exception as e:
                    logger.warning(f"Error fetching {data_type} for {ticker}: {e}")
                    financial_data[data_type] = []
            
            # Fetch dividends
            try:
                dividends_response = obb_client.equity.fundamental.dividends(
                    symbol=ticker, provider='fmp'
                )
                financial_data['dividends'] = dividends_response.results
            except Exception as e:
                logger.warning(f"No dividend data for {ticker}: {e}")
                financial_data['dividends'] = []
            
            # Extract metrics directly without trying to serialize the financial_data
            try:
                metrics_dict = extract_metrics_from_financial_data(financial_data)
                
                # Cache the result for future use
                metrics_cache[ticker] = metrics_dict
                
                # Manage cache size
                if len(metrics_cache) > CACHE_SIZE:
                    oldest_key = next(iter(metrics_cache))
                    del metrics_cache[oldest_key]
                    
                return (ticker, metrics_dict)
            except Exception as e:
                logger.error(f"Error extracting metrics for {ticker}: {e}")
                logger.error(traceback.format_exc())
                return (ticker, None)
                
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            return (ticker, None)

    # Use ThreadPoolExecutor to process tickers in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process all tickers and collect results
        for ticker, metrics in tqdm(executor.map(process_ticker_sync, tickers), 
                                   total=len(tickers), desc="Processing stocks"):
            if metrics is not None:
                all_metrics[ticker] = metrics
                valid_tickers.append(ticker)
    
    # If no valid tickers were processed, return empty results
    if not valid_tickers:
        logger.warning("No valid tickers could be processed.")
        return []
    
    # Calculate scores using the same approach as before
    normalized_metrics = normalize_data(all_metrics)
    z_scores = calculate_z_scores(normalized_metrics)
    
    results = []
    for ticker in valid_tickers:
        ticker_z_scores = z_scores[ticker]
        
        # Calculate category scores
        category_scores = {
            'profitability': calculate_weighted_score(ticker_z_scores, PROFITABILITY_WEIGHTS),
            'growth': calculate_weighted_score(ticker_z_scores, GROWTH_WEIGHTS),
            'financial_health': calculate_weighted_score(ticker_z_scores, FINANCIAL_HEALTH_WEIGHTS),
            'valuation': calculate_weighted_score(ticker_z_scores, VALUATION_WEIGHTS),
            'efficiency': calculate_weighted_score(ticker_z_scores, EFFICIENCY_WEIGHTS)
        }
        
        # Calculate composite score
        composite_score = sum(
            score * CATEGORY_WEIGHTS[category] 
            for category, score in category_scores.items()
        )
        
        # Create detailed results
        detailed_results = {
            'raw_metrics': all_metrics[ticker],
            'normalized_metrics': normalized_metrics[ticker],
            'z_scores': ticker_z_scores,
            'category_scores': category_scores,
            'composite_score': composite_score
        }
        
        results.append((ticker, composite_score, detailed_results))
    
    # Sort by composite score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Log results summary
    logger.info(f"Successfully screened {len(results)} stocks.")
    
    return results

def get_top_stocks(tickers: List[str], top_n: int = 10, max_workers: int = None) -> pd.DataFrame:
    """
    Get the top N stocks based on their fundamental scores.
    """
    results = screen_stocks(tickers, max_workers=max_workers)
    
    # Create DataFrame with results
    data = []
    
    for ticker, composite_score, detailed_results in results[:min(top_n, len(results))]:
        category_scores = detailed_results['category_scores']
        
        data.append({
            'Ticker': ticker,
            'Composite Score': round(composite_score, 2),
            'Profitability': round(category_scores['profitability'], 2),
            'Growth': round(category_scores['growth'], 2),
            'Financial Health': round(category_scores['financial_health'], 2),
            'Valuation': round(category_scores['valuation'], 2),
            'Efficiency': round(category_scores['efficiency'], 2)
        })
    
    return pd.DataFrame(data)

def get_metric_contributions(ticker: str) -> pd.DataFrame:
    """
    Get detailed breakdown of metric contributions to the composite score.
    """
    results = screen_stocks([ticker], max_workers=1)  # Only one ticker, so max_workers=1
    
    if not results:
        logger.warning(f"No data found for {ticker}")
        return pd.DataFrame()
    
    _, _, detailed_results = results[0]
    z_scores = detailed_results['z_scores']
    raw_metrics = detailed_results['raw_metrics']
    
    # Create DataFrame with metric contributions
    data = []
    
    # Process each category
    for category, weights in [
        ('Profitability', PROFITABILITY_WEIGHTS),
        ('Growth', GROWTH_WEIGHTS),
        ('Financial Health', FINANCIAL_HEALTH_WEIGHTS),
        ('Valuation', VALUATION_WEIGHTS),
        ('Efficiency', EFFICIENCY_WEIGHTS)
    ]:
        for metric, weight in weights.items():
            if metric in z_scores:
                z_score = z_scores[metric]
                raw_value = raw_metrics.get(metric)
                contribution = z_score * weight
                
                data.append({
                    'Category': category,
                    'Metric': metric,
                    'Raw Value': raw_value,
                    'Weight': weight,
                    'Z-Score': round(z_score, 2),
                    'Contribution': round(contribution, 3),
                    'Impact': 'Positive' if contribution > 0 else 'Negative'
                })
    
    # Sort by absolute contribution
    df = pd.DataFrame(data)
    df['Abs Contribution'] = df['Contribution'].abs()
    df = df.sort_values('Abs Contribution', ascending=False)
    df = df.drop('Abs Contribution', axis=1)
    
    return df

def generate_stock_report(ticker: str) -> Dict[str, Any]:
    """
    Generate a comprehensive fundamental analysis report for a stock.
    Synchronous version that avoids event loop complications.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with report data
    """
    # Get OpenBB client
    obb_client = get_openbb_client()
    financial_data = {}
    
    # Fetch financial data with proper error handling
    try:
        # Basic financial statements
        for data_type in ['income', 'balance', 'cash']:
            try:
                response = getattr(obb_client.equity.fundamental, data_type)(
                    symbol=ticker, period='annual', limit=5, provider='fmp'
                )
                financial_data[data_type] = response.results
            except Exception as e:
                logger.warning(f"Error fetching {data_type} for {ticker}: {e}")
                financial_data[data_type] = []
        
        # Growth metrics
        for data_type in ['income_growth', 'balance_growth', 'cash_growth']:
            try:
                response = getattr(obb_client.equity.fundamental, data_type)(
                    symbol=ticker, period='annual', limit=5, provider='fmp'
                )
                financial_data[data_type] = response.results
            except Exception as e:
                logger.warning(f"Error fetching {data_type} for {ticker}: {e}")
                financial_data[data_type] = []
        
        # Ratios and metrics
        for data_type in ['ratios', 'metrics']:
            try:
                response = getattr(obb_client.equity.fundamental, data_type)(
                    symbol=ticker, period='annual', limit=5, provider='fmp'
                )
                financial_data[data_type] = response.results
            except Exception as e:
                logger.warning(f"Error fetching {data_type} for {ticker}: {e}")
                financial_data[data_type] = []
        
        # Dividends
        try:
            dividends_response = obb_client.equity.fundamental.dividends(
                symbol=ticker, provider='fmp'
            )
            financial_data['dividends'] = dividends_response.results
        except Exception as e:
            logger.warning(f"No dividend data for {ticker}: {e}")
            financial_data['dividends'] = []
    except Exception as e:
        logger.error(f"Error fetching financial data for {ticker}: {e}")
    
    # Extract metrics
    metrics = extract_metrics_from_financial_data(financial_data)
    
    # Create report
    report = {
        'ticker': ticker,
        'composite_score': 0.0,
        'category_scores': {
            'profitability': 0.0,
            'growth': 0.0,
            'financial_health': 0.0,
            'valuation': 0.0,
            'efficiency': 0.0
        },
        'key_metrics': {
            'profitability': {
                'gross_margin': metrics.get('gross_profit_margin'),
                'operating_margin': metrics.get('operating_income_margin'),
                'net_margin': metrics.get('net_income_margin'),
                'roe': metrics.get('return_on_equity'),
                'roa': metrics.get('return_on_assets')
            },
            'growth': {
                'revenue_growth': metrics.get('growth_revenue'),
                'earnings_growth': metrics.get('growth_net_income'),
                'eps_growth': metrics.get('growth_eps')
            },
            'financial_health': {
                'current_ratio': metrics.get('current_ratio'),
                'debt_to_equity': metrics.get('debt_to_equity'),
                'interest_coverage': metrics.get('interest_coverage')
            },
            'valuation': {
                'pe_ratio': metrics.get('pe_ratio'),
                'price_to_book': metrics.get('price_to_book'),
                'ev_to_ebitda': metrics.get('ev_to_ebitda'),
                'dividend_yield': metrics.get('dividend_yield')
            }
        },
        'strengths': [],
        'weaknesses': [],
        'raw_metrics': metrics,
    }
    
    # Use benchmarks to identify strengths and weaknesses
    profitability_benchmarks = {
        'gross_profit_margin': 0.35,
        'operating_income_margin': 0.15,
        'net_income_margin': 0.10,
        'return_on_equity': 0.15,
        'return_on_assets': 0.05
    }
    
    financial_health_benchmarks = {
        'current_ratio': 1.5,
        'debt_to_equity': 1.0,
        'interest_coverage': 3.0
    }
    
    growth_benchmarks = {
        'growth_revenue': 0.10,
        'growth_net_income': 0.10,
        'growth_eps': 0.10
    }
    
    # Profitability
    for metric, benchmark in profitability_benchmarks.items():
        if metric in metrics and metrics.get(metric) is not None:
            if metrics.get(metric) > benchmark * 1.5:
                report['strengths'].append(f"{metric}: {metrics.get(metric):.2f}")
            elif metrics.get(metric) < benchmark * 0.5:
                report['weaknesses'].append(f"{metric}: {metrics.get(metric):.2f}")
    
    # Financial health
    for metric, benchmark in financial_health_benchmarks.items():
        if metric in metrics and metrics.get(metric) is not None:
            if metric == 'debt_to_equity':  # Lower is better
                if metrics.get(metric) < benchmark * 0.5:
                    report['strengths'].append(f"{metric}: {metrics.get(metric):.2f}")
                elif metrics.get(metric) > benchmark * 1.5:
                    report['weaknesses'].append(f"{metric}: {metrics.get(metric):.2f}")
            else:  # Higher is better
                if metrics.get(metric) > benchmark * 1.5:
                    report['strengths'].append(f"{metric}: {metrics.get(metric):.2f}")
                elif metrics.get(metric) < benchmark * 0.5:
                    report['weaknesses'].append(f"{metric}: {metrics.get(metric):.2f}")
    
    # Growth
    for metric, benchmark in growth_benchmarks.items():
        if metric in metrics and metrics.get(metric) is not None:
            if metrics.get(metric) > benchmark * 1.5:
                report['strengths'].append(f"{metric}: {metrics.get(metric):.2f}")
            elif metrics.get(metric) < 0:  # Negative growth is a weakness
                report['weaknesses'].append(f"{metric}: {metrics.get(metric):.2f}")
    
    return report
