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
from scipy.stats import spearmanr

# Import PyTorch to leverage MPS on Apple Silicon
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

def select_valid_record(records: List[Any], key: str, min_value: float = 0.001) -> Any:
    """
    Returns the first record with a nonzero value for `key` above min_value.
    If none is found, returns the first record.
    """
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

async def get_financial_data_async(ticker: str) -> Tuple[str, Dict[str, List]]:
    """
    Async version of get_financial_data using OpenBB API.
    Fetches all required financial data for a stock, handling provider fallbacks properly.
    """
    obb_client = get_openbb_client()
    financial_data = {}
    
    try:
        # Fetch essential financial statements first
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
        
        income_response, balance_response, cash_response = await asyncio.gather(
            income_task, balance_task, cash_task, 
            return_exceptions=True
        )
        
        # Check if essential data is available
        if isinstance(income_response, Exception) or isinstance(balance_response, Exception):
            logger.error(f"Essential data fetch failed for {ticker}")
            return (ticker, financial_data)
        
        financial_data['income'] = income_response.results if not isinstance(income_response, Exception) else []
        financial_data['balance'] = balance_response.results if not isinstance(balance_response, Exception) else []
        financial_data['cash'] = cash_response.results if not isinstance(cash_response, Exception) else []
        
        # If essential data is available, fetch additional data
        if financial_data['income'] and financial_data['balance']:
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
            historical_estimates_task = rate_limited_api_call(
                obb_client.equity.estimates.historical,
                symbol=ticker, provider='fmp'
            )
            
            results = await asyncio.gather(
                income_growth_task, balance_growth_task, cash_growth_task, 
                ratios_task, metrics_task, historical_estimates_task,
                return_exceptions=True
            )
            
            financial_data['income_growth'] = results[0].results if not isinstance(results[0], Exception) else []
            financial_data['balance_growth'] = results[1].results if not isinstance(results[1], Exception) else []
            financial_data['cash_growth'] = results[2].results if not isinstance(results[2], Exception) else []
            financial_data['ratios'] = results[3].results if not isinstance(results[3], Exception) else []
            financial_data['metrics'] = results[4].results if not isinstance(results[4], Exception) else []
            financial_data['historical_estimates'] = results[5].results if not isinstance(results[5], Exception) else []
            
            # Forward sales fallback chain: fmp > intrinio > default
            try:
                forward_sales_response = await rate_limited_api_call(
                    obb_client.equity.estimates.forward_sales,
                    symbol=ticker
                )
                financial_data['forward_sales'] = forward_sales_response.results
            except Exception as e:
                logger.warning(f"Error fetching forward sales for {ticker} from standard: \n{e}")
                try:
                    forward_sales_response = await rate_limited_api_call(
                        obb_client.equity.estimates.forward_sales,
                        symbol=ticker, provider='intrinio'
                    )
                    financial_data['forward_sales'] = forward_sales_response.results
                except Exception as e2:
                    logger.warning(f"Error fetching forward sales for {ticker} from Intrinio: \n{e2}")
                    try:
                        forward_sales_response = await rate_limited_api_call(
                            obb_client.equity.estimates.forward_sales,
                            symbol=ticker
                        )
                        financial_data['forward_sales'] = forward_sales_response.results
                    except Exception as e3:
                        logger.warning(f"Error fetching forward sales for {ticker} with default provider: \n{e3}")
                        financial_data['forward_sales'] = []
            
            # Forward EBITDA fallback chain: fmp > intrinio > default
            try:
                forward_ebitda_response = await rate_limited_api_call(
                    obb_client.equity.estimates.forward_ebitda,
                    symbol=ticker, fiscal_period='annual', provider='fmp'
                )
                financial_data['forward_ebitda'] = forward_ebitda_response.results
            except Exception as e:
                logger.warning(f"Error fetching forward EBITDA for {ticker} from FMP: \n{e}")
                try:
                    forward_ebitda_response = await rate_limited_api_call(
                        obb_client.equity.estimates.forward_ebitda,
                        symbol=ticker, fiscal_period='annual', provider='intrinio'
                    )
                    financial_data['forward_ebitda'] = forward_ebitda_response.results
                except Exception as e2:
                    logger.warning(f"Error fetching forward EBITDA for {ticker} from Intrinio: \n{e2}")
                    try:
                        forward_ebitda_response = await rate_limited_api_call(
                            obb_client.equity.estimates.forward_ebitda,
                            symbol=ticker, fiscal_period='annual'
                        )
                        financial_data['forward_ebitda'] = forward_ebitda_response.results
                    except Exception as e3:
                        logger.warning(f"Error fetching forward EBITDA for {ticker} with default provider: \n{e3}")
                        financial_data['forward_ebitda'] = []
            
            # Dividends
            try:
                dividends_response = await rate_limited_api_call(
                    obb_client.equity.fundamental.dividends,
                    symbol=ticker, provider='fmp'
                )
                financial_data['dividends'] = dividends_response.results
            except Exception as e:
                logger.warning(f"No dividend data for {ticker}: {e}")
                financial_data['dividends'] = []
            
            # Earnings
            try:
                if hasattr(obb_client.equity.fundamental, 'earnings'):
                    earnings_response = await rate_limited_api_call(
                        obb_client.equity.fundamental.earnings,
                        symbol=ticker, provider='fmp'
                    )
                    financial_data['earnings'] = earnings_response.results
                else:
                    logger.warning(f"Earnings endpoint not available for {ticker}, constructing from income data")
                    financial_data['earnings'] = construct_earnings_from_income(financial_data['income'])
            except Exception as e:
                logger.warning(f"Earnings endpoint not available for {ticker}, constructing from income data: {e}")
                financial_data['earnings'] = construct_earnings_from_income(financial_data['income'])
    
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        logger.error(traceback.format_exc())
    
    return (ticker, financial_data)

def get_attribute_value(obj: Any, attr_name: str, default=0) -> Any:
    """Safely get attribute value from an object."""
    if hasattr(obj, attr_name):
        value = getattr(obj, attr_name)
        if value is not None and not pd.isna(value):
            try:
                # For numerical values, ensure they're finite
                if isinstance(value, (int, float)):
                    if np.isfinite(value):
                        return value
                    return default
                return value
            except:
                return default
    return default

def calculate_estimate_accuracy(historical_estimates: List[Any], actual_results: List[Any]) -> Dict[str, float]:
    """
    Calculate how accurate analyst estimates have been historically using Spearman rank correlation.
    Handles case where actual_results might be constructed from income data.
    """
    import datetime as dt
    
    if not historical_estimates or not actual_results:
        return {
            'estimate_eps_accuracy': 0,
            'estimate_revenue_accuracy': 0,
            'estimate_ebitda_accuracy': 0
        }
    
    eps_data = []
    revenue_data = []
    ebitda_data = []
    
    for estimate in historical_estimates:
        est_date = get_attribute_value(estimate, 'date')
        if not est_date:
            continue
        
        # Parse date into a datetime.date object
        est_date_obj = None
        if isinstance(est_date, str):
            try:
                est_date_obj = dt.datetime.strptime(est_date, '%Y-%m-%d').date()
            except ValueError:
                continue
        elif isinstance(est_date, dt.datetime):
            est_date_obj = est_date.date()
        elif isinstance(est_date, dt.date):
            est_date_obj = est_date
        else:
            continue
        
        # Find matching actual result
        matching_result = None
        for result in actual_results:
            result_date = get_attribute_value(result, 'date', None)
            if result_date is None:
                result_date = get_attribute_value(result, 'period_ending', None)
            if not result_date:
                continue
            
            # Parse result date
            result_date_obj = None
            if isinstance(result_date, str):
                try:
                    result_date_obj = dt.datetime.strptime(result_date, '%Y-%m-%d').date()
                except ValueError:
                    continue
            elif isinstance(result_date, dt.datetime):
                result_date_obj = result_date.date()
            elif isinstance(result_date, dt.date):
                result_date_obj = result_date
            else:
                continue
            
            # Check if dates are close enough
            if est_date_obj and result_date_obj:
                date_diff = abs((est_date_obj - result_date_obj).days)
                if date_diff <= 90:
                    matching_result = result
                    break
        
        # If we found a matching result, extract data for correlation
        if matching_result:
            # EPS data
            est_eps = get_attribute_value(estimate, 'estimated_eps_avg')
            actual_eps = get_attribute_value(matching_result, 'eps')
            if est_eps is not None and actual_eps is not None and est_eps != 0 and actual_eps != 0:
                # Ensure values are valid numbers
                if np.isfinite(est_eps) and np.isfinite(actual_eps):
                    eps_data.append((est_eps, actual_eps))
            
            # Revenue data
            est_revenue = get_attribute_value(estimate, 'estimated_revenue_avg')
            actual_revenue = get_attribute_value(matching_result, 'revenue')
            if est_revenue is not None and actual_revenue is not None and est_revenue != 0 and actual_revenue != 0:
                if np.isfinite(est_revenue) and np.isfinite(actual_revenue):
                    revenue_data.append((est_revenue, actual_revenue))
            
            # EBITDA data
            est_ebitda = get_attribute_value(estimate, 'estimated_ebitda_avg')
            actual_ebitda = get_attribute_value(matching_result, 'ebitda')
            if est_ebitda is not None and actual_ebitda is not None and est_ebitda != 0 and actual_ebitda != 0:
                if np.isfinite(est_ebitda) and np.isfinite(actual_ebitda):
                    ebitda_data.append((est_ebitda, actual_ebitda))
    
    # Calculate correlations
    eps_accuracy = 0
    revenue_accuracy = 0
    ebitda_accuracy = 0
    
    # Need at least 3 data points for a meaningful correlation
    if len(eps_data) >= 3:
        est_eps_values, actual_eps_values = zip(*eps_data)
        try:
            correlation, _ = spearmanr(est_eps_values, actual_eps_values)
            if np.isfinite(correlation):
                eps_accuracy = max(0, correlation)
        except Exception as e:
            logger.warning(f"Error calculating EPS accuracy: {e}")
            eps_accuracy = 0
    
    if len(revenue_data) >= 3:
        est_revenue_values, actual_revenue_values = zip(*revenue_data)
        try:
            correlation, _ = spearmanr(est_revenue_values, actual_revenue_values)
            if np.isfinite(correlation):
                revenue_accuracy = max(0, correlation)
        except Exception as e:
            logger.warning(f"Error calculating revenue accuracy: {e}")
            revenue_accuracy = 0
    
    if len(ebitda_data) >= 3:
        est_ebitda_values, actual_ebitda_values = zip(*ebitda_data)
        try:
            correlation, _ = spearmanr(est_ebitda_values, actual_ebitda_values)
            if np.isfinite(correlation):
                ebitda_accuracy = max(0, correlation)
        except Exception as e:
            logger.warning(f"Error calculating EBITDA accuracy: {e}")
            ebitda_accuracy = 0
    
    return {
        'estimate_eps_accuracy': eps_accuracy,
        'estimate_revenue_accuracy': revenue_accuracy,
        'estimate_ebitda_accuracy': ebitda_accuracy
    }

def calculate_estimate_revision_momentum(forward_estimates: List[Any]) -> float:
    """
    Calculate the momentum of analyst estimate revisions.
    Positive value means estimates are being revised upward, negative means downward.
    """
    if not forward_estimates or len(forward_estimates) < 2:
        return 0
    
    # Sort estimates by date
    sorted_estimates = sorted(forward_estimates, key=lambda x: get_attribute_value(x, 'date'))
    revisions = []
    
    for i in range(1, len(sorted_estimates)):
        prev_mean = get_attribute_value(sorted_estimates[i-1], 'mean')
        curr_mean = get_attribute_value(sorted_estimates[i], 'mean')
        
        if prev_mean and curr_mean and prev_mean != 0 and np.isfinite(prev_mean) and np.isfinite(curr_mean):
            pct_change = (curr_mean - prev_mean) / abs(prev_mean)
            
            # Limit extreme values
            pct_change = max(min(pct_change, 2.0), -2.0)
            
            revisions.append(pct_change)
    
    if revisions:
        # Get the median change to avoid outlier influence
        momentum = np.median(revisions)
        return momentum
    else:
        return 0

def calculate_consensus_deviation(estimate: Any) -> float:
    """
    Calculate how much the mean estimate deviates from the range of estimates.
    Lower values are better (less dispersion among analysts).
    """
    if not estimate:
        return 0
    
    mean = get_attribute_value(estimate, 'mean')
    low = get_attribute_value(estimate, 'low_estimate')
    high = get_attribute_value(estimate, 'high_estimate')
    
    if (not np.isfinite(mean) or not np.isfinite(low) or not np.isfinite(high) or 
        low == high or mean == 0 or low == 0 or high == 0):
        return 0
    
    range_size = high - low
    if range_size == 0:
        return 0
    
    center = (high + low) / 2
    deviation = abs(mean - center) / range_size
    
    # Cap the deviation to avoid extreme values
    deviation = min(deviation, 1.0)
    
    return deviation

def extract_metrics_from_financial_data(financial_data):
    """
    Extract all relevant metrics from financial data, including analyst estimates.
    Modified to work with FMP data only and handle missing data gracefully.
    """
    metrics = {}
    
    # Process income statement data
    if isinstance(financial_data, dict) and financial_data.get('income') and len(financial_data['income']) > 0:
        income = select_valid_record(financial_data['income'], 'revenue') or financial_data['income'][0]
        
        # Profitability metrics
        metrics['gross_profit_margin'] = get_attribute_value(income, 'gross_profit_margin')
        metrics['operating_income_margin'] = get_attribute_value(income, 'operating_income_margin')
        metrics['net_income_margin'] = get_attribute_value(income, 'net_income_margin')
        metrics['ebitda_margin'] = get_attribute_value(income, 'ebitda_margin')
        
        # Revenue and earnings metrics
        metrics['revenue'] = get_attribute_value(income, 'revenue')
        metrics['gross_profit'] = get_attribute_value(income, 'gross_profit')
        metrics['operating_income'] = get_attribute_value(income, 'operating_income')
        metrics['net_income'] = get_attribute_value(income, 'net_income')
        metrics['ebitda'] = get_attribute_value(income, 'ebitda')
    
    # Process balance sheet data
    if isinstance(financial_data, dict) and financial_data.get('balance') and len(financial_data['balance']) > 0:
        balance = select_valid_record(financial_data['balance'], 'total_assets') or financial_data['balance'][0]
        
        # Balance sheet metrics
        metrics['total_assets'] = get_attribute_value(balance, 'total_assets')
        metrics['total_liabilities'] = get_attribute_value(balance, 'total_liabilities')
        metrics['total_shareholders_equity'] = get_attribute_value(balance, 'total_shareholders_equity')
        metrics['cash_and_cash_equivalents'] = get_attribute_value(balance, 'cash_and_cash_equivalents')
        metrics['total_debt'] = get_attribute_value(balance, 'total_debt')
        metrics['net_debt'] = get_attribute_value(balance, 'net_debt')
        
        # Calculate derived metrics
        if metrics.get('total_assets', 0) > 0:
            metrics['debt_to_assets'] = metrics.get('total_debt', 0) / metrics['total_assets']
        
        if metrics.get('total_shareholders_equity', 0) > 0:
            metrics['debt_to_equity'] = metrics.get('total_debt', 0) / metrics['total_shareholders_equity']
            if 'net_income' in metrics and metrics['net_income'] != 0:
                metrics['return_on_equity'] = metrics['net_income'] / metrics['total_shareholders_equity']
        
        if metrics.get('total_assets', 0) > 0 and 'net_income' in metrics and metrics['net_income'] != 0:
            metrics['return_on_assets'] = metrics['net_income'] / metrics['total_assets']
        
        # Cash to debt ratio with cap for zero debt cases
        if metrics.get('total_debt', 0) > 0:
            metrics['cash_to_debt'] = metrics.get('cash_and_cash_equivalents', 0) / metrics['total_debt']
            # Cap extremely high values
            metrics['cash_to_debt'] = min(metrics['cash_to_debt'], 10)
        else:
            metrics['cash_to_debt'] = 10  # Set to max value if no debt
    
    # Process cash flow data
    if isinstance(financial_data, dict) and financial_data.get('cash') and len(financial_data['cash']) > 0:
        cash_flow = select_valid_record(financial_data['cash'], 'operating_cash_flow') or financial_data['cash'][0]
        
        # Cash flow metrics
        metrics['operating_cash_flow'] = get_attribute_value(cash_flow, 'operating_cash_flow')
        metrics['capital_expenditure'] = get_attribute_value(cash_flow, 'capital_expenditure')
        metrics['free_cash_flow'] = get_attribute_value(cash_flow, 'free_cash_flow')
        
        # Capital expenditure to revenue ratio
        if 'revenue' in metrics and metrics['revenue'] > 0 and abs(metrics.get('capital_expenditure', 0)) > 0:
            metrics['capex_to_revenue'] = abs(metrics['capital_expenditure']) / metrics['revenue']
    
    # Process income growth data
    if isinstance(financial_data, dict) and financial_data.get('income_growth') and len(financial_data['income_growth']) > 0:
        income_growth = financial_data['income_growth'][0]
        
        # Growth metrics
        metrics['growth_revenue'] = get_attribute_value(income_growth, 'growth_revenue')
        metrics['growth_gross_profit'] = get_attribute_value(income_growth, 'growth_gross_profit')
        metrics['growth_ebitda'] = get_attribute_value(income_growth, 'growth_ebitda')
        metrics['growth_operating_income'] = get_attribute_value(income_growth, 'growth_operating_income')
        metrics['growth_net_income'] = get_attribute_value(income_growth, 'growth_net_income')
        metrics['growth_eps'] = get_attribute_value(income_growth, 'growth_eps')
        
        # Cap extreme growth values
        for growth_metric in ['growth_revenue', 'growth_gross_profit', 'growth_ebitda', 
                              'growth_operating_income', 'growth_net_income', 'growth_eps']:
            if growth_metric in metrics and metrics[growth_metric] is not None:
                # Cap growth metrics to avoid extreme values
                metrics[growth_metric] = max(min(metrics[growth_metric], 2.0), -1.0)
    
    # Process balance sheet growth data
    if isinstance(financial_data, dict) and financial_data.get('balance_growth') and len(financial_data['balance_growth']) > 0:
        balance_growth = financial_data['balance_growth'][0]
        
        # Balance sheet growth metrics
        metrics['growth_total_assets'] = get_attribute_value(balance_growth, 'growth_total_assets')
        metrics['growth_total_liabilities'] = get_attribute_value(balance_growth, 'growth_total_liabilities')
        metrics['growth_total_shareholders_equity'] = get_attribute_value(balance_growth, 'growth_total_shareholders_equity')
        metrics['growth_total_debt'] = get_attribute_value(balance_growth, 'growth_total_debt')
        metrics['growth_net_debt'] = get_attribute_value(balance_growth, 'growth_net_debt')
        
        # Cap extreme growth values
        for growth_metric in ['growth_total_assets', 'growth_total_liabilities', 
                              'growth_total_shareholders_equity', 'growth_total_debt', 'growth_net_debt']:
            if growth_metric in metrics and metrics[growth_metric] is not None:
                # Cap growth metrics to avoid extreme values
                metrics[growth_metric] = max(min(metrics[growth_metric], 2.0), -1.0)
    
    # Process ratio data
    if isinstance(financial_data, dict) and financial_data.get('ratios') and len(financial_data['ratios']) > 0:
        ratios = financial_data['ratios'][0]
        
        # Financial ratios
        metrics['current_ratio'] = get_attribute_value(ratios, 'current_ratio')
        metrics['quick_ratio'] = get_attribute_value(ratios, 'quick_ratio')
        metrics['interest_coverage'] = get_attribute_value(ratios, 'interest_coverage')
        metrics['asset_turnover'] = get_attribute_value(ratios, 'asset_turnover')
        metrics['inventory_turnover'] = get_attribute_value(ratios, 'inventory_turnover')
        metrics['receivables_turnover'] = get_attribute_value(ratios, 'receivables_turnover')
        metrics['cash_conversion_cycle'] = get_attribute_value(ratios, 'cash_conversion_cycle')
        
        # Apply caps to extreme ratio values to prevent outliers
        if 'current_ratio' in metrics and metrics['current_ratio'] is not None:
            metrics['current_ratio'] = min(metrics['current_ratio'], 10.0)
        
        if 'interest_coverage' in metrics and metrics['interest_coverage'] is not None:
            metrics['interest_coverage'] = min(metrics['interest_coverage'], 50.0)
            metrics['interest_coverage'] = max(metrics['interest_coverage'], -50.0)
            
        if 'asset_turnover' in metrics and metrics['asset_turnover'] is not None:
            metrics['asset_turnover'] = min(metrics['asset_turnover'], 5.0)
            
        if 'inventory_turnover' in metrics and metrics['inventory_turnover'] is not None:
            metrics['inventory_turnover'] = min(metrics['inventory_turnover'], 50.0)
            
        if 'receivables_turnover' in metrics and metrics['receivables_turnover'] is not None:
            metrics['receivables_turnover'] = min(metrics['receivables_turnover'], 50.0)
            
        if 'cash_conversion_cycle' in metrics and metrics['cash_conversion_cycle'] is not None:
            metrics['cash_conversion_cycle'] = max(min(metrics['cash_conversion_cycle'], 365.0), -365.0)
    
    # Process market metrics data
    if isinstance(financial_data, dict) and financial_data.get('metrics') and len(financial_data['metrics']) > 0:
        market_metrics = financial_data['metrics'][0]
        
        # Valuation metrics
        metrics['pe_ratio'] = get_attribute_value(market_metrics, 'pe_ratio')
        metrics['price_to_book'] = get_attribute_value(market_metrics, 'price_to_book')
        metrics['price_to_sales'] = get_attribute_value(market_metrics, 'price_to_sales')
        metrics['ev_to_ebitda'] = get_attribute_value(market_metrics, 'ev_to_ebitda')
        metrics['peg_ratio'] = get_attribute_value(market_metrics, 'peg_ratio')
        metrics['market_cap'] = get_attribute_value(market_metrics, 'market_cap')
        metrics['price'] = get_attribute_value(market_metrics, 'price')
        
        # Apply caps to extreme valuation metrics
        if 'pe_ratio' in metrics and metrics['pe_ratio'] is not None:
            # PE ratio can be negative but limit extreme values
            metrics['pe_ratio'] = max(min(metrics['pe_ratio'], 200.0), -200.0)
            
        if 'price_to_book' in metrics and metrics['price_to_book'] is not None:
            metrics['price_to_book'] = min(max(metrics['price_to_book'], 0.0), 50.0)
            
        if 'price_to_sales' in metrics and metrics['price_to_sales'] is not None:
            metrics['price_to_sales'] = min(max(metrics['price_to_sales'], 0.0), 50.0)
            
        if 'ev_to_ebitda' in metrics and metrics['ev_to_ebitda'] is not None:
            metrics['ev_to_ebitda'] = max(min(metrics['ev_to_ebitda'], 100.0), -100.0)
            
        if 'peg_ratio' in metrics and metrics['peg_ratio'] is not None:
            metrics['peg_ratio'] = max(min(metrics['peg_ratio'], 10.0), -10.0)
    
    # Process dividend data
    if isinstance(financial_data, dict) and financial_data.get('dividends'):
        import datetime as dt
        dividends = financial_data['dividends']
        annual_dividend = 0
        
        if dividends:
            one_year_ago = (dt.datetime.now() - dt.timedelta(days=365)).date()
            recent_dividends = []
            
            for div in dividends:
                try:
                    if hasattr(div, 'ex_dividend_date') and hasattr(div, 'amount'):
                        div_date = div.ex_dividend_date
                        if isinstance(div_date, str):
                            try:
                                div_date = dt.datetime.strptime(div_date, '%Y-%m-%d').date()
                            except ValueError:
                                continue
                        if isinstance(div_date, dt.datetime):
                            div_date = div_date.date()
                        elif not isinstance(div_date, dt.date):
                            continue
                        
                        if div_date >= one_year_ago:
                            amount = div.amount
                            if isinstance(amount, (int, float)) and np.isfinite(amount):
                                recent_dividends.append(amount)
                except Exception as e:
                    logger.warning(f"Error processing dividend entry: {e}")
                    continue
            
            # Sum up recent dividends to get annual amount
            annual_dividend = sum(recent_dividends)
        
        # Calculate dividend yield
        if annual_dividend > 0 and 'price' in metrics and metrics['price'] > 0:
            metrics['dividend_yield'] = annual_dividend / metrics['price']
            # Cap extreme dividend yields
            metrics['dividend_yield'] = min(metrics['dividend_yield'], 0.25)  # Cap at 25%
        else:
            metrics['dividend_yield'] = 0
    
    # Initialize estimates-related metrics
    metrics['estimate_eps_accuracy'] = 0
    metrics['estimate_revenue_accuracy'] = 0
    metrics['estimate_ebitda_accuracy'] = 0
    metrics['estimate_consensus_deviation'] = 0
    metrics['estimate_revision_momentum'] = 0
    metrics['forward_sales_growth'] = 0
    metrics['forward_ebitda_growth'] = 0
    metrics['forward_sales_consensus_deviation'] = 0
    metrics['forward_ebitda_consensus_deviation'] = 0
    metrics['forward_sales_revision_momentum'] = 0
    metrics['forward_ebitda_revision_momentum'] = 0
    
    # Process historical estimate accuracy
    if isinstance(financial_data, dict) and financial_data.get('historical_estimates') and financial_data.get('earnings'):
        try:
            accuracy_metrics = calculate_estimate_accuracy(
                financial_data['historical_estimates'],
                financial_data['earnings']
            )
            metrics.update(accuracy_metrics)
        except Exception as e:
            logger.error(f"Error calculating estimate accuracy: {e}")
            logger.error(traceback.format_exc())
    
    # Process forward sales estimates
    if isinstance(financial_data, dict) and financial_data.get('forward_sales'):
        try:
            if financial_data['forward_sales']:
                latest_estimate = financial_data['forward_sales'][0]
                metrics['forward_sales_consensus_deviation'] = calculate_consensus_deviation(latest_estimate)
                metrics['forward_sales_revision_momentum'] = calculate_estimate_revision_momentum(financial_data['forward_sales'])
                
                if 'revenue' in metrics and metrics['revenue'] > 0:
                    forward_revenue = get_attribute_value(latest_estimate, 'mean')
                    if forward_revenue and np.isfinite(forward_revenue):
                        growth = (forward_revenue - metrics['revenue']) / metrics['revenue']
                        # Cap extreme growth values
                        metrics['forward_sales_growth'] = max(min(growth, 2.0), -1.0)
        except Exception as e:
            logger.warning(f"Error processing forward sales data: {e}")
    
    # Process forward EBITDA estimates
    if isinstance(financial_data, dict) and financial_data.get('forward_ebitda'):
        try:
            if financial_data['forward_ebitda']:
                latest_estimate = financial_data['forward_ebitda'][0]
                metrics['forward_ebitda_consensus_deviation'] = calculate_consensus_deviation(latest_estimate)
                metrics['forward_ebitda_revision_momentum'] = calculate_estimate_revision_momentum(financial_data['forward_ebitda'])
                
                if 'ebitda' in metrics and metrics['ebitda'] > 0:
                    forward_ebitda = get_attribute_value(latest_estimate, 'mean')
                    if forward_ebitda and np.isfinite(forward_ebitda):
                        growth = (forward_ebitda - metrics['ebitda']) / metrics['ebitda']
                        # Cap extreme growth values
                        metrics['forward_ebitda_growth'] = max(min(growth, 2.0), -1.0)
        except Exception as e:
            logger.warning(f"Error processing forward EBITDA data: {e}")
    
    # Ensure all metrics are finite
    for key in list(metrics.keys()):
        if key in metrics and metrics[key] is not None:
            try:
                value = metrics[key]
                if isinstance(value, (int, float)) and not np.isfinite(value):
                    metrics[key] = 0
            except:
                metrics[key] = 0
    
    return metrics

def preprocess_data(data_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Preprocess data to handle missing values and outliers.
    This is a more focused preprocessing step than the full normalization.
    """
    preprocessed = {ticker: {} for ticker in data_dict}
    
    # First pass: collect valid values for each metric
    metrics_data = {}
    for ticker, metrics in data_dict.items():
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value) and not pd.isna(value):
                if metric not in metrics_data:
                    metrics_data[metric] = []
                metrics_data[metric].append(value)
    
    # Second pass: apply winsorization using percentiles
    for metric, values in metrics_data.items():
        if len(values) >= 5:  # Need enough values to compute meaningful percentiles
            values_array = np.array(values)
            p05 = np.percentile(values_array, 5)
            p95 = np.percentile(values_array, 95)
            
            for ticker, metrics in data_dict.items():
                if metric in metrics and isinstance(metrics[metric], (int, float)) and np.isfinite(metrics[metric]):
                    # Winsorize the value
                    preprocessed[ticker][metric] = max(min(metrics[metric], p95), p05)
                    # For metrics where 0 is a special value (like debt metrics for debt-free companies),
                    # preserve actual zeros
                    if metrics[metric] == 0 and (
                        'debt' in metric or 
                        'liability' in metric or 
                        'dividend' in metric or 
                        'interest' in metric
                    ):
                        preprocessed[ticker][metric] = 0
        else:
            # Not enough data for winsorization, just copy over
            for ticker, metrics in data_dict.items():
                if metric in metrics and isinstance(metrics[metric], (int, float)) and np.isfinite(metrics[metric]):
                    preprocessed[ticker][metric] = metrics[metric]
    
    return preprocessed

def calculate_z_scores(data_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate z-scores for each metric using PyTorch for consistency.
    Values are clamped to [-3, 3] to prevent extreme outliers from dominating.
    """
    z_scores = {ticker: {} for ticker in data_dict}
    metrics_dict = {}
    
    # First pass: group metrics by ticker
    for ticker, metrics in data_dict.items():
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value) and not pd.isna(value):
                metrics_dict.setdefault(metric, []).append((ticker, value))
    
    # Second pass: calculate z-scores for each metric
    for metric, ticker_values in metrics_dict.items():
        if len(ticker_values) < 2:
            # If only one ticker has this metric, assign z-score of 0
            for ticker, value in ticker_values:
                z_scores[ticker][metric] = 0.0
            continue
        
        tickers, values = zip(*ticker_values)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
        
        # Use robust standardization for skewed distributions
        if len(values) > 4 and torch.std(values_tensor) > 0:
            mean_val = torch.mean(values_tensor)
            std_val = torch.std(values_tensor)
            
            # Check for skewness
            skewness = torch.abs(torch.mean(((values_tensor - mean_val) / std_val) ** 3)).item()
            
            if skewness > 2:
                # Robustly standardize using median and IQR for skewed data
                median_val = torch.median(values_tensor)
                q1 = torch.quantile(values_tensor, 0.25)
                q3 = torch.quantile(values_tensor, 0.75)
                iqr = q3 - q1
                robust_std = max((iqr / 1.349).item(), 1e-6)  # 1.349 is a statistical constant for normal distribution
                
                metric_z_scores = (values_tensor - median_val) / robust_std
            else:
                # Use standard z-score for normally distributed data
                metric_z_scores = (values_tensor - mean_val) / max(std_val, torch.tensor(1e-6, device=device))
        else:
            # Not enough data or zero variance, use basic standardization
            mean_val = torch.mean(values_tensor)
            std_val = torch.std(values_tensor)
            metric_z_scores = (values_tensor - mean_val) / max(std_val, torch.tensor(1e-6, device=device))
        
        # Clamp extreme values
        metric_z_scores = torch.clamp(metric_z_scores, -3.0, 3.0)
        
        # Store calculated z-scores
        for ticker, z_score in zip(tickers, metric_z_scores):
            z_scores[ticker][metric] = z_score.item()
    
    return z_scores

def calculate_weighted_score(z_scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Calculate weighted score based on z-scores and weights.
    Handles missing metrics gracefully with default values.
    """
    if not weights:
        return 0
    
    score = 0
    total_weight = 0
    
    for metric, weight in weights.items():
        weight_abs = abs(weight)
        
        if metric in z_scores and np.isfinite(z_scores[metric]):
            score += z_scores[metric] * weight
        else:
            # For missing metrics:
            # - For positive weights, assume z-score of -1 (below average)
            # - For negative weights, assume z-score of 1 (above average)
            score += (-1 if weight > 0 else 1) * weight_abs
        
        total_weight += weight_abs
    
    if total_weight == 0:
        return 0
    
    # Return score normalized by total weight
    return score / total_weight

def construct_earnings_from_income(income_data):
    """
    Construct earnings data structure from income statement data.
    This is used as a fallback when the earnings endpoint is not available.
    """
    earnings_list = []
    
    if not income_data:
        return earnings_list
    
    for income_item in income_data:
        earnings_item = type('EarningsItem', (), {})()
        
        # Map essential fields
        earnings_item.date = getattr(income_item, 'period_ending', None)
        earnings_item.period_ending = getattr(income_item, 'period_ending', None)
        earnings_item.eps = getattr(income_item, 'diluted_earnings_per_share', 
                                   getattr(income_item, 'basic_earnings_per_share', None))
        earnings_item.revenue = getattr(income_item, 'revenue', None)
        earnings_item.ebitda = getattr(income_item, 'ebitda', None)
        earnings_item.net_income = getattr(income_item, 'consolidated_net_income', 
                                          getattr(income_item, 'net_income', None))
        earnings_item.fiscal_period = getattr(income_item, 'fiscal_period', None)
        earnings_item.fiscal_year = getattr(income_item, 'fiscal_year', None)
        
        # Only add items with sufficient data
        if (earnings_item.date or earnings_item.period_ending) and (
            earnings_item.eps or earnings_item.revenue or 
            earnings_item.ebitda or earnings_item.net_income
        ):
            earnings_list.append(earnings_item)
    
    return earnings_list

async def process_ticker_async(ticker: str) -> Tuple[str, Optional[Dict[str, float]]]:
    """
    Process a single ticker asynchronously with improved error handling.
    """
    try:
        # Check metrics cache with proper locking
        with metrics_cache_lock:
            if ticker in metrics_cache:
                return (ticker, metrics_cache[ticker])
        
        # Fetch financial data
        ticker, financial_data = await get_financial_data_async(ticker)
        
        # Check if we have essential data
        if not financial_data.get('income') or not financial_data.get('balance'):
            logger.warning(f"Insufficient financial data for {ticker}. Skipping...")
            return (ticker, None)
        
        try:
            # Extract metrics
            metrics_dict = extract_metrics_from_financial_data(financial_data)
            
            # Update cache with proper locking
            with metrics_cache_lock:
                metrics_cache[ticker] = metrics_dict
                if len(metrics_cache) > CACHE_SIZE:
                    oldest_key = next(iter(metrics_cache))
                    del metrics_cache[oldest_key]
            
            logger.debug(f"Successfully processed {ticker}")
            return (ticker, metrics_dict)
        
        except Exception as e:
            logger.error(f"Error extracting metrics for {ticker}: {e}")
            logger.error(traceback.format_exc())
            return (ticker, None)
    
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        logger.error(traceback.format_exc())
        return (ticker, None)

def process_ticker_sync(ticker: str) -> Tuple[str, Optional[Dict[str, float]]]:
    """
    Synchronous wrapper for process_ticker_async.
    """
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_ticker_async(ticker))
        loop.close()
        return result
    except Exception as e:
        logger.error(f"Error in process_ticker_sync for {ticker}: {e}")
        return (ticker, None)

async def screen_stocks_async(tickers: List[str], max_concurrent: int = os.cpu_count()*2) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Asynchronously screen stocks with improved consistency.
    """
    # Set a consistent random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    results = []
    all_metrics = {}
    valid_tickers = []
    
    # Create a semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Define a function to process each ticker with the semaphore
    async def process_with_semaphore(ticker):
        async with semaphore:
            return await process_ticker_async(ticker)
    
    # Create tasks for all tickers
    tasks = [process_with_semaphore(ticker) for ticker in tickers]
    
    # Process tasks as they complete and show progress
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        result = await task
        ticker, metrics_dict = result
        
        # Update progress
        progress = i / len(tickers) * 100
        if i % 5 == 0 or i == 1 or i == len(tickers):
            print(f"Processing stocks: {progress:.0f}% completed ({i}/{len(tickers)})", end='\r')
        
        # Store valid results
        if metrics_dict is not None:
            all_metrics[ticker] = metrics_dict
            valid_tickers.append(ticker)
    
    print()  # New line after progress updates
    
    if not valid_tickers:
        logger.warning("No valid tickers could be processed.")
        return []
    
    # Preprocess data to handle outliers consistently
    preprocessed_metrics = preprocess_data(all_metrics)
    
    # Calculate z-scores
    z_scores = calculate_z_scores(preprocessed_metrics)
    
    # Calculate scores for each ticker
    for ticker in valid_tickers:
        ticker_z_scores = z_scores[ticker]
        
        # Calculate category scores
        profitability_score = calculate_weighted_score(ticker_z_scores, PROFITABILITY_WEIGHTS)
        growth_score = calculate_weighted_score(ticker_z_scores, GROWTH_WEIGHTS)
        financial_health_score = calculate_weighted_score(ticker_z_scores, FINANCIAL_HEALTH_WEIGHTS)
        valuation_score = calculate_weighted_score(ticker_z_scores, VALUATION_WEIGHTS)
        efficiency_score = calculate_weighted_score(ticker_z_scores, EFFICIENCY_WEIGHTS)
        analyst_estimates_score = calculate_weighted_score(ticker_z_scores, ANALYST_ESTIMATES_WEIGHTS)
        
        # Calculate composite score with consistent weighting
        composite_score = (
            profitability_score * CATEGORY_WEIGHTS['profitability'] +
            growth_score * CATEGORY_WEIGHTS['growth'] +
            financial_health_score * CATEGORY_WEIGHTS['financial_health'] +
            valuation_score * CATEGORY_WEIGHTS['valuation'] +
            efficiency_score * CATEGORY_WEIGHTS['efficiency'] +
            analyst_estimates_score * CATEGORY_WEIGHTS['analyst_estimates']
        )
        
        # Store detailed results for this ticker
        detailed_results = {
            'raw_metrics': all_metrics[ticker],
            'preprocessed_metrics': preprocessed_metrics[ticker],
            'z_scores': ticker_z_scores,
            'category_scores': {
                'profitability': profitability_score,
                'growth': growth_score,
                'financial_health': financial_health_score,
                'valuation': valuation_score,
                'efficiency': efficiency_score,
                'analyst_estimates': analyst_estimates_score
            },
            'composite_score': composite_score
        }
        
        results.append((ticker, composite_score, detailed_results))
    
    # Sort results by composite score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Successfully screened {len(results)} stocks.")
    return results

def screen_stocks(tickers: List[str], max_workers: int = None) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Synchronously screen stocks with consistent results.
    """
    # Set a consistent random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    if max_workers is None:
        max_workers = min(32, os.cpu_count() * 4)
    
    all_metrics = {}
    valid_tickers = []
    
    # Process tickers with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for ticker, metrics in tqdm(executor.map(process_ticker_sync, tickers), total=len(tickers), desc="Processing stocks"):
            if metrics is not None:
                all_metrics[ticker] = metrics
                valid_tickers.append(ticker)
    
    if not valid_tickers:
        logger.warning("No valid tickers could be processed.")
        return []
    
    # Preprocess data to handle outliers consistently
    preprocessed_metrics = preprocess_data(all_metrics)
    
    # Calculate z-scores with consistent handling
    z_scores = calculate_z_scores(preprocessed_metrics)
    
    results = []
    
    # Calculate scores for each ticker
    for ticker in valid_tickers:
        ticker_z_scores = z_scores[ticker]
        
        # Calculate category scores
        category_scores = {
            'profitability': calculate_weighted_score(ticker_z_scores, PROFITABILITY_WEIGHTS),
            'growth': calculate_weighted_score(ticker_z_scores, GROWTH_WEIGHTS),
            'financial_health': calculate_weighted_score(ticker_z_scores, FINANCIAL_HEALTH_WEIGHTS),
            'valuation': calculate_weighted_score(ticker_z_scores, VALUATION_WEIGHTS),
            'efficiency': calculate_weighted_score(ticker_z_scores, EFFICIENCY_WEIGHTS),
            'analyst_estimates': calculate_weighted_score(ticker_z_scores, ANALYST_ESTIMATES_WEIGHTS)
        }
        
        # Calculate composite score
        composite_score = sum(score * CATEGORY_WEIGHTS[cat] for cat, score in category_scores.items())
        
        # Store detailed results
        detailed_results = {
            'raw_metrics': all_metrics[ticker],
            'preprocessed_metrics': preprocessed_metrics[ticker],
            'z_scores': ticker_z_scores,
            'category_scores': category_scores,
            'composite_score': composite_score
        }
        
        results.append((ticker, composite_score, detailed_results))
    
    # Sort results by composite score (descending)
    results.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Successfully screened {len(results)} stocks.")
    return results

def get_metric_contributions(ticker: str) -> pd.DataFrame:
    """
    Get detailed breakdown of metric contributions to the composite score.
    Uses fixed random seed for consistent results.
    """
    # Set seed for consistency
    np.random.seed(42)
    torch.manual_seed(42)
    
    results = screen_stocks([ticker], max_workers=1)
    
    if not results:
        logger.warning(f"No data found for {ticker}")
        return pd.DataFrame()
    
    _, _, detailed_results = results[0]
    z_scores = detailed_results['z_scores']
    raw_metrics = detailed_results['raw_metrics']
    
    data = []
    
    # Process each category and its metrics
    for category, weights in [
        ('Profitability', PROFITABILITY_WEIGHTS),
        ('Growth', GROWTH_WEIGHTS),
        ('Financial Health', FINANCIAL_HEALTH_WEIGHTS),
        ('Valuation', VALUATION_WEIGHTS),
        ('Efficiency', EFFICIENCY_WEIGHTS),
        ('Analyst Estimates', ANALYST_ESTIMATES_WEIGHTS)
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
    
    # Create DataFrame and sort by absolute contribution
    df = pd.DataFrame(data)
    df['Abs Contribution'] = df['Contribution'].abs()
    df = df.sort_values('Abs Contribution', ascending=False)
    df = df.drop('Abs Contribution', axis=1)
    
    return df

def get_estimate_accuracy_report(ticker: str) -> pd.DataFrame:
    """
    Generate a report on analyst estimate accuracy and forward projections for a stock.
    """
    obb_client = get_openbb_client()
    financial_data = {}
    
    try:
        # Fetch historical estimates
        historical_estimates_response = obb_client.equity.estimates.historical(
            symbol=ticker, provider='fmp'
        )
        financial_data['historical_estimates'] = historical_estimates_response.results
        
        # Fetch earnings data
        earnings_response = obb_client.equity.fundamental.earnings(
            symbol=ticker, provider='fmp'
        )
        financial_data['earnings'] = earnings_response.results
        
        # Fetch forward sales estimates
        try:
            forward_sales_response = obb_client.equity.estimates.forward_sales(
                symbol=ticker
            )
        except Exception as e:
            logger.warning(f"Error fetching forward sales for {ticker} in estimate report from FMP: {e}")
            try:
                forward_sales_response = obb_client.equity.estimates.forward_sales(
                    symbol=ticker, provider='intrinio'
                )
            except Exception as e2:
                logger.warning(f"Error fetching forward sales for {ticker} in estimate report from Intrinio: {e2}")
                forward_sales_response = None
        
        if forward_sales_response is not None:
            financial_data['forward_sales'] = forward_sales_response.results
        else:
            financial_data['forward_sales'] = []
        
        # Fetch forward EBITDA estimates
        try:
            forward_ebitda_response = obb_client.equity.estimates.forward_ebitda(
                symbol=ticker, fiscal_period='annual', provider='fmp'
            )
        except Exception as e:
            logger.warning(f"Error fetching forward EBITDA for {ticker} in estimate report from FMP: {e}")
            try:
                forward_ebitda_response = obb_client.equity.estimates.forward_ebitda(
                    symbol=ticker, fiscal_period='annual', provider='intrinio'
                )
            except Exception as e2:
                logger.warning(f"Error fetching forward EBITDA for {ticker} in estimate report from Intrinio: {e2}")
                forward_ebitda_response = None
        
        if forward_ebitda_response is not None:
            financial_data['forward_ebitda'] = forward_ebitda_response.results
        else:
            financial_data['forward_ebitda'] = []
    
    except Exception as e:
        logger.error(f"Error fetching estimate data for {ticker}: {e}")
        return pd.DataFrame()
    
    if not financial_data.get('historical_estimates') or not financial_data.get('earnings'):
        logger.warning(f"No historical estimates or earnings data for {ticker}")
        return pd.DataFrame()
    
    # Calculate historical accuracy metrics
    accuracy_metrics = calculate_estimate_accuracy(
        financial_data['historical_estimates'],
        financial_data['earnings']
    )
    
    # Process forward estimates
    forward_metrics = {}
    if financial_data.get('forward_sales'):
        future_estimates = [est for est in financial_data['forward_sales'] 
                           if hasattr(est, 'fiscal_year') and 
                           est.fiscal_year > datetime.now().year]
        
        if future_estimates:
            future_estimates.sort(key=lambda x: x.fiscal_year)
            for i, estimate in enumerate(future_estimates[:3]):  # Limit to next 3 years
                year = estimate.fiscal_year
                mean = get_attribute_value(estimate, 'mean')
                low = get_attribute_value(estimate, 'low_estimate')
                high = get_attribute_value(estimate, 'high_estimate')
                std_dev = get_attribute_value(estimate, 'standard_deviation')
                analysts = get_attribute_value(estimate, 'number_of_analysts')
                
                forward_metrics[f'forward_sales_{year}'] = mean
                forward_metrics[f'forward_sales_low_{year}'] = low
                forward_metrics[f'forward_sales_high_{year}'] = high
                forward_metrics[f'forward_sales_std_dev_{year}'] = std_dev
                forward_metrics[f'forward_sales_analysts_{year}'] = analysts
    
    if financial_data.get('forward_ebitda'):
        future_estimates = [est for est in financial_data['forward_ebitda'] 
                           if hasattr(est, 'fiscal_year') and 
                           est.fiscal_year > datetime.now().year]
        
        if future_estimates:
            future_estimates.sort(key=lambda x: x.fiscal_year)
            for i, estimate in enumerate(future_estimates[:3]):  # Limit to next 3 years
                year = estimate.fiscal_year
                mean = get_attribute_value(estimate, 'mean')
                low = get_attribute_value(estimate, 'low_estimate')
                high = get_attribute_value(estimate, 'high_estimate')
                std_dev = get_attribute_value(estimate, 'standard_deviation')
                analysts = get_attribute_value(estimate, 'number_of_analysts')
                
                forward_metrics[f'forward_ebitda_{year}'] = mean
                forward_metrics[f'forward_ebitda_low_{year}'] = low
                forward_metrics[f'forward_ebitda_high_{year}'] = high
                forward_metrics[f'forward_ebitda_std_dev_{year}'] = std_dev
                forward_metrics[f'forward_ebitda_analysts_{year}'] = analysts
    
    # Combine all metrics and create dataframe
    all_metrics = {**accuracy_metrics, **forward_metrics}
    data = []
    
    for metric, value in all_metrics.items():
        if isinstance(value, (int, float)) and np.isfinite(value):
            metric_type = 'Accuracy' if 'accuracy' in metric else 'Forward Estimate'
            data.append({
                'Metric': metric,
                'Value': value,
                'Type': metric_type
            })
    
    return pd.DataFrame(data)

def generate_stock_report(ticker: str) -> Dict[str, Any]:
    """
    Generate a comprehensive fundamental analysis report for a stock.
    Uses fixed random seed for consistent results.
    """
    # Set seed for consistency
    np.random.seed(42)
    torch.manual_seed(42)
    
    obb_client = get_openbb_client()
    financial_data = {}
    
    try:
        # Fetch fundamental financial data
        for data_type in ['income', 'balance', 'cash']:
            try:
                response = getattr(obb_client.equity.fundamental, data_type)(
                    symbol=ticker, period='annual', limit=5, provider='fmp'
                )
                financial_data[data_type] = response.results
            except Exception as e:
                logger.warning(f"Error fetching {data_type} for {ticker}: {e}")
                financial_data[data_type] = []
        
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
        
        # Fetch additional metrics
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
        
        # Fetch forward estimates with fallbacks
        try:
            forward_sales_response = obb_client.equity.estimates.forward_sales(
                symbol=ticker
            )
            financial_data['forward_sales'] = forward_sales_response.results
        except Exception as e:
            logger.warning(f"Error fetching forward sales for {ticker} from standard: {e}")
            try:
                forward_sales_response = obb_client.equity.estimates.forward_sales(
                    symbol=ticker, provider='intrinio'
                )
                financial_data['forward_sales'] = forward_sales_response.results
            except Exception as e2:
                logger.warning(f"Error fetching forward sales for {ticker} from Intrinio: {e2}")
                financial_data['forward_sales'] = []
        
        try:
            forward_ebitda_response = obb_client.equity.estimates.forward_ebitda(
                symbol=ticker, fiscal_period='annual', provider='fmp'
            )
            financial_data['forward_ebitda'] = forward_ebitda_response.results
        except Exception as e:
            logger.warning(f"Error fetching forward EBITDA for {ticker} from FMP: {e}")
            try:
                forward_ebitda_response = obb_client.equity.estimates.forward_ebitda(
                    symbol=ticker, fiscal_period='annual', provider='intrinio'
                )
                financial_data['forward_ebitda'] = forward_ebitda_response.results
            except Exception as e2:
                logger.warning(f"Error fetching forward EBITDA for {ticker} from Intrinio: {e2}")
                financial_data['forward_ebitda'] = []
        
        # Fetch historical estimates
        try:
            historical_estimates_response = obb_client.equity.estimates.historical(
                symbol=ticker, provider='fmp'
            )
            financial_data['historical_estimates'] = historical_estimates_response.results
        except Exception as e:
            logger.warning(f"No historical estimates data for {ticker}: {e}")
            financial_data['historical_estimates'] = []
        
        # Fetch earnings
        try:
            if hasattr(obb_client.equity.fundamental, 'earnings'):
                earnings_response = obb_client.equity.fundamental.earnings(
                    symbol=ticker, provider='fmp'
                )
                financial_data['earnings'] = earnings_response.results
            else:
                logger.warning(f"Earnings endpoint not available for {ticker}, constructing from income data")
                financial_data['earnings'] = construct_earnings_from_income(financial_data['income'])
        except Exception as e:
            logger.warning(f"Error fetching earnings data for {ticker}: {e}")
            financial_data['earnings'] = construct_earnings_from_income(financial_data['income'])
    
    except Exception as e:
        logger.error(f"Error fetching financial data for {ticker}: {e}")
    
    # Extract metrics and calculate scores
    metrics = extract_metrics_from_financial_data(financial_data)
    
    # Process historical estimate accuracy
    estimate_accuracy = {}
    if financial_data.get('historical_estimates') and financial_data.get('earnings'):
        estimate_accuracy = calculate_estimate_accuracy(
            financial_data['historical_estimates'],
            financial_data['earnings']
        )
    
    # Get category scores by screening the ticker
    ticker_results = screen_stocks([ticker], max_workers=1)
    category_scores = {}
    composite_score = 0.0
    
    if ticker_results:
        _, composite_score, detailed_results = ticker_results[0]
        category_scores = detailed_results['category_scores']
    
    # Create report structure
    report = {
        'ticker': ticker,
        'composite_score': composite_score,
        'category_scores': category_scores,
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
                'eps_growth': metrics.get('growth_eps'),
                'forward_sales_growth': metrics.get('forward_sales_growth'),
                'forward_ebitda_growth': metrics.get('forward_ebitda_growth')
            },
            'financial_health': {
                'current_ratio': metrics.get('current_ratio'),
                'debt_to_equity': metrics.get('debt_to_equity'),
                'debt_to_assets': metrics.get('debt_to_assets'),
                'interest_coverage': metrics.get('interest_coverage'),
                'cash_to_debt': metrics.get('cash_to_debt')
            },
            'valuation': {
                'pe_ratio': metrics.get('pe_ratio'),
                'price_to_book': metrics.get('price_to_book'),
                'price_to_sales': metrics.get('price_to_sales'),
                'ev_to_ebitda': metrics.get('ev_to_ebitda'),
                'dividend_yield': metrics.get('dividend_yield'),
                'peg_ratio': metrics.get('peg_ratio')
            },
            'efficiency': {
                'asset_turnover': metrics.get('asset_turnover'),
                'inventory_turnover': metrics.get('inventory_turnover'),
                'receivables_turnover': metrics.get('receivables_turnover'),
                'cash_conversion_cycle': metrics.get('cash_conversion_cycle'),
                'capex_to_revenue': metrics.get('capex_to_revenue')
            },
            'analyst_estimates': {
                'estimate_eps_accuracy': estimate_accuracy.get('estimate_eps_accuracy', 0),
                'estimate_revenue_accuracy': estimate_accuracy.get('estimate_revenue_accuracy', 0),
                'forward_sales_growth': metrics.get('forward_sales_growth'),
                'forward_ebitda_growth': metrics.get('forward_ebitda_growth'),
                'estimate_revision_momentum': metrics.get('estimate_revision_momentum', 0),
                'estimate_consensus_deviation': metrics.get('estimate_consensus_deviation', 0)
            }
        },
        'strengths': [],
        'weaknesses': [],
        'raw_metrics': metrics,
        'z_scores': {}
    }
    
    # Add z-scores if available
    if ticker_results:
        report['z_scores'] = ticker_results[0][2]['z_scores']
    
    # Identify strengths and weaknesses based on industry benchmarks
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
        'interest_coverage': 3.0,
        'cash_to_debt': 0.5
    }
    
    growth_benchmarks = {
        'growth_revenue': 0.10,
        'growth_net_income': 0.10,
        'growth_eps': 0.10,
        'forward_sales_growth': 0.10,
        'forward_ebitda_growth': 0.10
    }
    
    estimate_benchmarks = {
        'estimate_eps_accuracy': 0.7,
        'estimate_revenue_accuracy': 0.7,
        'estimate_revision_momentum': 0.05
    }
    
    valuation_benchmarks = {
        'pe_ratio': 20.0,
        'price_to_book': 3.0,
        'price_to_sales': 2.0,
        'ev_to_ebitda': 12.0,
        'dividend_yield': 0.02
    }
    
    # Check profitability metrics against benchmarks
    for metric, benchmark in profitability_benchmarks.items():
        if metric in metrics and metrics.get(metric) is not None and np.isfinite(metrics.get(metric)):
            if metrics.get(metric) > benchmark * 1.5:
                report['strengths'].append(f"Strong {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
            elif metrics.get(metric) < benchmark * 0.5:
                report['weaknesses'].append(f"Low {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
    
    # Check financial health metrics against benchmarks
    for metric, benchmark in financial_health_benchmarks.items():
        if metric in metrics and metrics.get(metric) is not None and np.isfinite(metrics.get(metric)):
            if metric == 'debt_to_equity' or metric == 'debt_to_assets':
                if metrics.get(metric) < benchmark * 0.5:
                    report['strengths'].append(f"Low {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
                elif metrics.get(metric) > benchmark * 1.5:
                    report['weaknesses'].append(f"High {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
            else:
                if metrics.get(metric) > benchmark * 1.5:
                    report['strengths'].append(f"Strong {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
                elif metrics.get(metric) < benchmark * 0.5:
                    report['weaknesses'].append(f"Low {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
    
    # Check growth metrics against benchmarks
    for metric, benchmark in growth_benchmarks.items():
        if metric in metrics and metrics.get(metric) is not None and np.isfinite(metrics.get(metric)):
            if metrics.get(metric) > benchmark * 1.5:
                report['strengths'].append(f"Strong {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
            elif metrics.get(metric) < 0:
                report['weaknesses'].append(f"Negative {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
    
    # Check estimate metrics against benchmarks
    for metric, benchmark in estimate_benchmarks.items():
        if metric in metrics and metrics.get(metric) is not None and np.isfinite(metrics.get(metric)):
            if metrics.get(metric) > benchmark:
                report['strengths'].append(f"Strong {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
            elif metrics.get(metric) < 0:
                report['weaknesses'].append(f"Poor {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
    
    # Check valuation metrics against benchmarks
    for metric, benchmark in valuation_benchmarks.items():
        if metric in metrics and metrics.get(metric) is not None and np.isfinite(metrics.get(metric)):
            if metric == 'dividend_yield':
                if metrics.get(metric) > benchmark * 1.5:
                    report['strengths'].append(f"High {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
            elif metric in ['pe_ratio', 'price_to_book', 'price_to_sales', 'ev_to_ebitda']:
                if metrics.get(metric) < benchmark * 0.6:
                    report['strengths'].append(f"Low {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
                elif metrics.get(metric) > benchmark * 1.5:
                    report['weaknesses'].append(f"High {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
    
    return report

async def get_peers_async(ticker: str) -> List[str]:
    """
    Asynchronously fetch peers for a given ticker using OpenBB API.
    Returns a list of peer ticker symbols.
    """
    obb_client = get_openbb_client()
    
    try:
        response = await rate_limited_api_call(
            obb_client.equity.compare.peers, 
            symbol=ticker, 
            provider='fmp'
        )
        
        if response and hasattr(response, 'results'):
            result = response.results
            
            if isinstance(result, list) and len(result) > 0:
                if hasattr(result[0], 'peers_list'):
                    return result[0].peers_list
                elif isinstance(result[0], dict) and 'peers_list' in result[0]:
                    return result[0]['peers_list']
                else:
                    return []
            elif hasattr(result, 'peers_list'):
                return result.peers_list
            else:
                return []
        else:
            return []
    
    except Exception as e:
        logger.error(f"Error fetching peers for {ticker}: {e}")
        return []

async def analyze_ticker_with_peers(ticker: str, depth: int = 1, visited: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Analyze a ticker and its peers without recursion to avoid asyncio event loop issues.
    Returns peer comparison data for the ticker.
    """
    # Set seed for consistency
    np.random.seed(42)
    torch.manual_seed(42)
    
    if visited is None:
        visited = set()
    
    if ticker in visited:
        return {'ticker': ticker, 'peer_comparison': {}, 'peers': []}
    
    visited.add(ticker)
    
    # Screen the target ticker
    ticker_results = await screen_stocks_async([ticker])
    if not ticker_results:
        return {'ticker': ticker, 'peer_comparison': {}, 'peers': []}
    
    ticker_score = ticker_results[0][1]
    
    # Get peers for the ticker
    peers_list = await get_peers_async(ticker)
    if not peers_list:
        return {
            'ticker': ticker,
            'peer_comparison': {
                'average_score': ticker_score,
                'std_dev': 0.0,
                'count': 0,
                'percentile': 50.0  # Default to median
            },
            'peers': []
        }
    
    # Filter out already visited peers
    filtered_peers = [peer for peer in peers_list if peer not in visited]
    
    peer_scores = []
    peer_data = []
    
    # Process peers
    if filtered_peers:
        peers_results = await screen_stocks_async(filtered_peers)
        peers_dict = {res[0]: res[1] for res in peers_results}
        
        for peer in filtered_peers:
            if peer in peers_dict:
                peer_score = peers_dict[peer]
                peer_scores.append(peer_score)
                peer_data.append({
                    'ticker': peer,
                    'score': peer_score
                })
    
    # Calculate statistics
    if peer_scores:
        peer_scores.append(ticker_score)  # Include the target ticker in stats
        all_scores = np.array(peer_scores)
        
        average_score = np.mean(all_scores)
        std_dev = np.std(all_scores)
        
        # Calculate percentile rank of the ticker
        percentile = 100 * (len(np.where(all_scores < ticker_score)[0]) / len(all_scores))
    else:
        average_score = ticker_score
        std_dev = 0.0
        percentile = 50.0
    
    return {
        'ticker': ticker,
        'peer_comparison': {
            'average_score': average_score,
            'std_dev': std_dev,
            'count': len(peer_scores) - 1,  # Exclude target ticker from count
            'percentile': percentile
        },
        'peers': peer_data
    }

async def gather_peer_analysis(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Gather peer analysis for a list of tickers, all within a single async context.
    """
    # Set seed for consistency
    np.random.seed(42)
    torch.manual_seed(42)
    
    visited = set()
    
    tasks = [analyze_ticker_with_peers(ticker, depth=1, visited=visited) for ticker in tickers]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    peer_analysis = {}
    for i, ticker in enumerate(tickers):
        if isinstance(results[i], Exception):
            logger.error(f"Error in peer analysis for {ticker}: {results[i]}")
            peer_analysis[ticker] = {
                'peer_comparison': {
                    'average_score': 0.0,
                    'std_dev': 0.0,
                    'count': 0,
                    'percentile': 0.0
                },
                'peers': []
            }
        else:
            peer_analysis[ticker] = results[i]
    
    return peer_analysis

def run_stock_screen(tickers: List[str], use_async: bool = True) -> pd.DataFrame:
    """
    Run a complete stock screen and return a DataFrame with rankings and scores.
    
    Parameters:
    tickers (List[str]): List of stock ticker symbols to screen
    use_async (bool): Whether to use async processing (faster but may have issues in some environments)
    
    Returns:
    pd.DataFrame: DataFrame with screening results
    """
    # Set seed for consistency
    np.random.seed(42)
    torch.manual_seed(42)
    
    if use_async:
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(screen_stocks_async(tickers))
    else:
        results = screen_stocks(tickers)
    
    if not results:
        return pd.DataFrame()
    
    # Create DataFrame from results
    data = []
    for rank, (ticker, score, details) in enumerate(results, 1):
        category_scores = details['category_scores']
        
        data.append({
            'Rank': rank,
            'Ticker': ticker,
            'Composite Score': score,
            'Profitability': category_scores['profitability'],
            'Growth': category_scores['growth'],
            'Financial Health': category_scores['financial_health'],
            'Valuation': category_scores['valuation'],
            'Efficiency': category_scores['efficiency'],
            'Analyst Estimates': category_scores['analyst_estimates']
        })
    
    return pd.DataFrame(data)

def compare_stocks(tickers: List[str], use_async: bool = True) -> pd.DataFrame:
    """
    Compare a list of stocks on key metrics.
    
    Parameters:
    tickers (List[str]): List of stock ticker symbols to compare
    use_async (bool): Whether to use async processing
    
    Returns:
    pd.DataFrame: DataFrame with comparison of key metrics
    """
    # Set seed for consistency
    np.random.seed(42)
    torch.manual_seed(42)
    
    all_metrics = {}
    
    if use_async:
        loop = asyncio.get_event_loop()
        tasks = [process_ticker_async(ticker) for ticker in tickers]
        results = loop.run_until_complete(asyncio.gather(*tasks))
        
        for ticker, metrics in results:
            if metrics is not None:
                all_metrics[ticker] = metrics
    else:
        for ticker in tqdm(tickers, desc="Fetching data"):
            ticker, metrics = process_ticker_sync(ticker)
            if metrics is not None:
                all_metrics[ticker] = metrics
    
    if not all_metrics:
        return pd.DataFrame()
    
    # Define key metrics to compare
    key_metrics = [
        # Profitability
        'gross_profit_margin', 'operating_income_margin', 'net_income_margin', 'return_on_equity',
        # Growth
        'growth_revenue', 'growth_net_income', 'growth_eps',
        # Financial Health
        'current_ratio', 'debt_to_equity', 'interest_coverage',
        # Valuation
        'pe_ratio', 'price_to_book', 'ev_to_ebitda', 'dividend_yield',
        # Efficiency
        'asset_turnover', 'inventory_turnover',
        # Estimates
        'forward_sales_growth', 'estimate_revision_momentum'
    ]
    
    # Create comparison DataFrame
    comparison_data = {}
    
    for ticker, metrics in all_metrics.items():
        ticker_data = {}
        for metric in key_metrics:
            if metric in metrics and metrics[metric] is not None and np.isfinite(metrics[metric]):
                ticker_data[metric] = metrics[metric]
            else:
                ticker_data[metric] = np.nan
        
        comparison_data[ticker] = ticker_data
    
    # Convert to DataFrame and transpose for better display
    df = pd.DataFrame(comparison_data).T
    
    # Rename columns to be more reader-friendly
    df.columns = [col.replace('_', ' ').title() for col in df.columns]
    
    # Round numerical values
    for col in df.columns:
        df[col] = df[col].round(4)
    
    return df