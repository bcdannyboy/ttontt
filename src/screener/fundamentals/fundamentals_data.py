import asyncio
import traceback
import logging
import numpy as np
import json
import os
from datetime import datetime, timedelta

from src.screener.fundamentals.fundamentals_core import get_openbb_client, rate_limited_api_call, select_valid_record
from src.screener.fundamentals.fundamentals_metrics import construct_earnings_from_income

logger = logging.getLogger(__name__)

# Cache directory setup
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(ticker, data_type):
    """Get path for cached data file"""
    return os.path.join(CACHE_DIR, f"{ticker}_{data_type}_cache.json")

async def get_financial_data_async(ticker: str):
    """
    Async version of get_financial_data using OpenBB API with improved fallbacks.
    Fetches all required financial data for a stock, handling provider fallbacks properly.
    
    Returns:
        tuple: (ticker, financial_data dictionary)
    """
    obb_client = get_openbb_client()
    financial_data = {}
    
    # List of providers to try in order of preference
    providers = ['fmp', 'intrinio', 'yfinance', 'polygon', 'alphavantage']
    
    try:
        # Check if we have cached data that's recent enough (less than 24 hours old)
        cached_data = load_cached_data(ticker)
        if cached_data:
            logger.info(f"Using cached data for {ticker}")
            return (ticker, cached_data)
        
        # Try to fetch essential financial statements with multiple provider fallbacks
        income_data = None
        balance_data = None
        cash_data = None
        
        # Try each provider for income data
        for provider in providers:
            try:
                income_task = rate_limited_api_call(
                    obb_client.equity.fundamental.income,
                    symbol=ticker, period='annual', limit=5, provider=provider
                )
                income_response = await income_task
                if income_response and hasattr(income_response, 'results') and income_response.results:
                    income_data = income_response.results
                    logger.info(f"Successfully fetched income data for {ticker} using {provider}")
                    break
            except Exception as e:
                logger.warning(f"Failed to fetch income data for {ticker} using {provider}: {e}")
                continue
        
        # Try each provider for balance sheet data
        for provider in providers:
            try:
                balance_task = rate_limited_api_call(
                    obb_client.equity.fundamental.balance,
                    symbol=ticker, period='annual', limit=5, provider=provider
                )
                balance_response = await balance_task
                if balance_response and hasattr(balance_response, 'results') and balance_response.results:
                    balance_data = balance_response.results
                    logger.info(f"Successfully fetched balance data for {ticker} using {provider}")
                    break
            except Exception as e:
                logger.warning(f"Failed to fetch balance data for {ticker} using {provider}: {e}")
                continue
        
        # Try each provider for cash flow data
        for provider in providers:
            try:
                cash_task = rate_limited_api_call(
                    obb_client.equity.fundamental.cash,
                    symbol=ticker, period='annual', limit=5, provider=provider
                )
                cash_response = await cash_task
                if cash_response and hasattr(cash_response, 'results') and cash_response.results:
                    cash_data = cash_response.results
                    logger.info(f"Successfully fetched cash flow data for {ticker} using {provider}")
                    break
            except Exception as e:
                logger.warning(f"Failed to fetch cash flow data for {ticker} using {provider}: {e}")
                continue
        
        # Check if essential data is available
        if not income_data or not balance_data:
            # Last resort: Try to fetch summary data that might have minimum required info
            try:
                summary_response = await rate_limited_api_call(
                    obb_client.equity.fundamental.overview, 
                    symbol=ticker
                )
                if summary_response and hasattr(summary_response, 'results') and summary_response.results:
                    # Create minimal income and balance data from overview
                    if not income_data:
                        income_data = construct_minimal_income_data(summary_response.results)
                    if not balance_data:
                        balance_data = construct_minimal_balance_data(summary_response.results)
                    logger.info(f"Constructed minimal data for {ticker} from overview")
            except Exception as e:
                logger.warning(f"Failed to fetch overview data for {ticker}: {e}")
        
        # If still no data, then we can't proceed
        if not income_data or not balance_data:
            logger.error(f"Essential data fetch failed for {ticker}")
            return (ticker, financial_data)
        
        # Store successfully fetched basic data
        financial_data['income'] = income_data
        financial_data['balance'] = balance_data
        financial_data['cash'] = cash_data or []
        
        # Since we have essential data, fetch additional data with proper error handling
        additional_data_tasks = []
        
        # Add tasks for additional financial data
        for data_type, method in [
            ('income_growth', obb_client.equity.fundamental.income_growth),
            ('balance_growth', obb_client.equity.fundamental.balance_growth),
            ('cash_growth', obb_client.equity.fundamental.cash_growth),
            ('ratios', obb_client.equity.fundamental.ratios),
            ('metrics', obb_client.equity.fundamental.metrics),
            ('historical_estimates', obb_client.equity.estimates.historical)
        ]:
            for provider in providers[:2]:  # Only try the first two providers for additional data
                additional_data_tasks.append(
                    ('additional', data_type, provider, rate_limited_api_call(
                        method, symbol=ticker, period='annual' if data_type != 'historical_estimates' else None, 
                        limit=5 if data_type != 'historical_estimates' else None, provider=provider
                    ))
                )
        
        # Add tasks for forward estimates
        for data_type, method in [
            ('forward_sales', obb_client.equity.estimates.forward_sales),
            ('forward_ebitda', obb_client.equity.estimates.forward_ebitda)
        ]:
            for provider in providers[:2]:  # Only try the first two providers for estimates
                additional_data_tasks.append(
                    ('estimates', data_type, provider, rate_limited_api_call(
                        method, symbol=ticker, fiscal_period='annual' if data_type == 'forward_ebitda' else None, 
                        provider=provider
                    ))
                )
        
        # Add task for dividends
        additional_data_tasks.append(
            ('dividends', 'dividends', 'fmp', rate_limited_api_call(
                obb_client.equity.fundamental.dividends, symbol=ticker, provider='fmp'
            ))
        )
        
        # Add task for earnings (or construct from income if not available)
        try:
            if hasattr(obb_client.equity.fundamental, 'earnings'):
                additional_data_tasks.append(
                    ('earnings', 'earnings', 'fmp', rate_limited_api_call(
                        obb_client.equity.fundamental.earnings, symbol=ticker, provider='fmp'
                    ))
                )
        except Exception:
            logger.warning(f"Earnings endpoint not available for {ticker}, will construct from income data")
        
        # Process all additional data tasks
        processed_types = set()
        for category, data_type, provider, task in additional_data_tasks:
            if data_type in processed_types:
                continue  # Skip if we already have this data type
                
            try:
                response = await task
                if response and hasattr(response, 'results'):
                    results = response.results
                    if results:
                        financial_data[data_type] = results
                        processed_types.add(data_type)
                        logger.debug(f"Added {data_type} data for {ticker} using {provider}")
            except Exception as e:
                logger.warning(f"Error fetching {data_type} for {ticker} using {provider}: {str(e)}")
        
        # If earnings data wasn't retrieved, construct it from income data
        if 'earnings' not in financial_data:
            financial_data['earnings'] = construct_earnings_from_income(financial_data['income'])
            logger.info(f"Constructed earnings data for {ticker} from income data")
        
        # Cache the successfully fetched data
        save_to_cache(ticker, financial_data)
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        logger.error(traceback.format_exc())
    
    return (ticker, financial_data)

def load_cached_data(ticker):
    """Load data from cache if it exists and is recent"""
    cache_path = get_cache_path(ticker, "financial_data")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
                # Check if cache is less than 24 hours old
                if datetime.fromtimestamp(cached.get('timestamp', 0)) > datetime.now() - timedelta(hours=24):
                    return cached.get('data', {})
        except Exception as e:
            logger.warning(f"Failed to load cache for {ticker}: {e}")
    return None

def save_to_cache(ticker, data):
    """Save data to cache with timestamp"""
    cache_path = get_cache_path(ticker, "financial_data")
    try:
        with open(cache_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().timestamp(),
                'data': data
            }, f, default=lambda o: str(o) if hasattr(o, '__dict__') else o.__dict__ if hasattr(o, '__dict__') else str(o))
    except Exception as e:
        logger.warning(f"Failed to save cache for {ticker}: {e}")

def construct_minimal_income_data(overview_data):
    """Construct minimal income statement data from overview"""
    if not overview_data or not isinstance(overview_data, list) or not overview_data[0]:
        return None
    
    overview = overview_data[0]
    income_item = type('IncomeItem', (), {})()
    
    # Try to get relevant fields from overview
    income_item.period_ending = getattr(overview, 'fiscal_year_end', datetime.now().strftime('%Y-%m-%d'))
    income_item.revenue = getattr(overview, 'revenue', 0)
    income_item.net_income = getattr(overview, 'net_income', 0)
    income_item.ebitda = getattr(overview, 'ebitda', 0)
    
    # Calculate margins if possible
    if income_item.revenue and income_item.revenue > 0:
        income_item.gross_profit_margin = getattr(overview, 'gross_profit_margin', 0.5)
        income_item.operating_income_margin = getattr(overview, 'operating_margin', 0.1)
        income_item.net_income_margin = income_item.net_income / income_item.revenue if income_item.net_income else 0
        income_item.ebitda_margin = income_item.ebitda / income_item.revenue if income_item.ebitda else 0
    
    return [income_item]

def construct_minimal_balance_data(overview_data):
    """Construct minimal balance sheet data from overview"""
    if not overview_data or not isinstance(overview_data, list) or not overview_data[0]:
        return None
    
    overview = overview_data[0]
    balance_item = type('BalanceItem', (), {})()
    
    # Try to get relevant fields from overview
    balance_item.period_ending = getattr(overview, 'fiscal_year_end', datetime.now().strftime('%Y-%m-%d'))
    balance_item.total_assets = getattr(overview, 'total_assets', 0)
    balance_item.total_liabilities = getattr(overview, 'total_liabilities', 0)
    balance_item.total_shareholders_equity = balance_item.total_assets - balance_item.total_liabilities
    balance_item.cash_and_cash_equivalents = getattr(overview, 'cash_and_cash_equivalents', 0)
    balance_item.total_debt = getattr(overview, 'total_debt', 0)
    balance_item.net_debt = balance_item.total_debt - balance_item.cash_and_cash_equivalents
    
    return [balance_item]