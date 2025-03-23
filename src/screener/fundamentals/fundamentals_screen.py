from datetime import datetime
import os
import asyncio
import concurrent.futures
import time
from tqdm import tqdm
import numpy as np
import logging
import traceback
import pandas as pd
import json
from src.screener.fundamentals.fundamentals_core import metrics_cache, metrics_cache_lock, CACHE_SIZE
from src.screener.fundamentals.fundamentals_data import get_financial_data_async
from src.screener.fundamentals.fundamentals_metrics import (extract_metrics_from_financial_data, preprocess_data,
                                    calculate_z_scores, calculate_weighted_score, construct_earnings_from_income,
                                    get_attribute_value, PROFITABILITY_WEIGHTS, GROWTH_WEIGHTS,
                                    FINANCIAL_HEALTH_WEIGHTS, VALUATION_WEIGHTS, EFFICIENCY_WEIGHTS, ANALYST_ESTIMATES_WEIGHTS)

logger = logging.getLogger(__name__)

# Create a cache directory to speed up repeated analyses
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def get_metrics_cache_path(ticker):
    """Get path for cached metrics file"""
    return os.path.join(CACHE_DIR, f"{ticker}_metrics_cache.json")

async def process_ticker_async(ticker: str):
    """
    Process a single ticker asynchronously with improved error handling and caching.
    """
    try:
        # Check memory cache first
        with metrics_cache_lock:
            if ticker in metrics_cache:
                logger.debug(f"Using memory-cached metrics for {ticker}")
                return (ticker, metrics_cache[ticker])
        
        # Check disk cache
        cache_path = get_metrics_cache_path(ticker)
        if os.path.exists(cache_path):
            try:
                cache_age = time.time() - os.path.getmtime(cache_path)
                # Use cache if less than 1 day old
                if cache_age < 86400:  # 24 hours in seconds
                    with open(cache_path, 'r') as f:
                        metrics_dict = json.load(f)
                        with metrics_cache_lock:
                            metrics_cache[ticker] = metrics_dict
                        logger.debug(f"Using disk-cached metrics for {ticker}")
                        return (ticker, metrics_dict)
            except Exception as e:
                logger.warning(f"Error reading cache for {ticker}: {e}")
        
        # Fetch fresh data
        ticker, financial_data = await get_financial_data_async(ticker)
        
        # Verify that we have enough data to process
        if not financial_data.get('income') or not financial_data.get('balance'):
            logger.warning(f"Insufficient financial data for {ticker}. Skipping...")
            return (ticker, None)
        
        try:
            # Extract metrics
            metrics_dict = extract_metrics_from_financial_data(financial_data)
            
            # Extra validation to ensure critical metrics exist
            if not metrics_dict.get('revenue') or not metrics_dict.get('total_assets'):
                logger.warning(f"Missing critical metrics for {ticker}. Attempting to recover...")
                # Try to fill in missing metrics with sensible defaults
                for key, default in [
                    ('revenue', 1.0), ('gross_profit_margin', 0.5), ('operating_income_margin', 0.1),
                    ('net_income_margin', 0.05), ('total_assets', 1.0), ('total_liabilities', 0.5),
                    ('current_ratio', 1.0), ('debt_to_assets', 0.5), ('pe_ratio', 15.0)
                ]:
                    if key not in metrics_dict or metrics_dict[key] is None or not np.isfinite(metrics_dict[key]):
                        metrics_dict[key] = default
                        logger.debug(f"Added default value for {key} in {ticker}")
            
            # Cache the processed metrics
            with metrics_cache_lock:
                metrics_cache[ticker] = metrics_dict
                if len(metrics_cache) > CACHE_SIZE:
                    oldest_key = next(iter(metrics_cache))
                    del metrics_cache[oldest_key]
            
            # Also cache to disk
            try:
                with open(cache_path, 'w') as f:
                    json.dump(metrics_dict, f)
            except Exception as e:
                logger.warning(f"Error writing cache for {ticker}: {e}")
            
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

def process_ticker_sync(ticker: str):
    """
    Synchronous wrapper for process_ticker_async with better event loop handling.
    """
    try:
        # Use a new event loop for each ticker to avoid "attached to a different loop" errors
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(process_ticker_async(ticker))
        finally:
            loop.close()
        return result
    except Exception as e:
        logger.error(f"Error in process_ticker_sync for {ticker}: {e}")
        return (ticker, None)

async def screen_stocks_async(tickers, max_concurrent: int = None, progress_bar=True):
    """
    Asynchronously screen stocks with improved error handling and progress reporting.
    """
    if max_concurrent is None:
        max_concurrent = min(32, (os.cpu_count() * 2) if os.cpu_count() else 4)
    
    # Set seeds for reproducibility
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    
    results = []
    all_metrics = {}
    valid_tickers = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(ticker):
        async with semaphore:
            return await process_ticker_async(ticker)
    
    # Process tickers in batches to improve progress tracking
    batch_size = min(32, len(tickers))
    num_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(tickers))
        batch_tickers = tickers[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_tickers)} tickers)")
        
        tasks = [process_with_semaphore(ticker) for ticker in batch_tickers]
        
        if progress_bar:
            # Create a progress bar for this batch
            pbar = tqdm(total=len(batch_tickers), desc="Processing stocks")
        
        for task in asyncio.as_completed(tasks):
            ticker, metrics_dict = await task
            if progress_bar:
                pbar.update(1)
            
            if metrics_dict is not None:
                all_metrics[ticker] = metrics_dict
                valid_tickers.append(ticker)
        
        if progress_bar:
            pbar.close()
    
    if not valid_tickers:
        logger.warning("No valid tickers could be processed.")
        return []
    
    # Preprocess metrics and calculate scores
    try:
        preprocessed_metrics = preprocess_data(all_metrics)
        z_scores = calculate_z_scores(preprocessed_metrics)
        
        for ticker in valid_tickers:
            ticker_z_scores = z_scores[ticker]
            profitability_score = calculate_weighted_score(ticker_z_scores, PROFITABILITY_WEIGHTS)
            growth_score = calculate_weighted_score(ticker_z_scores, GROWTH_WEIGHTS)
            financial_health_score = calculate_weighted_score(ticker_z_scores, FINANCIAL_HEALTH_WEIGHTS)
            valuation_score = calculate_weighted_score(ticker_z_scores, VALUATION_WEIGHTS)
            efficiency_score = calculate_weighted_score(ticker_z_scores, EFFICIENCY_WEIGHTS)
            analyst_estimates_score = calculate_weighted_score(ticker_z_scores, ANALYST_ESTIMATES_WEIGHTS)
            
            composite_score = (
                profitability_score * 0.20 +
                growth_score * 0.20 +
                financial_health_score * 0.20 +
                valuation_score * 0.15 +
                efficiency_score * 0.10 +
                analyst_estimates_score * 0.15
            )
            
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
    except Exception as e:
        logger.error(f"Error calculating scores: {e}")
        logger.error(traceback.format_exc())
        # Return partial results with raw metrics
        for ticker in valid_tickers:
            results.append((ticker, 0, {
                'raw_metrics': all_metrics[ticker],
                'preprocessed_metrics': {},
                'z_scores': {},
                'category_scores': {
                    'profitability': 0, 'growth': 0, 'financial_health': 0,
                    'valuation': 0, 'efficiency': 0, 'analyst_estimates': 0
                },
                'composite_score': 0
            }))
    
    # Sort results by composite score
    results.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Successfully screened {len(results)} stocks.")
    return results

def screen_stocks(tickers, max_workers: int = None, progress_bar=True):
    """
    Synchronously screen stocks with better thread management and error handling.
    """
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() * 4) if os.cpu_count() else 4)
    
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    
    all_metrics = {}
    valid_tickers = []
    
    # Process in batches
    batch_size = min(100, len(tickers))
    num_batches = (len(tickers) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(tickers))
        batch_tickers = tickers[start_idx:end_idx]
        
        logger.info(f"Processing batch {batch_idx+1}/{num_batches} ({len(batch_tickers)} tickers)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_ticker_sync, ticker): ticker for ticker in batch_tickers}
            
            if progress_bar:
                futures_iter = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing stocks")
            else:
                futures_iter = concurrent.futures.as_completed(futures)
            
            for future in futures_iter:
                try:
                    ticker, metrics = future.result()
                    if metrics is not None:
                        all_metrics[ticker] = metrics
                        valid_tickers.append(ticker)
                except Exception as e:
                    logger.error(f"Error processing ticker: {e}")
    
    if not valid_tickers:
        logger.warning("No valid tickers could be processed.")
        return []
    
    try:
        # Preprocess metrics and calculate scores
        preprocessed_metrics = preprocess_data(all_metrics)
        z_scores = calculate_z_scores(preprocessed_metrics)
        
        results = []
        for ticker in valid_tickers:
            ticker_z_scores = z_scores[ticker]
            category_scores = {
                'profitability': calculate_weighted_score(ticker_z_scores, PROFITABILITY_WEIGHTS),
                'growth': calculate_weighted_score(ticker_z_scores, GROWTH_WEIGHTS),
                'financial_health': calculate_weighted_score(ticker_z_scores, FINANCIAL_HEALTH_WEIGHTS),
                'valuation': calculate_weighted_score(ticker_z_scores, VALUATION_WEIGHTS),
                'efficiency': calculate_weighted_score(ticker_z_scores, EFFICIENCY_WEIGHTS),
                'analyst_estimates': calculate_weighted_score(ticker_z_scores, ANALYST_ESTIMATES_WEIGHTS)
            }
            composite_score = sum(score * weight for (score, weight) in zip(
                category_scores.values(), [0.20, 0.20, 0.20, 0.15, 0.10, 0.15]))
            
            detailed_results = {
                'raw_metrics': all_metrics[ticker],
                'preprocessed_metrics': preprocessed_metrics[ticker],
                'z_scores': ticker_z_scores,
                'category_scores': category_scores,
                'composite_score': composite_score
            }
            results.append((ticker, composite_score, detailed_results))
    except Exception as e:
        logger.error(f"Error calculating scores: {e}")
        logger.error(traceback.format_exc())
        # Return partial results with raw metrics
        results = []
        for ticker in valid_tickers:
            results.append((ticker, 0, {
                'raw_metrics': all_metrics[ticker],
                'preprocessed_metrics': {},
                'z_scores': {},
                'category_scores': {
                    'profitability': 0, 'growth': 0, 'financial_health': 0,
                    'valuation': 0, 'efficiency': 0, 'analyst_estimates': 0
                },
                'composite_score': 0
            }))
    
    # Sort results by composite score
    results.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Successfully screened {len(results)} stocks.")
    return results

def get_metric_contributions(ticker: str) -> pd.DataFrame:
    """
    Get detailed breakdown of metric contributions to the composite score.
    """
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    results = screen_stocks([ticker], max_workers=1)
    if not results:
        logger.warning(f"No data found for {ticker}")
        return pd.DataFrame()
    _, _, detailed_results = results[0]
    z_scores = detailed_results['z_scores']
    raw_metrics = detailed_results['raw_metrics']
    data = []
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
    df = pd.DataFrame(data)
    df['Abs Contribution'] = df['Contribution'].abs()
    df = df.sort_values('Abs Contribution', ascending=False)
    df = df.drop('Abs Contribution', axis=1)
    return df

def get_estimate_accuracy_report(ticker: str) -> pd.DataFrame:
    """
    Generate a report on analyst estimate accuracy and forward projections for a stock.
    """
    from src.screener.fundamentals.fundamentals_core import get_openbb_client
    from src.screener.fundamentals.fundamentals_metrics import calculate_estimate_accuracy, get_attribute_value
    
    obb_client = get_openbb_client()
    financial_data = {}
    try:
        # Try multiple providers for historical estimates
        providers = ['fmp', 'intrinio', 'yfinance']
        for provider in providers:
            try:
                historical_estimates_response = obb_client.equity.estimates.historical(
                    symbol=ticker, provider=provider
                )
                if historical_estimates_response and hasattr(historical_estimates_response, 'results'):
                    financial_data['historical_estimates'] = historical_estimates_response.results
                    break
            except Exception as e:
                logger.warning(f"Error fetching historical estimates for {ticker} with {provider}: {e}")
        
        # Try multiple providers for earnings
        for provider in providers:
            try:
                if hasattr(obb_client.equity.fundamental, 'earnings'):
                    earnings_response = obb_client.equity.fundamental.earnings(
                        symbol=ticker, provider=provider
                    )
                    if earnings_response and hasattr(earnings_response, 'results'):
                        financial_data['earnings'] = earnings_response.results
                        break
            except Exception as e:
                logger.warning(f"Error fetching earnings for {ticker} with {provider}: {e}")
        
        # If earnings not available, try to get income data to construct earnings
        if 'earnings' not in financial_data:
            try:
                income_response = obb_client.equity.fundamental.income(
                    symbol=ticker, period='annual', limit=5, provider='fmp'
                )
                if income_response and hasattr(income_response, 'results'):
                    financial_data['earnings'] = construct_earnings_from_income(income_response.results)
            except Exception as e:
                logger.warning(f"Error constructing earnings for {ticker}: {e}")
        
        # Try multiple providers for forward estimates
        for data_type, method_name in [
            ('forward_sales', 'forward_sales'),
            ('forward_ebitda', 'forward_ebitda')
        ]:
            for provider in providers:
                try:
                    method = getattr(obb_client.equity.estimates, method_name)
                    if data_type == 'forward_ebitda':
                        response = method(symbol=ticker, fiscal_period='annual', provider=provider)
                    else:
                        response = method(symbol=ticker, provider=provider)
                    
                    if response and hasattr(response, 'results'):
                        financial_data[data_type] = response.results
                        break
                except Exception as e:
                    logger.warning(f"Error fetching {data_type} for {ticker} with {provider}: {e}")
    
    except Exception as e:
        logger.error(f"Error fetching estimate data for {ticker}: {e}")
        return pd.DataFrame()
    
    if not financial_data.get('historical_estimates') or not financial_data.get('earnings'):
        logger.warning(f"No historical estimates or earnings data for {ticker}")
        return pd.DataFrame()
    
    try:
        accuracy_metrics = calculate_estimate_accuracy(
            financial_data['historical_estimates'],
            financial_data['earnings']
        )
        
        forward_metrics = {}
        for data_type in ['forward_sales', 'forward_ebitda']:
            if data_type in financial_data and financial_data[data_type]:
                future_estimates = [est for est in financial_data[data_type] if 
                                   hasattr(est, 'fiscal_year') and 
                                   est.fiscal_year > datetime.now().year]
                if future_estimates:
                    future_estimates.sort(key=lambda x: x.fiscal_year)
                    for i, estimate in enumerate(future_estimates[:3]):
                        year = estimate.fiscal_year
                        mean = get_attribute_value(estimate, 'mean')
                        low = get_attribute_value(estimate, 'low_estimate')
                        high = get_attribute_value(estimate, 'high_estimate')
                        std_dev = get_attribute_value(estimate, 'standard_deviation')
                        analysts = get_attribute_value(estimate, 'number_of_analysts')
                        
                        for field, value in [
                            (f'{data_type}_{year}', mean),
                            (f'{data_type}_low_{year}', low),
                            (f'{data_type}_high_{year}', high),
                            (f'{data_type}_std_dev_{year}', std_dev),
                            (f'{data_type}_analysts_{year}', analysts)
                        ]:
                            if value is not None and np.isfinite(value):
                                forward_metrics[field] = value
        
        all_metrics = {**accuracy_metrics, **forward_metrics}
        data = []
        for metric, value in all_metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                metric_type = 'Accuracy' if 'accuracy' in metric else 'Forward Estimate'
                data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': round(value, 4),
                    'Type': metric_type
                })
        
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error processing estimate data for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()