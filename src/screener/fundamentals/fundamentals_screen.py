import os
import asyncio
import concurrent.futures
import time
from tqdm import tqdm
import numpy as np
import logging
import traceback
import pandas as pd
from src.screener.fundamentals.fundamentals_core import metrics_cache, metrics_cache_lock, CACHE_SIZE
from src.screener.fundamentals.fundamentals_data import get_financial_data_async
from src.screener.fundamentals.fundamentals_metrics import (extract_metrics_from_financial_data, preprocess_data,
                                    calculate_z_scores, calculate_weighted_score, construct_earnings_from_income,
                                    get_attribute_value, PROFITABILITY_WEIGHTS, GROWTH_WEIGHTS,
                                    FINANCIAL_HEALTH_WEIGHTS, VALUATION_WEIGHTS, EFFICIENCY_WEIGHTS, ANALYST_ESTIMATES_WEIGHTS)

logger = logging.getLogger(__name__)

async def process_ticker_async(ticker: str):
    """
    Process a single ticker asynchronously with improved error handling.
    """
    try:
        with metrics_cache_lock:
            if ticker in metrics_cache:
                return (ticker, metrics_cache[ticker])
        
        ticker, financial_data = await get_financial_data_async(ticker)
        if not financial_data.get('income') or not financial_data.get('balance'):
            logger.warning(f"Insufficient financial data for {ticker}. Skipping...")
            return (ticker, None)
        try:
            metrics_dict = extract_metrics_from_financial_data(financial_data)
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

def process_ticker_sync(ticker: str):
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

async def screen_stocks_async(tickers, max_concurrent: int = None):
    """
    Asynchronously screen stocks.
    """
    if max_concurrent is None:
        max_concurrent = (os.cpu_count() * 2) if os.cpu_count() else 4
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
    
    tasks = [process_with_semaphore(ticker) for ticker in tickers]
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        result = await task
        ticker, metrics_dict = result
        progress = i / len(tickers) * 100
        if i % 5 == 0 or i == 1 or i == len(tickers):
            print(f"Processing stocks: {progress:.0f}% completed ({i}/{len(tickers)})", end='\r')
        if metrics_dict is not None:
            all_metrics[ticker] = metrics_dict
            valid_tickers.append(ticker)
    print()
    if not valid_tickers:
        logger.warning("No valid tickers could be processed.")
        return []
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
    results.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"Successfully screened {len(results)} stocks.")
    return results

def screen_stocks(tickers, max_workers: int = None):
    """
    Synchronously screen stocks.
    """
    if max_workers is None:
        max_workers = min(32, (os.cpu_count() * 4) if os.cpu_count() else 4)
    all_metrics = {}
    valid_tickers = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for ticker, metrics in tqdm(executor.map(process_ticker_sync, tickers), total=len(tickers), desc="Processing stocks"):
            if metrics is not None:
                all_metrics[ticker] = metrics
                valid_tickers.append(ticker)
    if not valid_tickers:
        logger.warning("No valid tickers could be processed.")
        return []
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
        composite_score = sum(score * weight for (score, weight) in zip(category_scores.values(), [0.20, 0.20, 0.20, 0.15, 0.10, 0.15]))
        detailed_results = {
            'raw_metrics': all_metrics[ticker],
            'preprocessed_metrics': preprocessed_metrics[ticker],
            'z_scores': ticker_z_scores,
            'category_scores': category_scores,
            'composite_score': composite_score
        }
        results.append((ticker, composite_score, detailed_results))
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
    from fundamentals_core import get_openbb_client
    obb_client = get_openbb_client()
    financial_data = {}
    try:
        historical_estimates_response = obb_client.equity.estimates.historical(
            symbol=ticker, provider='fmp'
        )
        financial_data['historical_estimates'] = historical_estimates_response.results
        earnings_response = obb_client.equity.fundamental.earnings(
            symbol=ticker, provider='fmp'
        )
        financial_data['earnings'] = earnings_response.results
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
    from fundamentals_metrics import calculate_estimate_accuracy, get_attribute_value
    accuracy_metrics = calculate_estimate_accuracy(
        financial_data['historical_estimates'],
        financial_data['earnings']
    )
    forward_metrics = {}
    if financial_data.get('forward_sales'):
        future_estimates = [est for est in financial_data['forward_sales'] if hasattr(est, 'fiscal_year') and est.fiscal_year > datetime.now().year]
        if future_estimates:
            future_estimates.sort(key=lambda x: x.fiscal_year)
            for i, estimate in enumerate(future_estimates[:3]):
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
        future_estimates = [est for est in financial_data['forward_ebitda'] if hasattr(est, 'fiscal_year') and est.fiscal_year > datetime.now().year]
        if future_estimates:
            future_estimates.sort(key=lambda x: x.fiscal_year)
            for i, estimate in enumerate(future_estimates[:3]):
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
