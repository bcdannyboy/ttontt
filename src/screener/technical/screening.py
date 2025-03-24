import asyncio
import os
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from .data_fetch import get_technical_data_async
from .indicator_extraction import extract_indicators_from_technical_data, normalize_data, calculate_z_scores
from .scoring import calculate_weighted_score

logger = logging.getLogger(__name__)

# Simple in-memory cache for indicators.
indicators_cache = {}

async def process_ticker_async(ticker: str):
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
                if len(indicators_cache) > 1000:
                    oldest_key = next(iter(indicators_cache))
                    del indicators_cache[oldest_key]
            logger.debug(f"Successfully processed {ticker}")
            return (ticker, indicators_dict)
        except Exception as e:
            logger.error(f"Error extracting indicators for {ticker}: {e}")
            return (ticker, None)
    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}")
        return (ticker, None)

async def screen_stocks_async(tickers, max_concurrent: int = os.cpu_count()*2):
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
        from .constants import TREND_WEIGHTS, MOMENTUM_WEIGHTS, VOLATILITY_WEIGHTS, VOLUME_WEIGHTS, CATEGORY_WEIGHTS
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

def process_ticker_sync(ticker: str):
    try:
        result = asyncio.run(process_ticker_async(ticker))
        return result
    except Exception as e:
        logger.error(f"Error in process_ticker_sync for {ticker}: {e}")
        return (ticker, None)

def screen_stocks(tickers):
    return asyncio.run(screen_stocks_async(tickers, max_concurrent=os.cpu_count()*2))
