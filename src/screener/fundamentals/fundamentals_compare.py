import asyncio
import pandas as pd
from tqdm import tqdm
import numpy as np
import logging
from src.screener.fundamentals.fundamentals_screen import process_ticker_async, process_ticker_sync, screen_stocks_async, screen_stocks

logger = logging.getLogger(__name__)

def run_stock_screen(tickers: list, use_async: bool = True) -> pd.DataFrame:
    """
    Run a complete stock screen and return a DataFrame with rankings and scores.
    """
    import numpy as np
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    if use_async:
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(screen_stocks_async(tickers))
    else:
        results = screen_stocks(tickers)
    if not results:
        return pd.DataFrame()
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

def compare_stocks(tickers: list, use_async: bool = True) -> pd.DataFrame:
    """
    Compare a list of stocks on key metrics.
    """
    import numpy as np
    np.random.seed(42)
    import torch
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
    key_metrics = [
        'gross_profit_margin', 'operating_income_margin', 'net_income_margin', 'return_on_equity',
        'growth_revenue', 'growth_net_income', 'growth_eps',
        'current_ratio', 'debt_to_equity', 'interest_coverage',
        'pe_ratio', 'price_to_book', 'ev_to_ebitda', 'dividend_yield',
        'asset_turnover', 'inventory_turnover',
        'forward_sales_growth', 'estimate_revision_momentum'
    ]
    comparison_data = {}
    for ticker, metrics in all_metrics.items():
        ticker_data = {}
        for metric in key_metrics:
            if metric in metrics and metrics[metric] is not None and np.isfinite(metrics[metric]):
                ticker_data[metric] = metrics[metric]
            else:
                ticker_data[metric] = np.nan
        comparison_data[ticker] = ticker_data
    df = pd.DataFrame(comparison_data).T
    df.columns = [col.replace('_', ' ').title() for col in df.columns]
    for col in df.columns:
        df[col] = df[col].round(4)
    return df
