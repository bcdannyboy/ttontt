import asyncio
import logging
import numpy as np
import traceback
from src.screener.fundamentals.fundamentals_core import get_openbb_client, rate_limited_api_call

logger = logging.getLogger(__name__)

async def get_peers_async(ticker: str):
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

async def analyze_ticker_with_peers(ticker: str, depth: int = 1, visited=None):
    """
    Analyze a ticker and its peers.
    Returns peer comparison data for the ticker.
    """
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    if visited is None:
        visited = set()
    if ticker in visited:
        return {'ticker': ticker, 'peer_comparison': {}, 'peers': []}
    visited.add(ticker)
    from fundamentals_screen import screen_stocks_async
    ticker_results = await screen_stocks_async([ticker])
    if not ticker_results:
        return {'ticker': ticker, 'peer_comparison': {}, 'peers': []}
    ticker_score = ticker_results[0][1]
    peers_list = await get_peers_async(ticker)
    if not peers_list:
        return {
            'ticker': ticker,
            'peer_comparison': {
                'average_score': ticker_score,
                'std_dev': 0.0,
                'count': 0,
                'percentile': 50.0
            },
            'peers': []
        }
    filtered_peers = [peer for peer in peers_list if peer not in visited]
    peer_scores = []
    peer_data = []
    if filtered_peers:
        peers_results = await screen_stocks_async(filtered_peers)
        peers_dict = {res[0]: res[1] for res in peers_results}
        for peer in filtered_peers:
            if peer in peers_dict:
                peer_score = peers_dict[peer]
                peer_scores.append(peer_score)
                peer_data.append({'ticker': peer, 'score': peer_score})
    if peer_scores:
        peer_scores.append(ticker_score)
        all_scores = np.array(peer_scores)
        average_score = np.mean(all_scores)
        std_dev = np.std(all_scores)
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
            'count': len(peer_scores) - 1,
            'percentile': percentile
        },
        'peers': peer_data
    }

async def gather_peer_analysis(tickers: list):
    """
    Gather peer analysis for a list of tickers.
    """
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    
    visited = set()
    peer_analysis = {}
    
    # Process tickers in smaller batches to avoid overwhelming the event loop
    batch_size = 10
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_tasks = []
        
        for ticker in batch:
            if ticker in visited:
                continue
            batch_tasks.append(analyze_ticker_with_peers(ticker, depth=1, visited=visited))
        
        # Process this batch in parallel if there are tasks
        if batch_tasks:
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            batch_idx = 0
            for j, ticker in enumerate(batch):
                if ticker in visited and ticker not in peer_analysis:
                    continue
                
                if batch_idx >= len(batch_results):
                    continue  # Skip if no result for this ticker
                    
                result = batch_results[batch_idx]
                batch_idx += 1
                
                if isinstance(result, Exception):
                    logger.error(f"Error in peer analysis for {ticker}: {result}")
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
                    peer_analysis[ticker] = result
        
        # Small delay between batches to prevent rate limiting
        if i + batch_size < len(tickers):
            await asyncio.sleep(0.5)
    
    return peer_analysis
