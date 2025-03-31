# tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'AVGO', 'TSLA', 'LLY', 'TSM']
# tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN']
# tickers = [ 'BX', 'KKR', 'APO', 'CG', 'TPG', 'ARES', 'EQT', 'PGHN.SW', 'BAM', 'ARCC', 'OBDC', 'BXSL', 'FSK', 'MAIN', 'GBDC', 'HTGC', 'TSLX', 'PSEC', 'GSBD', 'OCSL', 'MFIC', 'NMFC', 'KBDC', 'CSWC', 'BBDC', 'TRIN', 'PFLT', 'SLR', 'CGBDC', 'MSIF', 'FDUS', 'CCAP', 'TCPC', 'GLAD', 'CION', 'GAIN', 'PNNT', 'RWAY', 'SCM', 'HRZN', 'TPVG', 'WHF', 'OXSQ', 'MRCC', 'PTMN', 'SSSS', 'OFS', 'GECC', 'PFX', 'LRFC', 'RAND', 'ICMB', 'EQS', 'NVDA', 'MSFT', 'ARM', 'AMD' ];
# tickers = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'LLY', 'TSM', 'WMT', 'JPM', 'V', 'MA', 'UNH', 'ORCL', 'NFLX', 'COST', 'JNJ', 'PG', 'ABBV', 'HD']
tickers = [
   # Technology
   "NVDA", "MSFT", "ADBE", "ASML", "TSM",
   
   # Energy Transition and Infrastructure
   "NEE", "ENPH", "PCG", "BLK", "EQIX",
   
   # Healthcare
   "UNH", "LLY", "ISRG", "VEEV", "DHR",
   
   # Financial Services
   "JPM", "MA", "COIN", "BAC", "SCHW",
   
   # Manufacturing & Industrial
   "CAT", "DE", "HON", "EMR", "GE",
   
   # Consumer & Retail
   "AMZN", "COST", "WMT", "HD", "TGT",
   
   # Materials & Commodities
   "FCX", "NUE", "APD", "ALB", "BHP",
   
   # Real Estate
   "AMT", "PLD", "DLR", "CBRE", "PSA",
   
   # Defense & Aerospace
   "LMT", "RTX", "NOC", "PLTR", "LDOS",
   
   # Transportation & Logistics
   "UPS", "CSX", "JBHT", "UAL", "ODFL"
]
from openbb import obb

def get_active_tickers():
    """
    Fetch tickers from multiple OpenBB equity discovery endpoints:
      - growth_tech
      - undervalued_large_caps
      - aggressive_small_caps
      - undervalued_growth
      - gainers
      - losers
      - top_retail
      - active

    Each endpoint returns an object (OBBject) with a .results attribute.
    Each result is assumed to have a 'symbol' key (or attribute) representing the ticker.

    Returns:
        list: A unique list of ticker symbols.
    """
    # Call the API endpoints with example parameters (adjust as needed)
    endpoints = [
        obb.equity.discovery.growth_tech(provider='yfinance', sort='desc', limit=10),
        obb.equity.discovery.undervalued_large_caps(provider='yfinance', sort='desc', limit=10),
        obb.equity.discovery.aggressive_small_caps(provider='yfinance', sort='desc'),
        obb.equity.discovery.undervalued_growth(provider='yfinance', sort='desc', limit=10),
        obb.equity.discovery.gainers(provider='yfinance', sort='desc', limit=10),
        obb.equity.discovery.losers(provider='yfinance', sort='desc', limit=10),
        obb.equity.discovery.active(provider='yfinance', sort='desc', limit=10)
    ]
    
    # Use a set to avoid duplicate tickers.
    tickers_set = set()
    
    # Helper to extract tickers from each result set.
    for endpoint in endpoints:
        for item in endpoint.results:
            # Depending on the type of each result, try to get the symbol.
            ticker = None
            if isinstance(item, dict):
                ticker = item.get('symbol')
            else:
                ticker = getattr(item, 'symbol', None)
            if ticker:
                tickers_set.add(ticker)
    
    # Convert to a list; sorting if a consistent order is needed.
    tickers = sorted(list(tickers_set))
    return tickers


# Example usage:
if __name__ == "__main__":
    tickers = get_active_tickers()
    print(tickers)
