from openbb import obb

from src.volatility.historical import get_combined_historical_volatility

if __name__ == "__main__":
    ascii_art = r"""
  _____     _____     ___      _   _    _____     _____ 
 |_   _|   |_   _|   / _ \    | \ | |  |_   _|   |_   _|
   | |       | |    | | | |   |  \| |    | |       | |  
   | |       | |    | |_| |   | |\  |    | |       | |  
   |_|       |_|     \___/    |_| \_|    |_|       |_|  
 ═══════════════════════════════════════════════════════
 To Trade                  or               Not to Trade 
"""
    print(ascii_art)

    # Example usage:
    stock_data = obb.equity.price.historical(symbol='TSLA', start_date='2023-01-01', provider='yfinance')
    combined_volatility, best_tf = get_combined_historical_volatility(stock_data.results)
    
    # Print the combined volatility dictionary by window size.
    for window, values in combined_volatility.items():
        print(f"Window: {window}")
        for key, value in values.items():
            print(f"  {key}: {value}")
    
    # Print the best timeframe indicator.
    print("\nBest timeframe indicator for trades:")
    print(f"Window: {best_tf['window']}")
    print(f"Score: {best_tf['score']}")
    print("Details:")
    for key, value in best_tf["details"].items():
        print(f"  {key}: {value}")
