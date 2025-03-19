from openbb import obb
import statistics
from scipy.stats import skew, kurtosis

def get_combined_historical_volatility(data, lower_q=0.25, upper_q=0.75, trading_periods=252, is_crypto=False, index="date"):
    """
    Calculate the realized volatility cones using all available volatility models,
    combine the results by window size, and compute summary statistics including
    average, standard deviation bounds, skew, and kurtosis for each volatility metric.
    
    The returned dictionary is keyed by the window size. For each window, the dictionary
    contains the following keys from all volatility models:
        - 'realized'
        - 'min'
        - 'lower_25%'
        - 'median'
        - 'upper_75%'
        - 'max'
    
    For each metric above, additional keys are added:
        - 'avg_<metric>': the average of the array.
        - 'min<metric>': the lower bound (average minus the standard deviation).
        - 'max<metric>': the upper bound (average plus the standard deviation).
        - 'skew_<metric>': the skew of the array.
        - 'kurtosis_<metric>': the kurtosis of the array.
    
    Additionally, this function computes a "timeframe indicator" based on the realized volatility,
    its skew, and its kurtosis. In this example the score is defined as:
    
        score = avg_realized + abs(skew_realized) + abs(kurtosis_realized)
    
    The best timeframe is the one with the lowest score.
    
    Parameters:
        data (list[dict]): Price data to use for the calculation.
        lower_q (float): Lower quantile value for calculations.
        upper_q (float): Upper quantile value for calculations.
        trading_periods (int): Number of trading periods in a year (default: 252).
        is_crypto (bool): Whether the data is crypto (True uses 365 days instead of 252).
        index (str): The index column name to use from the data (default: "date").
    
    Returns:
        tuple: (combined_vols, best_timeframe)
            - combined_vols: dict with window keys and raw/statistical values.
            - best_timeframe: dict with the selected window, its score, and details.

    Example usage:
        ```
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
        ```

    """
    models = [
        "std", 
        "parkinson", 
        "garman_klass", 
        "hodges_tompkins", 
        "rogers_satchell", 
        "yang_zhang"
    ]
    combined_vols = {}
    
    # Collect arrays from each volatility model for each window.
    for model in models:
        cones_data = obb.technical.cones(
            data=data,
            lower_q=lower_q,
            upper_q=upper_q,
            model=model,
            trading_periods=trading_periods,
            is_crypto=is_crypto,
            index=index
        )
        for cone in cones_data.results:
            try:
                window = cone.window
            except AttributeError:
                print("Cone object does not have attribute 'window':", cone)
                continue

            if window not in combined_vols:
                combined_vols[window] = {
                    "realized": [],
                    "min": [],
                    "lower_25%": [],
                    "median": [],
                    "upper_75%": [],
                    "max": []
                }
            combined_vols[window]["realized"].append(getattr(cone, "realized", None))
            combined_vols[window]["min"].append(getattr(cone, "min", None))
            combined_vols[window]["lower_25%"].append(getattr(cone, "lower_25%", None))
            combined_vols[window]["median"].append(getattr(cone, "median", None))
            combined_vols[window]["upper_75%"].append(getattr(cone, "upper_75%", None))
            combined_vols[window]["max"].append(getattr(cone, "max", None))
    
    # Define the metrics for which to compute statistics.
    metrics = ["realized", "min", "lower_25%", "median", "upper_75%", "max"]
    
    # For each window and each metric, compute average, bounds, skew, and kurtosis.
    for window, values_dict in combined_vols.items():
        for metric in metrics:
            # Filter out any None values.
            arr = [v for v in values_dict[metric] if v is not None]
            if not arr:
                continue
            avg_val = statistics.mean(arr)
            std_val = statistics.stdev(arr) if len(arr) > 1 else 0
            values_dict[f"avg_{metric}"] = avg_val
            values_dict[f"min{metric}"] = avg_val - std_val
            values_dict[f"max{metric}"] = avg_val + std_val
            values_dict[f"skew_{metric}"] = skew(arr)
            values_dict[f"kurtosis_{metric}"] = kurtosis(arr)
    
    # Compute the best timeframe indicator.
    # Here we use the 'realized' metric as the key measure.
    best_window = None
    best_score = None
    for window, values in combined_vols.items():
        if "avg_realized" in values and "skew_realized" in values and "kurtosis_realized" in values:
            # Define a simple scoring function: lower is better.
            score = values["avg_realized"] + abs(values["skew_realized"]) + abs(values["kurtosis_realized"])
            if best_score is None or score < best_score:
                best_score = score
                best_window = window
    
    best_timeframe = {
        "window": best_window,
        "score": best_score,
        "details": combined_vols.get(best_window, {})
    }
    
    return combined_vols, best_timeframe