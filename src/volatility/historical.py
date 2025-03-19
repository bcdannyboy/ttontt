from openbb import obb

def get_combined_historical_volatility(data, lower_q=0.25, upper_q=0.75, trading_periods=252, is_crypto=False, index="date"):
    """
    Calculate the realized volatility cones using all available volatility models
    and combine the results by window size. The returned dictionary is keyed by the 
    window size, and for each window the values are a dictionary where each key 
    (e.g., 'realized', 'min', 'lower_25%', 'median', 'upper_75%', 'max') maps to 
    an array of values from each volatility model.
    
    Parameters:
        data (list[dict]): Price data to use for the calculation.
        lower_q (float): Lower quantile value for calculations.
        upper_q (float): Upper quantile value for calculations.
        trading_periods (int): Number of trading periods in a year (default: 252).
        is_crypto (bool): Whether the data is crypto (True uses 365 days instead of 252).
        index (str): The index column name to use from the data (default: "date").
    
    Returns:
        dict: A dictionary where each key is a window size (time frame) and the 
              corresponding value is a dictionary with keys 'realized', 'min', 
              'lower_25%', 'median', 'upper_75%', and 'max'. Each key maps to a list
              of values aggregated from all volatility models.
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
        # Iterate over each Data object in the results.
        for cone in cones_data.results:
            try:
                # Access attributes directly.
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
            
    return combined_vols
