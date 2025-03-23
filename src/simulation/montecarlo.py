"""
Monte Carlo Stock Price Simulator
================================

A comprehensive stochastic price simulation tool that evaluates future price movements
based on historical volatility and generates probabilistic price forecasts.

Overall Approach:
----------------
1. Historical Data Analysis: Analyzes historical price and volatility data to calibrate
   the simulation parameters, including drift, volatility, and mean reversion tendencies.

2. Brownian Motion Simulation: Uses geometric Brownian motion as the foundation for 
   simulating future price paths with appropriate random noise components.

3. Heston Model Integration: Incorporates the Heston stochastic volatility model to account
   for volatility clustering and mean reversion in volatility.

4. Multi-threaded Execution: Leverages parallel processing through multi-threading and
   GPU acceleration where available to run thousands of simulation paths efficiently.

5. Statistical Analysis: Aggregates simulation results to compute expected prices,
   confidence intervals, and probability distributions for various time horizons.

6. Risk Assessment: Provides probabilistic assessment of potential upside and downside
   scenarios across different time frames.

Statistical Methods:
-------------------
- Geometric Brownian Motion for price path simulation
- Heston stochastic volatility model
- Monte Carlo simulation with multiple paths
- Statistical analysis of simulation results (mean, quantiles, etc.)
- Normal and log-normal distribution analysis
- Volatility calibration using historical data

Key Components:
--------------
- Historical Data Analysis: Calibrates model parameters from historical price data
- Simulation Engine: Generates thousands of potential price paths
- Statistical Analyzer: Computes statistics from simulation results
- Visualization: Presents results in tabular format for different time frames

This simulation provides a robust, data-driven method for evaluating potential
future price movements and associated probabilities.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import concurrent.futures
import threading
import traceback
import time
from tqdm import tqdm
import asyncio
import math
from scipy.stats import norm, lognorm, skew, kurtosis
from openbb import obb

# Import PyTorch to leverage MPS on Apple Silicon
import torch

from src.volatility.historical import get_combined_historical_volatility

# Set device to mps if available; otherwise use cpu
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    logger_device = "MPS"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger_device = "CUDA"
else:
    device = torch.device("cpu")
    logger_device = "CPU"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.info(f"Using device: {logger_device}")

# Thread-local storage for OpenBB API session
thread_local = threading.local()

# Helper functions for statistics
def calculate_statistics(data: np.ndarray) -> Dict[str, Any]:
    """
    Calculate descriptive statistics for the simulation results.
    
    Args:
        data: Array of simulation results
        
    Returns:
        Dictionary containing statistics
    """
    stats = {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q1": float(np.percentile(data, 25)),
        "q3": float(np.percentile(data, 75)),
        "skew": float(skew(data)),
        "kurtosis": float(kurtosis(data))
    }
    return stats

def get_openbb_client():
    """Returns a thread-local OpenBB client instance."""
    if not hasattr(thread_local, "openbb_client"):
        thread_local.openbb_client = obb
    return thread_local.openbb_client

class MonteCarloSimulator:
    """
    A class to perform Monte Carlo simulations for stock price movements
    using both Geometric Brownian Motion and the Heston stochastic volatility model,
    with support for multiple volatility models.
    """
    
    def __init__(self, ticker: str, num_simulations: int = 1000, 
                 time_horizons: List[int] = None,
                 use_heston: bool = True, random_seed: Optional[int] = None,
                 use_multiple_volatility_models: bool = True):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            ticker: Stock ticker symbol
            num_simulations: Number of simulation paths to generate
            time_horizons: List of time horizons (in trading days) to simulate
            use_heston: Whether to use the Heston model for stochastic volatility
            random_seed: Random seed for reproducibility
            use_multiple_volatility_models: Whether to use multiple volatility models
        """
        self.ticker = ticker
        self.num_simulations = num_simulations
        
        # Default time horizons if not specified
        if time_horizons is None:
            self.time_horizons = [1, 5, 10, 21, 63, 126, 252]  # 1D, 1W, 2W, 1M, 3M, 6M, 1Y
        else:
            self.time_horizons = time_horizons
            
        self.use_heston = use_heston
        self.use_multiple_volatility_models = use_multiple_volatility_models
        
        # Initialize random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        # Initialize data attributes
        self.price_history = None
        self.current_price = None
        self.annualized_drift = None
        self.annualized_volatility = None
        
        # Store volatility models' results
        self.volatility_models = None
        self.best_timeframe = None
        
        # Heston model parameters (will be calibrated from data)
        self.kappa = 2.0  # Mean reversion speed of volatility
        self.theta = 0.04  # Long-term volatility
        self.sigma = 0.3  # Volatility of volatility
        self.rho = -0.7  # Correlation between asset returns and volatility
        self.v0 = None  # Initial variance (will be set from data)
        
        # Results storage
        self.simulation_results = {}
        
    async def fetch_historical_data(self, days: int = 252) -> pd.DataFrame:
        """
        Fetch historical price data for the ticker.
        
        Args:
            days: Number of historical trading days to fetch
            
        Returns:
            DataFrame containing price history
        """
        obb_client = get_openbb_client()
        
        try:
            # Calculate start date based on calendar days (roughly 1.4x trading days)
            start_date = (datetime.now() - timedelta(days=int(days * 1.4))).strftime('%Y-%m-%d')
            
            # Fetch historical price data - using synchronous call instead of await
            price_history_response = obb_client.equity.price.historical(
                symbol=self.ticker, start_date=start_date, provider='fmp'
            )
            
            if hasattr(price_history_response, 'results'):
                # Convert to pandas DataFrame for easier manipulation
                price_data = []
                for record in price_history_response.results:
                    price_data.append({
                        'date': getattr(record, 'date', None),
                        'open': getattr(record, 'open', None),
                        'high': getattr(record, 'high', None),
                        'low': getattr(record, 'low', None),
                        'close': getattr(record, 'close', None),
                        'volume': getattr(record, 'volume', None)
                    })
                
                df = pd.DataFrame(price_data)
                
                # Ensure we have the minimum required data
                if len(df) < 30:
                    logger.warning(f"Insufficient historical data for {self.ticker}. Need at least 30 days.")
                    return pd.DataFrame()
                
                # Calculate log returns for volatility estimation
                df['log_return'] = np.log(df['close'] / df['close'].shift(1))
                df = df.dropna()
                
                return df
            else:
                logger.warning(f"No results returned for {self.ticker}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {self.ticker}: {e}")
            return pd.DataFrame()

    async def calibrate_model_parameters(self) -> bool:
        """
        Calibrate model parameters based on historical data.
        
        Returns:
            True if calibration was successful, False otherwise.
        """
        try:
            # Fetch historical data (using at least 252 days or the maximum time horizon)
            self.price_history = await self.fetch_historical_data(days=max(252, max(self.time_horizons)))
            if self.price_history.empty:
                logger.warning(f"No historical data available for {self.ticker}. Cannot calibrate model.")
                return False

            # Extract current price from the last closing value
            self.current_price = float(self.price_history['close'].iloc[-1])
            
            # Calculate annualized drift using log returns (mean daily return * 252 trading days)
            daily_drift = self.price_history['log_return'].mean()
            self.annualized_drift = daily_drift * 252

            # If using multiple volatility models, attempt to combine their estimates
            if self.use_multiple_volatility_models:
                try:
                    data = []
                    # Iterate over the DataFrame rows and use the actual 'date' column value
                    for _, row in self.price_history.iterrows():
                        # Use the value in the 'date' column; if it isn’t a string, try to format it
                        date_val = row['date'] if 'date' in row else str(row.name)
                        if not isinstance(date_val, str) and hasattr(date_val, 'strftime'):
                            date_val = date_val.strftime('%Y-%m-%d')
                        else:
                            date_val = str(date_val)
                        data.append({
                            'date': date_val,
                            'close': row['close'],
                            'high': row.get('high', row['close']),
                            'low': row.get('low', row['close']),
                            'open': row.get('open', row['close']),
                            'volume': row.get('volume', 0)
                        })

                    # Call the volatility combination function using the correctly formatted data
                    self.volatility_models, self.best_timeframe = get_combined_historical_volatility(
                        data=data,
                        lower_q=0.25,
                        upper_q=0.75,
                        trading_periods=252,
                        is_crypto=False,
                        index="date"
                    )

                    # Use the best timeframe’s realized volatility if available
                    best_window = self.best_timeframe.get('window')
                    if best_window and 'avg_realized' in self.volatility_models.get(best_window, {}):
                        self.annualized_volatility = self.volatility_models[best_window]['avg_realized']
                    else:
                        # Fallback: use standard calculation (daily volatility scaled to annual)
                        daily_volatility = self.price_history['log_return'].std()
                        self.annualized_volatility = daily_volatility * np.sqrt(252)
                except Exception as e:
                    logger.error(f"Error calculating combined volatility for {self.ticker}: {e}")
                    logger.error(traceback.format_exc())
                    # Fallback to standard volatility calculation
                    daily_volatility = self.price_history['log_return'].std()
                    self.annualized_volatility = daily_volatility * np.sqrt(252)
            else:
                # Standard volatility calculation from log returns
                daily_volatility = self.price_history['log_return'].std()
                self.annualized_volatility = daily_volatility * np.sqrt(252)

            # Set the initial variance for the Heston model as the square of the annualized volatility
            self.v0 = self.annualized_volatility ** 2

            # If using the Heston model, perform a heuristic calibration of its parameters
            if self.use_heston:
                # Use a rolling window (up to 60 days or half the available days)
                vol_window = min(60, len(self.price_history) // 2)
                rolling_vol = self.price_history['log_return'].rolling(window=vol_window).std() * np.sqrt(252)

                if len(rolling_vol.dropna()) > 10:
                    vol_diff = rolling_vol.diff().dropna()
                    vol_lag = rolling_vol.shift(1).dropna()

                    if len(vol_diff) == len(vol_lag) and len(vol_diff) > 10:
                        # Simple linear regression (slope) to estimate mean reversion speed
                        vol_diff_arr = vol_diff.values[-len(vol_lag):]
                        vol_lag_arr = vol_lag.values
                        if np.var(vol_lag_arr) > 0:
                            beta = np.cov(vol_diff_arr, vol_lag_arr)[0, 1] / np.var(vol_lag_arr)
                            self.kappa = -beta * 252  # Annualize the estimated slope
                            self.kappa = max(0.1, min(10, self.kappa))
                        
                        # Estimate the long-term variance (theta) as the square of the mean rolling volatility
                        self.theta = rolling_vol.mean() ** 2
                        # Estimate volatility of volatility (sigma) using a heuristic scaling
                        self.sigma = rolling_vol.std() * 2
                        self.sigma = max(0.01, min(2, self.sigma))
                
                # Estimate correlation (rho) between returns and volatility changes
                if len(self.price_history) > 20:
                    log_returns = self.price_history['log_return'].values
                    # Use the difference of the rolling volatility values for volatility changes
                    vol_changes = np.diff(np.concatenate([[0], rolling_vol.dropna().values]))
                    min_len = min(len(log_returns), len(vol_changes))
                    if min_len > 10:
                        corr_matrix = np.corrcoef(log_returns[-min_len:], vol_changes[-min_len:])
                        if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]):
                            self.rho = corr_matrix[0, 1]
                            self.rho = max(-0.95, min(0.95, self.rho))

            logger.info(f"Model calibration for {self.ticker} completed successfully.")
            logger.info(f"Current price: {self.current_price:.2f}")
            logger.info(f"Annualized drift: {self.annualized_drift:.4f}")
            logger.info(f"Annualized volatility: {self.annualized_volatility:.4f}")
            if self.use_heston:
                logger.info(f"Heston parameters - κ: {self.kappa:.4f}, θ: {self.theta:.4f}, σ: {self.sigma:.4f}, ρ: {self.rho:.4f}, v0: {self.v0:.4f}")
            return True

        except Exception as e:
            logger.error(f"Error calibrating model for {self.ticker}: {e}")
            logger.error(traceback.format_exc())
            return False


    def _simulate_gbm(self, time_horizon: int, num_paths: int, volatility: float = None) -> np.ndarray:
        """
        Simulate stock price paths using Geometric Brownian Motion.
        
        Args:
            time_horizon: Number of days to simulate
            num_paths: Number of paths to simulate
            volatility: Optional override for volatility (annualized)
            
        Returns:
            Array of final prices for each path
        """
        # Use provided volatility if specified, otherwise use the calibrated value
        vol_to_use = volatility if volatility is not None else self.annualized_volatility
        
        # Daily parameters
        daily_drift = self.annualized_drift / 252
        daily_volatility = vol_to_use / np.sqrt(252)
        
        # Generate random standard normal samples for all paths at once
        if device.type != 'cpu':
            # Use GPU if available
            random_samples = torch.normal(0, 1, size=(num_paths, time_horizon), device=device)
            random_samples = random_samples.cpu().numpy()
        else:
            # Use CPU
            random_samples = np.random.normal(0, 1, size=(num_paths, time_horizon))
        
        # Calculate price paths (vectorized)
        daily_returns = daily_drift + daily_volatility * random_samples
        cumulative_returns = np.cumsum(daily_returns, axis=1)
        price_paths = self.current_price * np.exp(cumulative_returns)
        
        # Return final prices
        return price_paths[:, -1]
    
    def _simulate_heston(self, time_horizon: int, num_paths: int, volatility: float = None, variance: float = None) -> np.ndarray:
        """
        Simulate stock price paths using the Heston stochastic volatility model.
        
        Args:
            time_horizon: Number of days to simulate
            num_paths: Number of paths to simulate
            volatility: Optional override for volatility (annualized)
            variance: Optional override for initial variance
            
        Returns:
            Array of final prices for each path
        """
        # Use provided volatility/variance if specified, otherwise use the calibrated values
        vol_to_use = volatility if volatility is not None else self.annualized_volatility
        v0_to_use = variance if variance is not None else (vol_to_use ** 2 if volatility is not None else self.v0)
        
        # Daily parameters
        dt = 1.0 / 252  # Daily time step
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays for price and variance paths
        prices = np.zeros((num_paths, time_horizon + 1))
        variances = np.zeros((num_paths, time_horizon + 1))
        
        # Set initial values
        prices[:, 0] = self.current_price
        variances[:, 0] = v0_to_use
        
        # Generate correlated random samples
        if device.type != 'cpu':
            # Use GPU for random number generation
            z1 = torch.normal(0, 1, size=(num_paths, time_horizon), device=device)
            z2 = torch.normal(0, 1, size=(num_paths, time_horizon), device=device)
            z1 = z1.cpu().numpy()
            z2 = z2.cpu().numpy()
        else:
            # Use CPU
            z1 = np.random.normal(0, 1, size=(num_paths, time_horizon))
            z2 = np.random.normal(0, 1, size=(num_paths, time_horizon))
        
        # Apply correlation
        z2_correlated = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2
        
        # Simulate paths
        for t in range(time_horizon):
            # Ensure variance stays positive
            variances[:, t] = np.maximum(variances[:, t], 1e-8)
            
            # Update price (log process for better numerical stability)
            price_drift = (self.annualized_drift - 0.5 * variances[:, t]) * dt
            price_diffusion = np.sqrt(variances[:, t]) * sqrt_dt * z1[:, t]
            prices[:, t+1] = prices[:, t] * np.exp(price_drift + price_diffusion)
            
            # Update variance
            var_drift = self.kappa * (self.theta - variances[:, t]) * dt
            var_diffusion = self.sigma * np.sqrt(variances[:, t]) * sqrt_dt * z2_correlated[:, t]
            variances[:, t+1] = variances[:, t] + var_drift + var_diffusion
        
        # Return final prices
        return prices[:, -1]
    
    async def run_simulation(self) -> Dict[str, Any]:
        """
        Run the Monte Carlo simulation for all specified time horizons.
        
        Returns:
            Dictionary containing simulation results
        """
        # If using multiple volatility models, distribute simulations among models
        if self.use_multiple_volatility_models and self.volatility_models:
            return await self._run_simulation_with_multiple_models()
        else:
            return await self._run_simulation_with_single_model()
    
    async def _run_simulation_with_single_model(self) -> Dict[str, Any]:
        """
        Run the Monte Carlo simulation with a single volatility model.
        
        Returns:
            Dictionary containing simulation results
        """
        # Determine number of simulations based on device
        if device.type != 'cpu':
            # For GPU, we can run more simulations
            num_simulations = min(self.num_simulations, 5000)
        else:
            # For CPU, limit to a reasonable number
            num_simulations = min(self.num_simulations, 1000)
        
        # Simulate for each time horizon
        for horizon in self.time_horizons:
            logger.info(f"Running {num_simulations} simulations for {horizon} day horizon")
            
            # Choose simulation method based on configuration
            if self.use_heston:
                final_prices = self._simulate_heston(horizon, num_simulations)
            else:
                final_prices = self._simulate_gbm(horizon, num_simulations)
            
            # Calculate expected price and confidence intervals
            expected_price = np.mean(final_prices)
            price_std = np.std(final_prices)
            
            # Calculate various percentiles
            percentiles = np.percentile(final_prices, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            
            # Calculate probability of price increase
            prob_increase = np.mean(final_prices > self.current_price)
            
            # Calculate probability of significant moves
            prob_up_5pct = np.mean(final_prices > self.current_price * 1.05)
            prob_up_10pct = np.mean(final_prices > self.current_price * 1.10)
            prob_up_20pct = np.mean(final_prices > self.current_price * 1.20)
            prob_down_5pct = np.mean(final_prices < self.current_price * 0.95)
            prob_down_10pct = np.mean(final_prices < self.current_price * 0.90)
            prob_down_20pct = np.mean(final_prices < self.current_price * 0.80)
            
            # Store results
            self.simulation_results[horizon] = {
                "expected_price": float(expected_price),
                "price_std": float(price_std),
                "current_price": self.current_price,
                "percentiles": {
                    "p1": float(percentiles[0]),
                    "p5": float(percentiles[1]),
                    "p10": float(percentiles[2]),
                    "p25": float(percentiles[3]),
                    "p50": float(percentiles[4]),
                    "p75": float(percentiles[5]),
                    "p90": float(percentiles[6]),
                    "p95": float(percentiles[7]),
                    "p99": float(percentiles[8])
                },
                "probabilities": {
                    "increase": float(prob_increase),
                    "up_5pct": float(prob_up_5pct),
                    "up_10pct": float(prob_up_10pct),
                    "up_20pct": float(prob_up_20pct),
                    "down_5pct": float(prob_down_5pct),
                    "down_10pct": float(prob_down_10pct),
                    "down_20pct": float(prob_down_20pct)
                }
            }
        
        return self.simulation_results

    async def _run_simulation_with_multiple_models(self) -> Dict[str, Any]:
        """
        Run the Monte Carlo simulation with multiple volatility models.
        
        Returns:
            Dictionary containing aggregated simulation results
        """
        # Determine number of simulations per model
        num_volatility_models = len(self.volatility_models)
        
        # Determine total simulations based on device
        if device.type != 'cpu':
            # For GPU, we can run more simulations
            total_simulations = min(self.num_simulations, 5000)
        else:
            # For CPU, limit to a reasonable number
            total_simulations = min(self.num_simulations, 1000)
        
        # Calculate simulations per model (at least 10 per model)
        simulations_per_model = max(10, total_simulations // num_volatility_models)
        
        # Store results for each model
        all_model_results = {}
        
        # Simulate for each time horizon and each volatility model
        for horizon in self.time_horizons:
            logger.info(f"Running simulations for {horizon} day horizon with {num_volatility_models} volatility models")
            
            # Initialize arrays to store all final prices for this horizon
            all_final_prices = []
            
            # Run simulations for each volatility model
            for window, vol_data in self.volatility_models.items():
                # Skip if avg_realized is not available
                if 'avg_realized' not in vol_data:
                    continue
                
                # Use this model's realized volatility
                model_volatility = vol_data['avg_realized']
                model_variance = model_volatility ** 2
                
                # Skip if volatility is invalid
                if not np.isfinite(model_volatility) or model_volatility <= 0:
                    continue
                
                # Run simulations with this volatility
                logger.info(f"  - Window {window}: Running {simulations_per_model} simulations with volatility {model_volatility:.4f}")
                
                # Choose simulation method based on configuration
                if self.use_heston:
                    model_final_prices = self._simulate_heston(
                        horizon, simulations_per_model, volatility=model_volatility, variance=model_variance
                    )
                else:
                    model_final_prices = self._simulate_gbm(
                        horizon, simulations_per_model, volatility=model_volatility
                    )
                
                # Store results for this model
                all_final_prices.extend(model_final_prices)
                
                # Also store model-specific results for potential later use
                if horizon not in all_model_results:
                    all_model_results[horizon] = {}
                
                all_model_results[horizon][window] = {
                    "volatility": model_volatility,
                    "expected_price": float(np.mean(model_final_prices)),
                    "price_std": float(np.std(model_final_prices)),
                    "num_simulations": len(model_final_prices)
                }
            
            # Check if we have any valid simulations
            if not all_final_prices:
                logger.warning(f"No valid simulations for horizon {horizon}")
                continue
            
            # Convert to numpy array
            all_final_prices = np.array(all_final_prices)
            
            # Calculate expected price and confidence intervals
            expected_price = np.mean(all_final_prices)
            price_std = np.std(all_final_prices)
            
            # Calculate various percentiles
            percentiles = np.percentile(all_final_prices, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            
            # Calculate probability of price increase
            prob_increase = np.mean(all_final_prices > self.current_price)
            
            # Calculate probability of significant moves
            prob_up_5pct = np.mean(all_final_prices > self.current_price * 1.05)
            prob_up_10pct = np.mean(all_final_prices > self.current_price * 1.10)
            prob_up_20pct = np.mean(all_final_prices > self.current_price * 1.20)
            prob_down_5pct = np.mean(all_final_prices < self.current_price * 0.95)
            prob_down_10pct = np.mean(all_final_prices < self.current_price * 0.90)
            prob_down_20pct = np.mean(all_final_prices < self.current_price * 0.80)
            
            # Store aggregated results
            self.simulation_results[horizon] = {
                "expected_price": float(expected_price),
                "price_std": float(price_std),
                "current_price": self.current_price,
                "percentiles": {
                    "p1": float(percentiles[0]),
                    "p5": float(percentiles[1]),
                    "p10": float(percentiles[2]),
                    "p25": float(percentiles[3]),
                    "p50": float(percentiles[4]),
                    "p75": float(percentiles[5]),
                    "p90": float(percentiles[6]),
                    "p95": float(percentiles[7]),
                    "p99": float(percentiles[8])
                },
                "probabilities": {
                    "increase": float(prob_increase),
                    "up_5pct": float(prob_up_5pct),
                    "up_10pct": float(prob_up_10pct),
                    "up_20pct": float(prob_up_20pct),
                    "down_5pct": float(prob_down_5pct),
                    "down_10pct": float(prob_down_10pct),
                    "down_20pct": float(prob_down_20pct)
                },
                "model_details": all_model_results[horizon]  # Store individual model results
            }
        
        return self.simulation_results

    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the simulation results for reporting purposes.
        
        Returns:
            Dictionary containing a summary of the simulation results including
            current price, volatility metrics, and horizon-specific forecasts
        """
        if not self.simulation_results:
            logger.warning(f"No simulation results available for {self.ticker}")
            return None
        
        summary = {
            "ticker": self.ticker,
            "current_price": self.current_price,
            "annualized_drift": self.annualized_drift,
            "annualized_volatility": self.annualized_volatility,
            "horizons": {}
        }
        
        # Add information about volatility models if available
        if self.use_multiple_volatility_models and hasattr(self, 'volatility_models') and self.volatility_models and hasattr(self, 'best_timeframe') and self.best_timeframe:
            summary["volatility_models"] = {
                "num_models": len(self.volatility_models),
                "best_timeframe": self.best_timeframe
            }
        
        for horizon, results in self.simulation_results.items():
            # Calculate summary metrics for this horizon
            expected_price = results["expected_price"]
            expected_change_pct = ((expected_price / self.current_price) - 1) * 100
            
            # Get percentiles - ensure we're using the right keys based on the simulation results
            if "percentiles" in results:
                lower_bound = results["percentiles"].get("p5", None)
                upper_bound = results["percentiles"].get("p95", None)
            else:
                # Fallback if percentiles are stored differently
                lower_bound = None
                upper_bound = None
            
            # Get probability values - ensure they're converted to percentages
            if "probabilities" in results:
                prob_increase = results["probabilities"].get("increase", 0.5) * 100
                prob_up_10pct = results["probabilities"].get("up_10pct", 0) * 100
                prob_down_10pct = results["probabilities"].get("down_10pct", 0) * 100
            else:
                # Fallback if probabilities are stored differently
                prob_increase = 50
                prob_up_10pct = 0
                prob_down_10pct = 0
            
            summary["horizons"][horizon] = {
                "days": horizon,
                "expected_price": expected_price,
                "expected_change_pct": expected_change_pct,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "prob_increase": prob_increase,  # Already converted to percentage
                "prob_up_10pct": prob_up_10pct,  # Already converted to percentage
                "prob_down_10pct": prob_down_10pct  # Already converted to percentage
            }
        
        return summary

async def generate_stock_report(ticker: str) -> Dict[str, Any]:
    """
    Generate a comprehensive Monte Carlo report for a stock.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary containing report data
    """
    try:
        # Initialize simulator
        simulator = MonteCarloSimulator(
            ticker=ticker,
            num_simulations=1000,
            time_horizons=[1, 5, 10, 21, 63, 126, 252],  # 1D, 1W, 2W, 1M, 3M, 6M, 1Y
            use_heston=True
        )
        
        # Run simulation
        await simulator.run_simulation()
        
        # Get results summary
        summary = simulator.get_results_summary()
        
        if not summary:
            logger.warning(f"No simulation results for {ticker}")
            return {"error": f"Failed to generate simulation for {ticker}"}
        
        # Format report
        report = {
            "ticker": ticker,
            "current_price": summary["current_price"],
            "annualized_volatility": summary["annualized_volatility"],
            "time_horizons": [],
            "raw_simulations": simulator.simulation_results
        }
        
        # Add each time horizon to the report
        for days, horizon_data in summary["horizons"].items():
            report["time_horizons"].append({
                "days": days,
                "label": _format_time_horizon(days),
                "expected_price": horizon_data["expected_price"],
                "expected_change_pct": horizon_data["expected_change_pct"],
                "lower_bound": horizon_data["lower_bound"],
                "upper_bound": horizon_data["upper_bound"],
                "prob_increase": horizon_data["prob_increase"],
                "prob_up_10pct": horizon_data["prob_up_10pct"],
                "prob_down_10pct": horizon_data["prob_down_10pct"]
            })
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating Monte Carlo report for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def _format_time_horizon(days: int) -> str:
    """
    Format a time horizon in days to a human-readable string.
    
    Args:
        days: Number of days
        
    Returns:
        Formatted string (e.g., "1D", "1W", "1M", etc.)
    """
    if days == 1:
        return "1D"
    elif days <= 5:
        return "1W"
    elif days <= 10:
        return "2W"
    elif days <= 21:
        return "1M"
    elif days <= 63:
        return "3M"
    elif days <= 126:
        return "6M"
    elif days <= 252:
        return "1Y"
    else:
        return f"{days}D"

async def run_monte_carlo_analysis(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Run Monte Carlo analysis for multiple stocks in parallel.
    
    Args:
        tickers: List of stock ticker symbols
        
    Returns:
        Dictionary mapping ticker symbols to their simulation results
    """
    results = {}
    
    # Create tasks for all tickers
    tasks = [generate_stock_report(ticker) for ticker in tickers]
    
    # Run all tasks concurrently
    for i, ticker in enumerate(tickers):
        try:
            results[ticker] = await tasks[i]
        except Exception as e:
            logger.error(f"Error in Monte Carlo analysis for {ticker}: {e}")
            results[ticker] = {"error": str(e)}
    
    return results

def screen_stocks(tickers: List[str]) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Screen stocks using Monte Carlo simulation to predict future price movements.
    
    Args:
        tickers: List of stock ticker symbols
        
    Returns:
        List of tuples (ticker, score, details) for each stock
    """
    # Run Monte Carlo analysis
    monte_carlo_results = asyncio.run(run_monte_carlo_analysis(tickers))
    
    # Process results and calculate scores
    results = []
    for ticker, report in monte_carlo_results.items():
        if "error" in report:
            logger.warning(f"Skipping {ticker} due to error: {report['error']}")
            continue
        
        # Get the 1-month, 3-month, and 6-month horizons for scoring
        time_horizons = report.get("time_horizons", [])
        horizon_1m = next((h for h in time_horizons if h["days"] == 21), None)
        horizon_3m = next((h for h in time_horizons if h["days"] == 63), None)
        horizon_6m = next((h for h in time_horizons if h["days"] == 126), None)
        
        if not all([horizon_1m, horizon_3m, horizon_6m]):
            logger.warning(f"Skipping {ticker} due to missing time horizons")
            continue
        
        # Calculate score based on expected returns and probabilities
        score_components = [
            horizon_1m["expected_change_pct"] * 0.2,  # 1-month expected return (20% weight)
            horizon_3m["expected_change_pct"] * 0.3,  # 3-month expected return (30% weight)
            horizon_6m["expected_change_pct"] * 0.5,  # 6-month expected return (50% weight)
            horizon_1m["prob_increase"] * 0.1,          # Raw probability of price increase
            horizon_3m["prob_increase"] * 0.15,
            horizon_6m["prob_increase"] * 0.25,
            (horizon_1m["prob_up_10pct"] - horizon_1m["prob_down_10pct"]) * 0.2  # Skew adjustment
        ]


        
        score = sum(score_components)
        
        # Create detailed results
        details = {
            "current_price": report["current_price"],
            "annualized_volatility": report["annualized_volatility"],
            "time_horizons": {h["label"]: h for h in time_horizons},
            "raw_simulations": report.get("raw_simulations", {}),
            "category_scores": {
                "expected_return": (
                    horizon_1m["expected_change_pct"] * 0.2 +
                    horizon_3m["expected_change_pct"] * 0.3 +
                    horizon_6m["expected_change_pct"] * 0.5
                ),
                "probability": (
                    (horizon_1m["prob_increase"] - 50) * 0.25 +
                    (horizon_3m["prob_increase"] - 50) * 0.35 +
                    (horizon_6m["prob_increase"] - 50) * 0.4
                ),
                "risk_reward": (
                    (horizon_1m["prob_up_10pct"] - horizon_1m["prob_down_10pct"]) * 0.3 +
                    (horizon_3m["prob_up_10pct"] - horizon_3m["prob_down_10pct"]) * 0.3 +
                    (horizon_6m["prob_up_10pct"] - horizon_6m["prob_down_10pct"]) * 0.4
                )
            }
        }
        
        results.append((ticker, score, details))
    
    # Sort results by score in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

def generate_stock_report_task(ticker: str) -> Dict[str, Any]:
    """
    Generate a Monte Carlo report for a stock, suitable for concurrent execution.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary containing report data
    """
    try:
        # Initialize simulator
        simulator = MonteCarloSimulator(
            ticker=ticker,
            num_simulations=1000,
            time_horizons=[1, 5, 10, 21, 63, 126, 252],  # 1D, 1W, 2W, 1M, 3M, 6M, 1Y
            use_heston=True
        )
        
        # Run calibration synchronously
        calibration_success = False
        try:
            # Get the historical data
            price_history = simulator.fetch_historical_data(days=max(252, max(simulator.time_horizons)))
            
            if price_history.empty:
                logger.warning(f"No historical data available for {ticker}. Cannot calibrate model.")
                return {"error": f"No historical data available for {ticker}"}
                
            # Extract current price
            simulator.current_price = float(price_history['close'].iloc[-1])
            
            # Calculate annualized drift (average daily return * 252)
            daily_drift = price_history['log_return'].mean()
            simulator.annualized_drift = daily_drift * 252
            
            # Calculate annualized volatility (std dev of daily returns * sqrt(252))
            daily_volatility = price_history['log_return'].std()
            simulator.annualized_volatility = daily_volatility * np.sqrt(252)
            
            # Set initial variance for Heston model
            simulator.v0 = simulator.annualized_volatility ** 2
            
            # If using Heston, calibrate Heston model parameters
            if simulator.use_heston:
                # Simple heuristic-based calibration
                vol_window = min(60, len(price_history) // 2)
                rolling_vol = price_history['log_return'].rolling(window=vol_window).std() * np.sqrt(252)
                
                if len(rolling_vol.dropna()) > 10:
                    # Estimate mean reversion parameters
                    vol_diff = rolling_vol.diff().dropna()
                    vol_lag = rolling_vol.shift(1).dropna()
                    
                    if len(vol_diff) == len(vol_lag) and len(vol_diff) > 10:
                        # Simple linear regression for mean reversion speed
                        vol_diff_arr = vol_diff.values[-len(vol_lag):]
                        vol_lag_arr = vol_lag.values
                        
                        # Avoid division by zero
                        if np.var(vol_lag_arr) > 0:
                            beta = np.cov(vol_diff_arr, vol_lag_arr)[0, 1] / np.var(vol_lag_arr)
                            simulator.kappa = -beta * 252  # Annualize
                            simulator.kappa = max(0.1, min(10, simulator.kappa))  # Keep in reasonable range
                        
                        # Estimate long-term volatility
                        simulator.theta = rolling_vol.mean() ** 2  # Convert to variance
                        
                        # Estimate volatility of volatility
                        simulator.sigma = rolling_vol.std() * 2  # Heuristic scaling
                        simulator.sigma = max(0.01, min(2, simulator.sigma))  # Keep in reasonable range
                
                # Estimate correlation between returns and volatility changes
                if len(price_history) > 20:
                    log_returns = price_history['log_return'].values
                    vol_changes = np.diff(np.concatenate([[0], rolling_vol.dropna().values]))
                    min_len = min(len(log_returns), len(vol_changes))
                    
                    if min_len > 10:
                        corr_matrix = np.corrcoef(log_returns[-min_len:], vol_changes[-min_len:])
                        if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]):
                            simulator.rho = corr_matrix[0, 1]
                            simulator.rho = max(-0.95, min(0.95, simulator.rho))  # Keep in reasonable range
            
            simulator.price_history = price_history
            calibration_success = True
            
        except Exception as e:
            logger.error(f"Error calibrating model for {ticker}: {e}")
            logger.error(traceback.format_exc())
            return {"error": f"Failed to calibrate model for {ticker}: {str(e)}"}
        
        if not calibration_success:
            return {"error": f"Failed to calibrate model for {ticker}"}
        
        # Run simulation synchronously
        simulation_results = {}
        
        # Determine number of simulations based on device
        if device.type != 'cpu':
            # For GPU, we can run more simulations
            num_simulations = min(simulator.num_simulations, 5000)
        else:
            # For CPU, limit to a reasonable number
            num_simulations = min(simulator.num_simulations, 1000)
        
        # Simulate for each time horizon
        for horizon in simulator.time_horizons:
            logger.info(f"Running {num_simulations} simulations for {horizon} day horizon")
            
            # Choose simulation method based on configuration
            if simulator.use_heston:
                final_prices = simulator._simulate_heston(horizon, num_simulations)
            else:
                final_prices = simulator._simulate_gbm(horizon, num_simulations)
            
            # Calculate expected price and confidence intervals
            expected_price = np.mean(final_prices)
            price_std = np.std(final_prices)
            
            # Calculate various percentiles
            percentiles = np.percentile(final_prices, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            
            # Calculate probability of price increase
            prob_increase = np.mean(final_prices > simulator.current_price)
            
            # Calculate probability of significant moves
            prob_up_5pct = np.mean(final_prices > simulator.current_price * 1.05)
            prob_up_10pct = np.mean(final_prices > simulator.current_price * 1.10)
            prob_up_20pct = np.mean(final_prices > simulator.current_price * 1.20)
            prob_down_5pct = np.mean(final_prices < simulator.current_price * 0.95)
            prob_down_10pct = np.mean(final_prices < simulator.current_price * 0.90)
            prob_down_20pct = np.mean(final_prices < simulator.current_price * 0.80)
            
            # Store results
            simulation_results[horizon] = {
                "expected_price": float(expected_price),
                "price_std": float(price_std),
                "current_price": simulator.current_price,
                "percentiles": {
                    "p1": float(percentiles[0]),
                    "p5": float(percentiles[1]),
                    "p10": float(percentiles[2]),
                    "p25": float(percentiles[3]),
                    "p50": float(percentiles[4]),
                    "p75": float(percentiles[5]),
                    "p90": float(percentiles[6]),
                    "p95": float(percentiles[7]),
                    "p99": float(percentiles[8])
                },
                "probabilities": {
                    "increase": float(prob_increase),
                    "up_5pct": float(prob_up_5pct),
                    "up_10pct": float(prob_up_10pct),
                    "up_20pct": float(prob_up_20pct),
                    "down_5pct": float(prob_down_5pct),
                    "down_10pct": float(prob_down_10pct),
                    "down_20pct": float(prob_down_20pct)
                }
            }
        
        # Get results summary
        summary = {
            "ticker": ticker,
            "current_price": simulator.current_price,
            "annualized_drift": simulator.annualized_drift,
            "annualized_volatility": simulator.annualized_volatility,
            "horizons": {}
        }
        
        for horizon, results in simulation_results.items():
            # Calculate summary metrics for this horizon
            expected_change_pct = (results["expected_price"] / simulator.current_price - 1) * 100
            lower_bound = results["percentiles"]["p5"]
            upper_bound = results["percentiles"]["p95"]
            
            summary["horizons"][horizon] = {
                "days": horizon,
                "expected_price": results["expected_price"],
                "expected_change_pct": expected_change_pct,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "prob_increase": results["probabilities"]["increase"] * 100,
                "prob_up_10pct": results["probabilities"]["up_10pct"] * 100,
                "prob_down_10pct": results["probabilities"]["down_10pct"] * 100
            }
            
        # Format report
        report = {
            "ticker": ticker,
            "current_price": summary["current_price"],
            "annualized_volatility": summary["annualized_volatility"],
            "time_horizons": [],
            "raw_simulations": simulation_results
        }
        
        # Add each time horizon to the report
        for days, horizon_data in summary["horizons"].items():
            report["time_horizons"].append({
                "days": days,
                "label": _format_time_horizon(days),
                "expected_price": horizon_data["expected_price"],
                "expected_change_pct": horizon_data["expected_change_pct"],
                "lower_bound": horizon_data["lower_bound"],
                "upper_bound": horizon_data["upper_bound"],
                "prob_increase": horizon_data["prob_increase"],
                "prob_up_10pct": horizon_data["prob_up_10pct"],
                "prob_down_10pct": horizon_data["prob_down_10pct"]
            })
        
        return report
            
    except Exception as e:
        logger.error(f"Error generating Monte Carlo report for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def generate_stock_reports(tickers: List[str], max_workers: int = None) -> Dict[str, Dict[str, Any]]:
    """
    Generate Monte Carlo reports for multiple stocks using ThreadPoolExecutor.
    
    Args:
        tickers: List of stock ticker symbols
        max_workers: Maximum number of worker threads
        
    Returns:
        Dictionary mapping ticker symbols to their reports
    """
    if max_workers is None:
        max_workers = min(32, os.cpu_count() * 2)
    
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(generate_stock_report_task, ticker): ticker for ticker in tickers}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_ticker), total=len(future_to_ticker), desc="Generating Monte Carlo simulations"):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                logger.error(f"Error in Monte Carlo analysis for {ticker}: {e}")
                results[ticker] = {"error": str(e)}
    
    return results

def integrate_with_technicals(technical_results: List[Tuple[str, float, Dict[str, Any]]]) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Integrate Monte Carlo simulations with existing technical analysis results.
    
    Args:
        technical_results: Results from technical analysis
        
    Returns:
        Enhanced technical analysis results with Monte Carlo data
    """
    # Extract tickers from technical results
    tickers = [ticker for ticker, _, _ in technical_results]
    
    # Run Monte Carlo analysis
    monte_carlo_results = generate_stock_reports(tickers)
    
    # Integrate results
    enhanced_results = []
    for ticker, score, details in technical_results:
        mc_report = monte_carlo_results.get(ticker, {})
        
        if "error" in mc_report:
            # If Monte Carlo analysis failed, keep original technical analysis
            logger.warning(f"Using only technical analysis for {ticker} due to Monte Carlo error: {mc_report['error']}")
            enhanced_results.append((ticker, score, details))
            continue
        
        # Get the most relevant time horizons
        time_horizons = mc_report.get("time_horizons", [])
        
        # Get the 1-month horizon
        horizon_1m = next((h for h in time_horizons if h["days"] == 21), None)
        
        if horizon_1m:
            # Update raw indicators
            if "raw_indicators" in details:
                details["raw_indicators"]["mc_expected_price"] = horizon_1m["expected_price"]
                details["raw_indicators"]["mc_lower_bound"] = horizon_1m["lower_bound"]
                details["raw_indicators"]["mc_upper_bound"] = horizon_1m["upper_bound"]
                details["raw_indicators"]["mc_prob_increase"] = horizon_1m["prob_increase"]
            
            # Add Monte Carlo data to details
            details["monte_carlo"] = {
                "expected_price": horizon_1m["expected_price"],
                "lower_bound": horizon_1m["lower_bound"],
                "upper_bound": horizon_1m["upper_bound"],
                "prob_increase": horizon_1m["prob_increase"],
                "prob_up_10pct": horizon_1m["prob_up_10pct"],
                "prob_down_10pct": horizon_1m["prob_down_10pct"],
                "time_horizons": {h["label"]: h for h in time_horizons}
            }
            
            # Adjust score using Monte Carlo results (optional)
            mc_score_adjustment = (
                (horizon_1m["expected_change_pct"] / 10)  # Expected return component
                + ((horizon_1m["prob_increase"] - 50) / 25)  # Probability component
                + ((horizon_1m["prob_up_10pct"] - horizon_1m["prob_down_10pct"]) / 15)  # Risk-reward component
            )
            
            # Blend technical score with Monte Carlo adjustment
            # Weight can be adjusted based on preference (here using 80% technical, 20% Monte Carlo)
            blended_score = score * 0.8 + mc_score_adjustment * 0.2
            
            enhanced_results.append((ticker, blended_score, details))
        else:
            # If no 1-month horizon, keep original technical analysis
            enhanced_results.append((ticker, score, details))
    
    # Re-sort based on new scores
    enhanced_results.sort(key=lambda x: x[1], reverse=True)
    
    return enhanced_results

def create_monte_carlo_table_data(ticker: str, mc_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create data for a tabular visualization of Monte Carlo simulation results.
    
    Args:
        ticker: Stock ticker symbol
        mc_report: Monte Carlo simulation report
        
    Returns:
        List of dictionaries for table display
    """
    if "error" in mc_report:
        return []
    
    # Extract data for table
    table_data = []
    time_horizons = sorted(mc_report.get("time_horizons", []), key=lambda x: x["days"])
    
    for horizon in time_horizons:
        table_row = {
            "ticker": ticker,
            "horizon": horizon["label"],
            "days": horizon["days"],
            "expected_price": round(horizon["expected_price"], 2),
            "expected_change_pct": round(horizon["expected_change_pct"], 2),
            "lower_bound": round(horizon["lower_bound"], 2),
            "upper_bound": round(horizon["upper_bound"], 2),
            "range_width_pct": round((horizon["upper_bound"] - horizon["lower_bound"]) / mc_report["current_price"] * 100, 2),
            "prob_increase": round(horizon["prob_increase"], 2),
            "prob_up_10pct": round(horizon["prob_up_10pct"], 2),
            "prob_down_10pct": round(horizon["prob_down_10pct"], 2)
        }
        table_data.append(table_row)
    
    return table_data

def create_combined_table_data(technical_results: List[Tuple[str, float, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Create a combined table of technical and Monte Carlo results.
    
    Args:
        technical_results: Results from technical analysis
        
    Returns:
        List of dictionaries for combined table display
    """
    table_data = []
    
    for ticker, score, details in technical_results:
        # Get technical indicators
        raw_indicators = details.get("raw_indicators", {})
        current_price = raw_indicators.get("current_price")
        
        if not current_price:
            continue
        
        # Get Monte Carlo data
        monte_carlo = details.get("monte_carlo", {})
        
        # Create row for each ticker
        table_row = {
            "ticker": ticker,
            "composite_score": round(score, 4),
            "current_price": round(current_price, 2),
            "technical_expected": raw_indicators.get("expected_price", current_price),
            "technical_low": raw_indicators.get("probable_low", current_price * 0.9),
            "technical_high": raw_indicators.get("probable_high", current_price * 1.1),
            "mc_expected": monte_carlo.get("expected_price", current_price),
            "mc_low": monte_carlo.get("lower_bound", current_price * 0.9),
            "mc_high": monte_carlo.get("upper_bound", current_price * 1.1),
            "mc_prob_increase": monte_carlo.get("prob_increase", 50),
            "mc_prob_up_10pct": monte_carlo.get("prob_up_10pct", 0),
            "mc_prob_down_10pct": monte_carlo.get("prob_down_10pct", 0)
        }
        
        # Add category scores if available
        if "category_scores" in details:
            for category, score in details["category_scores"].items():
                table_row[f"score_{category}"] = round(score, 4)
        
        table_data.append(table_row)
    
    return table_data

def generate_price_path_sample(ticker: str, time_horizon: int = 63, num_paths: int = 10) -> Dict[str, Any]:
    """
    Generate a sample of price paths for visualization purposes.
    
    Args:
        ticker: Stock ticker symbol
        time_horizon: Time horizon in days
        num_paths: Number of sample paths to generate
        
    Returns:
        Dictionary containing sample paths and metadata
    """
    try:
        # Initialize simulator
        simulator = MonteCarloSimulator(
            ticker=ticker,
            num_simulations=num_paths,
            time_horizons=[time_horizon],
            use_heston=True
        )
        
        # Calibrate model parameters
        calibration_success = asyncio.run(simulator.calibrate_model_parameters())
        if not calibration_success:
            return {"error": f"Failed to calibrate model for {ticker}"}
        
        # Generate price paths
        dt = 1.0 / 252  # Daily time step
        sqrt_dt = np.sqrt(dt)
        
        # Initialize arrays for price and variance paths
        prices = np.zeros((num_paths, time_horizon + 1))
        variances = np.zeros((num_paths, time_horizon + 1))
        
        # Set initial values
        prices[:, 0] = simulator.current_price
        variances[:, 0] = simulator.v0
        
        # Generate correlated random samples
        z1 = np.random.normal(0, 1, size=(num_paths, time_horizon))
        z2 = np.random.normal(0, 1, size=(num_paths, time_horizon))
        
        # Apply correlation
        z2_correlated = simulator.rho * z1 + np.sqrt(1 - simulator.rho**2) * z2
        
        # Simulate paths
        for t in range(time_horizon):
            # Ensure variance stays positive
            variances[:, t] = np.maximum(variances[:, t], 1e-8)
            
            # Update price (log process for better numerical stability)
            price_drift = (simulator.annualized_drift - 0.5 * variances[:, t]) * dt
            price_diffusion = np.sqrt(variances[:, t]) * sqrt_dt * z1[:, t]
            prices[:, t+1] = prices[:, t] * np.exp(price_drift + price_diffusion)
            
            # Update variance
            var_drift = simulator.kappa * (simulator.theta - variances[:, t]) * dt
            var_diffusion = simulator.sigma * np.sqrt(variances[:, t]) * sqrt_dt * z2_correlated[:, t]
            variances[:, t+1] = variances[:, t] + var_drift + var_diffusion
        
        # Convert to list format for easy serialization
        days = list(range(time_horizon + 1))
        paths = []
        
        for i in range(num_paths):
            path = {
                "path_id": i,
                "days": days,
                "prices": prices[i, :].tolist()
            }
            paths.append(path)
        
        # Add statistics
        final_prices = prices[:, -1]
        price_stats = calculate_statistics(final_prices)
        
        # Calculate percentiles
        percentiles = np.percentile(final_prices, [5, 25, 50, 75, 95])
        
        return {
            "ticker": ticker,
            "current_price": simulator.current_price,
            "time_horizon": time_horizon,
            "time_horizon_label": _format_time_horizon(time_horizon),
            "annualized_drift": simulator.annualized_drift,
            "annualized_volatility": simulator.annualized_volatility,
            "heston_parameters": {
                "kappa": simulator.kappa,
                "theta": simulator.theta,
                "sigma": simulator.sigma,
                "rho": simulator.rho,
                "v0": simulator.v0
            },
            "price_paths": paths,
            "final_price_stats": price_stats,
            "percentiles": {
                "p5": float(percentiles[0]),
                "p25": float(percentiles[1]),
                "p50": float(percentiles[2]),
                "p75": float(percentiles[3]),
                "p95": float(percentiles[4])
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating price paths for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def analyze_historical_volatility(ticker: str) -> Dict[str, Any]:
    """
    Analyze historical volatility patterns to enhance Monte Carlo simulations.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary containing volatility analysis
    """
    try:        
        # Initialize OpenBB client
        obb_client = get_openbb_client()
        
        # Fetch historical price data
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        price_history_response = obb_client.equity.price.historical(
            symbol=ticker, start_date=one_year_ago, provider='fmp'
        )
        
        if not hasattr(price_history_response, 'results') or not price_history_response.results:
            return {"error": f"No historical price data available for {ticker}"}
        
        # Run volatility analysis
        combined_vols, best_timeframe = get_combined_historical_volatility(
            data=price_history_response.results,
            lower_q=0.25,
            upper_q=0.75,
            trading_periods=252,
            is_crypto=False,
            index="date"
        )
        
        # Format results
        volatility_analysis = {
            "ticker": ticker,
            "best_timeframe": best_timeframe,
            "volatility_by_window": {}
        }
        
        # Add combined volatility data
        for window, values in combined_vols.items():
            # Convert NumPy arrays and other non-serializable values to Python types
            volatility_analysis["volatility_by_window"][window] = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in values.items()
                if not isinstance(v, (list, np.ndarray))
            }
        
        return volatility_analysis
        
    except Exception as e:
        logger.error(f"Error analyzing historical volatility for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def analyze_normality(ticker: str) -> Dict[str, Any]:
    """
    Analyze the normality of returns to improve simulation accuracy.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary containing normality test results
    """
    try:
        # Initialize OpenBB client
        obb_client = get_openbb_client()
        
        # Fetch historical price data
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        price_history_response = obb_client.equity.price.historical(
            symbol=ticker, start_date=one_year_ago, provider='fmp'
        )
        
        if not hasattr(price_history_response, 'results') or not price_history_response.results:
            return {"error": f"No historical price data available for {ticker}"}
        
        # Convert to DataFrame
        price_data = []
        for record in price_history_response.results:
            price_data.append({
                'date': getattr(record, 'date', None),
                'close': getattr(record, 'close', None)
            })
        
        df = pd.DataFrame(price_data)
        
        # Calculate returns
        df['return'] = df['close'].pct_change()
        df = df.dropna()
        
        if len(df) < 30:
            return {"error": f"Insufficient historical data for {ticker}. Need at least 30 days."}
        
        # Run normality tests
        normality_results = obb_client.quantitative.normality(
            data=df.to_dict('records'), target='return'
        )
        
        # Extract results
        if hasattr(normality_results, 'results'):
            results = normality_results.results
            
            # Convert to dictionary
            normality_analysis = {
                "ticker": ticker,
                "jarque_bera": {
                    "statistic": float(getattr(results, 'jarque_bera_stat', 0)),
                    "p_value": float(getattr(results, 'jarque_bera_p', 0)),
                    "is_normal": bool(getattr(results, 'jarque_bera_normal', False))
                },
                "shapiro": {
                    "statistic": float(getattr(results, 'shapiro_stat', 0)),
                    "p_value": float(getattr(results, 'shapiro_p', 0)),
                    "is_normal": bool(getattr(results, 'shapiro_normal', False))
                },
                "kolmogorov": {
                    "statistic": float(getattr(results, 'kolmogorov_stat', 0)),
                    "p_value": float(getattr(results, 'kolmogorov_p', 0)),
                    "is_normal": bool(getattr(results, 'kolmogorov_normal', False))
                },
                "kurtosis": {
                    "value": float(getattr(results, 'kurtosis', 0)),
                    "is_normal": bool(getattr(results, 'kurtosis_normal', False))
                },
                "skewness": {
                    "value": float(getattr(results, 'skew', 0)),
                    "is_normal": bool(getattr(results, 'skew_normal', False))
                },
                "overall_normal": bool(getattr(results, 'is_normal', False))
            }
            
            return normality_analysis
        else:
            return {"error": "Failed to run normality tests"}
        
    except Exception as e:
        logger.error(f"Error analyzing normality for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return {"error": str(e)}
