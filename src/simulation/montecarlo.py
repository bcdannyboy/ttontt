"""
Monte Carlo Stock Price Simulator
================================

A comprehensive stochastic price simulation tool that evaluates future price movements
based on historical volatility and generates probabilistic price forecasts.
"""

import os
import numpy as np
import pandas as pd
import logging
import traceback
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import torch
from functools import lru_cache

# Import simulation components
from src.simulation.utils import (
    initialize_openbb, calculate_statistics, _format_time_horizon, 
    CPU_COUNT, DEFAULT_SIMULATION_COUNT, DEFAULT_TIME_HORIZONS, DEVICE
)
from src.simulation.parallel import (
    simulate_ticker, simulate_ticker_batch, run_parallel_simulation
)
from src.simulation.models import simulate_gbm, simulate_heston, simulate_sabr_cgmy

# Configure logging
logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """
    A class to perform Monte Carlo simulations for stock price movements
    using multiple stochastic models. This class uses optimized parallelism
    for efficient simulation across multiple threads with GPU acceleration.
    """
    
    def __init__(self, ticker: str, num_simulations: int = DEFAULT_SIMULATION_COUNT, 
                 time_horizons: List[int] = None,
                 use_heston: bool = True, random_seed: Optional[int] = None,
                 use_multiple_volatility_models: bool = True,
                 max_threads: Optional[int] = None):
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            ticker: Stock ticker symbol.
            num_simulations: Number of simulation paths.
            time_horizons: List of time horizons (trading days) to simulate.
            use_heston: Whether to use the Heston model.
            random_seed: Random seed for reproducibility.
            use_multiple_volatility_models: Whether to use multiple volatility models.
            max_threads: Maximum number of threads to use for internal parallelism.
        """
        self.ticker = ticker
        self.num_simulations = num_simulations
        
        if time_horizons is None:
            self.time_horizons = DEFAULT_TIME_HORIZONS
        else:
            self.time_horizons = time_horizons
            
        self.use_heston = use_heston
        self.use_multiple_volatility_models = use_multiple_volatility_models
        
        # Set maximum threads for internal parallelism - reduced to avoid oversaturation
        self.max_threads = max_threads or min(8, max(2, CPU_COUNT // 2))
        
        self.risk_free_rate = 0.05
        
        # Default SABR parameters
        self.sabr_alpha = 0.3
        self.sabr_beta = 0.5
        self.sabr_rho = -0.5
        
        # Default CGMY parameters
        self.cgm_C = 1.0
        self.cgm_G = 5.0
        self.cgm_M = 10.0
        self.cgm_Y = 0.5
        
        if random_seed is not None:
            np.random.seed(random_seed)
            if torch.cuda.is_available() or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                torch.manual_seed(random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(random_seed)
        
        self.price_history = None
        self.current_price = None
        self.annualized_drift = None
        self.annualized_volatility = None
        self.volatility_models = None
        self.best_timeframe = None
        
        # Heston model parameters
        self.kappa = 2.0
        self.theta = 0.04
        self.sigma = 0.3
        self.rho = -0.7
        self.v0 = None
        
        self.simulation_results = {}
        self._semaphore = None  # Will be initialized in run_simulation
    
    def __getstate__(self):
        """
        Customize pickling to handle unpickleable objects.
        """
        state = self.__dict__.copy()
        # Remove unpickleable objects
        if '_semaphore' in state:
            del state['_semaphore']
        return state

    async def fetch_historical_data(self, days: int = 252) -> pd.DataFrame:
        """
        Fetch historical price data for the ticker asynchronously.
        """
        initialize_openbb()
        try:
            from openbb import obb
            start_date = (datetime.now() - timedelta(days=int(days * 1.4))).strftime('%Y-%m-%d')
            
            # Run potentially blocking OpenBB call in a thread pool
            loop = asyncio.get_event_loop()
            price_history_response = await loop.run_in_executor(
                None,
                lambda: obb.equity.price.historical(
                    symbol=self.ticker, start_date=start_date, provider='fmp'
                )
            )
            
            if hasattr(price_history_response, 'results'):
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
                if len(df) < 30:
                    logger.warning(f"Insufficient historical data for {self.ticker}. Need at least 30 days.")
                    return pd.DataFrame()
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
        """
        try:
            # Import utilities with updated OpenBB access logic
            from src.simulation.utils import get_openbb_client, openbb_has_technical
            
            self.price_history = await self.fetch_historical_data(days=max(252, max(self.time_horizons)))
            if self.price_history.empty:
                logger.warning(f"No historical data available for {self.ticker}. Cannot calibrate model.")
                return False

            self.current_price = float(self.price_history['close'].iloc[-1])
            daily_drift = self.price_history['log_return'].mean()
            self.annualized_drift = daily_drift * 252

            # Run volatility model calculation in thread pool to avoid blocking
            if self.use_multiple_volatility_models:
                try:
                    # Prepare data
                    data = []
                    for _, row in self.price_history.iterrows():
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
                    
                    # Run computation in thread pool
                    # Check if we can use OpenBB's technical module
                    if openbb_has_technical():
                        # Run using OpenBB if technical module is available
                        from src.volatility.historical import get_combined_historical_volatility
                        loop = asyncio.get_event_loop()
                        vol_result = await loop.run_in_executor(
                            None,
                            lambda: get_combined_historical_volatility(
                                data=data,
                                lower_q=0.25,
                                upper_q=0.75,
                                trading_periods=252,
                                is_crypto=False,
                                index="date"
                            )
                        )
                    else:
                        # Use fallback implementation if technical module is not available
                        from src.volatility.historical import _get_combined_historical_volatility_fallback
                        loop = asyncio.get_event_loop()
                        vol_result = await loop.run_in_executor(
                            None,
                            lambda: _get_combined_historical_volatility_fallback(
                                data=data,
                                lower_q=0.25,
                                upper_q=0.75,
                                trading_periods=252,
                                is_crypto=False,
                                index="date"
                            )
                        )
                    
                    self.volatility_models, self.best_timeframe = vol_result
                    best_window = self.best_timeframe.get('window')
                    if best_window and 'avg_realized' in self.volatility_models.get(best_window, {}):
                        self.annualized_volatility = self.volatility_models[best_window]['avg_realized']
                    else:
                        daily_volatility = self.price_history['log_return'].std()
                        self.annualized_volatility = daily_volatility * np.sqrt(252)
                except Exception as e:
                    logger.error(f"Error calculating combined volatility for {self.ticker}: {e}")
                    logger.error(traceback.format_exc())
                    daily_volatility = self.price_history['log_return'].std()
                    self.annualized_volatility = daily_volatility * np.sqrt(252)
            else:
                daily_volatility = self.price_history['log_return'].std()
                self.annualized_volatility = daily_volatility * np.sqrt(252)

            self.v0 = self.annualized_volatility ** 2

            # Calibrate Heston parameters if needed
            if self.use_heston:
                await self._calibrate_heston_parameters()
            
            logger.info(f"Model calibration for {self.ticker} completed successfully.")
            logger.info(f"Current price: {self.current_price:.2f}")
            logger.info(f"Annualized drift: {self.annualized_drift:.4f}")
            logger.info(f"Annualized volatility: {self.annualized_volatility:.4f}")
            return True
        except Exception as e:
            logger.error(f"Error calibrating model for {self.ticker}: {e}")
            logger.error(traceback.format_exc())
            return False

    async def _calibrate_heston_parameters(self):
        """
        Calibrate Heston model parameters in a separate async method.
        """
        vol_window = min(60, len(self.price_history) // 2)
        rolling_vol = self.price_history['log_return'].rolling(window=vol_window).std() * np.sqrt(252)
        
        if len(rolling_vol.dropna()) > 10:
            vol_diff = rolling_vol.diff().dropna()
            vol_lag = rolling_vol.shift(1).dropna()
            
            if len(vol_diff) == len(vol_lag) and len(vol_diff) > 10:
                vol_diff_arr = vol_diff.values[-len(vol_lag):]
                vol_lag_arr = vol_lag.values
                
                if np.var(vol_lag_arr) > 0:
                    beta = np.cov(vol_diff_arr, vol_lag_arr)[0, 1] / np.var(vol_lag_arr)
                    self.kappa = -beta * 252
                    self.kappa = max(0.1, min(10, self.kappa))
                
                self.theta = rolling_vol.mean() ** 2
                self.sigma = rolling_vol.std() * 2
                self.sigma = max(0.01, min(2, self.sigma))
        
        if len(self.price_history) > 20:
            log_returns = self.price_history['log_return'].values
            vol_changes = np.diff(np.concatenate([[0], rolling_vol.dropna().values]))
            min_len = min(len(log_returns), len(vol_changes))
            
            if min_len > 10:
                corr_matrix = np.corrcoef(log_returns[-min_len:], vol_changes[-min_len:])
                if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]):
                    self.rho = corr_matrix[0, 1]
                    self.rho = max(-0.95, min(0.95, self.rho))
        
        logger.info(f"Heston parameters - κ: {self.kappa:.4f}, θ: {self.theta:.4f}, σ: {self.sigma:.4f}, ρ: {self.rho:.4f}, v0: {self.v0:.4f}")
   
    # Delegating simulation models to the models.py module
    def simulate_gbm(self, time_horizon: int, num_paths: int, volatility: float = None) -> np.ndarray:
        """Wrapper for the GBM simulation function."""
        vol_to_use = volatility if volatility is not None else self.annualized_volatility
        return simulate_gbm(
            self.current_price, 
            self.annualized_drift,
            vol_to_use, 
            time_horizon, 
            num_paths
        )

    def simulate_heston(self, time_horizon: int, num_paths: int, 
                      volatility: float = None, variance: float = None) -> np.ndarray:
        """Wrapper for the Heston simulation function."""
        vol_to_use = volatility if volatility is not None else self.annualized_volatility
        v0_to_use = variance if variance is not None else (vol_to_use ** 2 if volatility is not None else self.v0)
        return simulate_heston(
            self.current_price,
            self.annualized_drift,
            v0_to_use,
            self.kappa,
            self.theta,
            self.sigma,
            self.rho,
            time_horizon,
            num_paths
        )

    def simulate_sabr_cgmy(self, time_horizon: int, num_paths: int, 
                       vol_override: Optional[float] = None) -> np.ndarray:
        """Wrapper for the SABR-CGMY simulation function."""
        initial_vol = vol_override if vol_override is not None else self.annualized_volatility
        return simulate_sabr_cgmy(
            self.current_price,
            initial_vol,
            self.risk_free_rate,
            self.sabr_alpha,
            self.sabr_beta,
            self.sabr_rho,
            self.cgm_C,
            self.cgm_G,
            self.cgm_M,
            self.cgm_Y,
            time_horizon,
            num_paths
        )

    # ---------------- Main Simulation Methods ---------------- #

    async def run_simulation(self) -> Dict[str, Any]:
        """
        Run the Monte Carlo simulation for all specified time horizons.
        """
        if self.current_price is None:
            success = await self.calibrate_model_parameters()
            if not success:
                logger.error(f"Failed to calibrate model for {self.ticker}")
                return {}
        
        # Initialize semaphore to control concurrency
        max_concurrent = min(4, self.max_threads)
        self._semaphore = asyncio.Semaphore(max_concurrent)
                
        if self.use_multiple_volatility_models and self.volatility_models:
            results = await self._run_simulation_with_multiple_models()
        else:
            results = await self._run_simulation_with_single_model()
            
        return results

    async def _run_simulation_with_single_model(self) -> Dict[str, Any]:
        """
        Run simulation using a single default model (Heston/GBM) and SABR-CGMY, then combine results.
        Optimized to run separate time horizons concurrently but controlled with a semaphore.
        """
        # Adjust number of simulations based on device
        if DEVICE.type != 'cpu':
            logger.info(f"Running simulations with GPU-enabled device: {DEVICE}")
            num_simulations = min(self.num_simulations, 5000)  # Increased for GPU
        else:
            num_simulations = min(self.num_simulations, 1000)  # Decreased for CPU
        
        # Create tasks for each time horizon
        tasks = []
        for horizon in self.time_horizons:
            task = asyncio.create_task(self._simulate_single_horizon_controlled(horizon, num_simulations))
            tasks.append(task)
            
        # Wait for all tasks to complete
        await asyncio.gather(*tasks)
        
        return self.simulation_results
    
    async def _simulate_single_horizon_controlled(self, horizon: int, num_simulations: int) -> None:
        """
        Simulate a single time horizon with semaphore control for concurrency.
        """
        async with self._semaphore:
            logger.info(f"Running {num_simulations} simulations for {horizon}-day horizon")
            
            # Run in an executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            
            # Run default model simulations
            if self.use_heston:
                default_final_prices = await loop.run_in_executor(
                    None,
                    lambda: run_parallel_simulation(
                        self.simulate_heston, horizon, num_simulations, self.max_threads
                    )
                )
            else:
                default_final_prices = await loop.run_in_executor(
                    None,
                    lambda: run_parallel_simulation(
                        self.simulate_gbm, horizon, num_simulations, self.max_threads
                    )
                )
                
            # Run SABR-CGMY simulations
            sabr_cgmy_final_prices = await loop.run_in_executor(
                None,
                lambda: run_parallel_simulation(
                    self.simulate_sabr_cgmy, horizon, num_simulations, self.max_threads
                )
            )
                
            # Combine model results
            combined_final_prices = np.concatenate([default_final_prices, sabr_cgmy_final_prices])
            
            # Calculate statistics on combined results
            self._store_simulation_results(horizon, combined_final_prices)
        
    async def _run_simulation_with_multiple_models(self) -> Dict[str, Any]:
        """
        Run simulation with multiple volatility models efficiently.
        Creates controlled concurrent tasks for each time horizon.
        """
        # Get number of valid volatility models
        num_volatility_models = len([
            window for window, data in self.volatility_models.items() 
            if 'avg_realized' in data and np.isfinite(data['avg_realized']) and data['avg_realized'] > 0
        ])
        
        if num_volatility_models == 0:
            logger.warning(f"No valid volatility models for {self.ticker}. Falling back to single model.")
            return await self._run_simulation_with_single_model()
        
        # Adjust simulations based on device and model count
        if DEVICE.type != 'cpu':
            logger.info(f"Running multi-model simulations with GPU-enabled device: {DEVICE}")
            simulations_per_model = min(1000, self.num_simulations // num_volatility_models)
        else:
            simulations_per_model = min(200, self.num_simulations // num_volatility_models)
        
        # Create tasks for each time horizon with controlled concurrency
        tasks = []
        for horizon in self.time_horizons:
            task = asyncio.create_task(
                self._simulate_multimodel_horizon_controlled(
                    horizon, simulations_per_model, num_volatility_models
                )
            )
            tasks.append(task)
            
        # Wait for all horizon simulations to complete
        await asyncio.gather(*tasks)
        
        return self.simulation_results
    
    async def _simulate_multimodel_horizon_controlled(self, horizon: int, simulations_per_model: int, 
                                                 num_volatility_models: int) -> None:
        """
        Simulate a single time horizon with multiple volatility models,
        with controlled concurrency using semaphores.
        """
        async with self._semaphore:
            logger.info(f"Running simulations for {horizon}-day horizon with {num_volatility_models} volatility models")
            
            loop = asyncio.get_event_loop()
            
            # Run the simulation in a thread pool to avoid blocking
            simulation_result = await loop.run_in_executor(
                None,
                lambda: self._simulate_multimodel_horizon_sync(
                    horizon, simulations_per_model, num_volatility_models
                )
            )
            
            # Process results
            all_final_prices, all_model_results = simulation_result
            
            if not all_final_prices:
                logger.warning(f"No valid simulations for horizon {horizon}")
                return
            
            # Calculate and store statistics
            self._store_simulation_results(horizon, all_final_prices, all_model_results.get(horizon))
    
    def _simulate_multimodel_horizon_sync(self, horizon: int, simulations_per_model: int, 
                                        num_volatility_models: int) -> Tuple[List[float], Dict]:
        """
        Synchronous implementation of multi-model simulation for a horizon.
        This runs in a thread pool to avoid blocking the event loop.
        """
        import concurrent.futures
        all_final_prices = []
        all_model_results = {}
        
        # Run all volatility models in parallel using a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, num_volatility_models)) as executor:
            future_to_model = {}
            
            # Submit simulations for each volatility model
            for window, vol_data in self.volatility_models.items():
                if 'avg_realized' not in vol_data:
                    continue
                    
                model_volatility = vol_data['avg_realized']
                if not np.isfinite(model_volatility) or model_volatility <= 0:
                    continue
                
                model_variance = model_volatility ** 2
                
                # Submit the simulation task
                future = executor.submit(
                    self._simulate_model_combination, 
                    horizon, 
                    simulations_per_model, 
                    model_volatility,
                    model_variance
                )
                future_to_model[future] = (window, model_volatility)
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_model):
                window, model_volatility = future_to_model[future]
                try:
                    combined_prices, _ = future.result()
                    
                    # Skip empty results
                    if len(combined_prices) == 0:
                        continue
                        
                    # Add to combined results
                    all_final_prices.extend(combined_prices)
                    
                    # Store model-specific results
                    if horizon not in all_model_results:
                        all_model_results[horizon] = {}
                        
                    all_model_results[horizon][window] = {
                        "volatility": float(model_volatility),
                        "expected_price": float(np.mean(combined_prices)),
                        "price_std": float(np.std(combined_prices)),
                        "num_simulations": len(combined_prices)
                    }
                except Exception as e:
                    logger.error(f"Error in model simulation for window {window}: {e}")
                    logger.error(traceback.format_exc())
        
        return all_final_prices, all_model_results
    
    def _simulate_model_combination(self, horizon: int, num_paths: int, 
                                 model_volatility: float, model_variance: float) -> Tuple[List[float], float]:
        """
        Run simulation for a specific volatility model combination.
        """
        try:
            logger.info(f"  - Simulating with volatility {model_volatility:.4f} for {horizon} days")
            
            # For GPU simulations, run with larger batch size
            if DEVICE.type != 'cpu' and num_paths * horizon > 5000:
                # Run default model
                if self.use_heston:
                    default_model_prices = self.simulate_heston(
                        horizon, num_paths, volatility=model_volatility, variance=model_variance
                    )
                else:
                    default_model_prices = self.simulate_gbm(
                        horizon, num_paths, volatility=model_volatility
                    )
                
                # Run SABR-CGMY
                sabr_cgmy_prices = self.simulate_sabr_cgmy(
                    horizon, num_paths, vol_override=model_volatility
                )
            else:
                # Run with parallelization for CPU
                if self.use_heston:
                    default_model_prices = run_parallel_simulation(
                        self.simulate_heston, horizon, num_paths, max(2, self.max_threads // 2),
                        volatility=model_volatility, variance=model_variance
                    )
                else:
                    default_model_prices = run_parallel_simulation(
                        self.simulate_gbm, horizon, num_paths, max(2, self.max_threads // 2),
                        volatility=model_volatility
                    )
                
                # Run SABR-CGMY with parallelization
                sabr_cgmy_prices = run_parallel_simulation(
                    self.simulate_sabr_cgmy, horizon, num_paths, max(2, self.max_threads // 2),
                    vol_override=model_volatility
                )
            
            # Combine results
            combined_prices = np.concatenate([default_model_prices, sabr_cgmy_prices])
            
            return combined_prices, model_volatility
            
        except Exception as e:
            logger.error(f"Error in model combination simulation: {e}")
            logger.error(traceback.format_exc())
            return [], model_volatility
    
    def _store_simulation_results(self, horizon: int, final_prices: Union[List, np.ndarray], model_details: Dict = None) -> None:
        """
        Calculate statistics and store simulation results for a time horizon.
        Includes proper type checking and conversion.
        """
        # Ensure final_prices is a numpy array
        if isinstance(final_prices, list):
            if not final_prices:  # Empty list
                logger.warning(f"No valid simulation results for horizon {horizon}")
                return
            final_prices = np.array(final_prices, dtype=np.float64)
        
        # Calculate statistics
        try:
            expected_price = float(np.mean(final_prices))
            price_std = float(np.std(final_prices))
            percentiles = np.percentile(final_prices, [1, 5, 10, 25, 50, 75, 90, 95, 99])
            
            # Calculate probability metrics
            prob_increase = float(np.mean(final_prices > self.current_price))
            prob_up_5pct = float(np.mean(final_prices > self.current_price * 1.05))
            prob_up_10pct = float(np.mean(final_prices > self.current_price * 1.10))
            prob_up_20pct = float(np.mean(final_prices > self.current_price * 1.20))
            prob_down_5pct = float(np.mean(final_prices < self.current_price * 0.95))
            prob_down_10pct = float(np.mean(final_prices < self.current_price * 0.90))
            prob_down_20pct = float(np.mean(final_prices < self.current_price * 0.80))
            
            # Store results
            result = {
                "expected_price": expected_price,
                "price_std": price_std,
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
                    "increase": prob_increase,
                    "up_5pct": prob_up_5pct,
                    "up_10pct": prob_up_10pct,
                    "up_20pct": prob_up_20pct,
                    "down_5pct": prob_down_5pct,
                    "down_10pct": prob_down_10pct,
                    "down_20pct": prob_down_20pct
                }
            }
            
            # Add model details if available
            if model_details:
                result["model_details"] = model_details
            
            # Store in results dictionary
            self.simulation_results[horizon] = result
        except Exception as e:
            logger.error(f"Error calculating statistics for horizon {horizon}: {e}")
            logger.error(traceback.format_exc())

    def get_results_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the simulation results.
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
        
        if self.use_multiple_volatility_models and self.volatility_models and self.best_timeframe:
            summary["volatility_models"] = {
                "num_models": len(self.volatility_models),
                "best_timeframe": self.best_timeframe
            }
        
        for horizon, results in self.simulation_results.items():
            expected_price = results["expected_price"]
            expected_change_pct = ((expected_price / self.current_price) - 1) * 100
            lower_bound = results["percentiles"].get("p5", None)
            upper_bound = results["percentiles"].get("p95", None)
            prob_increase = results["probabilities"].get("increase", 0.5) * 100
            prob_up_10pct = results["probabilities"].get("up_10pct", 0) * 100
            prob_down_10pct = results["probabilities"].get("down_10pct", 0) * 100
            summary["horizons"][horizon] = {
                "days": horizon,
                "expected_price": expected_price,
                "expected_change_pct": expected_change_pct,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "prob_increase": prob_increase,
                "prob_up_10pct": prob_up_10pct,
                "prob_down_10pct": prob_down_10pct
            }
        
        return summary