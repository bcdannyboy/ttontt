"""
Visualization and Reporting Functions for Monte Carlo Simulation
==============================================================

This module provides functions for generating visualizations and reports from
Monte Carlo simulation results.
"""

import os
import numpy as np
import logging
import asyncio
from typing import List, Dict, Any, Tuple, Union, Optional

# Configure logging
logger = logging.getLogger(__name__)

async def generate_stock_report(ticker: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive Monte Carlo report for a stock.
    
    Args:
        ticker: Stock ticker symbol
        config: Optional configuration parameters
        
    Returns:
        Dictionary with report data
    """
    try:
        from src.simulation.utils import _format_time_horizon
        from src.simulation.montecarlo import MonteCarloSimulator
        
        # Use default config if none provided
        if config is None:
            config = {}
            
        # Create simulator
        simulator = MonteCarloSimulator(
            ticker=ticker,
            num_simulations=config.get('num_simulations', 500),
            time_horizons=config.get('time_horizons', [1, 5, 10, 21, 63, 126, 252]),
            use_heston=config.get('use_heston', True),
            max_threads=config.get('max_threads', min(os.cpu_count(), 24))
        )
        
        # Run calibration and simulation
        if not await simulator.calibrate_model_parameters():
            logger.warning(f"Failed to calibrate model for {ticker}")
            return {"error": f"Failed to calibrate model for {ticker}"}
            
        await simulator.run_simulation()
        
        # Get summary results
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
        
        # Format time horizons
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
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}

def screen_stocks(tickers: List[str], config: Dict[str, Any] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Screen stocks using Monte Carlo simulation to predict future price movements.
    
    Args:
        tickers: List of ticker symbols to screen
        config: Optional configuration parameters
        
    Returns:
        List of tuples with (ticker, score, details)
    """
    from src.simulation.parallel import run_monte_carlo_analysis
    
    # Run Monte Carlo analysis for all tickers
    monte_carlo_results = run_monte_carlo_analysis(tickers, config)
    
    results = []
    for ticker, report in monte_carlo_results.items():
        # Skip tickers with errors
        if "error" in report:
            logger.warning(f"Skipping {ticker} due to error: {report['error']}")
            continue
        
        # Get time horizons
        time_horizons = report.get("horizons", {})
        
        # Find key horizons
        horizon_1m = next((h for t, h in time_horizons.items() if h["days"] == 21), None)
        horizon_3m = next((h for t, h in time_horizons.items() if h["days"] == 63), None)
        horizon_6m = next((h for t, h in time_horizons.items() if h["days"] == 126), None)
        
        # Skip if missing horizons
        if not all([horizon_1m, horizon_3m, horizon_6m]):
            logger.warning(f"Skipping {ticker} due to missing time horizons")
            continue
        
        # Calculate score components
        score_components = [
            horizon_1m["expected_change_pct"] * 0.2,
            horizon_3m["expected_change_pct"] * 0.3,
            horizon_6m["expected_change_pct"] * 0.5,
            horizon_1m["prob_increase"] * 0.1,
            horizon_3m["prob_increase"] * 0.15,
            horizon_6m["prob_increase"] * 0.25,
            (horizon_1m["prob_up_10pct"] - horizon_1m["prob_down_10pct"]) * 0.2
        ]
        
        # Calculate total score
        score = sum(score_components)
        
        # Prepare details
        details = {
            "current_price": report["current_price"],
            "annualized_volatility": report["annualized_volatility"],
            "time_horizons": {_format_time_horizon(h["days"]): h for h in time_horizons.values()},
            "raw_simulations": report.get("raw_simulations", {}),
            "category_scores": {
                "expected_return": (horizon_1m["expected_change_pct"] * 0.2 +
                                  horizon_3m["expected_change_pct"] * 0.3 +
                                  horizon_6m["expected_change_pct"] * 0.5),
                "probability": ((horizon_1m["prob_increase"] - 50) * 0.25 +
                              (horizon_3m["prob_increase"] - 50) * 0.35 +
                              (horizon_6m["prob_increase"] - 50) * 0.4),
                "risk_reward": ((horizon_1m["prob_up_10pct"] - horizon_1m["prob_down_10pct"]) * 0.3 +
                              (horizon_3m["prob_up_10pct"] - horizon_3m["prob_down_10pct"]) * 0.3 +
                              (horizon_6m["prob_up_10pct"] - horizon_6m["prob_down_10pct"]) * 0.4)
            }
        }
        
        # Add to results
        results.append((ticker, score, details))
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

def _format_time_horizon(days: int) -> str:
    """
    Format a time horizon in days to a human-readable string.
    Helper function imported from utils to avoid circular imports.
    
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

def create_enhanced_monte_carlo_visualization(technical_results: List[tuple]) -> Dict[str, Any]:
    """
    Create an enhanced visualization of Monte Carlo simulation results
    integrated with technical analysis results.
    
    Args:
        technical_results: List of technical analysis results
        
    Returns:
        Dictionary with visualization data
    """
    visualization_data = {
        "tickers": [],
        "scores": [],
        "current_prices": [],
        "expected_prices": [],
        "lower_bounds": [],
        "upper_bounds": [],
        "prob_increases": [],
        "category_scores": {},
        "signals": [],
        "warnings": []
    }
    
    for ticker, score, details in technical_results:
        raw_indicators = details.get("raw_indicators", {})
        current_price = raw_indicators.get("current_price")
        
        if not current_price:
            continue
            
        monte_carlo = details.get("monte_carlo", {})
        
        # Add data to visualization
        visualization_data["tickers"].append(ticker)
        visualization_data["scores"].append(round(score, 4))
        visualization_data["current_prices"].append(round(current_price, 2))
        visualization_data["expected_prices"].append(monte_carlo.get("expected_price", current_price))
        visualization_data["lower_bounds"].append(monte_carlo.get("lower_bound", current_price * 0.9))
        visualization_data["upper_bounds"].append(monte_carlo.get("upper_bound", current_price * 1.1))
        visualization_data["prob_increases"].append(monte_carlo.get("prob_increase", 50))
        
        # Add signals and warnings
        signals = details.get("signals", [])
        warnings = details.get("warnings", [])
        visualization_data["signals"].append(signals[:2] if signals else [])
        visualization_data["warnings"].append(warnings[:2] if warnings else [])
        
        # Add category scores
        if "category_scores" in details:
            for category, score_val in details["category_scores"].items():
                if category not in visualization_data["category_scores"]:
                    visualization_data["category_scores"][category] = []
                visualization_data["category_scores"][category].append(round(score_val, 4))
    
    return visualization_data

def generate_price_path_sample(ticker: str, time_horizon: int = 63, num_paths: int = 10) -> Dict[str, Any]:
    """
    Generate a sample of price paths for visualization purposes.
    Imported from utils.py to avoid circular imports.
    
    Args:
        ticker: Stock ticker symbol
        time_horizon: Number of days to simulate
        num_paths: Number of paths to generate
        
    Returns:
        Dictionary with price path data
    """
    try:
        import asyncio
        from src.simulation.montecarlo import MonteCarloSimulator
        from src.simulation.utils import calculate_statistics
        
        # Create simulator
        simulator = MonteCarloSimulator(
            ticker=ticker,
            num_simulations=num_paths,
            time_horizons=[time_horizon],
            use_heston=True
        )
        
        # Calibrate model
        calibration_success = asyncio.run(simulator.calibrate_model_parameters())
        if not calibration_success:
            return {"error": f"Failed to calibrate model for {ticker}"}
        
        # Set simulation parameters
        dt = 1.0 / 252
        sqrt_dt = np.sqrt(dt)
        
        # Pre-allocate arrays
        prices = np.zeros((num_paths, time_horizon + 1))
        variances = np.zeros((num_paths, time_horizon + 1))
        
        # Set initial values
        prices[:, 0] = simulator.current_price
        variances[:, 0] = simulator.v0
        
        # Generate random variables for all paths at once
        z1 = np.random.normal(0, 1, size=(num_paths, time_horizon))
        z2 = np.random.normal(0, 1, size=(num_paths, time_horizon))
        z2_correlated = simulator.rho * z1 + np.sqrt(1 - simulator.rho**2) * z2
        
        # Simulate paths
        for t in range(time_horizon):
            # Ensure positive variance
            variances[:, t] = np.maximum(variances[:, t], 1e-8)
            
            # Price update
            price_drift = (simulator.annualized_drift - 0.5 * variances[:, t]) * dt
            price_diffusion = np.sqrt(variances[:, t]) * sqrt_dt * z1[:, t]
            prices[:, t+1] = prices[:, t] * np.exp(price_drift + price_diffusion)
            
            # Variance update
            var_drift = simulator.kappa * (simulator.theta - variances[:, t]) * dt
            var_diffusion = simulator.sigma * np.sqrt(variances[:, t]) * sqrt_dt * z2_correlated[:, t]
            variances[:, t+1] = variances[:, t] + var_drift + var_diffusion
        
        # Create path data for visualization
        days = list(range(time_horizon + 1))
        paths = []
        
        for i in range(num_paths):
            paths.append({
                "path_id": i,
                "days": days,
                "prices": prices[i, :].tolist()
            })
        
        # Calculate statistics
        final_prices = prices[:, -1]
        price_stats = calculate_statistics(final_prices)
        percentiles = np.percentile(final_prices, [5, 25, 50, 75, 95])
        
        # Return results
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
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}