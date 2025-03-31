# /src/simulation/report.py
"""
Report generation for Monte Carlo simulations.
"""

import traceback
import logging
import numpy as np
from typing import Dict, Any, List, Optional

from src.simulation.utils import format_time_horizon
from ttontt import run_async_in_new_loop

logger = logging.getLogger(__name__)

def generate_monte_carlo_report_task(ticker: str, score: float, details: dict) -> dict:
    """
    Generate a Monte Carlo simulation report for a ticker.
    
    Args:
        ticker (str): The stock ticker symbol.
        score (float): The technical score.
        details (dict): Details from technical analysis.
        
    Returns:
        dict: Technical report enhanced with Monte Carlo data.
    """
    try:
        # Import the Monte Carlo simulator class
        from src.simulation.montecarlo import MonteCarloSimulator
        import asyncio

        # Initialize simulator with multiple volatility models
        simulator = MonteCarloSimulator(
            ticker=ticker,
            num_simulations=1000,
            time_horizons=[1, 5, 10, 21, 63, 126, 252],  # 1D, 1W, 2W, 1M, 3M, 6M, 1Y
            use_heston=True,
            use_multiple_volatility_models=True
        )
        
        # Run calibration with extended timeout (300 seconds)
        async def run_calibration():
            return await simulator.calibrate_model_parameters()
        
        calibration_success = run_async_in_new_loop(run_calibration(), timeout=300)
        if not calibration_success:
            logger.warning(f"Failed to calibrate model for {ticker}. Cannot run simulation.")
            return {
                "ticker": ticker,
                "technical_score": float(score),
                "category_scores": details.get("category_scores", {}),
                "signals": details.get("signals", []),
                "warnings": details.get("warnings", []) + ["Monte Carlo error: Failed to calibrate model"],
                "report": details,
                "monte_carlo_error": "Failed to calibrate model"
            }
        
        # Run simulation with extended timeout (300 seconds)
        async def run_simulation():
            return await simulator.run_simulation()
        
        simulation_results = run_async_in_new_loop(run_simulation(), timeout=300)
        if not simulation_results:
            logger.warning(f"No simulation results for {ticker}")
            return {
                "ticker": ticker,
                "technical_score": float(score),
                "category_scores": details.get("category_scores", {}),
                "signals": details.get("signals", []),
                "warnings": details.get("warnings", []) + ["Monte Carlo error: No simulation results"],
                "report": details,
                "monte_carlo_error": "No simulation results"
            }
        
        # Get results summary
        summary = simulator.get_results_summary()
        
        # Format report
        mc_report = {
            "ticker": ticker,
            "current_price": summary["current_price"],
            "annualized_volatility": summary["annualized_volatility"],
            "time_horizons": [],
            "simulation_details": {
                # (Assuming here the config tier information is set elsewhere.)
                "config_tier": "comprehensive" if simulator.use_multiple_volatility_models else "single",
                "num_simulations": simulator.num_simulations,
                "time_horizons": simulator.time_horizons,
                "use_heston": simulator.use_heston
            }
        }
        
        if "volatility_models" in summary:
            mc_report["volatility_models"] = summary["volatility_models"]
        
        # Add each time horizon to the report
        for days, horizon_data in summary["horizons"].items():
            mc_report["time_horizons"].append({
                "days": days,
                "label": format_time_horizon(days),
                "expected_price": horizon_data["expected_price"],
                "expected_change_pct": horizon_data["expected_change_pct"],
                "lower_bound": horizon_data["lower_bound"],
                "upper_bound": horizon_data["upper_bound"],
                "prob_increase": horizon_data["prob_increase"],
                "prob_up_10pct": horizon_data["prob_up_10pct"],
                "prob_down_10pct": horizon_data["prob_down_10pct"]
            })
        
        # Get the 1-month horizon as the primary projection
        time_horizons = mc_report.get("time_horizons", [])
        horizon_1m = next((h for h in time_horizons if h["days"] == 21), None)
        
        # Enhance raw indicators with Monte Carlo data
        raw_indicators = details.get("raw_indicators", {})
        if horizon_1m:
            raw_indicators["mc_expected_price"] = horizon_1m["expected_price"]
            raw_indicators["mc_expected_change_pct"] = horizon_1m["expected_change_pct"]
            raw_indicators["mc_lower_bound"] = horizon_1m["lower_bound"]
            raw_indicators["mc_upper_bound"] = horizon_1m["upper_bound"]
            raw_indicators["mc_prob_increase"] = horizon_1m["prob_increase"]
            raw_indicators["mc_prob_up_10pct"] = horizon_1m["prob_up_10pct"]
            raw_indicators["mc_prob_down_10pct"] = horizon_1m["prob_down_10pct"]
        
        # Create enhanced report
        enhanced_report = {
            "ticker": ticker,
            "technical_score": float(score),
            "category_scores": details.get("category_scores", {}),
            "signals": details.get("signals", []),
            "warnings": details.get("warnings", []),
            "report": {
                "raw_indicators": raw_indicators,
                "category_scores": details.get("category_scores", {}),
                "monte_carlo": {
                    "expected_price": horizon_1m["expected_price"] if horizon_1m else None,
                    "expected_change_pct": horizon_1m["expected_change_pct"] if horizon_1m else None,
                    "lower_bound": horizon_1m["lower_bound"] if horizon_1m else None,
                    "upper_bound": horizon_1m["upper_bound"] if horizon_1m else None,
                    "prob_increase": horizon_1m["prob_increase"] if horizon_1m else None,
                    "prob_up_10pct": horizon_1m["prob_up_10pct"] if horizon_1m else None,
                    "prob_down_10pct": horizon_1m["prob_down_10pct"] if horizon_1m else None,
                    "time_horizons": time_horizons,
                    "volatility_models": mc_report.get("volatility_models"),
                    "simulation_details": mc_report.get("simulation_details")
                }
            }
        }
        
        return enhanced_report
        
    except Exception as e:
        logger.error(f"Error generating Monte Carlo report for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return {
            "ticker": ticker,
            "technical_score": float(score),
            "category_scores": details.get("category_scores", {}),
            "signals": details.get("signals", []),
            "warnings": details.get("warnings", []) + [f"Monte Carlo error: {str(e)}"],
            "report": details,
            "monte_carlo_error": str(e)
        }   


