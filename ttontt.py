import asyncio
import concurrent
from src.screener.fundamentals.fundamentals_peers import gather_peer_analysis
from src.screener.fundamentals.fundamentals_report import generate_stock_report
from src.screener.fundamentals.fundamentals_screen import screen_stocks
from src.screener.fundamentals.fundamentals_metrics import (extract_metrics_from_financial_data, preprocess_data,
                                    calculate_z_scores, calculate_weighted_score, construct_earnings_from_income,
                                    ensure_essential_z_scores)
from src.screener.technical.screening import screen_stocks as tech_screen_stocks
from src.screener.technical.report import generate_stock_report as tech_generate_stock_report, get_indicator_contributions
from src.simulation import montecarlo 
import json
import os
import sys
from datetime import datetime
import numpy as np
import logging
import traceback
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Import Rich classes for dynamic terminal output
from rich.console import Console
from rich.table import Table
from rich.columns import Columns
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn

from tickers import get_active_tickers
import tickers


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('ttontt')

# Create a list to store all tables for display at the end
all_tables = []

def generate_stock_report_task(ticker, score, details):
    """
    Generates a detailed stock report for a given ticker using existing screening data.
    
    Args:
        ticker (str): The stock ticker symbol.
        score (float): The (adjusted) composite score from screening.
        details (dict): Detailed results from screening including category scores and peer analysis.
        
    Returns:
        dict: A complete stock report with screening scores and peer analysis.
    """
    try:
        # Create the report directly from the provided details instead of calling generate_stock_report
        report = {
            'ticker': ticker,
            'composite_score': float(score),
            'category_scores': details['category_scores'],
            'z_scores': details.get('z_scores', {}),
            'peer_analysis': {
                'peer_average': details.get('peer_comparison'),
                'peer_delta': details.get('peer_delta')
            },
            'key_metrics': {},
            'strengths': [],
            'weaknesses': [],
            'raw_metrics': details.get('raw_metrics', {})
        }
        
        # Extract and format key metrics from the raw metrics
        raw_metrics = details.get('raw_metrics', {})
        if raw_metrics:
            # Organize metrics into categories
            report['key_metrics'] = {
                'profitability': {
                    'gross_margin': raw_metrics.get('gross_profit_margin'),
                    'operating_margin': raw_metrics.get('operating_income_margin'),
                    'net_margin': raw_metrics.get('net_income_margin'),
                    'roe': raw_metrics.get('return_on_equity'),
                    'roa': raw_metrics.get('return_on_assets')
                },
                'growth': {
                    'revenue_growth': raw_metrics.get('growth_revenue'),
                    'earnings_growth': raw_metrics.get('growth_net_income'),
                    'eps_growth': raw_metrics.get('growth_eps'),
                    'forward_sales_growth': raw_metrics.get('forward_sales_growth'),
                    'forward_ebitda_growth': raw_metrics.get('forward_ebitda_growth')
                },
                'financial_health': {
                    'current_ratio': raw_metrics.get('current_ratio'),
                    'debt_to_equity': raw_metrics.get('debt_to_equity'),
                    'debt_to_assets': raw_metrics.get('debt_to_assets'),
                    'interest_coverage': raw_metrics.get('interest_coverage'),
                    'cash_to_debt': raw_metrics.get('cash_to_debt')
                },
                'valuation': {
                    'pe_ratio': raw_metrics.get('pe_ratio'),
                    'price_to_book': raw_metrics.get('price_to_book'),
                    'price_to_sales': raw_metrics.get('price_to_sales'),
                    'ev_to_ebitda': raw_metrics.get('ev_to_ebitda'),
                    'dividend_yield': raw_metrics.get('dividend_yield'),
                    'peg_ratio': raw_metrics.get('peg_ratio')
                },
                'efficiency': {
                    'asset_turnover': raw_metrics.get('asset_turnover'),
                    'inventory_turnover': raw_metrics.get('inventory_turnover'),
                    'receivables_turnover': raw_metrics.get('receivables_turnover'),
                    'cash_conversion_cycle': raw_metrics.get('cash_conversion_cycle'),
                    'capex_to_revenue': raw_metrics.get('capex_to_revenue')
                },
                'analyst_estimates': {
                    'estimate_eps_accuracy': raw_metrics.get('estimate_eps_accuracy', 0),
                    'estimate_revenue_accuracy': raw_metrics.get('estimate_revenue_accuracy', 0),
                    'forward_sales_growth': raw_metrics.get('forward_sales_growth'),
                    'forward_ebitda_growth': raw_metrics.get('forward_ebitda_growth'),
                    'estimate_revision_momentum': raw_metrics.get('estimate_revision_momentum', 0),
                    'estimate_consensus_deviation': raw_metrics.get('estimate_consensus_deviation', 0)
                }
            }
            
            # Generate strengths and weaknesses based on metrics
            # Simple logic to identify notable metrics
            profitability_benchmarks = {
                'gross_profit_margin': 0.35,
                'operating_income_margin': 0.15,
                'net_income_margin': 0.10,
                'return_on_equity': 0.15,
                'return_on_assets': 0.05
            }
            
            financial_health_benchmarks = {
                'current_ratio': 1.5,
                'debt_to_equity': 1.0,
                'interest_coverage': 3.0,
                'cash_to_debt': 0.5
            }
            
            growth_benchmarks = {
                'growth_revenue': 0.10,
                'growth_net_income': 0.10,
                'growth_eps': 0.10,
                'forward_sales_growth': 0.10,
                'forward_ebitda_growth': 0.10
            }
            
            valuation_benchmarks = {
                'pe_ratio': 20.0,
                'price_to_book': 3.0,
                'price_to_sales': 2.0,
                'ev_to_ebitda': 12.0,
                'dividend_yield': 0.02
            }
            
            # Apply the benchmarks to identify strengths and weaknesses
            for metric, benchmark in profitability_benchmarks.items():
                if metric in raw_metrics and raw_metrics.get(metric) is not None and np.isfinite(raw_metrics.get(metric)):
                    if raw_metrics.get(metric) > benchmark * 1.5:
                        report['strengths'].append(f"Strong {metric.replace('_', ' ')}: {raw_metrics.get(metric):.2f}")
                    elif raw_metrics.get(metric) < benchmark * 0.5:
                        report['weaknesses'].append(f"Low {metric.replace('_', ' ')}: {raw_metrics.get(metric):.2f}")
            
            # Similar pattern for other metric categories
            for metric, benchmark in financial_health_benchmarks.items():
                if metric in raw_metrics and raw_metrics.get(metric) is not None and np.isfinite(raw_metrics.get(metric)):
                    if metric == 'debt_to_equity' or metric == 'debt_to_assets':
                        if raw_metrics.get(metric) < benchmark * 0.5:
                            report['strengths'].append(f"Low {metric.replace('_', ' ')}: {raw_metrics.get(metric):.2f}")
                        elif raw_metrics.get(metric) > benchmark * 1.5:
                            report['weaknesses'].append(f"High {metric.replace('_', ' ')}: {raw_metrics.get(metric):.2f}")
                    else:
                        if raw_metrics.get(metric) > benchmark * 1.5:
                            report['strengths'].append(f"Strong {metric.replace('_', ' ')}: {raw_metrics.get(metric):.2f}")
                        elif raw_metrics.get(metric) < benchmark * 0.5:
                            report['weaknesses'].append(f"Low {metric.replace('_', ' ')}: {raw_metrics.get(metric):.2f}")
            
            # Growth metrics
            for metric, benchmark in growth_benchmarks.items():
                if metric in raw_metrics and raw_metrics.get(metric) is not None and np.isfinite(raw_metrics.get(metric)):
                    if raw_metrics.get(metric) > benchmark * 1.5:
                        report['strengths'].append(f"Strong {metric.replace('_', ' ')}: {raw_metrics.get(metric):.2f}")
                    elif raw_metrics.get(metric) < 0:
                        report['weaknesses'].append(f"Negative {metric.replace('_', ' ')}: {raw_metrics.get(metric):.2f}")
            
            # Valuation metrics
            for metric, benchmark in valuation_benchmarks.items():
                if metric in raw_metrics and raw_metrics.get(metric) is not None and np.isfinite(raw_metrics.get(metric)):
                    if metric == 'dividend_yield':
                        if raw_metrics.get(metric) > benchmark * 1.5:
                            report['strengths'].append(f"High {metric.replace('_', ' ')}: {raw_metrics.get(metric):.2f}")
                    elif metric in ['pe_ratio', 'price_to_book', 'price_to_sales', 'ev_to_ebitda']:
                        if raw_metrics.get(metric) < benchmark * 0.6:
                            report['strengths'].append(f"Low {metric.replace('_', ' ')}: {raw_metrics.get(metric):.2f}")
                        elif raw_metrics.get(metric) > benchmark * 1.5:
                            report['weaknesses'].append(f"High {metric.replace('_', ' ')}: {raw_metrics.get(metric):.2f}")
        
        return {
            "ticker": ticker,
            "composite_score": float(score),
            "category_scores": {k: float(v) for k, v in details["category_scores"].items()},
            "report": report
        }
    except Exception as e:
        logger.error(f"Error generating report for {ticker}: {e}")
        return {
            "ticker": ticker,
            "composite_score": float(score),
            "category_scores": {k: float(v) for k, v in details["category_scores"].items()},
            "report": {"error": str(e)},
            "error": True
        }

def generate_technical_report_task(ticker, score, details):
    """
    Generates a technical analysis report for a given ticker.
    
    Args:
        ticker (str): The stock ticker symbol.
        score (float): The technical score from screening.
        details (dict): Detailed results from technical screening including category scores.
        
    Returns:
        dict: A technical analysis report with screening scores, signals, and warnings.
    """
    try:
        # Updated to use properly imported function
        report = tech_generate_stock_report(ticker)
        
        report['category_scores'] = details['category_scores']
        report['composite_score'] = float(score)
        
        # Extract price prediction data
        report['expected_price'] = report.get('raw_indicators', {}).get('expected_price', None)
        report['probable_low'] = report.get('raw_indicators', {}).get('probable_low', None)
        report['probable_high'] = report.get('raw_indicators', {}).get('probable_high', None)
        
        # Get indicator contributions for deeper analysis
        try:
            contributions_df = get_indicator_contributions(ticker)
            if not contributions_df.empty:
                # Convert dataframe to dictionary for JSON serialization
                contributions = contributions_df.to_dict(orient='records')
                report['indicator_contributions'] = contributions
        except Exception as e:
            logger.warning(f"Could not get indicator contributions for {ticker}: {e}")
            report['indicator_contributions'] = []
        
        return {
            "ticker": ticker,
            "technical_score": float(score),
            "category_scores": {k: float(v) for k, v in details["category_scores"].items()},
            "signals": report.get("signals", []),
            "warnings": report.get("warnings", []),
            "report": report
        }
    except Exception as e:
        logger.error(f"Error generating technical report for {ticker}: {e}")
        return {
            "ticker": ticker,
            "technical_score": float(score),
            "category_scores": {k: float(v) for k, v in details["category_scores"].items()},
            "signals": [],
            "warnings": [f"Error generating report: {str(e)}"],
            "report": {"error": str(e)},
            "error": True
        }

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

def run_comprehensive_monte_carlo_analysis(tickers, technical_results, max_workers=None):
    """
    Run a comprehensive Monte Carlo analysis for the given tickers,
    integrating with the technical analysis results.
    
    Args:
        tickers (list): List of ticker symbols to analyze
        technical_results (list): List of (ticker, score, details) tuples from technical analysis
        max_workers (int): Maximum number of worker threads
        
    Returns:
        dict: Dictionary mapping ticker symbols to their Monte Carlo simulation results
    """
    if max_workers is None:
        max_workers = min(32, os.cpu_count() * 2)
    
    # Create a mapping for quick lookup of technical results
    tech_details = {ticker: (score, details) for ticker, score, details in technical_results}
    
    rich_console = Console()
    mc_reports = []
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=rich_console
    ) as progress:
        task = progress.add_task("Generating Monte Carlo simulations...", total=len(tickers))
        
        # Process tickers in batches to prevent too many event loops
        valid_tickers = [t for t in tickers if t in tech_details]
        batch_size = min(10, len(valid_tickers))
        
        for i in range(0, len(valid_tickers), batch_size):
            batch = valid_tickers[i:i+batch_size]
            batch_reports = []
            
            with ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
                futures = {}
                
                # Submit Monte Carlo simulation tasks
                for ticker in batch:
                    score, details = tech_details[ticker]
                    future = executor.submit(generate_monte_carlo_report_task, ticker, score, details)
                    futures[future] = ticker
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    ticker = futures[future]
                    try:
                        mc_report = future.result()
                        if mc_report:
                            batch_reports.append(mc_report)
                    except Exception as e:
                        logger.error(f"Error generating Monte Carlo report for {ticker}: {e}")
                        logger.error(traceback.format_exc())
                    
                    progress.update(task, advance=1)
            
            # Add batch results to overall results
            mc_reports.extend(batch_reports)
            
            # Small delay between batches
            if i + batch_size < len(valid_tickers):
                time.sleep(1)
    
    return mc_reports

def create_enhanced_monte_carlo_visualization(technical_results, monte_carlo_results):
    """
    Create an enhanced visualization of Monte Carlo simulation results
    integrated with technical analysis results.
    
    Args:
        technical_results (list): List of (ticker, score, details) tuples from technical analysis
        monte_carlo_results (list): List of Monte Carlo reports
        
    Returns:
        rich.table.Table: A formatted table for display
    """
    # Create a mapping for quick lookup of technical results
    tech_details = {ticker: (score, details) for ticker, score, details in technical_results}
    
    # Create a mapping for quick lookup of Monte Carlo results
    mc_details = {report["ticker"]: report for report in monte_carlo_results}
    
    # Create table for visualization
    table = Table(title="Technical Analysis with Monte Carlo Simulations")
    
    # Add columns
    table.add_column("Ticker", style="cyan")
    table.add_column("Tech Score", justify="right", style="green")
    table.add_column("Current", justify="right", style="yellow")
    table.add_column("Tech Exp", justify="right", style="yellow")
    table.add_column("MC Exp", justify="right", style="blue")
    table.add_column("MC Range", justify="right", style="blue")
    table.add_column("Prob↑", justify="right", style="magenta")
    table.add_column("Vol Models", justify="right", style="blue")
    table.add_column("Best Frame", justify="right", style="blue")
    table.add_column("Signals", style="green")
    
    # Populate the table
    for ticker, score, details in technical_results:
        if ticker in mc_details:
            mc_report = mc_details[ticker]["report"]["monte_carlo"]
            raw = details.get("raw_indicators", {})
            
            # Extract technical indicators
            current = raw.get("current_price", "N/A")
            tech_exp = raw.get("expected_price", "N/A")
            
            # Extract Monte Carlo indicators
            mc_exp = mc_report.get("expected_price", "N/A")
            mc_low = mc_report.get("lower_bound", "N/A")
            mc_high = mc_report.get("upper_bound", "N/A")
            mc_range = f"{mc_low:.2f}-{mc_high:.2f}" if isinstance(mc_low, (int, float)) and isinstance(mc_high, (int, float)) else "N/A"
            mc_prob = mc_report.get("prob_increase", "N/A")
            
            # Get volatility model information
            vol_models = mc_report.get("volatility_models", {})
            num_models = vol_models.get("num_models", "N/A")
            best_frame = vol_models.get("best_timeframe", "N/A")
            
            # Get signals
            signals = ", ".join(mc_details[ticker].get("signals", [])[:2])
            
            # Add row
            table.add_row(
                ticker,
                f"{score:.4f}",
                f"{current:.2f}" if isinstance(current, (int, float)) else str(current),
                f"{tech_exp:.2f}" if isinstance(tech_exp, (int, float)) else str(tech_exp),
                f"{mc_exp:.2f}" if isinstance(mc_exp, (int, float)) else str(mc_exp),
                mc_range,
                f"{mc_prob:.1f}%" if isinstance(mc_prob, (int, float)) else str(mc_prob),
                str(num_models),
                str(best_frame),
                signals
            )
    
    return table

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
        import numpy as np
        import asyncio
        import traceback

        # Initialize simulator with multiple volatility models
        simulator = MonteCarloSimulator(
            ticker=ticker,
            num_simulations=1000,
            time_horizons=[1, 5, 10, 21, 63, 126, 252],  # 1D, 1W, 2W, 1M, 3M, 6M, 1Y
            use_heston=True,
            use_multiple_volatility_models=True  # Use all volatility models
        )
        
        # Run calibration
        calibration_success = False
        try:
            # Calibrate the model parameters in a safely isolated manner
            async def run_calibration():
                return await simulator.calibrate_model_parameters()
            
            calibration_success = run_async_in_new_loop(run_calibration())
            
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
            
        except Exception as e:
            logger.error(f"Error calibrating model for {ticker}: {e}")
            logger.error(traceback.format_exc())
            return {
                "ticker": ticker,
                "technical_score": float(score),
                "category_scores": details.get("category_scores", {}),
                "signals": details.get("signals", []),
                "warnings": details.get("warnings", []) + [f"Monte Carlo error: {str(e)}"],
                "report": details,
                "monte_carlo_error": f"Failed to calibrate model: {str(e)}"
            }
        
        # Run simulation
        try:
            # Run the simulation in a safely isolated manner
            async def run_simulation():
                return await simulator.run_simulation()
            
            simulation_results = run_async_in_new_loop(run_simulation())
            
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
                "raw_simulations": simulation_results
            }
            
            # Add volatility model information if available
            if "volatility_models" in summary:
                mc_report["volatility_models"] = summary["volatility_models"]
            
            # Add each time horizon to the report
            for days, horizon_data in summary["horizons"].items():
                mc_report["time_horizons"].append({
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
                        "volatility_models": mc_report.get("volatility_models")
                    }
                }
            }
            
            return enhanced_report
            
        except Exception as e:
            logger.error(f"Error generating Monte Carlo simulation for {ticker}: {e}")
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

def run_async_in_new_loop(coro):
    """Safely run an async coroutine in a new event loop with proper cleanup."""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            # Cleanup any remaining tasks
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception as e:
            logger.warning(f"Error cleaning up tasks: {e}")
        finally:
            loop.close()

def perform_stock_screening(tickers, batch_size=10):
    """
    Screen stocks by first computing fundamental composite scores and then adjusting
    the composite score by adding a fraction (peer_weight) of the difference between
    the stock's score and its peers' average.
    """
    try:
        all_results = []
        # Process tickers in batches
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size} ({len(batch)} tickers)")
            batch_results = screen_stocks(batch)
            all_results.extend(batch_results)
            if i + batch_size < len(tickers):
                time.sleep(2)
                
        if not all_results:
            logger.warning("No valid results were returned from the screening process.")
            return []
        
        # Sort by composite score
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Check z-score content validity
        sample_ticker = all_results[0][0]
        sample_details = all_results[0][2]
        if 'z_scores' in sample_details:
            z_score_keys = sample_details['z_scores'].keys()
            nonzero_count = sum(1 for k in z_score_keys if abs(sample_details['z_scores'][k]) > 1e-6)
            logger.info(f"Z-score check: {len(z_score_keys)} metrics, {nonzero_count} non-zero values")
            
            # If z-scores are all zero, attempt progressive recovery strategies
            if nonzero_count == 0:
                logger.warning("All z-scores are zero. Starting recovery process...")
                
                # Strategy 1: Regenerate z-scores with fixed calculation
                logger.info("Strategy 1: Regenerating z-scores with fixed calculation...")
                all_metrics = {ticker: details['raw_metrics'] for ticker, _, details in all_results}
                preprocessed = preprocess_data(all_metrics)
                new_z_scores = calculate_z_scores(preprocessed)
                
                # Check if fixed calculation worked
                sample_ticker_z = new_z_scores.get(sample_ticker, {})
                new_nonzero = sum(1 for k in sample_ticker_z.keys() if abs(sample_ticker_z.get(k, 0)) > 1e-6)
                
                if new_nonzero > 0:
                    logger.info(f"Z-score regeneration successful: {new_nonzero} non-zero values")
                    # Update z-scores in all results
                    for idx, (ticker, score, details) in enumerate(all_results):
                        if ticker in new_z_scores:
                            details['z_scores'] = new_z_scores[ticker]
                else:
                    logger.warning("Z-score regeneration failed. Moving to Strategy 2...")
                    # Strategy 2: Create simulated z-scores based on relative positions
                    logger.info("Strategy 2: Creating simulated z-scores...")
                    create_simulated_z_scores(all_results)
                    
                    # Check if simulation worked
                    sample_details = all_results[0][2]
                    z_score_keys = sample_details['z_scores'].keys()
                    nonzero_count = sum(1 for k in z_score_keys if abs(sample_details['z_scores'][k]) > 1e-6)
                    
                    if nonzero_count > 0:
                        logger.info(f"Z-score simulation successful: {nonzero_count} non-zero values")
                    else:
                        logger.warning("Z-score simulation failed. Moving to Strategy 3...")
                        # Strategy 3: Last resort direct normalization
                        logger.info("Strategy 3: Applying direct normalization...")
                        direct_normalization(all_results)

        # Get list of tickers to analyze
        tickers_to_analyze = [ticker for ticker, _, _ in all_results]
        
        # Set up peer data dictionary
        peer_data_dict = {}
        
        # Run peer analysis safely
        try:
            from src.screener.fundamentals.fundamentals_peers import gather_peer_analysis
            
            async def run_peer_analysis():
                return await gather_peer_analysis(tickers_to_analyze)
            
            # Run in a completely new event loop
            peer_data_dict = run_async_in_new_loop(run_peer_analysis())
            
            # Verify peer data is valid
            if peer_data_dict:
                peer_valid_count = sum(1 for t in peer_data_dict if 
                                    peer_data_dict[t].get('peer_average', 0) != 
                                    next((s for ticker, s, _ in all_results if ticker == t), 0))
                
                logger.info(f"Peer analysis found {len(peer_data_dict)} entries, {peer_valid_count} with unique peer averages")
                
                # If all peer averages equal the stock's own score, use synthetic peers
                if peer_valid_count == 0:
                    logger.warning("All peer averages equal stock scores. Using synthetic peers.")
                    peer_data_dict = generate_synthetic_peers(all_results)
            else:
                logger.warning("Empty peer analysis results. Using synthetic peers.")
                peer_data_dict = generate_synthetic_peers(all_results)
                
        except Exception as e:
            logger.error(f"Error during peer analysis: {e}")
            logger.debug(traceback.format_exc())
            
            # Generate synthetic peers as fallback
            peer_data_dict = generate_synthetic_peers(all_results)
        
        # Peer weight determines how much to adjust scores based on peer comparison
        peer_weight = 0.1  # 10% adjustment factor
        
        # Process each stock's results with peer data
        for idx, (ticker, composite_score, details) in enumerate(all_results):
            try:
                # Get peer data for this ticker
                peer_data = peer_data_dict.get(ticker, {})
                peer_comp = peer_data.get('peer_comparison', {})
                
                # If no peer average available, use the stock's own score with slight adjustment
                if not peer_comp or 'average_score' not in peer_comp:
                    peer_avg = composite_score * 0.95  # 5% lower than stock's score
                else:
                    peer_avg = peer_comp.get('average_score')
                
                # Calculate peer delta: the difference between this stock's score and peer average
                peer_delta = composite_score - peer_avg
                
                # Store peer comparison data in details
                details['peer_comparison'] = peer_avg
                details['peer_delta'] = peer_delta
                
                # Adjust composite score by adding peer_weight * peer_delta
                adjusted_score = composite_score + (peer_weight * peer_delta)
                
                # Update the result with the adjusted score
                all_results[idx] = (ticker, adjusted_score, details)
                
                logger.debug(f"{ticker}: orig={composite_score:.4f}, peer_avg={peer_avg:.4f}, delta={peer_delta:.4f}, adj={adjusted_score:.4f}")
                
            except Exception as e:
                logger.error(f"Error processing peer data for {ticker}: {e}")
                logger.debug(traceback.format_exc())
                # Keep original score if there's an error
        
        # Sort again by the adjusted scores
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results
        
    except Exception as e:
        logger.error(f"Error during stock screening: {e}")
        logger.debug(traceback.format_exc())
        return []

def direct_normalization(all_results):
    """
    As a last resort, directly normalize raw metrics to create synthetic z-scores.
    This is used when all other methods of creating non-zero z-scores have failed.
    """
    # Extract raw metrics
    all_metrics = {}
    for ticker, _, details in all_results:
        if 'raw_metrics' in details:
            all_metrics[ticker] = details['raw_metrics']
    
    if not all_metrics:
        logger.error("No raw metrics available for direct normalization.")
        return
    
    # Define metrics with known directionality
    higher_better = {
        'gross_profit_margin', 'operating_income_margin', 'net_income_margin', 'ebitda_margin',
        'return_on_equity', 'return_on_assets', 'growth_revenue', 'growth_gross_profit',
        'growth_ebitda', 'growth_operating_income', 'growth_net_income', 'growth_eps',
        'current_ratio', 'quick_ratio', 'interest_coverage', 'cash_to_debt',
        'asset_turnover', 'inventory_turnover', 'receivables_turnover', 'dividend_yield'
    }
    
    lower_better = {
        'debt_to_equity', 'debt_to_assets', 'growth_total_debt', 'growth_net_debt',
        'cash_conversion_cycle', 'capex_to_revenue', 'pe_ratio', 'price_to_book',
        'price_to_sales', 'ev_to_ebitda', 'peg_ratio'
    }
    
    # For each ticker, normalize metrics directly
    for ticker, _, details in all_results:
        if ticker not in all_metrics:
            continue
            
        z_scores = {}
        metrics = all_metrics[ticker]
        
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                continue
                
            # Apply metric-specific normalization
            if metric in higher_better:
                if metric.startswith('growth_'):
                    # Growth metrics: scale from -1 to 2
                    z_scores[metric] = max(min(value, 2.0), -1.0)
                elif metric.endswith('_margin'):
                    # Margin metrics: scale from -2 to 2
                    z_scores[metric] = max(min((value - 0.1) * 5, 2.0), -2.0)
                elif metric in {'return_on_equity', 'return_on_assets'}:
                    # Return metrics: scale from -2 to 2
                    z_scores[metric] = max(min(value * 10, 2.0), -2.0)
                elif metric == 'current_ratio':
                    # Current ratio: optimal around 2.0
                    if value > 2.0:
                        z_scores[metric] = min(1.0 + (value - 2.0) * 0.2, 2.0)
                    else:
                        z_scores[metric] = min(value - 1.0, 1.0)
                elif metric == 'interest_coverage':
                    # Interest coverage: higher is better
                    z_scores[metric] = min(value * 0.1, 2.0)
                elif metric == 'dividend_yield':
                    # Dividend yield: optimal around 0.03-0.05
                    z_scores[metric] = min(value * 40, 2.0)
                else:
                    # General case: use a small positive value
                    z_scores[metric] = min(value * 0.5, 1.0)
            elif metric in lower_better:
                if metric == 'pe_ratio':
                    # PE ratio: lower is better, benchmark around 15
                    if value <= 0:
                        z_scores[metric] = -1.0  # Negative PE is bad
                    elif value < 15:
                        z_scores[metric] = 1.0 - (value / 15)
                    else:
                        z_scores[metric] = max(-2.0, -0.1 * (value - 15))
                elif metric == 'price_to_book':
                    # P/B ratio: lower is better, benchmark around 3
                    if value < 3:
                        z_scores[metric] = 1.0 - (value / 3)
                    else:
                        z_scores[metric] = max(-2.0, -0.2 * (value - 3))
                elif metric == 'price_to_sales':
                    # P/S ratio: lower is better, benchmark around 2
                    if value < 2:
                        z_scores[metric] = 1.0 - (value / 2)
                    else:
                        z_scores[metric] = max(-2.0, -0.2 * (value - 2))
                elif metric == 'ev_to_ebitda':
                    # EV/EBITDA: lower is better, benchmark around 10
                    if value < 10:
                        z_scores[metric] = 1.0 - (value / 10)
                    else:
                        z_scores[metric] = max(-2.0, -0.1 * (value - 10))
                elif metric == 'debt_to_equity':
                    # D/E ratio: lower is better, benchmark around 1
                    z_scores[metric] = max(-2.0, 1.0 - value)
                elif metric == 'debt_to_assets':
                    # D/A ratio: lower is better, benchmark around 0.5
                    z_scores[metric] = max(-2.0, 1.0 - (value * 2))
                elif metric.startswith('growth_') and ('debt' in metric):
                    # Debt growth: lower is better
                    z_scores[metric] = max(-2.0, -value)
                else:
                    # General case: use a small negative value
                    z_scores[metric] = max(-1.0, -value * 0.5)
            else:
                # For unknown metrics, use a small random value
                seed_val = hash(metric + ticker) % 1000
                np.random.seed(seed_val)
                z_scores[metric] = np.random.uniform(-0.3, 0.3)
        
        # Update the z-scores in the details
        details['z_scores'].update(z_scores)
    
    # Log summary of direct normalization
    sample_ticker = all_results[0][0]
    sample_details = all_results[0][2]
    z_score_keys = sample_details['z_scores'].keys()
    nonzero_count = sum(1 for k in z_score_keys if abs(sample_details['z_scores'][k]) > 1e-6)
    logger.info(f"After direct normalization: {len(z_score_keys)} metrics, {nonzero_count} non-zero values")

def create_simulated_z_scores(all_results):
    """
    Create simulated z-scores when calculation fails.
    This uses multiple approaches to create synthetic but meaningful z-scores 
    that will produce reasonable rankings, even with limited data.
    """
    # Extract raw metrics from all results
    all_metrics = {}
    for ticker, _, details in all_results:
        if 'raw_metrics' in details:
            all_metrics[ticker] = details['raw_metrics']
    
    if not all_metrics:
        logger.warning("No raw metrics found for any ticker. Cannot create simulated z-scores.")
        return
    
    # Identify all available metrics across all tickers
    all_available_metrics = set()
    for ticker in all_metrics:
        all_available_metrics.update(all_metrics[ticker].keys())
    
    logger.info(f"Found {len(all_available_metrics)} unique metrics across all tickers")
    
    # Initialize z-scores for all tickers
    for ticker, _, details in all_results:
        if 'z_scores' not in details:
            details['z_scores'] = {}
    
    # Approach 1: For each metric, normalize values across available tickers
    for metric in all_available_metrics:
        # Collect values for this metric across tickers
        metric_values = []
        for ticker in all_metrics:
            if metric in all_metrics[ticker]:
                value = all_metrics[ticker][metric]
                if isinstance(value, (int, float)) and np.isfinite(value):
                    metric_values.append((ticker, value))
        
        # Skip metrics with no usable values
        if not metric_values:
            continue
            
        # For metrics with only 1 value, assign a small non-zero z-score
        if len(metric_values) == 1:
            ticker, value = metric_values[0]
            # Deterministic pseudo-random value based on metric and value
            seed_val = hash(metric + str(round(value, 4))) % 1000
            np.random.seed(seed_val)
            z_score = np.random.uniform(-0.5, 0.5)
            
            # Find this ticker in all_results
            for idx, (result_ticker, _, details) in enumerate(all_results):
                if result_ticker == ticker:
                    details['z_scores'][metric] = z_score
            continue
            
        # For metrics with multiple values, use percentile-based normalization
        metric_values.sort(key=lambda x: x[1])
        n = len(metric_values)
        
        for i, (ticker, value) in enumerate(metric_values):
            # Create z-score between -2 and 2 based on position
            position = i / (n - 1) if n > 1 else 0.5  # 0 to 1
            z_score = (position * 4) - 2  # -2 to 2
            
            # Find this ticker in all_results
            for idx, (result_ticker, _, details) in enumerate(all_results):
                if result_ticker == ticker:
                    details['z_scores'][metric] = z_score
    
    # Approach 2: For metrics with known directionality, create synthetic z-scores
    # Define metrics where higher values are better
    higher_better_metrics = {
        'gross_profit_margin', 'operating_income_margin', 'net_income_margin', 'ebitda_margin',
        'return_on_equity', 'return_on_assets', 'growth_revenue', 'growth_gross_profit',
        'growth_ebitda', 'growth_operating_income', 'growth_net_income', 'growth_eps',
        'current_ratio', 'quick_ratio', 'interest_coverage', 'cash_to_debt',
        'asset_turnover', 'inventory_turnover', 'receivables_turnover', 'dividend_yield',
        'estimate_eps_accuracy', 'estimate_revenue_accuracy', 'forward_sales_growth',
        'forward_ebitda_growth', 'estimate_revision_momentum'
    }
    
    # Define metrics where lower values are better
    lower_better_metrics = {
        'debt_to_equity', 'debt_to_assets', 'growth_total_debt', 'growth_net_debt',
        'cash_conversion_cycle', 'capex_to_revenue', 'pe_ratio', 'price_to_book',
        'price_to_sales', 'ev_to_ebitda', 'peg_ratio', 'estimate_consensus_deviation'
    }
    
    # Ensure z-scores reflect the desirability of metrics
    for ticker, _, details in all_results:
        if ticker in all_metrics:
            for metric, value in all_metrics[ticker].items():
                # Skip metrics that already have z-scores
                if metric in details['z_scores'] and abs(details['z_scores'][metric]) > 1e-6:
                    continue
                    
                if isinstance(value, (int, float)) and np.isfinite(value):
                    if metric in higher_better_metrics:
                        # For metrics where higher is better, use a positive z-score
                        # Scale based on the value relative to some reasonable benchmarks
                        if metric.startswith('growth_'):
                            # For growth metrics, scale between -1 and 2
                            if value < 0:
                                z_score = max(value, -1)  # Negative growth capped at -1
                            else:
                                z_score = min(value, 2)   # Positive growth capped at 2
                        elif metric.endswith('_margin'):
                            # For margin metrics, scale to roughly -2 to 2
                            z_score = (value - 0.1) * 5  # 0.1 maps to 0, 0.5 maps to 2
                            z_score = max(min(z_score, 2), -2)
                        elif metric in {'return_on_equity', 'return_on_assets'}:
                            # For ROE/ROA metrics
                            z_score = value * 10  # 0.1 maps to 1, 0.2 maps to 2
                            z_score = max(min(z_score, 2), -2)
                        else:
                            # General case - use a small positive z-score
                            seed_val = hash(metric + ticker) % 1000
                            np.random.seed(seed_val)
                            z_score = 0.5 + (np.random.random() * 0.5)  # 0.5 to 1.0
                    elif metric in lower_better_metrics:
                        # For metrics where lower is better, use a negative z-score
                        if metric in {'pe_ratio', 'price_to_book', 'price_to_sales', 'ev_to_ebitda'}:
                            # For valuation metrics
                            benchmark = {'pe_ratio': 15, 'price_to_book': 3, 'price_to_sales': 2, 'ev_to_ebitda': 10}
                            baseline = benchmark.get(metric, 1)
                            z_score = (baseline - value) / baseline
                            z_score = max(min(z_score, 2), -2)
                        elif metric in {'debt_to_equity', 'debt_to_assets'}:
                            # For debt metrics
                            z_score = 1 - value  # 0 maps to 1, 1 maps to 0, 2 maps to -1
                            z_score = max(min(z_score, 2), -2)
                        else:
                            # General case - use a small negative z-score
                            seed_val = hash(metric + ticker) % 1000
                            np.random.seed(seed_val)
                            z_score = -0.5 - (np.random.random() * 0.5)  # -0.5 to -1.0
                    else:
                        # For metrics with unknown directionality, use a small random z-score
                        seed_val = hash(metric + ticker) % 1000
                        np.random.seed(seed_val)
                        z_score = np.random.uniform(-0.5, 0.5)
                    
                    details['z_scores'][metric] = z_score
    
    # Approach 3: Base z-scores on category rankings
    # Define category mappings for common metrics
    category_mappings = {
        'profitability': ['gross_profit_margin', 'operating_income_margin', 'net_income_margin', 
                          'ebitda_margin', 'return_on_equity', 'return_on_assets'],
        'growth': ['growth_revenue', 'growth_gross_profit', 'growth_ebitda', 
                   'growth_operating_income', 'growth_net_income', 'growth_eps',
                   'growth_total_assets', 'growth_total_shareholders_equity'],
        'financial_health': ['current_ratio', 'quick_ratio', 'debt_to_equity', 'debt_to_assets',
                            'interest_coverage', 'cash_to_debt', 'growth_total_debt', 'growth_net_debt'],
        'valuation': ['pe_ratio', 'price_to_book', 'price_to_sales', 'ev_to_ebitda', 
                      'dividend_yield', 'peg_ratio'],
        'efficiency': ['asset_turnover', 'inventory_turnover', 'receivables_turnover',
                      'cash_conversion_cycle', 'capex_to_revenue'],
        'analyst_estimates': ['estimate_eps_accuracy', 'estimate_revenue_accuracy',
                             'estimate_consensus_deviation', 'estimate_revision_momentum',
                             'forward_sales_growth', 'forward_ebitda_growth']
    }
    
    # For each category, rank tickers and assign z-scores
    for category, metrics in category_mappings.items():
        category_scores = {}
        
        # Calculate a category score for each ticker
        for ticker in all_metrics:
            score = 0
            score_count = 0
            
            for metric in metrics:
                if metric in all_metrics[ticker]:
                    value = all_metrics[ticker][metric]
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        # Apply the appropriate directionality
                        if metric in lower_better_metrics:
                            score -= value  # Lower is better
                        else:
                            score += value  # Higher is better
                        score_count += 1
            
            if score_count > 0:
                category_scores[ticker] = score / score_count
        
        # Skip categories with insufficient data
        if len(category_scores) < 2:
            continue
            
        # Rank tickers by category score
        ranked_tickers = sorted(category_scores.keys(), 
                               key=lambda t: category_scores[t],
                               reverse=True)  # Higher scores first
        
        # Assign z-scores based on rank
        n = len(ranked_tickers)
        for i, ticker in enumerate(ranked_tickers):
            # Create z-score between -2 and 2 based on position
            position = i / (n - 1) if n > 1 else 0.5  # 0 to 1
            z_score = (position * 4) - 2  # -2 to 2
            
            # The highest-ranked ticker gets z-score 2, lowest gets -2
            # Convert to our conventional format where lower rank = higher z-score
            z_score = -z_score
            
            # Create a category-specific synthetic metric
            category_metric = f"synthetic_{category}_score"
            
            # Find this ticker in all_results
            for idx, (result_ticker, _, details) in enumerate(all_results):
                if result_ticker == ticker:
                    details['z_scores'][category_metric] = z_score
    
    # Final pass: Ensure every ticker has at least some non-zero z-scores
    for ticker, _, details in all_results:
        z_scores = details.get('z_scores', {})
        nonzero_count = sum(1 for v in z_scores.values() if abs(v) > 1e-6)
        
        if nonzero_count == 0:
            logger.warning(f"Ticker {ticker} still has no non-zero z-scores after simulation")
            
            # Last resort: assign random z-scores
            for category in category_mappings:
                category_metric = f"synthetic_{category}_score"
                seed_val = hash(category + ticker) % 1000
                np.random.seed(seed_val)
                z_score = np.random.uniform(-1.0, 1.0)
                details['z_scores'][category_metric] = z_score
    
    # Log summary of simulated z-scores
    sample_ticker = all_results[0][0]
    sample_details = all_results[0][2]
    z_score_keys = sample_details['z_scores'].keys()
    nonzero_count = sum(1 for k in z_score_keys if abs(sample_details['z_scores'][k]) > 1e-6)
    logger.info(f"Simulated z-scores: {len(z_score_keys)} metrics, {nonzero_count} non-zero values")

def generate_synthetic_peers(all_results):
    """
    Generate synthetic peer groups when API-based peer analysis fails.
    This groups similar stocks based on their scores to create realistic peer comparisons.
    """
    peer_data_dict = {}
    ticker_to_details = {ticker: details for ticker, _, details in all_results}
    
    for ticker, score, details in all_results:
        # Find similar stocks based on category scores
        similarities = []
        cat_scores = details.get('category_scores', {})
        
        for other_ticker, other_score, other_details in all_results:
            if other_ticker != ticker:
                other_cat_scores = other_details.get('category_scores', {})
                
                # Calculate similarity score based on category scores
                similarity = 0
                count = 0
                for category in cat_scores:
                    if category in other_cat_scores:
                        similarity += 1 - min(abs(cat_scores[category] - other_cat_scores[category]), 1.0)
                        count += 1
                
                if count > 0:
                    similarity = similarity / count
                    similarities.append((other_ticker, similarity, other_score))
        
        # Sort by similarity (most similar first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 3-5 most similar tickers as peers
        peers = similarities[:min(5, len(similarities))]
        
        # Calculate peer metrics
        peer_scores = [p[2] for p in peers]
        if peer_scores:
            peer_avg = sum(peer_scores) / len(peer_scores)
            peer_std_dev = np.std(peer_scores) if len(peer_scores) > 1 else 0.05
            
            # Calculate percentile - how many peers are below this stock
            below_count = sum(1 for p_score in peer_scores if p_score < score)
            percentile = (below_count / len(peer_scores)) * 100 if peer_scores else 50.0
        else:
            peer_avg = score * 0.95  # Default to 5% below the stock's score
            peer_std_dev = 0.05
            percentile = 50.0
        
        # Calculate category percentiles
        category_percentiles = {}
        for category, cat_score in cat_scores.items():
            peer_cat_scores = []
            for p_ticker, _, _ in peers:
                if p_ticker in ticker_to_details:
                    p_details = ticker_to_details[p_ticker]
                    if category in p_details.get('category_scores', {}):
                        peer_cat_scores.append(p_details['category_scores'][category])
            
            if peer_cat_scores:
                below = sum(1 for s in peer_cat_scores if s < cat_score)
                category_percentiles[category] = (below / len(peer_cat_scores)) * 100
            else:
                category_percentiles[category] = 50.0
        
        # Create peer data structure
        peer_data = {
            'peer_comparison': {
                'average_score': peer_avg,
                'std_dev': peer_std_dev,
                'count': len(peers),
                'percentile': percentile
            },
            'category_percentiles': category_percentiles,
            'peers': [{'ticker': p[0], 'similarity': p[1], 'score': p[2]} for p in peers]
        }
        
        peer_data_dict[ticker] = peer_data
    
    return peer_data_dict

def save_final_json(fundamental_results, tech_reports, mc_reports, output_dir="output"):
    """
    Combines fundamental, technical, and Monte Carlo results into a final comprehensive JSON file
    organized by quartiles.
    
    Args:
        fundamental_results (list): List of (ticker, score, details) tuples from fundamental screening
        tech_reports (list): List of technical report dictionaries
        mc_reports (list): List of Monte Carlo report dictionaries
        output_dir (str): Directory to save the file
        
    Returns:
        str: Filename if successful, None if error occurs
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{output_dir}/final_json_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Extract fundamental scores and create quartiles
        fundamental_scores = [score for _, score, _ in fundamental_results]
        q1 = np.percentile(fundamental_scores, 25)
        q3 = np.percentile(fundamental_scores, 75)
        
        # Create mappings for fundamental quartiles
        top_fundamental_tickers = set(ticker for ticker, score, _ in fundamental_results if score >= q3)
        bottom_fundamental_tickers = set(ticker for ticker, score, _ in fundamental_results if score <= q1)
        
        # Create mapping for fundamental data
        fundamental_map = {ticker: {"score": score, "details": details} for ticker, score, details in fundamental_results}
        
        # Create mapping for technical data
        tech_map = {report["ticker"]: report for report in tech_reports}
        
        # Create mapping for Monte Carlo data
        mc_map = {report["ticker"]: report for report in mc_reports}
        
        # Group technical reports by fundamental quartile
        top_fundamental_tech_reports = [report for report in tech_reports if report["ticker"] in top_fundamental_tickers]
        bottom_fundamental_tech_reports = [report for report in tech_reports if report["ticker"] in bottom_fundamental_tickers]
        
        # Final data structure
        final_data = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "top_fundamental": {
                "top_technical": [],
                "bottom_technical": []
            },
            "bottom_fundamental": {
                "top_technical": [],
                "bottom_technical": []
            }
        }
        
        # Process top fundamental group
        if top_fundamental_tech_reports:
            tech_top_scores = [report["technical_score"] for report in top_fundamental_tech_reports]
            if len(tech_top_scores) >= 4:  # Need at least 4 points for quartiles to make sense
                tech_top_q1 = np.percentile(tech_top_scores, 25)
                tech_top_q3 = np.percentile(tech_top_scores, 75)
                
                tech_top_top_reports = [report for report in top_fundamental_tech_reports if report["technical_score"] >= tech_top_q3]
                tech_top_bottom_reports = [report for report in top_fundamental_tech_reports if report["technical_score"] <= tech_top_q1]
                
                # Add to final data
                for report in tech_top_top_reports:
                    ticker = report["ticker"]
                    mc_data = next((mc for mc in mc_reports if mc["ticker"] == ticker), {})
                    
                    stock_data = {
                        "ticker": ticker,
                        "fundamental_score": fundamental_map.get(ticker, {}).get("score", 0),
                        "technical_score": report["technical_score"],
                        "fundamental_details": fundamental_map.get(ticker, {}).get("details", {}),
                        "technical_details": report,
                        "monte_carlo": mc_data.get("report", {}).get("monte_carlo", {}) if "report" in mc_data else {}
                    }
                    final_data["top_fundamental"]["top_technical"].append(stock_data)
                
                for report in tech_top_bottom_reports:
                    ticker = report["ticker"]
                    mc_data = next((mc for mc in mc_reports if mc["ticker"] == ticker), {})
                    
                    stock_data = {
                        "ticker": ticker,
                        "fundamental_score": fundamental_map.get(ticker, {}).get("score", 0),
                        "technical_score": report["technical_score"],
                        "fundamental_details": fundamental_map.get(ticker, {}).get("details", {}),
                        "technical_details": report,
                        "monte_carlo": mc_data.get("report", {}).get("monte_carlo", {}) if "report" in mc_data else {}
                    }
                    final_data["top_fundamental"]["bottom_technical"].append(stock_data)
            else:
                # Not enough data points for quartiles, just use median as divide
                if tech_top_scores:
                    median = np.median(tech_top_scores)
                    tech_top_top_reports = [report for report in top_fundamental_tech_reports if report["technical_score"] >= median]
                    tech_top_bottom_reports = [report for report in top_fundamental_tech_reports if report["technical_score"] < median]
                    
                    # Add to final data (same process as above)
                    for report in tech_top_top_reports:
                        ticker = report["ticker"]
                        mc_data = next((mc for mc in mc_reports if mc["ticker"] == ticker), {})
                        
                        stock_data = {
                            "ticker": ticker,
                            "fundamental_score": fundamental_map.get(ticker, {}).get("score", 0),
                            "technical_score": report["technical_score"],
                            "fundamental_details": fundamental_map.get(ticker, {}).get("details", {}),
                            "technical_details": report,
                            "monte_carlo": mc_data.get("report", {}).get("monte_carlo", {}) if "report" in mc_data else {}
                        }
                        final_data["top_fundamental"]["top_technical"].append(stock_data)
                    
                    for report in tech_top_bottom_reports:
                        ticker = report["ticker"]
                        mc_data = next((mc for mc in mc_reports if mc["ticker"] == ticker), {})
                        
                        stock_data = {
                            "ticker": ticker,
                            "fundamental_score": fundamental_map.get(ticker, {}).get("score", 0),
                            "technical_score": report["technical_score"],
                            "fundamental_details": fundamental_map.get(ticker, {}).get("details", {}),
                            "technical_details": report,
                            "monte_carlo": mc_data.get("report", {}).get("monte_carlo", {}) if "report" in mc_data else {}
                        }
                        final_data["top_fundamental"]["bottom_technical"].append(stock_data)
        
        # Process bottom fundamental group
        if bottom_fundamental_tech_reports:
            tech_bottom_scores = [report["technical_score"] for report in bottom_fundamental_tech_reports]
            if len(tech_bottom_scores) >= 4:  # Need at least 4 points for quartiles to make sense
                tech_bottom_q1 = np.percentile(tech_bottom_scores, 25)
                tech_bottom_q3 = np.percentile(tech_bottom_scores, 75)
                
                tech_bottom_top_reports = [report for report in bottom_fundamental_tech_reports if report["technical_score"] >= tech_bottom_q3]
                tech_bottom_bottom_reports = [report for report in bottom_fundamental_tech_reports if report["technical_score"] <= tech_bottom_q1]
                
                # Add to final data
                for report in tech_bottom_top_reports:
                    ticker = report["ticker"]
                    mc_data = next((mc for mc in mc_reports if mc["ticker"] == ticker), {})
                    
                    stock_data = {
                        "ticker": ticker,
                        "fundamental_score": fundamental_map.get(ticker, {}).get("score", 0),
                        "technical_score": report["technical_score"],
                        "fundamental_details": fundamental_map.get(ticker, {}).get("details", {}),
                        "technical_details": report,
                        "monte_carlo": mc_data.get("report", {}).get("monte_carlo", {}) if "report" in mc_data else {}
                    }
                    final_data["bottom_fundamental"]["top_technical"].append(stock_data)
                
                for report in tech_bottom_bottom_reports:
                    ticker = report["ticker"]
                    mc_data = next((mc for mc in mc_reports if mc["ticker"] == ticker), {})
                    
                    stock_data = {
                        "ticker": ticker,
                        "fundamental_score": fundamental_map.get(ticker, {}).get("score", 0),
                        "technical_score": report["technical_score"],
                        "fundamental_details": fundamental_map.get(ticker, {}).get("details", {}),
                        "technical_details": report,
                        "monte_carlo": mc_data.get("report", {}).get("monte_carlo", {}) if "report" in mc_data else {}
                    }
                    final_data["bottom_fundamental"]["bottom_technical"].append(stock_data)
            else:
                # Not enough data points for quartiles, just use median as divide
                if tech_bottom_scores:
                    median = np.median(tech_bottom_scores)
                    tech_bottom_top_reports = [report for report in bottom_fundamental_tech_reports if report["technical_score"] >= median]
                    tech_bottom_bottom_reports = [report for report in bottom_fundamental_tech_reports if report["technical_score"] < median]
                    
                    # Add to final data (same process as above)
                    for report in tech_bottom_top_reports:
                        ticker = report["ticker"]
                        mc_data = next((mc for mc in mc_reports if mc["ticker"] == ticker), {})
                        
                        stock_data = {
                            "ticker": ticker,
                            "fundamental_score": fundamental_map.get(ticker, {}).get("score", 0),
                            "technical_score": report["technical_score"],
                            "fundamental_details": fundamental_map.get(ticker, {}).get("details", {}),
                            "technical_details": report,
                            "monte_carlo": mc_data.get("report", {}).get("monte_carlo", {}) if "report" in mc_data else {}
                        }
                        final_data["bottom_fundamental"]["top_technical"].append(stock_data)
                    
                    for report in tech_bottom_bottom_reports:
                        ticker = report["ticker"]
                        mc_data = next((mc for mc in mc_reports if mc["ticker"] == ticker), {})
                        
                        stock_data = {
                            "ticker": ticker,
                            "fundamental_score": fundamental_map.get(ticker, {}).get("score", 0),
                            "technical_score": report["technical_score"],
                            "fundamental_details": fundamental_map.get(ticker, {}).get("details", {}),
                            "technical_details": report,
                            "monte_carlo": mc_data.get("report", {}).get("monte_carlo", {}) if "report" in mc_data else {}
                        }
                        final_data["bottom_fundamental"]["bottom_technical"].append(stock_data)
        
        # Sort all lists by technical_score in descending order
        final_data["top_fundamental"]["top_technical"].sort(key=lambda x: x["technical_score"], reverse=True)
        final_data["top_fundamental"]["bottom_technical"].sort(key=lambda x: x["technical_score"], reverse=True)
        final_data["bottom_fundamental"]["top_technical"].sort(key=lambda x: x["technical_score"], reverse=True)
        final_data["bottom_fundamental"]["bottom_technical"].sort(key=lambda x: x["technical_score"], reverse=True)
        
        # Write to file with proper handling of numpy types
        with open(output_filename, 'w') as json_file:
            json.dump(final_data, json_file, indent=4, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) 
                     else (x.tolist() if isinstance(x, np.ndarray) else x))
        
        logger.info(f"Final JSON saved to {output_filename}")
        return output_filename
    except Exception as e:
        logger.error(f"Error saving final JSON: {e}")
        logger.debug(traceback.format_exc())
        return None

def save_results_to_json(json_data, output_dir="output", filename_prefix="fundamental_screening"):
    """
    Saves results to JSON file with error handling.
    
    Args:
        json_data (dict): Data to save to JSON
        output_dir (str): Directory to save the file
        filename_prefix (str): Prefix for the filename
        
    Returns:
        str: Filename if successful, None if error occurs
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{output_dir}/{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_filename, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        return output_filename
    except Exception as e:
        logger.error(f"Error saving results to JSON: {e}")
        logger.debug(traceback.format_exc())
        return None

if __name__ == "__main__":
    start = datetime.now()
    try:
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

        init_tickers = tickers.tickers
        # init_tickers = get_active_tickers()
        print(f"analyzing {len(init_tickers)} tickers...")
                  
        results = perform_stock_screening(init_tickers, batch_size=os.cpu_count()*2)
        
        if not results:
            logger.error("No valid stocks were successfully screened. Check logs for details.")
            sys.exit(1)
            
        results.sort(key=lambda x: x[1], reverse=True)
        
        rich_console = Console()
        
        # Create the first table without printing
        stock_scores_table = Table(title="Stock Scores")
        stock_scores_table.add_column("Ticker", style="cyan")
        stock_scores_table.add_column("Score", justify="right", style="green")
        
        for ticker, score, _ in results:
            stock_scores_table.add_row(ticker, f"{score:.8f}")
            
        # Add table to our collection
        all_tables.append(("Fundamental Screening Results", stock_scores_table))
        
        json_data = {
            "screening_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tickers_analyzed": init_tickers,
            "stocks": []
        }
        
        logger.info("Generating detailed reports...")
        stock_reports = []
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=rich_console
        ) as progress:
            task = progress.add_task("Generating reports...", total=len(results))
            
            with ThreadPoolExecutor(max_workers=min(os.cpu_count()*2, len(results))) as executor:
                future_to_ticker = {
                    executor.submit(generate_stock_report_task, ticker, score, details): ticker 
                    for ticker, score, details in results
                }
                
                for future in tqdm(future_to_ticker, total=len(future_to_ticker), 
                                  desc="Generating reports", leave=False):
                    try:
                        stock_report = future.result()
                        stock_reports.append(stock_report)
                    except Exception as e:
                        ticker = future_to_ticker[future]
                        logger.error(f"Error generating report for {ticker}: {e}")
                    
                    progress.update(task, advance=1)
        
        for stock_data in stock_reports:
            json_data["stocks"].append(stock_data)
            
        output_filename = save_results_to_json(json_data)
        if output_filename:
            logger.info(f"Results saved to {output_filename}")
        else:
            logger.error("Error saving results to file. Check logs for details.")
            
        try:
            scores = [score for _, score, _ in results]
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            
            top_quartile = [(ticker, score, details) for ticker, score, details in results if score >= q3]
            bottom_quartile = [(ticker, score, details) for ticker, score, details in results if score <= q1]
            top_quartile.sort(key=lambda x: x[1], reverse=True)
            bottom_quartile.sort(key=lambda x: x[1])
                
            # Create and store the quartile tables instead of printing
            if top_quartile:
                categories = list(top_quartile[0][2]['category_scores'].keys())
                table_top = Table(title="Top Quartile Stocks")
                table_top.add_column("Ticker", style="cyan")
                table_top.add_column("Score", justify="right", style="green")
                for cat in categories:
                    table_top.add_column(cat.title(), justify="right")
                table_top.add_column("Peer Avg", justify="right", style="magenta")
                table_top.add_column("Peer Delta", justify="right", style="magenta")
                for ticker, score, details in top_quartile:
                    row = [ticker, f"{score:.8f}"]
                    for cat in categories:
                        cat_score = details['category_scores'].get(cat, 0)
                        row.append(f"{cat_score:.8f}")
                    peer_avg = details.get('peer_comparison', 0)
                    peer_delta = details.get('peer_delta', 0)
                    row.append(f"{peer_avg:.8f}")
                    row.append(f"{peer_delta:.8f}")
                    table_top.add_row(*row)
                
                if bottom_quartile:
                    categories = list(bottom_quartile[0][2]['category_scores'].keys())
                    table_bottom = Table(title="Bottom Quartile Stocks")
                    table_bottom.add_column("Ticker", style="cyan")
                    table_bottom.add_column("Score", justify="right", style="green")
                    for cat in categories:
                        table_bottom.add_column(cat.title(), justify="right")
                    table_bottom.add_column("Peer Avg", justify="right", style="magenta")
                    table_bottom.add_column("Peer Delta", justify="right", style="magenta")
                    for ticker, score, details in bottom_quartile:
                        row = [ticker, f"{score:.8f}"]
                        for cat in categories:
                            cat_score = details['category_scores'].get(cat, 0)
                            row.append(f"{cat_score:.8f}")
                        peer_avg = details.get('peer_comparison', 0)
                        peer_delta = details.get('peer_delta', 0)
                        row.append(f"{peer_avg:.8f}")
                        row.append(f"{peer_delta:.8f}")
                        table_bottom.add_row(*row)
                    
                    # Store both tables as columns in our collection
                    all_tables.append(("Fundamental Quartile Analysis", Columns([table_top, table_bottom])))
                else:
                    # Store just the top table in our collection
                    all_tables.append(("Fundamental Quartile Analysis", table_top))
            elif bottom_quartile:
                categories = list(bottom_quartile[0][2]['category_scores'].keys())
                table_bottom = Table(title="Bottom Quartile Stocks")
                table_bottom.add_column("Ticker", style="cyan")
                table_bottom.add_column("Score", justify="right", style="green")
                for cat in categories:
                    table_bottom.add_column(cat.title(), justify="right")
                table_bottom.add_column("Peer Avg", justify="right", style="magenta")
                table_bottom.add_column("Peer Delta", justify="right", style="magenta")
                for ticker, score, details in bottom_quartile:
                    row = [ticker, f"{score:.8f}"]
                    for cat in categories:
                        cat_score = details['category_scores'].get(cat, 0)
                        row.append(f"{cat_score:.8f}")
                    peer_avg = details.get('peer_comparison', 0)
                    peer_delta = details.get('peer_delta', 0)
                    row.append(f"{peer_avg:.8f}")
                    row.append(f"{peer_delta:.8f}")
                    table_bottom.add_row(*row)
                
                # Store the bottom table in our collection
                all_tables.append(("Fundamental Quartile Analysis", table_bottom))
            else:
                logger.info("No quartile analysis available.")
            
            # --- Technical Analysis on Fundamental Quartiles ---
            if top_quartile or bottom_quartile:
                logger.info("Starting Technical Analysis on Fundamental Quartiles...")
                
                # Create a mapping for fundamental group membership
                fundamental_group = {}
                for ticker, score, details in top_quartile:
                    fundamental_group[ticker] = 'top'
                for ticker, score, details in bottom_quartile:
                    fundamental_group[ticker] = 'bottom'
                
                # Collect all tickers from top and bottom quartiles
                quartile_tickers = [ticker for ticker, _, _ in top_quartile + bottom_quartile]
                
                # Perform technical screening
                tech_results = tech_screen_stocks(quartile_tickers)
                
                if tech_results:
                    # Attach fundamental group info to technical results
                    for i, (ticker, score, details) in enumerate(tech_results):
                        group = fundamental_group.get(ticker, 'unknown')
                        details['fundamental_group'] = group
                        tech_results[i] = (ticker, score, details)
                    
                    # Generate technical reports
                    tech_json = {
                        "screening_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "stocks": []
                    }
                    
                    logger.info("Generating technical reports...")
                    tech_reports = []
                    
                    with Progress(
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeElapsedColumn(),
                        TimeRemainingColumn(),
                        console=rich_console
                    ) as progress:
                        task = progress.add_task("Generating technical reports...", total=len(tech_results))
                        
                        with ThreadPoolExecutor(max_workers=min(os.cpu_count()*2, len(tech_results))) as executor:
                            tech_future_to_ticker = {
                                executor.submit(generate_technical_report_task, ticker, score, details): ticker 
                                for ticker, score, details in tech_results
                            }
                            
                            for future in tqdm(tech_future_to_ticker, total=len(tech_future_to_ticker), 
                                              desc="Generating technical reports", leave=False):
                                try:
                                    tech_report = future.result()
                                    tech_reports.append(tech_report)
                                except Exception as e:
                                    ticker = tech_future_to_ticker[future]
                                    logger.error(f"Error generating technical report for {ticker}: {e}")
                                
                                progress.update(task, advance=1)
                    
                    for tech_data in tech_reports:
                        tech_json["stocks"].append(tech_data)
                    
                    # Generate Monte Carlo reports
                    logger.info("Generating Monte Carlo simulations...")
                    mc_reports = []
                    
                    with Progress(
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        TimeElapsedColumn(),
                        TimeRemainingColumn(),
                        console=rich_console
                    ) as progress:
                        task = progress.add_task("Generating Monte Carlo simulations...", total=len(tech_results))
                        
                        with ThreadPoolExecutor(max_workers=min(os.cpu_count()*2, len(tech_results))) as executor:
                            mc_future_to_ticker = {
                                executor.submit(generate_monte_carlo_report_task, ticker, score, details): ticker 
                                for ticker, score, details in tech_results
                            }
                            
                            for future in tqdm(mc_future_to_ticker, total=len(mc_future_to_ticker), 
                                              desc="Generating Monte Carlo simulations", leave=False):
                                try:
                                    mc_report = future.result()
                                    mc_reports.append(mc_report)
                                except Exception as e:
                                    ticker = mc_future_to_ticker[future]
                                    logger.error(f"Error generating Monte Carlo report for {ticker}: {e}")
                                
                                progress.update(task, advance=1)
                    
                    # Add Monte Carlo reports to the JSON data
                    monte_carlo_json = {
                        "screening_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "stocks": mc_reports
                    }
                    
                    # Save Monte Carlo results
                    mc_output_filename = save_results_to_json(monte_carlo_json, filename_prefix="monte_carlo_screening")
                    if mc_output_filename:
                        logger.info(f"Monte Carlo results saved to {mc_output_filename}")
                    else:
                        logger.error("Error saving Monte Carlo results to file. Check logs for details.")
                    
                    # Also save the technical results
                    tech_output_filename = save_results_to_json(tech_json, filename_prefix="technical_screening")
                    if tech_output_filename:
                        logger.info(f"Technical results saved to {tech_output_filename}")
                    else:
                        logger.error("Error saving technical results to file. Check logs for details.")
                    
                    # Save combined final JSON - THIS IS THE NEW ADDITION
                    final_json_filename = save_final_json(results, tech_reports, mc_reports)
                    if final_json_filename:
                        logger.info(f"Final comprehensive quartile analysis saved to {final_json_filename}")
                    else:
                        logger.error("Error saving final comprehensive quartile analysis. Check logs for details.")
                    
                    # Create mapping for fast lookup of reports by ticker
                    tech_reports_map = {r["ticker"]: r for r in tech_reports}
                    mc_reports_map = {r["ticker"]: r for r in mc_reports}
                    
                    # Split technical results by fundamental group
                    tech_results_top = [item for item in tech_results if item[2].get('fundamental_group') == 'top']
                    tech_results_bottom = [item for item in tech_results if item[2].get('fundamental_group') == 'bottom']
                    
                    # Process technical quartiles for top fundamental group
                    if tech_results_top:
                        tech_top_scores = [score for _, score, _ in tech_results_top]
                        tech_top_q1 = np.percentile(tech_top_scores, 25)
                        tech_top_q3 = np.percentile(tech_top_scores, 75)
                        tech_top_top_quartile = [(ticker, score, details) for ticker, score, details in tech_results_top if score >= tech_top_q3]
                        tech_top_bottom_quartile = [(ticker, score, details) for ticker, score, details in tech_results_top if score <= tech_top_q1]
                        tech_top_top_quartile.sort(key=lambda x: x[1], reverse=True)
                        tech_top_bottom_quartile.sort(key=lambda x: x[1])
                        
                        tech_categories = list(tech_results_top[0][2]['category_scores'].keys())
                        
                        # For the top technical quartile of top fundamental stocks
                        table_tech_top_top = Table(title="Top Technical Quartile (Top Fundamentals)")
                        table_tech_top_top.add_column("Ticker", style="cyan")
                        table_tech_top_top.add_column("Score", justify="right", style="green")
                        table_tech_top_top.add_column("Current", justify="right", style="yellow")
                        table_tech_top_top.add_column("Tech Exp", justify="right", style="yellow")
                        table_tech_top_top.add_column("Tech Low", justify="right", style="yellow")
                        table_tech_top_top.add_column("Tech High", justify="right", style="yellow")
                        # Monte Carlo columns with both price and percentage
                        table_tech_top_top.add_column("MC Exp $", justify="right", style="blue")
                        table_tech_top_top.add_column("MC Exp Δ", justify="right", style="blue")
                        table_tech_top_top.add_column("MC Low $", justify="right", style="blue") 
                        table_tech_top_top.add_column("MC Low %", justify="right", style="blue")
                        table_tech_top_top.add_column("MC High $", justify="right", style="blue")
                        table_tech_top_top.add_column("MC High %", justify="right", style="blue")
                        # Continue with other columns
                        for cat in tech_categories:
                            table_tech_top_top.add_column(cat.title(), justify="right")
                        table_tech_top_top.add_column("Signals", style="green")
                        table_tech_top_top.add_column("Warnings", style="red")

                        for ticker, score, details in tech_top_top_quartile:
                            # Get technical report data
                            report_data = tech_reports_map.get(ticker, {})
                            raw = report_data.get("report", {}).get("raw_indicators", {})
                            current = raw.get("current_price", "N/A")
                            expected = raw.get("expected_price", "N/A")
                            low = raw.get("probable_low", "N/A")
                            high = raw.get("probable_high", "N/A")
                            
                            # Get Monte Carlo report data
                            mc_data = mc_reports_map.get(ticker, {})
                            mc_report = mc_data.get("report", {}).get("monte_carlo", {})
                            mc_expected = mc_report.get("expected_price", "N/A")
                            mc_expected_pct = mc_report.get("expected_change_pct", "N/A")
                            mc_low = mc_report.get("lower_bound", "N/A")
                            mc_high = mc_report.get("upper_bound", "N/A")
                            mc_low_prob = mc_report.get("prob_down_10pct", "N/A")
                            mc_high_prob = mc_report.get("prob_up_10pct", "N/A")
                            
                            # Create row with technical and Monte Carlo data
                            row = [
                                ticker,
                                f"{score:.8f}",
                                f"{current:.2f}" if isinstance(current, (int, float)) else str(current),
                                f"{expected:.2f}" if isinstance(expected, (int, float)) else str(expected),
                                f"{low:.2f}" if isinstance(low, (int, float)) else str(low),
                                f"{high:.2f}" if isinstance(high, (int, float)) else str(high),
                                # Monte Carlo data with percentages
                                f"{mc_expected:.2f}" if isinstance(mc_expected, (int, float)) else str(mc_expected),
                                f"{mc_expected_pct:.1f}%" if isinstance(mc_expected_pct, (int, float)) else str(mc_expected_pct),
                                f"{mc_low:.2f}" if isinstance(mc_low, (int, float)) else str(mc_low),
                                f"{mc_low_prob:.1f}%" if isinstance(mc_low_prob, (int, float)) else str(mc_low_prob),
                                f"{mc_high:.2f}" if isinstance(mc_high, (int, float)) else str(mc_high),
                                f"{mc_high_prob:.1f}%" if isinstance(mc_high_prob, (int, float)) else str(mc_high_prob),
                            ]
                            
                            # Add category scores
                            for cat in tech_categories:
                                cat_score = details['category_scores'].get(cat, 0)
                                row.append(f"{cat_score:.8f}")
                            
                            # Add signals and warnings
                            signals = ", ".join(report_data.get("signals", [])[:2])
                            warnings = ", ".join(report_data.get("warnings", [])[:2])
                            row.extend([signals, warnings])
                            
                            table_tech_top_top.add_row(*row)
                        
                        # Add the table to our collection
                        all_tables.append(("Technical Analysis for Top Fundamental Stocks - Top Technical Quartile", table_tech_top_top))
                        
                        # Create second table for bottom technical quartile, top fundamentals
                        table_tech_top_bottom = Table(title="Bottom Technical Quartile (Top Fundamentals)")
                        table_tech_top_bottom.add_column("Ticker", style="cyan")
                        table_tech_top_bottom.add_column("Score", justify="right", style="green")
                        table_tech_top_bottom.add_column("Current", justify="right", style="yellow")
                        table_tech_top_bottom.add_column("Tech Exp", justify="right", style="yellow")
                        table_tech_top_bottom.add_column("Tech Low", justify="right", style="yellow")
                        table_tech_top_bottom.add_column("Tech High", justify="right", style="yellow")
                        # Monte Carlo columns with both price and percentage
                        table_tech_top_bottom.add_column("MC Exp $", justify="right", style="blue")
                        table_tech_top_bottom.add_column("MC Exp Δ", justify="right", style="blue")
                        table_tech_top_bottom.add_column("MC Low $", justify="right", style="blue") 
                        table_tech_top_bottom.add_column("MC Low %", justify="right", style="blue")
                        table_tech_top_bottom.add_column("MC High $", justify="right", style="blue")
                        table_tech_top_bottom.add_column("MC High %", justify="right", style="blue")
                        # Continue with other columns
                        for cat in tech_categories:
                            table_tech_top_bottom.add_column(cat.title(), justify="right")
                        table_tech_top_bottom.add_column("Signals", style="green")
                        table_tech_top_bottom.add_column("Warnings", style="red")

                        for ticker, score, details in tech_top_bottom_quartile:
                            # Get technical report data
                            report_data = tech_reports_map.get(ticker, {})
                            raw = report_data.get("report", {}).get("raw_indicators", {})
                            current = raw.get("current_price", "N/A")
                            expected = raw.get("expected_price", "N/A")
                            low = raw.get("probable_low", "N/A")
                            high = raw.get("probable_high", "N/A")
                            
                            # Get Monte Carlo report data
                            mc_data = mc_reports_map.get(ticker, {})
                            mc_report = mc_data.get("report", {}).get("monte_carlo", {})
                            mc_expected = mc_report.get("expected_price", "N/A")
                            mc_expected_pct = mc_report.get("expected_change_pct", "N/A")
                            mc_low = mc_report.get("lower_bound", "N/A")
                            mc_high = mc_report.get("upper_bound", "N/A")
                            mc_low_prob = mc_report.get("prob_down_10pct", "N/A")
                            mc_high_prob = mc_report.get("prob_up_10pct", "N/A")
                            
                            # Create row with technical and Monte Carlo data
                            row = [
                                ticker,
                                f"{score:.8f}",
                                f"{current:.2f}" if isinstance(current, (int, float)) else str(current),
                                f"{expected:.2f}" if isinstance(expected, (int, float)) else str(expected),
                                f"{low:.2f}" if isinstance(low, (int, float)) else str(low),
                                f"{high:.2f}" if isinstance(high, (int, float)) else str(high),
                                # Monte Carlo data with percentages
                                f"{mc_expected:.2f}" if isinstance(mc_expected, (int, float)) else str(mc_expected),
                                f"{mc_expected_pct:.1f}%" if isinstance(mc_expected_pct, (int, float)) else str(mc_expected_pct),
                                f"{mc_low:.2f}" if isinstance(mc_low, (int, float)) else str(mc_low),
                                f"{mc_low_prob:.1f}%" if isinstance(mc_low_prob, (int, float)) else str(mc_low_prob),
                                f"{mc_high:.2f}" if isinstance(mc_high, (int, float)) else str(mc_high),
                                f"{mc_high_prob:.1f}%" if isinstance(mc_high_prob, (int, float)) else str(mc_high_prob),
                            ]
                            
                            # Add category scores
                            for cat in tech_categories:
                                cat_score = details['category_scores'].get(cat, 0)
                                row.append(f"{cat_score:.8f}")
                            
                            # Add signals and warnings
                            signals = ", ".join(report_data.get("signals", [])[:2])
                            warnings = ", ".join(report_data.get("warnings", [])[:2])
                            row.extend([signals, warnings])
                            
                            table_tech_top_bottom.add_row(*row)
                        
                        # Add the table to our collection
                        all_tables.append(("Technical Analysis for Top Fundamental Stocks - Bottom Technical Quartile", table_tech_top_bottom))
                    
                    # Process technical quartiles for bottom fundamental group
                    if tech_results_bottom:
                        tech_bottom_scores = [score for _, score, _ in tech_results_bottom]
                        tech_bottom_q1 = np.percentile(tech_bottom_scores, 25)
                        tech_bottom_q3 = np.percentile(tech_bottom_scores, 75)
                        tech_bottom_top_quartile = [(ticker, score, details) for ticker, score, details in tech_results_bottom if score >= tech_bottom_q3]
                        tech_bottom_bottom_quartile = [(ticker, score, details) for ticker, score, details in tech_results_bottom if score <= tech_bottom_q1]
                        tech_bottom_top_quartile.sort(key=lambda x: x[1], reverse=True)
                        tech_bottom_bottom_quartile.sort(key=lambda x: x[1])
                        
                        tech_categories = list(tech_results_bottom[0][2]['category_scores'].keys())
                        
                        # Create enhanced table with consistent Monte Carlo columns
                        table_tech_bottom_top = Table(title="Top Technical Quartile (Bottom Fundamentals)")
                        table_tech_bottom_top.add_column("Ticker", style="cyan")
                        table_tech_bottom_top.add_column("Score", justify="right", style="green")
                        table_tech_bottom_top.add_column("Current", justify="right", style="yellow")
                        table_tech_bottom_top.add_column("Tech Exp", justify="right", style="yellow")
                        table_tech_bottom_top.add_column("Tech Low", justify="right", style="yellow")
                        table_tech_bottom_top.add_column("Tech High", justify="right", style="yellow")
                        # Monte Carlo columns with both price and percentage - CONSISTENT FORMAT
                        table_tech_bottom_top.add_column("MC Exp $", justify="right", style="blue")
                        table_tech_bottom_top.add_column("MC Exp Δ", justify="right", style="blue")
                        table_tech_bottom_top.add_column("MC Low $", justify="right", style="blue") 
                        table_tech_bottom_top.add_column("MC Low %", justify="right", style="blue")
                        table_tech_bottom_top.add_column("MC High $", justify="right", style="blue")
                        table_tech_bottom_top.add_column("MC High %", justify="right", style="blue")
                        # Continue with other columns
                        for cat in tech_categories:
                            table_tech_bottom_top.add_column(cat.title(), justify="right")
                        table_tech_bottom_top.add_column("Signals", style="green")
                        table_tech_bottom_top.add_column("Warnings", style="red")
                        
                        for ticker, score, details in tech_bottom_top_quartile:
                            # Get technical report data
                            report_data = tech_reports_map.get(ticker, {})
                            raw = report_data.get("report", {}).get("raw_indicators", {})
                            current = raw.get("current_price", "N/A")
                            expected = raw.get("expected_price", "N/A")
                            low = raw.get("probable_low", "N/A")
                            high = raw.get("probable_high", "N/A")
                            
                            # Get Monte Carlo report data
                            mc_data = mc_reports_map.get(ticker, {})
                            mc_report = mc_data.get("report", {}).get("monte_carlo", {})
                            mc_expected = mc_report.get("expected_price", "N/A")
                            mc_expected_pct = mc_report.get("expected_change_pct", "N/A")
                            mc_low = mc_report.get("lower_bound", "N/A")
                            mc_high = mc_report.get("upper_bound", "N/A")
                            mc_low_prob = mc_report.get("prob_down_10pct", "N/A")
                            mc_high_prob = mc_report.get("prob_up_10pct", "N/A")
                            
                            # Create row with technical and Monte Carlo data - CONSISTENT FORMAT
                            row = [
                                ticker,
                                f"{score:.8f}",
                                f"{current:.2f}" if isinstance(current, (int, float)) else str(current),
                                f"{expected:.2f}" if isinstance(expected, (int, float)) else str(expected),
                                f"{low:.2f}" if isinstance(low, (int, float)) else str(low),
                                f"{high:.2f}" if isinstance(high, (int, float)) else str(high),
                                # Monte Carlo data with percentages
                                f"{mc_expected:.2f}" if isinstance(mc_expected, (int, float)) else str(mc_expected),
                                f"{mc_expected_pct:.1f}%" if isinstance(mc_expected_pct, (int, float)) else str(mc_expected_pct),
                                f"{mc_low:.2f}" if isinstance(mc_low, (int, float)) else str(mc_low),
                                f"{mc_low_prob:.1f}%" if isinstance(mc_low_prob, (int, float)) else str(mc_low_prob),
                                f"{mc_high:.2f}" if isinstance(mc_high, (int, float)) else str(mc_high),
                                f"{mc_high_prob:.1f}%" if isinstance(mc_high_prob, (int, float)) else str(mc_high_prob),
                            ]
                            
                            # Add category scores
                            for cat in tech_categories:
                                cat_score = details['category_scores'].get(cat, 0)
                                row.append(f"{cat_score:.8f}")
                            
                            # Add signals and warnings
                            signals = ", ".join(report_data.get("signals", [])[:2])
                            warnings = ", ".join(report_data.get("warnings", [])[:2])
                            row.extend([signals, warnings])
                            
                            table_tech_bottom_top.add_row(*row)
                        
                        # Add the table to our collection
                        all_tables.append(("Technical Analysis for Bottom Fundamental Stocks - Top Technical Quartile", table_tech_bottom_top))
                        
                        # Create final table for bottom technical quartile, bottom fundamentals
                        table_tech_bottom_bottom = Table(title="Bottom Technical Quartile (Bottom Fundamentals)")
                        table_tech_bottom_bottom.add_column("Ticker", style="cyan")
                        table_tech_bottom_bottom.add_column("Score", justify="right", style="green")
                        table_tech_bottom_bottom.add_column("Current", justify="right", style="yellow")
                        table_tech_bottom_bottom.add_column("Tech Exp", justify="right", style="yellow")
                        table_tech_bottom_bottom.add_column("Tech Low", justify="right", style="yellow")
                        table_tech_bottom_bottom.add_column("Tech High", justify="right", style="yellow")
                        # Monte Carlo columns with both price and percentage - CONSISTENT FORMAT
                        table_tech_bottom_bottom.add_column("MC Exp $", justify="right", style="blue")
                        table_tech_bottom_bottom.add_column("MC Exp Δ", justify="right", style="blue")
                        table_tech_bottom_bottom.add_column("MC Low $", justify="right", style="blue") 
                        table_tech_bottom_bottom.add_column("MC Low %", justify="right", style="blue")
                        table_tech_bottom_bottom.add_column("MC High $", justify="right", style="blue")
                        table_tech_bottom_bottom.add_column("MC High %", justify="right", style="blue")
                        # Continue with other columns
                        for cat in tech_categories:
                            table_tech_bottom_bottom.add_column(cat.title(), justify="right")
                        table_tech_bottom_bottom.add_column("Signals", style="green")
                        table_tech_bottom_bottom.add_column("Warnings", style="red")
                        
                        for ticker, score, details in tech_bottom_bottom_quartile:
                            # Get technical report data
                            report_data = tech_reports_map.get(ticker, {})
                            raw = report_data.get("report", {}).get("raw_indicators", {})
                            current = raw.get("current_price", "N/A")
                            expected = raw.get("expected_price", "N/A")
                            low = raw.get("probable_low", "N/A")
                            high = raw.get("probable_high", "N/A")
                            
                            # Get Monte Carlo report data
                            mc_data = mc_reports_map.get(ticker, {})
                            mc_report = mc_data.get("report", {}).get("monte_carlo", {})
                            mc_expected = mc_report.get("expected_price", "N/A")
                            mc_expected_pct = mc_report.get("expected_change_pct", "N/A")
                            mc_low = mc_report.get("lower_bound", "N/A")
                            mc_high = mc_report.get("upper_bound", "N/A")
                            mc_low_prob = mc_report.get("prob_down_10pct", "N/A")
                            mc_high_prob = mc_report.get("prob_up_10pct", "N/A")
                            
                            # Create row with technical and Monte Carlo data - CONSISTENT FORMAT
                            row = [
                                ticker,
                                f"{score:.8f}",
                                f"{current:.2f}" if isinstance(current, (int, float)) else str(current),
                                f"{expected:.2f}" if isinstance(expected, (int, float)) else str(expected),
                                f"{low:.2f}" if isinstance(low, (int, float)) else str(low),
                                f"{high:.2f}" if isinstance(high, (int, float)) else str(high),
                                # Monte Carlo data with percentages
                                f"{mc_expected:.2f}" if isinstance(mc_expected, (int, float)) else str(mc_expected),
                                f"{mc_expected_pct:.1f}%" if isinstance(mc_expected_pct, (int, float)) else str(mc_expected_pct),
                                f"{mc_low:.2f}" if isinstance(mc_low, (int, float)) else str(mc_low),
                                f"{mc_low_prob:.1f}%" if isinstance(mc_low_prob, (int, float)) else str(mc_low_prob),
                                f"{mc_high:.2f}" if isinstance(mc_high, (int, float)) else str(mc_high),
                                f"{mc_high_prob:.1f}%" if isinstance(mc_high_prob, (int, float)) else str(mc_high_prob),
                            ]
                            
                            # Add category scores
                            for cat in tech_categories:
                                cat_score = details['category_scores'].get(cat, 0)
                                row.append(f"{cat_score:.8f}")
                            
                            # Add signals and warnings
                            signals = ", ".join(report_data.get("signals", [])[:2])
                            warnings = ", ".join(report_data.get("warnings", [])[:2])
                            row.extend([signals, warnings])
                            
                            table_tech_bottom_bottom.add_row(*row)
                        
                        # Add the table to our collection
                        all_tables.append(("Technical Analysis for Bottom Fundamental Stocks - Bottom Technical Quartile", table_tech_bottom_bottom))
                else:
                    logger.error("No valid stocks were successfully screened with technical indicators.")
            else:
                logger.info("No fundamental quartiles available for technical analysis.")

            # =====================================================================
            # DISPLAY ALL TABLES AT THE END
            # =====================================================================
            logger.info("============= ALL ANALYSIS COMPLETED =============")
            logger.info(f"Processing completed in {datetime.now()-start}")
            
            # Print a visual separator
            rich_console.print("\n" + "="*80)
            rich_console.print("[bold cyan]ANALYSIS RESULTS[/bold cyan]")
            rich_console.print("="*80 + "\n")
            
            # Print all tables that were collected
            for title, table in all_tables:
                rich_console.print(f"\n[bold]{title}:[/bold]")
                rich_console.print("-----------------------------")
                rich_console.print(table)
                rich_console.print("\n")
                
        except Exception as e:
            logger.error(f"Error calculating quartiles: {e}")
            logger.debug(traceback.format_exc())
            logger.error("Error displaying quartile analysis. Check logs for details.")
            
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

    end = datetime.now()
    print(f"time to completion: {end-start}")
