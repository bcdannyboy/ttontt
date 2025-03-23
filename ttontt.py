import asyncio

import concurrent
from src.screener import technicals
from src.screener.fundamentals.fundamentals_peers import gather_peer_analysis
from src.screener.fundamentals.fundamentals_report import generate_stock_report
from src.screener.fundamentals.fundamentals_screen import screen_stocks
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

# Optional GPU check: using TensorFlow with macOS optimizations
try:
    import tensorflow as tf
    # Enable Metal GPU acceleration on macOS
    if 'tensorflow-metal' in tf.__version__ or (hasattr(tf, 'config') and hasattr(tf.config, 'experimental')):
        logger.info("Enabling Metal GPU acceleration for macOS...")
        tf.config.experimental.set_visible_devices([], 'CPU')
        physical_gpus = tf.config.list_physical_devices('GPU')
        if physical_gpus:
            try:
                for gpu in physical_gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"Metal GPU acceleration enabled. Found {len(physical_gpus)} GPU(s)")
            except Exception as e:
                logger.warning(f"Error configuring Metal GPU: {e}")
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info("GPU(s) detected:")
            for gpu in gpus:
                logger.info(f" - {gpu}")
        else:
            logger.info("No GPU detected.")
except ImportError:
    logger.info("TensorFlow not installed; skipping GPU check.")


def generate_stock_report_task(ticker, score, details):
    """
    Generates a detailed stock report for a given ticker.
    In addition to the fundamentals, this version includes the peer analysis data
    (peer average and the delta) in the report.
    
    Args:
        ticker (str): The stock ticker symbol.
        score (float): The (adjusted) composite score from screening.
        details (dict): Detailed results from screening including category scores and peer analysis.
        
    Returns:
        dict: A complete stock report with screening scores and peer analysis.
    """
    try:
        report = generate_stock_report(ticker)
        report['category_scores'] = details['category_scores']
        report['composite_score'] = float(score)
        # Include peer analysis in the report.
        report['peer_analysis'] = {
            'peer_average': details.get('peer_comparison'),
            'peer_delta': details.get('peer_delta')
        }
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
    try:
        report = technicals.generate_stock_report(ticker)
        report['category_scores'] = details['category_scores']
        report['composite_score'] = float(score)
        report['expected_price'] = report.get('raw_indicators', {}).get('expected_price', None)
        report['probable_low'] = report.get('raw_indicators', {}).get('probable_low', None)
        report['probable_high'] = report.get('raw_indicators', {}).get('probable_high', None)
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
                        "volatility_models": mc_report.get("volatility_models")  # Include volatility model info
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
    """
    Safely run an async coroutine in a new event loop.
    This function can be called from sync code to execute an async coroutine.
    """
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
        # Process tickers in batches.
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size} ({len(batch)} tickers)")
            batch_results = screen_stocks(batch)
            all_results.extend(batch_results)
            if i + batch_size < len(tickers):
                time.sleep(2)
        if not all_results:
            logger.warning("No valid results were returned from the screening process.")
        
        all_results.sort(key=lambda x: x[1], reverse=True)
        # (Optional) Remove outliers based on a threshold.
        scores = [score for _, score, _ in all_results]
        median_score = np.median(scores)
        threshold = abs(median_score) * 1000 if median_score != 0 else 1000
        all_results = [(t, s, d) for t, s, d in all_results if abs(s) < threshold]

        # --- Integrate peer analysis ---
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
            
        except Exception as e:
            logger.error(f"Error during peer analysis: {e}")
            logger.debug(traceback.format_exc())
        
        # Peer weight determines how much to adjust scores based on peer comparison
        peer_weight = 0.1  # 10% adjustment factor
        
        # Process each stock's results with peer data
        for idx, (ticker, composite_score, details) in enumerate(all_results):
            try:
                # Get peer data for this ticker
                peer_data = peer_data_dict.get(ticker, {})
                peer_comp = peer_data.get('peer_comparison', {})
                
                # If no peer average available, use the stock's own score
                peer_avg = peer_comp.get('average_score', composite_score)
                
                # Calculate peer delta: the difference between this stock's score and peer average
                peer_delta = composite_score - peer_avg
                
                # Store peer comparison data in details
                details['peer_comparison'] = peer_avg
                details['peer_delta'] = peer_delta
                
                # Adjust composite score by adding peer_weight * peer_delta
                adjusted_score = composite_score + (peer_weight * peer_delta)
                
                # Update the result with the adjusted score
                all_results[idx] = (ticker, adjusted_score, details)
            except Exception as e:
                logger.error(f"Error processing peer data for {ticker}: {e}")
                logger.debug(traceback.format_exc())
                # Keep original score if there's an error
        
        return all_results
    except Exception as e:
        logger.error(f"Error during stock screening: {e}")
        logger.debug(traceback.format_exc())
        return []

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
        rich_console.print("\n[bold]Fundamental Screening Results:[/bold]")
        rich_console.print("-----------------------------")
        
        table = Table(title="Stock Scores")
        table.add_column("Ticker", style="cyan")
        table.add_column("Score", justify="right", style="green")
        
        for ticker, score, _ in results:
            table.add_row(ticker, f"{score:.8f}")
            
        rich_console.print(table)
        
        json_data = {
            "screening_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tickers_analyzed": init_tickers,
            "stocks": []
        }
        
        rich_console.print("\n[bold]Generating detailed reports...[/bold]")
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
            rich_console.print(f"\n[bold green]Results saved to {output_filename}[/bold green]")
        else:
            rich_console.print("\n[bold red]Error saving results to file. Check logs for details.[/bold red]")
            
        try:
            scores = [score for _, score, _ in results]
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            
            top_quartile = [(ticker, score, details) for ticker, score, details in results if score >= q3]
            bottom_quartile = [(ticker, score, details) for ticker, score, details in results if score <= q1]
            top_quartile.sort(key=lambda x: x[1], reverse=True)
            bottom_quartile.sort(key=lambda x: x[1])
            
            rich_console.print("\n[bold]Fundamental Quartile Analysis:[/bold]")
            
            if top_quartile:
                categories = list(top_quartile[0][2]['category_scores'].keys())
                # Create table with additional peer columns.
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
                    
                    rich_console.print(Columns([table_top, table_bottom]))
                else:
                    rich_console.print(table_top)
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
                
                rich_console.print(table_bottom)
            else:
                rich_console.print("No quartile analysis available.")
            
            # --- Technical Analysis on Fundamental Quartiles ---
            if top_quartile or bottom_quartile:
                rich_console.print("\n[bold]Technical Analysis on Fundamental Quartiles:[/bold]")
                
                # Create a mapping for fundamental group membership
                fundamental_group = {}
                for ticker, score, details in top_quartile:
                    fundamental_group[ticker] = 'top'
                for ticker, score, details in bottom_quartile:
                    fundamental_group[ticker] = 'bottom'
                
                # Collect all tickers from top and bottom quartiles
                quartile_tickers = [ticker for ticker, _, _ in top_quartile + bottom_quartile]
                
                # Perform technical screening
                tech_results = technicals.screen_stocks(quartile_tickers)
                
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
                    
                    rich_console.print("\n[bold]Generating technical reports...[/bold]")
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
                    rich_console.print("\n[bold]Generating Monte Carlo simulations...[/bold]")
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
                        rich_console.print(f"\n[bold green]Monte Carlo results saved to {mc_output_filename}[/bold green]")
                    else:
                        rich_console.print("\n[bold red]Error saving Monte Carlo results to file. Check logs for details.[/bold red]")
                    
                    # Also save the technical results
                    tech_output_filename = save_results_to_json(tech_json, filename_prefix="technical_screening")
                    if tech_output_filename:
                        rich_console.print(f"\n[bold green]Technical results saved to {tech_output_filename}[/bold green]")
                    else:
                        rich_console.print("\n[bold red]Error saving technical results to file. Check logs for details.[/bold red]")
                    
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
                        
                        rich_console.print("\n[bold]Technical Analysis for Top Fundamental Stocks:[/bold]")
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
                        
                        rich_console.print(table_tech_top_top)
                        
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
                        
                        rich_console.print(table_tech_top_bottom)
                    
                    # Process technical quartiles for bottom fundamental group
                    if tech_results_bottom:
                        tech_bottom_scores = [score for _, score, _ in tech_results_bottom]
                        tech_bottom_q1 = np.percentile(tech_bottom_scores, 25)
                        tech_bottom_q3 = np.percentile(tech_bottom_scores, 75)
                        tech_bottom_top_quartile = [(ticker, score, details) for ticker, score, details in tech_results_bottom if score >= tech_bottom_q3]
                        tech_bottom_bottom_quartile = [(ticker, score, details) for ticker, score, details in tech_results_bottom if score <= tech_bottom_q1]
                        tech_bottom_top_quartile.sort(key=lambda x: x[1], reverse=True)
                        tech_bottom_bottom_quartile.sort(key=lambda x: x[1])
                        
                        rich_console.print("\n[bold]Technical Analysis for Bottom Fundamental Stocks:[/bold]")
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
                        
                        rich_console.print(table_tech_bottom_top)
                        
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
                        
                        rich_console.print(table_tech_bottom_bottom)
                else:
                    rich_console.print("[bold red]No valid stocks were successfully screened with technical indicators.[/bold red]")
            else:
                rich_console.print("No fundamental quartiles available for technical analysis.")

        except Exception as e:
            logger.error(f"Error calculating quartiles: {e}")
            logger.debug(traceback.format_exc())
            rich_console.print("\n[bold red]Error displaying quartile analysis. Check logs for details.[/bold red]")
            
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

