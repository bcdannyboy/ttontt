import asyncio
from src.screener import fundamentals, technicals
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
        report = fundamentals.generate_stock_report(ticker)
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
    """
    Generates a detailed technical analysis report for a given ticker.
    
    Args:
        ticker (str): The stock ticker symbol.
        score (float): The composite technical score from screening.
        details (dict): Detailed results from screening including category scores.
        
    Returns:
        dict: A complete technical analysis report with scores, signals, and warnings.
    """
    try:
        report = technicals.generate_stock_report(ticker)
        report['category_scores'] = details['category_scores']
        report['composite_score'] = float(score)
        
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
            batch_results = fundamentals.screen_stocks(batch)
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
        
        # Run peer analysis in a dedicated event loop
        try:
            # Important: Create and use a new event loop specifically for peer analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Execute the peer analysis within this loop by correctly referencing the function
            try:
                peer_data_dict = loop.run_until_complete(fundamentals.gather_peer_analysis(tickers_to_analyze))
            except Exception as inner_e:
                logger.error(f"Error during peer analysis execution: {inner_e}")
                logger.debug(traceback.format_exc())
            finally:
                loop.close()
        except Exception as outer_e:
            logger.error(f"Error setting up peer analysis loop: {outer_e}")
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
            
            with ThreadPoolExecutor(max_workers=min(10, len(results))) as executor:
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
            
            # Perform technical analysis on the quartile stocks from fundamental analysis
            if top_quartile or bottom_quartile:
                rich_console.print("\n[bold]Technical Analysis on Fundamental Quartiles:[/bold]")
                
                # Collect all tickers from top and bottom quartiles
                quartile_tickers = [ticker for ticker, _, _ in top_quartile + bottom_quartile]
                
                # Perform technical screening
                tech_results = technicals.screen_stocks(quartile_tickers)
                
                if tech_results:
                    # Add technical results to JSON data
                    tech_json = {
                        "screening_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "stocks": []
                    }
                    
                    # Generate technical reports
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
                        
                        with ThreadPoolExecutor(max_workers=min(10, len(tech_results))) as executor:
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
                    
                    # Save technical results to JSON
                    tech_output_filename = save_results_to_json(tech_json, filename_prefix="technical_screening")
                    if tech_output_filename:
                        rich_console.print(f"\n[bold green]Technical results saved to {tech_output_filename}[/bold green]")
                    else:
                        rich_console.print("\n[bold red]Error saving technical results to file. Check logs for details.[/bold red]")
                    
                    # Calculate quartiles for technical scores
                    tech_scores = [score for _, score, _ in tech_results]
                    tech_q1 = np.percentile(tech_scores, 25)
                    tech_q3 = np.percentile(tech_scores, 75)
                    
                    tech_top_quartile = [(ticker, score, details) for ticker, score, details in tech_results if score >= tech_q3]
                    tech_bottom_quartile = [(ticker, score, details) for ticker, score, details in tech_results if score <= tech_q1]
                    
                    tech_top_quartile.sort(key=lambda x: x[1], reverse=True)
                    tech_bottom_quartile.sort(key=lambda x: x[1])
                    
                    # Display technical quartiles
                    rich_console.print("\n[bold]Technical Analysis Quartiles:[/bold]")
                    
                    # Create mapping for fast signal and warning lookup
                    tech_reports_map = {r["ticker"]: r for r in tech_reports}
                    
                    if tech_top_quartile:
                        tech_categories = list(tech_top_quartile[0][2]['category_scores'].keys())
                        # Create table for top technical quartile
                        table_tech_top = Table(title="Top Technical Quartile Stocks")
                        table_tech_top.add_column("Ticker", style="cyan")
                        table_tech_top.add_column("Score", justify="right", style="green")
                        
                        for cat in tech_categories:
                            table_tech_top.add_column(cat.title(), justify="right")
                        
                        table_tech_top.add_column("Signals", style="green")
                        table_tech_top.add_column("Warnings", style="red")
                        
                        for ticker, score, details in tech_top_quartile:
                            # Get report data for signals and warnings
                            report_data = tech_reports_map.get(ticker, {})
                            
                            row = [ticker, f"{score:.8f}"]
                            for cat in tech_categories:
                                cat_score = details['category_scores'].get(cat, 0)
                                row.append(f"{cat_score:.8f}")
                            
                            signals = ", ".join(report_data.get("signals", [])[:2])  # Show at most 2 signals
                            warnings = ", ".join(report_data.get("warnings", [])[:2])  # Show at most 2 warnings
                            
                            row.append(signals)
                            row.append(warnings)
                            
                            table_tech_top.add_row(*row)
                        
                        if tech_bottom_quartile:
                            # Create table for bottom technical quartile
                            table_tech_bottom = Table(title="Bottom Technical Quartile Stocks")
                            table_tech_bottom.add_column("Ticker", style="cyan")
                            table_tech_bottom.add_column("Score", justify="right", style="green")
                            
                            for cat in tech_categories:
                                table_tech_bottom.add_column(cat.title(), justify="right")
                            
                            table_tech_bottom.add_column("Signals", style="green")
                            table_tech_bottom.add_column("Warnings", style="red")
                            
                            for ticker, score, details in tech_bottom_quartile:
                                # Get report data for signals and warnings
                                report_data = tech_reports_map.get(ticker, {})
                                
                                row = [ticker, f"{score:.8f}"]
                                for cat in tech_categories:
                                    cat_score = details['category_scores'].get(cat, 0)
                                    row.append(f"{cat_score:.8f}")
                                
                                signals = ", ".join(report_data.get("signals", [])[:2])  # Show at most 2 signals
                                warnings = ", ".join(report_data.get("warnings", [])[:2])  # Show at most 2 warnings
                                
                                row.append(signals)
                                row.append(warnings)
                                
                                table_tech_bottom.add_row(*row)
                            
                            rich_console.print(Columns([table_tech_top, table_tech_bottom]))
                        else:
                            rich_console.print(table_tech_top)
                    elif tech_bottom_quartile:
                        tech_categories = list(tech_bottom_quartile[0][2]['category_scores'].keys())
                        # Create table for bottom technical quartile only
                        table_tech_bottom = Table(title="Bottom Technical Quartile Stocks")
                        table_tech_bottom.add_column("Ticker", style="cyan")
                        table_tech_bottom.add_column("Score", justify="right", style="green")
                        
                        for cat in tech_categories:
                            table_tech_bottom.add_column(cat.title(), justify="right")
                        
                        table_tech_bottom.add_column("Signals", style="green")
                        table_tech_bottom.add_column("Warnings", style="red")
                        
                        for ticker, score, details in tech_bottom_quartile:
                            # Get report data for signals and warnings
                            report_data = tech_reports_map.get(ticker, {})
                            
                            row = [ticker, f"{score:.8f}"]
                            for cat in tech_categories:
                                cat_score = details['category_scores'].get(cat, 0)
                                row.append(f"{cat_score:.8f}")
                            
                            signals = ", ".join(report_data.get("signals", [])[:2])  # Show at most 2 signals
                            warnings = ", ".join(report_data.get("warnings", [])[:2])  # Show at most 2 warnings
                            
                            row.append(signals)
                            row.append(warnings)
                            
                            table_tech_bottom.add_row(*row)
                        
                        rich_console.print(table_tech_bottom)
                    else:
                        rich_console.print("No technical quartile analysis available.")
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