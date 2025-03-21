from src.screener import fundamentals
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
    Synchronous version that's thread-safe.
    
    Args:
        ticker (str): The stock ticker symbol
        score (float): The composite score from screening
        details (dict): Detailed results from screening
        
    Returns:
        dict: Complete stock report with screening scores
    """
    try:
        report = fundamentals.generate_stock_report(ticker)
        report['category_scores'] = details['category_scores']
        report['composite_score'] = float(score)
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

def perform_stock_screening(tickers, batch_size=10):
    """
    Performs stock screening with batching to avoid overwhelming the API.
    
    Args:
        tickers (list): List of ticker symbols to screen
        batch_size (int): Number of tickers to process in each batch
        
    Returns:
        list: Screening results or empty list if error occurs
    """
    try:
        all_results = []
        
        # Process tickers in batches to avoid overwhelming the API
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size} ({len(batch)} tickers)")
            
            # Run the screening with optimized parameters
            batch_results = fundamentals.screen_stocks(batch)
            all_results.extend(batch_results)
            
            # Add a small delay between batches to avoid rate limiting
            if i + batch_size < len(tickers):
                time.sleep(2)
        
        if not all_results:
            logger.warning("No valid results were returned from the screening process.")
        
        # IMPORTANT: Sort the combined results by composite score (descending)
        all_results.sort(key=lambda x: x[1], reverse=True)
                
        # Remove any extremely large outliers that might be due to calculation errors
        if all_results:
            # Find median score
            scores = [score for _, score, _ in all_results]
            median_score = np.median(scores)
            
            # Filter out entries with absurdly high scores (e.g., >1000x median)
            threshold = abs(median_score) * 1000 if median_score != 0 else 1000
            filtered_results = [(ticker, score, details) for ticker, score, details in all_results 
                              if abs(score) < threshold]
            
            # Log if we removed any outliers
            if len(filtered_results) < len(all_results):
                logger.info(f"Removed {len(all_results) - len(filtered_results)} outlier scores")
                all_results = filtered_results
                
        return all_results
    except Exception as e:
        logger.error(f"Error during stock screening: {e}")
        logger.debug(traceback.format_exc())
        return []

def save_results_to_json(json_data, output_dir="output"):
    """
    Saves results to JSON file with error handling.
    
    Args:
        json_data (dict): Data to save to JSON
        output_dir (str): Directory to save the file
        
    Returns:
        str: Filename if successful, None if error occurs
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        output_filename = f"{output_dir}/fundamental_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save JSON file
        with open(output_filename, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        
        return output_filename
    except Exception as e:
        logger.error(f"Error saving results to JSON: {e}")
        logger.debug(traceback.format_exc())
        return None

if __name__ == "__main__":
    try:
        # Display ASCII art header
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
        
        # Define tickers to screen
        tickers = ['BX', 'KKR', 'APO', 'CG', 'TPG', 'ARES', 'EQT', 'PGHN.SW', 'BAM', 'ARCC', 
                  'OBDC', 'BXSL', 'FSK', 'MAIN', 'GBDC', 'HTGC', 'TSLX', 'PSEC', 'GSBD', 'OCSL', 
                  'MFIC', 'NMFC', 'KBDC', 'CSWC', 'BBDC', 'TRIN', 'PFLT', 'SLR', 'CGBDC', 'MSIF', 
                  'FDUS', 'CCAP', 'TCPC', 'GLAD', 'CION', 'GAIN', 'PNNT', 'RWAY', 'SCM', 'HRZN', 
                  'SARS', 'TPVG', 'WHF', 'OXSQ', 'MRCC', 'PTMN', 'SSSS', 'OFS', 'GECC', 'PFX', 
                  'LRFC', 'RAND', 'ICMB', 'EQS']
                  
        # Screen stocks with batching and error handling
        results = perform_stock_screening(tickers, batch_size=5)
        
        if not results:
            logger.error("No valid stocks were successfully screened. Check logs for details.")
            sys.exit(1)
            
        # Ensure the results are sorted by composite score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Process results and generate reports
        rich_console = Console()
        rich_console.print("\n[bold]Fundamental Screening Results:[/bold]")
        rich_console.print("-----------------------------")
        
        # Format results as a table (ordered by composite score descending)
        table = Table(title="Stock Scores")
        table.add_column("Ticker", style="cyan")
        table.add_column("Score", justify="right", style="green")
        
        for ticker, score, _ in results:
            table.add_row(ticker, f"{score:.2f}")
            
        rich_console.print(table)
        
        # Prepare data for JSON export
        json_data = {
            "screening_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tickers_analyzed": tickers,
            "stocks": []
        }
        
        # Generate reports with ThreadPoolExecutor for improved performance
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
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=min(10, len(results))) as executor:
                # Map the function over all results
                future_to_ticker = {
                    executor.submit(generate_stock_report_task, ticker, score, details): ticker 
                    for ticker, score, details in results
                }
                
                # Process completed futures
                for future in tqdm(future_to_ticker, total=len(future_to_ticker), 
                                  desc="Generating reports", leave=False):
                    try:
                        stock_report = future.result()
                        stock_reports.append(stock_report)
                    except Exception as e:
                        ticker = future_to_ticker[future]
                        logger.error(f"Error generating report for {ticker}: {e}")
                    
                    # Update progress
                    progress.update(task, advance=1)
        
        # Add to JSON data
        for stock_data in stock_reports:
            json_data["stocks"].append(stock_data)
            
        # Save results to JSON file
        output_filename = save_results_to_json(json_data)
        if output_filename:
            rich_console.print(f"\n[bold green]Results saved to {output_filename}[/bold green]")
        else:
            rich_console.print("\n[bold red]Error saving results to file. Check logs for details.[/bold red]")
            
        # Calculate quartiles for display
        try:
            scores = [score for _, score, _ in results]
            q1 = np.percentile(scores, 25)
            q3 = np.percentile(scores, 75)
            
            # Identify top and bottom quartile stocks and sort them by composite score
            top_quartile = [(ticker, score, details) for ticker, score, details in results if score >= q3]
            bottom_quartile = [(ticker, score, details) for ticker, score, details in results if score <= q1]
            top_quartile.sort(key=lambda x: x[1], reverse=True)
            bottom_quartile.sort(key=lambda x: x[1])
            
            # Display quartile analysis
            rich_console.print("\n[bold]Quartile Analysis:[/bold]")
            
            # Build tables and display them
            if top_quartile:
                # Get category names
                categories = list(top_quartile[0][2]['category_scores'].keys())
                table_top = Table(title="Top Quartile Stocks")
                table_top.add_column("Ticker", style="cyan")
                table_top.add_column("Score", justify="right", style="green")
                for cat in categories:
                    table_top.add_column(cat.title(), justify="right")
                for ticker, score, details in top_quartile:
                    row = [ticker, f"{score:.2f}"]
                    for cat in categories:
                        cat_score = details['category_scores'].get(cat, 0)
                        row.append(f"{cat_score:.2f}")
                    table_top.add_row(*row)
                
                if bottom_quartile:
                    # Get category names
                    categories = list(bottom_quartile[0][2]['category_scores'].keys())
                    table_bottom = Table(title="Bottom Quartile Stocks")
                    table_bottom.add_column("Ticker", style="cyan")
                    table_bottom.add_column("Score", justify="right", style="green")
                    for cat in categories:
                        table_bottom.add_column(cat.title(), justify="right")
                    for ticker, score, details in bottom_quartile:
                        row = [ticker, f"{score:.2f}"]
                        for cat in categories:
                            cat_score = details['category_scores'].get(cat, 0)
                            row.append(f"{cat_score:.2f}")
                        table_bottom.add_row(*row)
                    
                    # Display tables side by side
                    rich_console.print(Columns([table_top, table_bottom]))
                else:
                    rich_console.print(table_top)
            elif bottom_quartile:
                # Only bottom quartile exists
                categories = list(bottom_quartile[0][2]['category_scores'].keys())
                table_bottom = Table(title="Bottom Quartile Stocks")
                table_bottom.add_column("Ticker", style="cyan")
                table_bottom.add_column("Score", justify="right", style="green")
                for cat in categories:
                    table_bottom.add_column(cat.title(), justify="right")
                for ticker, score, details in bottom_quartile:
                    row = [ticker, f"{score:.2f}"]
                    for cat in categories:
                        cat_score = details['category_scores'].get(cat, 0)
                        row.append(f"{cat_score:.2f}")
                    table_bottom.add_row(*row)
                
                rich_console.print(table_bottom)
            else:
                rich_console.print("No quartile analysis available.")
                
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
