# /src/simulation/analysis.py
"""
Comprehensive analysis functions for Monte Carlo simulations.
"""

import asyncio
import concurrent.futures
import logging
import time
import traceback
from typing import List, Dict, Any
from tqdm import tqdm

from src.simulation.report import generate_monte_carlo_report_task

logger = logging.getLogger(__name__)

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
        import os
        max_workers = min(32, os.cpu_count() * 2)
    
    # Create a mapping for quick lookup of technical results
    tech_details = {ticker: (score, details) for ticker, score, details in technical_results}
    
    from rich.console import Console
    from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
    
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
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(batch))) as executor:
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