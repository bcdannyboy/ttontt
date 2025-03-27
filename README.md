# TTONTT (To Trade Or Not To Trade)

A comprehensive algorithmic trading analysis platform that combines fundamental and technical analysis with Monte Carlo simulations to evaluate stock opportunities.

## Overview

TTONTT is a Python-based system that helps traders make informed decisions using multiple analytical approaches:

1. **Fundamental Analysis** - Evaluates stocks based on financial metrics and ratios
2. **Technical Analysis** - Analyzes price patterns and market indicators
3. **Monte Carlo Simulation** - Projects potential price paths using stochastic models

The system processes a list of stock tickers, fetches financial data through the OpenBB SDK, and provides a comprehensive analysis with clear buy/sell signals.

## Features

- **Multi-factor Fundamental Screening**:
  - Profitability metrics (margins, ROE, ROA)
  - Growth indicators (revenue, earnings, EPS growth)
  - Financial health analysis (debt ratios, liquidity)
  - Valuation metrics (P/E, P/B, EV/EBITDA)
  - Efficiency metrics (turnover ratios)
  - Analyst estimates analysis

- **Technical Analysis**:
  - Trend indicators (SMA/EMA crosses, price relative to moving averages)
  - Momentum indicators (RSI, stochastic, CCI)
  - Volatility analysis (Bollinger Bands, ATR)
  - Volume indicators (OBV, AD Line, VWAP)

- **Advanced Monte Carlo Simulations**:
  - Geometric Brownian Motion (GBM)
  - Heston Stochastic Volatility Model
  - SABR-CGMY Model

- **Peer Comparison**:
  - Compare metrics against industry peers
  - Relative performance analysis

- **Rich Terminal Output**:
  - Colored, formatted tables
  - Progress bars
  - Detailed analysis reports

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ttontt.git
cd ttontt
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install OpenBB SDK:
```bash
pip install openbb
```

## Usage

Run the main script:

```bash
python ttontt.py
```

By default, the system will analyze a predefined list of tickers in `tickers.py`. You can modify this file to analyze different stocks.

### Customization

- Edit `tickers.py` to change the list of stocks to analyze
- Adjust scoring weights in the `fundamentals_core.py` file
- Modify simulation parameters in the `montecarlo.py` file

## Cache System

TTONTT implements a sophisticated caching system to improve performance and reduce API calls:

- **Memory Cache**: Stores financial data in memory during program execution
- **Disk Cache**: Persists financial data to disk in the `cache` directory
- **Cache TTL**: Cache entries expire after 24 hours by default
- **Metrics Cache**: Separate caching for calculated metrics

The cache system automatically handles API rate limiting and provides fallback mechanisms when data providers are unavailable.

## Output

### Console Output

The system displays rich, color-coded terminal output with multiple tables:

1. **Fundamental Screening Results** - Basic ranking of all analyzed stocks
2. **Fundamental Quartile Analysis** - Detailed breakdown by financial categories
3. **Technical Analysis Tables**:
   - Top Technical Quartile (Top Fundamentals)
   - Bottom Technical Quartile (Top Fundamentals)
   - Top Technical Quartile (Bottom Fundamentals)
   - Bottom Technical Quartile (Bottom Fundamentals)
4. **Monte Carlo Projections** with price targets and probabilities

### Quartile Analysis

The quartile analysis is a key feature that:

- Divides stocks into top and bottom quartiles based on fundamental scores
- Further subdivides each quartile based on technical analysis scores
- Creates a matrix of four groups (top/bottom fundamental × top/bottom technical)
- Helps identify stocks with the best combination of fundamentals and technicals
- Highlights potential reversal candidates (e.g., strong fundamentals with weak technicals)

### JSON Output

Results are saved to JSON files in the `output` directory:

1. **fundamental_screening_[timestamp].json** - Complete fundamental analysis
2. **technical_screening_[timestamp].json** - Technical indicator analysis
3. **monte_carlo_screening_[timestamp].json** - Monte Carlo simulation results
4. **final_json_[timestamp].json** - Comprehensive quartile-based analysis that combines all results

These JSON files can be used for further analysis or integration with other systems.

## Project Structure

- `ttontt.py` - Main entry point
- `tickers.py` - List of stocks to analyze
- `metal_coordinator.py` - Device management (CPU/GPU)
- `src/screener/` - Fundamental and technical analysis
- `src/simulation/` - Monte Carlo simulation models
