# TTONTT (To Trade Or Not To Trade)

A comprehensive algorithmic trading analysis platform that combines fundamental and technical analysis with Monte Carlo simulations to evaluate stock opportunities.

## Overview

TTONTT is a Python-based system that helps traders make informed decisions using multiple analytical approaches:

1. **Fundamental Analysis** - Evaluates stocks based on financial metrics and ratios
2. **Technical Analysis** - Analyzes price patterns and market indicators
3. **Monte Carlo Simulation** - Projects potential price paths using stochastic models

The system processes a list of stock tickers, fetches financial data through the OpenBB SDK, and provides a comprehensive analysis with clear buy/sell signals.

## API Key Requirement

**Important**: This system requires an API key from Financial Modeling Prep (FMP) to fetch financial data through OpenBB. 

1. Sign up for an API key at [Financial Modeling Prep](https://site.financialmodelingprep.com/developer/docs/)
2. Configure OpenBB with your API key following the [OpenBB Documentation](https://docs.openbb.co/)

Without a valid API key, the system will be unable to fetch fundamental and technical data.

## Features

### Multi-factor Fundamental Screening

The system evaluates stocks across 6 major categories with specific metrics:

#### 1. Profitability Metrics
- Gross Profit Margin
- Operating Income Margin
- Net Income Margin
- EBITDA Margin
- Return on Equity (ROE)
- Return on Assets (ROA)

#### 2. Growth Indicators
- Revenue Growth
- Gross Profit Growth
- EBITDA Growth
- Operating Income Growth
- Net Income Growth
- EPS Growth
- Total Assets Growth
- Shareholders' Equity Growth

#### 3. Financial Health
- Current Ratio
- Quick Ratio
- Debt-to-Equity Ratio
- Debt-to-Assets Ratio
- Interest Coverage Ratio
- Cash-to-Debt Ratio
- Total Debt Growth
- Net Debt Growth

#### 4. Valuation Metrics
- P/E Ratio
- Price-to-Book Ratio
- Price-to-Sales Ratio
- EV/EBITDA
- Dividend Yield
- PEG Ratio

#### 5. Efficiency Metrics
- Asset Turnover
- Inventory Turnover
- Receivables Turnover
- Cash Conversion Cycle
- CAPEX-to-Revenue Ratio

#### 6. Analyst Estimates
- EPS Estimate Accuracy
- Revenue Estimate Accuracy
- Forward Sales Growth
- Forward EBITDA Growth
- Estimate Revision Momentum
- Estimate Consensus Deviation

### Technical Analysis

The system calculates and analyzes numerous technical indicators:

#### 1. Trend Indicators
- SMA Cross Signal (50-day vs 200-day)
- EMA Cross Signal (12-day vs 26-day)
- Price Relative to SMA
- Price Relative to EMA
- ADX Trend Strength
- Ichimoku Cloud Signal
- MACD Signal
- Bollinger Band Position

#### 2. Momentum Indicators
- RSI Signal (14-day)
- Stochastic Oscillator Signal
- CCI Signal
- Clenow Momentum
- Fisher Transform
- 1-Month Price Performance
- 3-Month Price Performance

#### 3. Volatility Indicators
- ATR Percent
- Bollinger Band Width
- Keltner Channel Width
- Volatility Cones
- Donchian Channel Width
- Price Target Upside

#### 4. Volume Indicators
- OBV Trend
- A/D Line Trend
- A/D Oscillator Signal
- VWAP Position
- Volume Trend

### Advanced Monte Carlo Simulations
- Geometric Brownian Motion (GBM)
- Heston Stochastic Volatility Model
- SABR-CGMY Model

### Peer Comparison
- Compare metrics against industry peers
- Relative performance analysis

### Rich Terminal Output
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

5. Configure OpenBB with your FMP API key:
```bash
# Through Python
from openbb import obb
obb.login(provider="fmp", api_key="YOUR_FMP_API_KEY")

# Or through OpenBB Terminal (if installed)
openbb login --provider fmp --key YOUR_FMP_API_KEY
```

## OpenBB Functions Used

The system leverages numerous OpenBB functions for data retrieval and analysis:

### Fundamental Data Functions
- `obb.equity.fundamental.income` - Income statement data
- `obb.equity.fundamental.balance` - Balance sheet data
- `obb.equity.fundamental.cash` - Cash flow statement data
- `obb.equity.fundamental.income_growth` - Income growth metrics
- `obb.equity.fundamental.balance_growth` - Balance sheet growth metrics
- `obb.equity.fundamental.cash_growth` - Cash flow growth metrics
- `obb.equity.fundamental.ratios` - Financial ratios
- `obb.equity.fundamental.metrics` - Market metrics
- `obb.equity.fundamental.dividends` - Dividend data
- `obb.equity.fundamental.earnings` - Earnings data
- `obb.equity.fundamental.overview` - Company overview data

### Estimates and Comparison Functions
- `obb.equity.estimates.historical` - Historical analyst estimates
- `obb.equity.estimates.forward_sales` - Forward sales estimates
- `obb.equity.estimates.forward_ebitda` - Forward EBITDA estimates
- `obb.equity.estimates.consensus` - Analyst consensus
- `obb.equity.estimates.price_target` - Price targets
- `obb.equity.compare.peers` - Peer company data

### Technical Analysis Functions
- `obb.equity.price.historical` - Historical price data
- `obb.equity.price.performance` - Price performance metrics
- `obb.technical.sma` - Simple Moving Average
- `obb.technical.ema` - Exponential Moving Average
- `obb.technical.bbands` - Bollinger Bands
- `obb.technical.kc` - Keltner Channels
- `obb.technical.macd` - MACD indicator
- `obb.technical.rsi` - Relative Strength Index
- `obb.technical.stoch` - Stochastic Oscillator
- `obb.technical.cci` - Commodity Channel Index
- `obb.technical.adx` - Average Directional Index
- `obb.technical.obv` - On-Balance Volume
- `obb.technical.ad` - Accumulation/Distribution Line
- `obb.technical.atr` - Average True Range
- `obb.technical.donchian` - Donchian Channels
- `obb.technical.fisher` - Fisher Transform
- `obb.technical.ichimoku` - Ichimoku Cloud
- `obb.technical.adosc` - A/D Oscillator
- `obb.technical.vwap` - Volume Weighted Average Price
- `obb.technical.clenow` - Clenow Momentum
- `obb.technical.cones` - Volatility Cones

### Discovery Functions
- `obb.equity.discovery.growth_tech` - Growth tech stocks
- `obb.equity.discovery.undervalued_large_caps` - Undervalued large caps
- `obb.equity.discovery.aggressive_small_caps` - Aggressive small caps
- `obb.equity.discovery.undervalued_growth` - Undervalued growth stocks
- `obb.equity.discovery.gainers` - Biggest gainers
- `obb.equity.discovery.losers` - Biggest losers
- `obb.equity.discovery.active` - Most active stocks

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
