# TTONTT (To Trade or Not To Trade)

A comprehensive fundamental stock screening and analysis tool designed to help identify investment opportunities based on quantitative financial metrics.

## Overview

TTONTT (To Trade or Not To Trade) is a robust stock screening platform that evaluates companies across multiple fundamental categories including profitability, growth, financial health, valuation, efficiency, and analyst estimates. The tool assigns composite scores to stocks, compares them against peers, and generates detailed reports to aid in investment decision-making.

## Features

- **Comprehensive Fundamental Analysis**: Evaluates stocks across six key categories using over 40 financial metrics
- **Peer Comparison**: Automatically identifies and compares stocks against sector peers
- **Composite Scoring System**: Standardizes and weights metrics to produce actionable composite scores
- **Analyst Estimate Integration**: Incorporates forward-looking analyst estimates and their historical accuracy
- **Rich Visualization**: Displays results with color-coded tables and detailed quartile analysis
- **Performance Optimization**: Utilizes PyTorch with MPS acceleration on compatible hardware
- **API Rate Limiting**: Intelligently manages API requests to prevent throttling

## Installation

### Prerequisites

- Python 3.8+
- OpenBB SDK
- PyTorch (optimized for Apple Silicon if available)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ttontt.git
cd ttontt
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up your OpenBB API credentials (if required).

## Usage

### Basic Usage

Run the main screening script to analyze the default set of tickers:

```bash
python ttontt.py
```

### Custom Ticker Lists

Edit the `tickers.py` file to modify the list of tickers to analyze:

```python
tickers = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'META', 'NVDA']
```

### Automatically Fetch Active Tickers

TTONTT can also fetch active tickers from various market segments:

```bash
# Uncomment in ttontt.py
# init_tickers = get_active_tickers()
```

## How It Works

### Screening Process

1. **Data Collection**: Retrieves comprehensive financial data for each ticker using the OpenBB API
2. **Metric Extraction**: Processes raw financial data to extract relevant metrics across all categories
3. **Standardization**: Transforms metrics into z-scores for fair comparison across different scales
4. **Weighted Scoring**: Applies category-specific weights to each metric based on their importance
5. **Peer Analysis**: Compares each stock against peers to highlight relative strength/weakness
6. **Report Generation**: Creates detailed reports with strengths, weaknesses, and composite scores

### Scoring Categories

The system evaluates stocks across six fundamental categories:

- **Profitability**: Gross margin, operating margin, net margin, ROE, ROA
- **Growth**: Revenue growth, earnings growth, asset growth, EPS growth
- **Financial Health**: Current ratio, debt-to-equity, interest coverage, debt-to-assets
- **Valuation**: P/E ratio, price-to-book, EV/EBITDA, dividend yield
- **Efficiency**: Asset turnover, inventory turnover, cash conversion cycle
- **Analyst Estimates**: Estimate accuracy, consensus deviation, revision momentum

## Configuration

### Category Weights

Adjust the importance of different categories by modifying the weights in `fundamentals.py`:

```python
CATEGORY_WEIGHTS = {
    'profitability': 0.20,
    'growth': 0.20,
    'financial_health': 0.20,
    'valuation': 0.15,
    'efficiency': 0.10,
    'analyst_estimates': 0.15
}
```

### API Rate Limiting

Configure API rate limiting parameters in `fundamentals.py`:

```python
API_CALLS_PER_MINUTE = 240
api_semaphore = asyncio.Semaphore(40)  # Max concurrent API calls
```

## Example Output

The tool produces color-coded tables showing top and bottom quartile stocks with their scores:

```
┌────────────────────────────────────┐
│      Stock Scores                   │
├────────┬─────────────────────────┬─┤
│ Ticker │ Score                    │
├────────┼─────────────────────────┼─┤
│ NVDA   │ 0.21381838              │
│ BX     │ 0.14663058              │
│ JPM    │ 0.08723661              │
│ V      │ -0.00131246             │
│ META   │ -0.08294178             │
│ MSFT   │ -0.12133162             │
│ ARES   │ -0.26030381             │
└────────┴─────────────────────────┴─┘
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
