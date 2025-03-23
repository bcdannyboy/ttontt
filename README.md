# TTONTT (To Trade or Not To Trade)

## A Comprehensive Stock Analysis Framework

TTONTT is an advanced financial analysis tool that integrates fundamental analysis, technical analysis, and Monte Carlo simulations to provide comprehensive stock screening and trading insights. The framework combines sophisticated mathematical models with efficient parallel processing to deliver in-depth analysis of stocks across multiple dimensions.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Core Components](#core-components)
- [Mathematical Models & Algorithms](#mathematical-models--algorithms)
- [Visualization & Reporting](#visualization--reporting)

## Overview

TTONTT aims to answer the eternal investor question: "To Trade or Not To Trade?" by combining multiple analytical approaches into a unified framework. The system:

1. Performs fundamental analysis to evaluate company financials and metrics
2. Applies technical analysis to identify price patterns and momentum
3. Runs stochastic Monte Carlo simulations to project future price movements
4. Integrates these approaches to provide a holistic view of investment opportunities

The system is optimized for performance with multi-threading, GPU acceleration where available, and efficient data processing techniques.

## Key Features

- **Multi-dimensional Stock Screening**: Evaluate stocks across fundamental, technical, and simulation-based metrics
- **Quartile Analysis**: Identify top and bottom performing stocks using statistical quartile breakdowns
- **Peer Comparison**: Compare stocks against their peers with adjustable weighting
- **Monte Carlo Simulation**: Project future price movements using multiple stochastic models
- **Rich Visualization**: Terminal-based tables and visualizations for analysis review
- **JSON Export**: Save analysis results for further processing or visualization
- **GPU Acceleration**: Leverage GPU/MPS capabilities for performance-intensive calculations
- **Parallel Processing**: Efficient multi-threaded and asynchronous execution

## Core Components

TTONTT consists of several specialized modules:

1. **Fundamental Analysis (fundamentals.py)**
   - Financial metrics extraction and scoring
   - Z-score normalization for cross-stock comparison
   - Category-based scoring system
   - Peer analysis integration

2. **Technical Analysis (technicals.py)**
   - Price pattern and momentum indicators
   - Volatility metrics
   - Volume and trend analysis
   - Technical signal detection

3. **Monte Carlo Simulation (montecarlo.py, models.py)**
   - Multiple stochastic models for price projection
   - Volatility model integration
   - Probability distribution analysis
   - Custom time horizon projections

4. **Parallel Execution (parallel.py)**
   - Optimized thread and process management
   - Task batching and workload distribution
   - Semaphore-controlled concurrency

5. **Historical Volatility Analysis (historical.py)**
   - Multiple volatility calculation methods
   - Timeframe optimization
   - Statistical analysis of volatility patterns

## Mathematical Models & Algorithms

### Fundamental Analysis

The fundamental analysis module evaluates companies based on several categories of financial metrics:

```
CATEGORY_WEIGHTS = {
    'profitability': 0.20,
    'growth': 0.20,
    'financial_health': 0.20,
    'valuation': 0.15,
    'efficiency': 0.10,
    'analyst_estimates': 0.15
}
```

Each category contains multiple metrics with specific weights. For example, profitability metrics include:

```
PROFITABILITY_WEIGHTS = {
    'gross_profit_margin': 0.20,
    'operating_income_margin': 0.20,
    'net_income_margin': 0.25,
    'ebitda_margin': 0.15,
    'return_on_equity': 0.10,
    'return_on_assets': 0.10
}
```

The system applies Z-score normalization to standardize metrics across different stocks:

1. Metrics are collected from financial data
2. Z-scores are calculated using robust standardization methods
3. Special handling for skewed distributions uses median and IQR instead of mean/std
4. Extreme outliers are clamped to a range of [-3, 3]

Peer comparison involves:
1. Identifying peer companies in the same sector
2. Calculating average scores for the peer group
3. Determining the delta between individual stock and peer average
4. Adjusting the composite score with a peer_weight factor (default 0.1)

### Technical Analysis

Technical analysis evaluates price patterns, trends, momentum, volatility, and volume indicators:

```
CATEGORY_WEIGHTS = {
    'trend': 0.30,
    'momentum': 0.30,
    'volatility': 0.20,
    'volume': 0.20
}
```

Key trend indicators include:
```
TREND_WEIGHTS = {
    'sma_cross_signal': 0.15,  # SMA crossover signal (50/200)
    'ema_cross_signal': 0.15,  # EMA crossover signal (12/26)
    'price_rel_sma': 0.10,     # Price relative to SMA (200-day)
    'price_rel_ema': 0.10,     # Price relative to EMA (50-day)
    'adx_trend': 0.15,         # ADX trend strength
    'ichimoku_signal': 0.10,   # Ichimoku Cloud signal
    'macd_signal': 0.15,       # MACD signal
    'bb_position': 0.10        # Position in Bollinger Bands
}
```

Similar weights exist for momentum, volatility, and volume categories. A unique aspect of the technical analysis is the handling of missing data, where the system:

1. Applies a missing data penalty
2. Requires a minimum number of valid metrics
3. Uses fallback calculations when specific indicators aren't available
4. Has a hierarchical dependency system for derived metrics

The system also calculates "expected price" and price ranges by combining different factors:

```
combined_factor = (w_pt * indicators.get('price_target_upside', 0) +
                   w_m * indicators.get('momentum_signal', 0) +
                   w_t * indicators.get('trend_signal', 0)) / (w_pt + w_m + w_t)
expected_price = current_price * (1 + combined_factor)
```

### Monte Carlo Simulation Models

The Monte Carlo module implements three stochastic models:

1. **Geometric Brownian Motion (GBM)**:
   The simplest model that assumes log-normal distribution of returns:
   ```
   dS = μS dt + σS dW
   ```
   Where:
   - dS is the change in stock price
   - μ is the drift (expected return)
   - σ is the volatility
   - dW is a Wiener process (random term)

2. **Heston Stochastic Volatility Model**:
   A more sophisticated model where volatility itself follows a stochastic process:
   ```
   dS = μS dt + √v S dW₁
   dv = κ(θ - v) dt + σ√v dW₂
   ```
   Where:
   - v is the variance
   - κ is the rate of mean reversion
   - θ is the long-term variance
   - σ is the volatility of volatility
   - dW₁ and dW₂ are correlated Wiener processes with correlation ρ

3. **SABR-CGMY Model**:
   A complex model combining stochastic volatility with jump processes:
   ```
   dS = μS dt + σ Sᵝ dW₁ + J
   dσ = α σ dW₂
   ```
   Where:
   - α is the volatility of volatility
   - β is the price elasticity parameter
   - J represents jumps following a CGMY process
   - dW₁ and dW₂ are correlated Wiener processes with correlation ρ

The CGMY process is defined by four parameters:
- C: overall level of activity
- G: rate of exponential decay for negative jumps
- M: rate of exponential decay for positive jumps
- Y: fine structure parameter controlling activity of small jumps

These models are implemented with GPU acceleration where available and include optimizations for efficient numerical computation.

### Historical Volatility Analysis

The system implements multiple volatility calculation methods:

1. **Realized Volatility**: Standard deviation of returns
2. **Parkinson Volatility**: Uses high-low price range
3. **Garman-Klass Volatility**: Incorporates OHLC data
4. **Rogers-Satchell Volatility**: Better for non-zero mean processes
5. **Yang-Zhang Volatility**: Accounts for overnight gaps

The system selects optimal volatility timeframes based on a scoring function:
```
score = avg_realized + abs(skew_realized) + abs(kurtosis_realized)
```
where lower scores are considered better for prediction purposes.

## Visualization & Reporting

TTONTT provides rich terminal-based visualization using the Rich library:

1. **Tables**: Displays results in formatted tables with color coding
2. **Progress Tracking**: Shows progress bars for long-running operations
3. **Quartile Analysis**: Visualizes top and bottom performing stocks
4. **Monte Carlo Results**: Displays simulation outcomes and probabilities
5. **JSON Export**: Saves comprehensive results in structured JSON format
