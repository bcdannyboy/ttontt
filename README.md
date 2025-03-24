# TTONTT (To Trade or Not To Trade)

## A Comprehensive Stock Analysis Framework

TTONTT is an advanced financial analysis tool that integrates fundamental analysis, technical analysis, and Monte Carlo simulations to provide comprehensive stock screening and trading insights. The framework combines sophisticated mathematical models with efficient parallel processing to deliver in-depth analysis of stocks across multiple dimensions.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Mathematical Models & Algorithms](#mathematical-models--algorithms)
- [Optimization & Parallelization](#optimization--parallelization)
- [Visualization & Reporting](#visualization--reporting)
- [Output Structure & Interpretation](#output-structure--interpretation)
- [Implementation Details](#implementation-details)

## Overview

TTONTT answers the eternal investor question: "To Trade or Not To Trade?" by combining multiple analytical approaches into a unified framework. The system:

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

## Mathematical Models & Algorithms

### Fundamental Analysis

The fundamental analysis module evaluates companies based on six categories of financial metrics with specific weightings:

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

Each category contains multiple metrics with specific weights. For example, profitability metrics include:

```python
PROFITABILITY_WEIGHTS = {
    'gross_profit_margin': 0.20,
    'operating_income_margin': 0.20,
    'net_income_margin': 0.25,
    'ebitda_margin': 0.15,
    'return_on_equity': 0.10,
    'return_on_assets': 0.10
}
```

Growth metrics focus on year-over-year improvements:

```python
GROWTH_WEIGHTS = {
    'growth_revenue': 0.20,
    'growth_gross_profit': 0.15,
    'growth_ebitda': 0.15,
    'growth_net_income': 0.20,
    'growth_eps': 0.15,
    'growth_total_assets': 0.05,
    'growth_total_shareholders_equity': 0.10
}
```

Financial health evaluates the company's stability:

```python
FINANCIAL_HEALTH_WEIGHTS = {
    'current_ratio': 0.15,
    'debt_to_equity': -0.20,  # Negative weight - lower is better
    'debt_to_assets': -0.15,
    'growth_total_debt': -0.15,
    'growth_net_debt': -0.15,
    'interest_coverage': 0.10,
    'cash_to_debt': 0.10
}
```

Valuation compares price to fundamentals:

```python
VALUATION_WEIGHTS = {
    'pe_ratio': -0.25,  # Negative weight - lower is better
    'price_to_book': -0.15,
    'price_to_sales': -0.15,
    'ev_to_ebitda': -0.20,
    'dividend_yield': 0.15,
    'peg_ratio': -0.10
}
```

Efficiency metrics evaluate company operations:

```python
EFFICIENCY_WEIGHTS = {
    'asset_turnover': 0.25,
    'inventory_turnover': 0.20,
    'receivables_turnover': 0.20,
    'cash_conversion_cycle': -0.20,
    'capex_to_revenue': -0.15
}
```

Analyst estimates evaluate forward-looking data quality:

```python
ANALYST_ESTIMATES_WEIGHTS = {
    'estimate_eps_accuracy': 0.35,
    'estimate_revenue_accuracy': 0.35,
    'estimate_consensus_deviation': -0.10,
    'estimate_revision_momentum': 0.20
}
```

The system applies Z-score normalization to standardize metrics across different stocks:

1. Raw metrics are collected from financial data sources (OpenBB API)
2. Z-scores are calculated using robust statistical methods:
   ```python
   z_score = (value - mean) / std_dev  # Standard calculation
   ```
3. For skewed distributions, the system uses median and IQR:
   ```python
   if skewness > 2:
       z_score = (value - median) / (IQR / 1.349)
   ```
4. Outliers are clamped to a range of [-3, 3]

The peer comparison process:
1. Identifies peer companies in the same sector
2. Calculates the average score for the peer group
3. Determines the delta between individual stock and peer average:
   ```python
   peer_delta = composite_score - peer_avg
   ```
4. Adjusts the final score with weighted peer comparison:
   ```python
   adjusted_score = composite_score + (peer_weight * peer_delta)
   ```

### Technical Analysis

Technical analysis evaluates price patterns, trends, momentum, volatility, and volume indicators:

```python
CATEGORY_WEIGHTS = {
    'trend': 0.30,
    'momentum': 0.30,
    'volatility': 0.20,
    'volume': 0.20
}
```

Trend indicators evaluate price direction and strength:

```python
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

Momentum indicators focus on the rate of price changes:

```python
MOMENTUM_WEIGHTS = {
    'rsi_signal': 0.20,             # RSI signal
    'stoch_signal': 0.15,           # Stochastic signal
    'cci_signal': 0.15,             # CCI signal
    'clenow_momentum': 0.15,        # Clenow Volatility Adjusted Momentum
    'fisher_transform': 0.10,       # Fisher Transform
    'price_performance_1m': 0.10,   # 1-month price performance
    'price_performance_3m': 0.15    # 3-month price performance
}
```

Volatility indicators assess price stability and potential ranges:

```python
VOLATILITY_WEIGHTS = {
    'atr_percent': -0.20,        # ATR as percentage of price (lower is better)
    'bb_width': -0.15,           # Bollinger Band width (narrower often better)
    'keltner_width': -0.15,      # Keltner Channel width
    'volatility_cones': -0.20,   # Volatility cones position
    'donchian_width': -0.15,     # Donchian channel width
    'price_target_upside': 0.15  # Price target upside potential
}
```

Volume indicators evaluate trading activity significance:

```python
VOLUME_WEIGHTS = {
    'obv_trend': 0.25,          # OBV trend
    'adl_trend': 0.20,          # Accumulation/Distribution Line trend
    'adosc_signal': 0.20,       # Accumulation/Distribution Oscillator
    'vwap_position': 0.20,      # Position relative to VWAP
    'volume_trend': 0.15        # Volume trend
}
```

The system calculates comprehensive price projections using a weighted combination:

```python
combined_factor = (w_pt * indicators.get('price_target_upside', 0) +
                   w_m * indicators.get('momentum_signal', 0) +
                   w_t * indicators.get('trend_signal', 0)) / (w_pt + w_m + w_t)
expected_price = current_price * (1 + combined_factor)
```

Missing data is handled with hierarchical fallbacks:
- Primary data → Derived indicators → Synthetic values
- Each fallback has data-specific logic based on known relationships
- A missing data penalty is applied to ensure conservative scoring

Signal generation includes both bullish and bearish indicators:
- Moving average crosses (golden/death cross)
- Overbought/oversold conditions (RSI, Stochastic)
- Support/resistance breaches
- Volume confirmation signals
- Pattern completion triggers

### Monte Carlo Simulation Models

The Monte Carlo module implements three stochastic models with increasing complexity:

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
   - β is the price elasticity parameter (default 0.5)
   - J represents jumps following a CGMY process
   - dW₁ and dW₂ are correlated Wiener processes with correlation ρ (default -0.5)

The CGMY process is defined by four parameters:
- C: overall level of activity (default 1.0)
- G: rate of exponential decay for negative jumps (default 5.0)
- M: rate of exponential decay for positive jumps (default 10.0)
- Y: fine structure parameter (default 0.5)

Implementation details:
- Uses vectorized operations for efficiency
- Implements Euler-Maruyama discretization for SDE simulation
- Applies antithetic variates for variance reduction
- Uses truncation and compensation for jump processes

The system calculates multiple statistics from simulations:
- Expected price (mean of simulations)
- Price range (various percentiles)
- Probability of increase/decrease
- Probability of specific percentage moves

### Historical Volatility Analysis

The system implements multiple volatility calculation methods:

1. **Realized Volatility**: Standard deviation of returns
   ```python
   realized_vol = np.std(returns) * np.sqrt(trading_periods)
   ```

2. **Parkinson Volatility**: Uses high-low price range
   ```python
   squared_log_range = (np.log(high / low) ** 2) / (4 * np.log(2))
   parkinson_vol = np.sqrt(np.mean(squared_log_range) * trading_periods)
   ```

3. **Garman-Klass Volatility**: Incorporates OHLC data
   ```python
   gk_estimator = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
   gk_vol = np.sqrt(np.mean(gk_estimator) * trading_periods)
   ```

4. **Rogers-Satchell Volatility**: Better for non-zero mean processes
   ```python
   rs_estimator = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
   rs_vol = np.sqrt(np.mean(rs_estimator) * trading_periods)
   ```

5. **Yang-Zhang Volatility**: Accounts for overnight gaps
   ```python
   sigma_yz = sigma_overnight + k * sigma_open_close + (1 - k) * sigma_rs
   yz_vol = np.sqrt(sigma_yz * trading_periods)
   ```

The system selects optimal volatility timeframes based on a scoring function:
```python
score = avg_realized + abs(skew_realized) + abs(kurtosis_realized)
```

Volatility cone analysis provides context for current volatility:
- Compares current values to historical ranges
- Identifies potential mean reversion opportunities
- Assesses volatility regime shifts
- Supports option pricing strategies

## Optimization & Parallelization

The codebase implements several optimization techniques:

1. **GPU Acceleration**
   - Uses PyTorch for tensor operations
   - Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU fallback
   - Automatically detects and utilizes available hardware
   - Prioritizes GPU for large tensor operations

2. **Thread and Process Management**
   - Maintains reusable thread pools to avoid creation overhead
   - Uses process pools for CPU-intensive tasks
   - Implements task batching for optimal workload distribution
   - Dynamically adjusts concurrency based on system resources

3. **Concurrency Control**
   - Uses semaphores to limit concurrent API calls
   - Implements provider-specific rate limiting
   - Controls event loop overloading with throttling
   - Manages resource contention with locks

4. **Memory Optimization**
   - Implements in-memory caching for API results
   - Uses disk caching for persistence between runs
   - Applies TTL (time-to-live) for cache entries
   - Prunes cache to limit memory usage

5. **API Optimization**
   - Implements provider-specific rate limiting
   - Uses fallback providers when rate limits are reached
   - Handles multiple data sources with unified interface
   - Optimizes request patterns to minimize API calls

## Visualization & Reporting

The framework uses terminal-based visualization and data export:

1. **Rich Terminal Interface**
   - Formatted tables with color coding for key metrics
   - Progress tracking with detailed statistics
   - Hierarchical layout for comparative analysis
   - Interactive display for large datasets

2. **Quartile Analysis Visualization**
   - Statistical breakdown of top and bottom performers
   - Multi-dimensional rating across categories
   - Combined technical and fundamental displays
   - Comparative peer metrics

3. **Monte Carlo Visualization**
   - Price projection ranges with probability metrics
   - Multiple time horizon analysis
   - Volatility model comparison
   - Risk assessment statistics

4. **JSON Export**
   - Structured data for external processing
   - Timestamped reports for tracking
   - Complete analysis results with all metrics
   - Integration-ready format

## Output Structure & Interpretation

The framework produces a multi-layered analysis output presented as formatted terminal tables:

### Fundamental Screening Results
The first table shows all analyzed tickers sorted by their composite fundamental scores. These scores represent overall financial strength based on the weighted metrics across all six fundamental categories.

### Fundamental Quartile Analysis
This section shows detailed tables for:
- **Top Quartile Stocks**: Companies in the top 25% of fundamental scores
- **Bottom Quartile Stocks**: Companies in the bottom 25% of fundamental scores

Each table provides the composite score plus the breakdown by category (Profitability, Growth, Financial Health, Valuation, Efficiency, and Analyst Estimates). The tables also include peer comparison data:
- **Peer Avg**: Average score of peer companies in the same sector
- **Peer Delta**: Difference between the stock's score and its peer average

### Technical & Monte Carlo Analysis Tables
These tables present stocks organized by both their fundamental and technical rankings. The system generates four tables:
- **Top Technical Quartile (Top Fundamentals)**: Strong technical signals with strong fundamentals
- **Bottom Technical Quartile (Top Fundamentals)**: Weak technical signals with strong fundamentals
- **Top Technical Quartile (Bottom Fundamentals)**: Strong technical signals with weak fundamentals
- **Bottom Technical Quartile (Bottom Fundamentals)**: Weak technical signals with weak fundamentals

Each table includes:

1. **Basic Information**:
   - Ticker: Stock symbol
   - Score: Technical analysis composite score
   - Current: Current stock price

2. **Technical Price Projections** (21-day horizon):
   - Tech Exp: Expected price based on technical indicators
   - Tech Low: Lower bound of the technical price projection
   - Tech High: Upper bound of the technical price projection

3. **Monte Carlo Projections** (21-day horizon):
   - MC Exp $: Expected price from Monte Carlo simulations
   - MC Exp Δ: Expected percentage change
   - MC Low $: Lower bound price (5th percentile)
   - MC Low %: Probability of dropping 10% or more
   - MC High $: Upper bound price (95th percentile)
   - MC High %: Probability of rising 10% or more

4. **Category Scores**:
   - Trend: Score for trend indicators
   - Momentum: Score for momentum indicators
   - Volatility: Score for volatility indicators
   - Volume: Score for volume indicators

5. **Signals & Warnings**:
   - Signals: Bullish technical indicators (e.g., "RSI is oversold")
   - Warnings: Bearish technical indicators (e.g., "Death Cross")

### Interpretation Guidelines

1. **Time Horizon**: All technical and Monte Carlo projections are for a **21-day (1-month)** time horizon.

2. **Trading Opportunities**:
   - Highest probability entries: Top Technical Quartile of Top Fundamental stocks
   - Potential reversals: Bottom Technical Quartile of Top Fundamental stocks
   - Short-term trades: Top Technical Quartile of Bottom Fundamental stocks
   - Potential shorts: Bottom Technical Quartile of Bottom Fundamental stocks

3. **Risk Assessment**:
   - Compare MC Low % and MC High % to evaluate risk-reward ratio
   - Higher probability differentials indicate stronger directional bias
   - Technical signals confirm Monte Carlo projections when aligned

4. **Decision Framework**:
   - Strong fundamentals + strong technicals + favorable Monte Carlo: Consider entry
   - Strong fundamentals + weak technicals: Consider watching for reversal
   - Weak fundamentals + strong technicals: Consider short-term trades only
   - Weak fundamentals + weak technicals: Consider avoiding or shorting

5. **Confidence Assessment**:
   - Higher agreement between technical projections and Monte Carlo results indicates higher confidence
   - Larger gaps between current price and projections suggest stronger potential moves
   - Multiple aligned signals reinforce the analysis

## Implementation Details

### Data Sources and API Integration

The system integrates with OpenBB (Open Bloomberg) API for financial data and uses multiple providers as fallbacks:

```python
# Provider priority order
providers = ['fmp', 'intrinio', 'yfinance', 'polygon', 'alphavantage']
```

Rate limits are carefully managed:

```python
API_CALLS_PER_MINUTE = {
    'default': 240,
    'fmp': 300,
    'intrinio': 100,
    'yfinance': 2000,
    'polygon': 50,
    'alphavantage': 75
}
```

### Error Handling and Fallbacks

The system implements robust error handling with fallback mechanisms:

1. Z-score calculation fallbacks:
   - Standard z-score calculation → Robust z-score with median/IQR → Min-max normalization
   
2. Missing data handling:
   - API-based peer analysis → Synthetic peer generation
   
3. Metric recovery strategies:
   - Primary metrics → Derived metrics → Synthetic metrics

### Peer Analysis Implementation

The peer analysis combines industry-standard methods with synthetic approaches:

1. API-based peer identification:
   ```python
   peers_list = await get_peers_async(ticker)
   ```

2. Cross-peer score calculation:
   ```python
   peer_average = np.mean(peer_scores)
   peer_std_dev = np.std(peer_scores)
   below_count = sum(1 for s in peer_scores if s < score)
   peer_percentile = (below_count / len(peer_scores)) * 100
   ```

3. Fallback with synthetic peers based on similarity metrics:
   ```python
   similarity = sum(1 - min(abs(cat_scores[category] - other_cat_scores[category]), 1.0) for category in cat_scores) / count
   ```

4. Score adjustment with configurable peer influence:
   ```python
   peer_delta = composite_score - peer_avg
   adjusted_score = composite_score + (peer_weight * peer_delta)
   ```

### Monte Carlo Simulation Pipeline

The Monte Carlo simulation process follows these steps:

1. Model calibration using historical data:
   - Drift calculation from returns
   - Volatility from multiple models
   - Heston parameter estimation
   - SABR-CGMY parameter fitting

2. Multi-model simulation:
   - Runs GBM, Heston, and SABR-CGMY in parallel
   - Combines results with equal weighting
   - Applies multiple time horizons
   - Uses GPU acceleration for large simulations

3. Result processing:
   - Calculates comprehensive statistics
   - Determines probability distributions
   - Generates expected price ranges
   - Computes risk metrics

4. Reporting integration:
   - Formats simulation results for display
   - Integrates with technical indicators
   - Combines with fundamental metrics
   - Provides comprehensive trading insights