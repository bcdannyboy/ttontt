# TTONTT (To Trade or Not To Trade)

## A Comprehensive Stock Analysis Framework

TTONTT is an advanced financial analysis tool that integrates fundamental analysis, technical analysis, and Monte Carlo simulations to provide comprehensive stock screening and trading insights. The framework combines sophisticated mathematical models with efficient parallel processing to deliver in-depth analysis of stocks across multiple dimensions.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Core Components](#core-components)
- [Mathematical Models & Algorithms](#mathematical-models--algorithms)
- [Optimization & Parallelization](#optimization--parallelization)
- [Visualization & Reporting](#visualization--reporting)
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

## Core Components

TTONTT consists of several specialized modules:

1. **Fundamental Analysis**
   - Financial metrics extraction and scoring (`fundamentals_metrics.py`)
   - Data retrieval and processing (`fundamentals_data.py`)
   - Z-score normalization for comparison (`fundamentals_metrics.py`)
   - Peer analysis for relative comparison (`fundamentals_peers.py`)
   - Stock screening and ranking (`fundamentals_screen.py`)
   - Comprehensive comparison tools (`fundamentals_compare.py`)

2. **Technical Analysis (`technicals.py`)**
   - Price pattern and momentum indicators
   - Volatility metrics calculation
   - Volume and trend analysis
   - Technical signal detection
   - Multi-factor scoring system

3. **Monte Carlo Simulation (`montecarlo.py`, `models.py`)**
   - Multiple stochastic models (GBM, Heston, SABR-CGMY)
   - Volatility model integration
   - Probability distribution analysis
   - Custom time horizon projections
   - GPU-accelerated simulations

4. **Parallel Execution (`parallel.py`)**
   - Optimized thread and process management
   - Task batching and workload distribution
   - Semaphore-controlled concurrency
   - Resource-aware scaling

5. **Historical Volatility Analysis (`historical.py`)**
   - Multiple volatility calculation methods
   - Timeframe optimization
   - Statistical analysis of volatility patterns

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

This comprehensive implementation provides a sophisticated multi-dimensional analysis framework for stock evaluation, combining fundamental metrics, technical indicators, and future price projections through stochastic simulations.