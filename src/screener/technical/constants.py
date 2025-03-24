import os
import torch

# Set device to MPS if available; otherwise use CPU.
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    logger_device = "MPS"
else:
    device = torch.device("cpu")
    logger_device = "CPU"

# Category weights for overall composite scoring.
CATEGORY_WEIGHTS = {
    'trend': 0.30,
    'momentum': 0.30,
    'volatility': 0.20,
    'volume': 0.20
}

# Weights for trend indicators.
TREND_WEIGHTS = {
    'sma_cross_signal': 0.15,
    'ema_cross_signal': 0.15,
    'price_rel_sma': 0.10,
    'price_rel_ema': 0.10,
    'adx_trend': 0.15,
    'ichimoku_signal': 0.10,
    'macd_signal': 0.15,
    'bb_position': 0.10
}

# Weights for momentum indicators.
MOMENTUM_WEIGHTS = {
    'rsi_signal': 0.20,
    'stoch_signal': 0.15,
    'cci_signal': 0.15,
    'clenow_momentum': 0.15,
    'fisher_transform': 0.10,
    'price_performance_1m': 0.10,
    'price_performance_3m': 0.15
}

# Weights for volatility indicators.
VOLATILITY_WEIGHTS = {
    'atr_percent': -0.20,
    'bb_width': -0.15,
    'keltner_width': -0.15,
    'volatility_cones': -0.20,
    'donchian_width': -0.15,
    'price_target_upside': 0.15
}

# Weights for volume indicators.
VOLUME_WEIGHTS = {
    'obv_trend': 0.25,
    'adl_trend': 0.20,
    'adosc_signal': 0.20,
    'vwap_position': 0.20,
    'volume_trend': 0.15
}

# API rate limiting and cache parameters.
API_CALLS_PER_MINUTE = 240
CACHE_SIZE = 1000

# Penalty values when data is missing.
MISSING_DATA_PENALTY = 1.0
MIN_VALID_METRICS = 3
