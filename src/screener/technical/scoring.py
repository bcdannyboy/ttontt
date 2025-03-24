from .constants import MISSING_DATA_PENALTY, MIN_VALID_METRICS

def calculate_weighted_score(z_scores, weights):
    if not z_scores or not weights:
        return -MISSING_DATA_PENALTY

    # Group metrics by category for fallback.
    category_metrics = {
        'trend': ['sma_cross_signal', 'ema_cross_signal', 'price_rel_sma', 'price_rel_ema',
                  'adx_trend', 'ichimoku_signal', 'macd_signal', 'bb_position', 'trend_signal'],
        'momentum': ['rsi_signal', 'stoch_signal', 'cci_signal', 'clenow_momentum',
                     'fisher_transform', 'price_performance_1m', 'price_performance_3m', 'momentum_signal'],
        'volatility': ['atr_percent', 'bb_width', 'keltner_width', 'volatility_cones',
                       'donchian_width', 'price_target_upside'],
        'volume': ['obv_trend', 'adl_trend', 'adosc_signal', 'vwap_position', 'volume_trend']
    }

    available_categories = set()
    for category, metrics in category_metrics.items():
        if any(metric in z_scores for metric in metrics):
            available_categories.add(category)

    fallback_used = False
    if len(available_categories) < 2:
        fallback_used = True
        if 'momentum' not in available_categories and ('price_performance_1m' in z_scores or 'price_performance_3m' in z_scores):
            available_categories.add('momentum')
        if 'volume' not in available_categories and 'volume_trend' in z_scores:
            available_categories.add('volume')

    score = 0
    total_weight = 0
    valid_metrics = 0

    common_metrics = set(z_scores.keys()) & set(weights.keys())
    for metric in common_metrics:
        weight = weights[metric]
        score += z_scores[metric] * weight
        total_weight += abs(weight)
        valid_metrics += 1

    if valid_metrics >= MIN_VALID_METRICS and total_weight > 0:
        base_score = score / total_weight
        total_metrics = len(weights)
        missing_count = total_metrics - valid_metrics
        penalty = (missing_count / total_metrics) * MISSING_DATA_PENALTY
        extra_penalty = MISSING_DATA_PENALTY if valid_metrics < MIN_VALID_METRICS else 0.0
        final_score = base_score - penalty - extra_penalty
        return final_score

    if fallback_used or valid_metrics < MIN_VALID_METRICS:
        category_scores = {}
        category_weights = {
            'trend': 0.30,
            'momentum': 0.30,
            'volatility': 0.20,
            'volume': 0.20
        }
        for category, metrics in category_metrics.items():
            available_metrics = [m for m in metrics if m in z_scores]
            if available_metrics:
                category_scores[category] = sum(z_scores[m] for m in available_metrics) / len(available_metrics)
        if category_scores:
            total_cat_weight = sum(category_weights[cat] for cat in category_scores)
            if total_cat_weight > 0:
                base_score = sum(category_scores[cat] * category_weights[cat] for cat in category_scores) / total_cat_weight
                final_score = base_score - MISSING_DATA_PENALTY
                return final_score

    if valid_metrics > 0:
        return (sum(z_scores[metric] for metric in common_metrics) / valid_metrics) - MISSING_DATA_PENALTY

    return -MISSING_DATA_PENALTY
