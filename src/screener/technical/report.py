import pandas as pd
import logging
import traceback
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def get_indicator_contributions(ticker: str) -> pd.DataFrame:
    from .screening import screen_stocks
    results = screen_stocks([ticker])
    if not results:
        logger.warning(f"No technical data found for {ticker}")
        return pd.DataFrame()
    _, _, detailed_results = results[0]
    z_scores = detailed_results['z_scores']
    raw_indicators = detailed_results['raw_indicators']
    data = []
    from .constants import TREND_WEIGHTS, MOMENTUM_WEIGHTS, VOLATILITY_WEIGHTS, VOLUME_WEIGHTS
    for category, weights in [
        ('Trend', TREND_WEIGHTS),
        ('Momentum', MOMENTUM_WEIGHTS),
        ('Volatility', VOLATILITY_WEIGHTS),
        ('Volume', VOLUME_WEIGHTS)
    ]:
        for indicator, weight in weights.items():
            if indicator in z_scores:
                z_score = z_scores[indicator]
                raw_value = raw_indicators.get(indicator)
                contribution = z_score * weight
                data.append({
                    'Category': category,
                    'Indicator': indicator,
                    'Raw Value': raw_value,
                    'Weight': weight,
                    'Z-Score': round(z_score, 2),
                    'Contribution': round(contribution, 3),
                    'Impact': 'Positive' if contribution > 0 else 'Negative'
                })
    df = pd.DataFrame(data)
    df['Abs Contribution'] = df['Contribution'].abs()
    df = df.sort_values('Abs Contribution', ascending=False)
    df = df.drop('Abs Contribution', axis=1)
    return df

def generate_stock_report(ticker: str) -> dict:
    from .utils import get_openbb_client
    obb_client = get_openbb_client()
    technical_data = {}
    try:
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        try:
            price_history_response = obb_client.equity.price.historical(
                symbol=ticker, start_date=one_year_ago, provider='fmp'
            )
            technical_data['price_history'] = price_history_response.results
        except Exception as e:
            logger.warning(f"Error fetching price history for {ticker}: {e}")
            technical_data['price_history'] = []
        try:
            price_perf_response = obb_client.equity.price.performance(
                symbol=ticker, provider='fmp'
            )
            technical_data['price_performance'] = price_perf_response.results
        except Exception as e:
            logger.warning(f"Error fetching price performance for {ticker}: {e}")
            technical_data['price_performance'] = []
        try:
            price_target_response = obb_client.equity.estimates.consensus(
                symbol=ticker, provider='fmp'
            )
            technical_data['price_target'] = price_target_response.results
        except Exception as e:
            logger.warning(f"Error fetching price target consensus for {ticker}: {e}")
            technical_data['price_target'] = []
        if technical_data.get('price_history'):
            price_data = technical_data['price_history']
            try:
                technical_data['sma_50'] = obb_client.technical.sma(data=price_data, target='close', length=50).results
            except Exception as e:
                logger.warning(f"Error calculating SMA 50 for {ticker}: {e}")
                technical_data['sma_50'] = []
            try:
                technical_data['sma_200'] = obb_client.technical.sma(data=price_data, target='close', length=200).results
            except Exception as e:
                logger.warning(f"Error calculating SMA 200 for {ticker}: {e}")
                technical_data['sma_200'] = []
            try:
                technical_data['ema_12'] = obb_client.technical.ema(data=price_data, target='close', length=12).results
            except Exception as e:
                logger.warning(f"Error calculating EMA 12 for {ticker}: {e}")
                technical_data['ema_12'] = []
            try:
                technical_data['ema_26'] = obb_client.technical.ema(data=price_data, target='close', length=26).results
            except Exception as e:
                logger.warning(f"Error calculating EMA 26 for {ticker}: {e}")
                technical_data['ema_26'] = []
            try:
                technical_data['rsi'] = obb_client.technical.rsi(data=price_data, target='close', length=14).results
            except Exception as e:
                logger.warning(f"Error calculating RSI for {ticker}: {e}")
                technical_data['rsi'] = []
            try:
                technical_data['macd'] = obb_client.technical.macd(data=price_data, target='close', fast=12, slow=26, signal=9).results
            except Exception as e:
                logger.warning(f"Error calculating MACD for {ticker}: {e}")
                technical_data['macd'] = []
            try:
                technical_data['bbands'] = obb_client.technical.bbands(data=price_data, target='close', length=20, std=2).results
            except Exception as e:
                logger.warning(f"Error calculating Bollinger Bands for {ticker}: {e}")
                technical_data['bbands'] = []
            try:
                technical_data['stoch'] = obb_client.technical.stoch(data=price_data, fast_k_period=14, slow_d_period=3).results
            except Exception as e:
                logger.warning(f"Error calculating Stochastic for {ticker}: {e}")
                technical_data['stoch'] = []
            try:
                technical_data['atr'] = obb_client.technical.atr(data=price_data, length=14).results
            except Exception as e:
                logger.warning(f"Error calculating ATR for {ticker}: {e}")
                technical_data['atr'] = []
            try:
                technical_data['adx'] = obb_client.technical.adx(data=price_data, length=14).results
            except Exception as e:
                logger.warning(f"Error calculating ADX for {ticker}: {e}")
                technical_data['adx'] = []
            try:
                technical_data['obv'] = obb_client.technical.obv(data=price_data).results
            except Exception as e:
                logger.warning(f"Error calculating OBV for {ticker}: {e}")
                technical_data['obv'] = []
            try:
                technical_data['ad'] = obb_client.technical.ad(data=price_data).results
            except Exception as e:
                logger.warning(f"Error calculating A/D Line for {ticker}: {e}")
                technical_data['ad'] = []
            try:
                technical_data['cci'] = obb_client.technical.cci(data=price_data, length=20).results
            except Exception as e:
                logger.warning(f"Error calculating CCI for {ticker}: {e}")
                technical_data['cci'] = []
            try:
                technical_data['adosc'] = obb_client.technical.adosc(data=price_data, fast=3, slow=10).results
            except Exception as e:
                logger.warning(f"Error calculating A/D Oscillator for {ticker}: {e}")
                technical_data['adosc'] = []
            try:
                technical_data['vwap'] = obb_client.technical.vwap(data=price_data, anchor='D').results
            except Exception as e:
                logger.warning(f"Error calculating VWAP for {ticker}: {e}")
                technical_data['vwap'] = []
            try:
                technical_data['kc'] = obb_client.technical.kc(data=price_data, length=20, scalar=2).results
            except Exception as e:
                logger.warning(f"Error calculating Keltner Channels for {ticker}: {e}")
                technical_data['kc'] = []
            try:
                technical_data['donchian'] = obb_client.technical.donchian(data=price_data, lower_length=20, upper_length=20).results
            except Exception as e:
                logger.warning(f"Error calculating Donchian Channels for {ticker}: {e}")
                technical_data['donchian'] = []
            try:
                technical_data['ichimoku'] = obb_client.technical.ichimoku(data=price_data, conversion=9, base=26).results
            except Exception as e:
                logger.warning(f"Error calculating Ichimoku Cloud for {ticker}: {e}")
                technical_data['ichimoku'] = []
            try:
                technical_data['clenow'] = obb_client.technical.clenow(data=price_data, period=90).results
            except Exception as e:
                logger.warning(f"Error calculating Clenow Momentum for {ticker}: {e}")
                technical_data['clenow'] = []
            try:
                technical_data['fisher'] = obb_client.technical.fisher(data=price_data, length=14).results
            except Exception as e:
                logger.warning(f"Error calculating Fisher Transform for {ticker}: {e}")
                technical_data['fisher'] = []
            try:
                technical_data['cones'] = obb_client.technical.cones(data=price_data, lower_q=0.25, upper_q=0.75, model='std').results
            except Exception as e:
                logger.warning(f"Error calculating Volatility Cones for {ticker}: {e}")
                technical_data['cones'] = []
            try:
                technical_data['price_targets'] = obb_client.equity.estimates.price_target(symbol=ticker, provider='fmp', limit=10).results
            except Exception as e:
                logger.warning(f"Error fetching price targets for {ticker}: {e}")
                technical_data['price_targets'] = []
    except Exception as e:
        logger.error(f"Error fetching technical data for {ticker}: {e}")
        logger.debug(traceback.format_exc())
    from .indicator_extraction import extract_indicators_from_technical_data
    indicators = extract_indicators_from_technical_data(technical_data)
    required_indicators = [
        'sma_cross_signal', 'ema_cross_signal', 'price_rel_sma', 'price_rel_ema',
        'adx_trend', 'ichimoku_signal', 'macd_signal', 'bb_position',
        'rsi_signal', 'stoch_signal', 'cci_signal', 'clenow_momentum',
        'fisher_transform', 'price_performance_1m', 'price_performance_3m',
        'atr_percent', 'bb_width', 'keltner_width', 'volatility_cones',
        'donchian_width', 'price_target_upside',
        'obv_trend', 'adl_trend', 'adosc_signal', 'vwap_position', 'volume_trend'
    ]
    for indicator in required_indicators:
        if indicator not in indicators:
            if indicator in ['sma_cross_signal', 'ema_cross_signal', 'macd_signal', 'trend_signal']:
                indicators[indicator] = 0.0
            elif indicator in ['rsi_signal', 'stoch_signal', 'cci_signal']:
                indicators[indicator] = 0.0
            elif indicator in ['atr_percent', 'bb_width', 'keltner_width', 'volatility_cones', 'donchian_width']:
                indicators[indicator] = 0.05
            elif indicator in ['price_target_upside']:
                indicators[indicator] = 0.05
            elif indicator in ['obv_trend', 'adl_trend', 'adosc_signal', 'vwap_position', 'volume_trend']:
                indicators[indicator] = 0.0
            else:
                indicators[indicator] = 0.0
    report = {
        'ticker': ticker,
        'composite_score': 0.0,
        'category_scores': {
            'trend': 0.0,
            'momentum': 0.0,
            'volatility': 0.0,
            'volume': 0.0
        },
        'key_indicators': {
            'trend': {
                'sma_cross': indicators.get('sma_cross_signal', 0.0),
                'price_rel_sma': indicators.get('price_rel_sma', 0.0),
                'macd_signal': indicators.get('macd_signal', 0.0),
                'adx_trend': indicators.get('adx_trend', 0.0)
            },
            'momentum': {
                'rsi': indicators.get('rsi', 50.0),
                'stoch_k': indicators.get('stoch_k', 50.0),
                'stoch_d': indicators.get('stoch_d', 50.0),
                'price_performance_1m': indicators.get('price_performance_1m', 0.0),
                'price_performance_3m': indicators.get('price_performance_3m', 0.0)
            },
            'volatility': {
                'atr_percent': indicators.get('atr_percent', 0.05),
                'bb_width': indicators.get('bb_width', 0.05),
                'price_target_upside': indicators.get('price_target_upside', 0.05)
            },
            'volume': {
                'obv_trend': indicators.get('obv_trend', 0.0),
                'adl_trend': indicators.get('adl_trend', 0.0),
                'volume_trend': indicators.get('volume_trend', 0.0)
            }
        },
        'signals': [],
        'warnings': [],
        'raw_indicators': indicators
    }
    if indicators.get('rsi') is not None:
        rsi_value = indicators['rsi']
        if rsi_value >= 70:
            report['warnings'].append(f"RSI is overbought at {rsi_value:.1f}")
        elif rsi_value <= 30:
            report['signals'].append(f"RSI is oversold at {rsi_value:.1f}")
    if indicators.get('macd_signal') is not None:
        macd_signal = indicators['macd_signal']
        if macd_signal > 0:
            report['signals'].append("MACD indicates bullish momentum")
        elif macd_signal < 0:
            report['warnings'].append("MACD indicates bearish momentum")
    if indicators.get('sma_cross_signal') is not None:
        sma_cross = indicators['sma_cross_signal']
        if sma_cross > 0:
            report['signals'].append("Golden Cross: 50-day SMA above 200-day SMA")
        elif sma_cross < 0:
            report['warnings'].append("Death Cross: 50-day SMA below 200-day SMA")
    if indicators.get('price_rel_sma_200') is not None:
        price_rel_sma = indicators['price_rel_sma_200']
        if price_rel_sma > 0.05:
            report['signals'].append(f"Price is {price_rel_sma*100:.1f}% above 200-day SMA")
        elif price_rel_sma < -0.05:
            report['warnings'].append(f"Price is {-price_rel_sma*100:.1f}% below 200-day SMA")
    if indicators.get('bb_position') is not None:
        bb_pos = indicators['bb_position']
        if bb_pos > 0.8:
            report['warnings'].append("Price near upper Bollinger Band, potentially overbought")
        elif bb_pos < -0.8:
            report['signals'].append("Price near lower Bollinger Band, potentially oversold")
    if indicators.get('price_target_upside') is not None:
        upside = indicators['price_target_upside']
        if upside > 0.15:
            report['signals'].append(f"Analyst consensus suggests {upside*100:.1f}% upside potential")
        elif upside < -0.15:
            report['warnings'].append(f"Analyst consensus suggests {-upside*100:.1f}% downside risk")
    if indicators.get('price_performance_1m') is not None:
        perf_1m = indicators['price_performance_1m']
        if perf_1m > 0.1:
            report['signals'].append(f"Strong 1-month performance: +{perf_1m*100:.1f}%")
        elif perf_1m < -0.1:
            report['warnings'].append(f"Weak 1-month performance: {perf_1m*100:.1f}%")
    if indicators.get('volume_trend') is not None:
        vol_trend = indicators['volume_trend']
        if vol_trend > 0.2:
            report['signals'].append(f"Increasing volume trend: +{vol_trend*100:.1f}%")
        elif vol_trend < -0.2:
            report['warnings'].append(f"Decreasing volume trend: {vol_trend*100:.1f}%")
    return report
