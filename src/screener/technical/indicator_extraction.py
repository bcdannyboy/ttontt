import numpy as np
import pandas as pd
import torch
import logging

from .utils import get_attribute_value
from .constants import device

logger = logging.getLogger(__name__)

def extract_indicators_from_technical_data(technical_data):
    indicators = {}
    if not technical_data or not isinstance(technical_data, dict):
        return {}
    has_price_history = bool(technical_data.get('price_history'))
    has_price_performance = bool(technical_data.get('price_performance'))
    has_sma_data = bool(technical_data.get('sma_50')) or bool(technical_data.get('sma_200'))
    has_ema_data = bool(technical_data.get('ema_12')) or bool(technical_data.get('ema_26')) or bool(technical_data.get('ema_50'))
    has_macd_data = bool(technical_data.get('macd'))
    has_rsi_data = bool(technical_data.get('rsi'))
    has_stoch_data = bool(technical_data.get('stoch'))
    has_volume_data = has_price_history

    def extract_value(data_obj):
        if data_obj is None:
            return None
        if isinstance(data_obj, (int, float)):
            return data_obj
        if hasattr(data_obj, 'value'):
            return data_obj.value
        if hasattr(data_obj, '__getitem__'):
            try:
                return data_obj['value']
            except (KeyError, TypeError):
                pass
        try:
            return float(data_obj)
        except (ValueError, TypeError):
            return None

    current_price = None
    if has_price_history:
        price_history = technical_data['price_history']
        if price_history and len(price_history) > 0:
            last_price = price_history[-1]
            if hasattr(last_price, 'close'):
                current_price = extract_value(last_price.close)
            if current_price is not None:
                indicators['current_price'] = current_price

    if not current_price and has_price_performance and technical_data.get('price_performance'):
        try:
            if technical_data.get('price_target') and len(technical_data['price_target']) > 0:
                target_data = technical_data['price_target'][0]
                if hasattr(target_data, 'target_consensus') and extract_value(target_data.target_consensus):
                    current_price = extract_value(target_data.target_consensus)
                    indicators['current_price'] = current_price
        except Exception:
            pass

    if not current_price:
        indicators['current_price'] = 100.0
        indicators['price_performance_1d'] = 0.0
        indicators['price_performance_1w'] = 0.0
        indicators['price_performance_1m'] = 0.0
        indicators['price_performance_3m'] = 0.0
        indicators['price_performance_ytd'] = 0.0
        indicators['price_performance_1y'] = 0.0
        indicators['sma_cross_signal'] = 0.0
        indicators['ema_cross_signal'] = 0.0
        indicators['price_rel_sma'] = 0.0
        indicators['price_rel_ema'] = 0.0
        indicators['adx_trend'] = 0.0
        indicators['ichimoku_signal'] = 0.0
        indicators['macd_signal'] = 0.0
        indicators['bb_position'] = 0.0
        indicators['rsi_signal'] = 0.0
        indicators['stoch_signal'] = 0.0
        indicators['cci_signal'] = 0.0
        indicators['clenow_momentum'] = 0.0
        indicators['fisher_transform'] = 0.0
        indicators['atr_percent'] = 0.02
        indicators['bb_width'] = 0.05
        indicators['keltner_width'] = 0.05
        indicators['volatility_cones'] = 0.0
        indicators['donchian_width'] = 0.05
        indicators['price_target_upside'] = 0.0
        indicators['obv_trend'] = 0.0
        indicators['adl_trend'] = 0.0
        indicators['adosc_signal'] = 0.0
        indicators['vwap_position'] = 0.0
        indicators['volume_trend'] = 0.0
        indicators['trend_signal'] = 0.0
        indicators['momentum_signal'] = 0.0
        return indicators

    if has_price_performance and technical_data.get('price_performance') and len(technical_data['price_performance']) > 0:
        perf_data = technical_data['price_performance'][0]
        one_day = get_attribute_value(perf_data, 'one_day', 0)
        one_week = get_attribute_value(perf_data, 'one_week', 0)
        one_month = get_attribute_value(perf_data, 'one_month', 0)
        three_month = get_attribute_value(perf_data, 'three_month', 0)
        year_to_date = get_attribute_value(perf_data, 'ytd', 0)
        one_year = get_attribute_value(perf_data, 'one_year', 0)
        indicators['price_performance_1d'] = one_day
        indicators['price_performance_1w'] = one_week
        indicators['price_performance_1m'] = one_month
        indicators['price_performance_3m'] = three_month
        indicators['price_performance_ytd'] = year_to_date
        indicators['price_performance_1y'] = one_year
        momentum_signal = 0.0
        if one_month is not None:
            momentum_signal += one_month * 0.6
        if three_month is not None:
            momentum_signal += three_month * 0.4
        indicators['momentum_signal'] = momentum_signal
        if momentum_signal is not None:
            pseudo_rsi = 50 + momentum_signal * 250
            pseudo_rsi = max(0, min(100, pseudo_rsi))
            indicators['rsi'] = pseudo_rsi
            if pseudo_rsi >= 70:
                indicators['rsi_signal'] = -1.0
            elif pseudo_rsi <= 30:
                indicators['rsi_signal'] = 1.0
            else:
                indicators['rsi_signal'] = (50 - pseudo_rsi) / 20
    else:
        indicators['price_performance_1d'] = 0.0
        indicators['price_performance_1w'] = 0.0
        indicators['price_performance_1m'] = 0.0
        indicators['price_performance_3m'] = 0.0
        indicators['price_performance_ytd'] = 0.0
        indicators['price_performance_1y'] = 0.0
        indicators['momentum_signal'] = 0.0
        indicators['rsi'] = 50.0
        indicators['rsi_signal'] = 0.0

    if has_price_history and len(technical_data['price_history']) > 5:
        price_data = technical_data['price_history']
        close_prices = []
        high_prices = []
        low_prices = []
        volumes = []
        for p in price_data:
            if hasattr(p, 'close'):
                close_val = extract_value(p.close)
                if close_val is not None:
                    close_prices.append(close_val)
            if hasattr(p, 'high'):
                high_val = extract_value(p.high)
                if high_val is not None:
                    high_prices.append(high_val)
            if hasattr(p, 'low'):
                low_val = extract_value(p.low)
                if low_val is not None:
                    low_prices.append(low_val)
            if hasattr(p, 'volume'):
                vol_val = extract_value(p.volume)
                if vol_val is not None:
                    volumes.append(vol_val)
        if not has_sma_data and len(close_prices) >= 200:
            if len(close_prices) >= 50:
                sma_50 = sum(close_prices[-50:]) / 50
                indicators['sma_50'] = sma_50
            if len(close_prices) >= 200:
                sma_200 = sum(close_prices[-200:]) / 200
                indicators['sma_200'] = sma_200
                if 'sma_50' in indicators:
                    indicators['sma_cross_signal'] = 1.0 if indicators['sma_50'] > sma_200 else -1.0
                indicators['price_rel_sma_200'] = (current_price / sma_200) - 1.0
                indicators['price_rel_sma'] = indicators['price_rel_sma_200']
        if not has_ema_data and len(close_prices) >= 26:
            def calculate_ema(prices, period):
                multiplier = 2 / (period + 1)
                ema = sum(prices[:period]) / period
                for price in prices[period:]:
                    ema = (price - ema) * multiplier + ema
                return ema
            if len(close_prices) >= 12:
                ema_12 = calculate_ema(close_prices, 12)
                indicators['ema_12'] = ema_12
            if len(close_prices) >= 26:
                ema_26 = calculate_ema(close_prices, 26)
                indicators['ema_26'] = ema_26
                if 'ema_12' in indicators:
                    indicators['ema_cross_signal'] = 1.0 if indicators['ema_12'] > ema_26 else -1.0
                indicators['price_rel_ema'] = (current_price / ema_26) - 1.0
        if len(close_prices) >= 14 and len(high_prices) >= 14 and len(low_prices) >= 14:
            tr_values = []
            for i in range(1, min(len(close_prices), len(high_prices), len(low_prices))):
                high = high_prices[i]
                low = low_prices[i]
                prev_close = close_prices[i-1]
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                tr_values.append(tr)
            if tr_values:
                atr = sum(tr_values[-14:]) / 14
                indicators['atr'] = atr
                if current_price > 0:
                    indicators['atr_percent'] = atr / current_price
        if volumes:
            if len(volumes) >= 10:
                recent_vol = sum(volumes[-5:]) / 5
                prev_vol = sum(volumes[-10:-5]) / 5
                if prev_vol > 0:
                    vol_change = (recent_vol / prev_vol) - 1.0
                    indicators['volume_trend'] = min(1.0, max(-1.0, vol_change))
                else:
                    indicators['volume_trend'] = 0.0
            else:
                indicators['volume_trend'] = 0.0
            if len(close_prices) >= 10 and len(volumes) >= 10:
                obv = 0
                prev_obv = 0
                for i in range(1, min(len(close_prices), len(volumes))):
                    if close_prices[i] > close_prices[i-1]:
                        obv += volumes[i]
                    elif close_prices[i] < close_prices[i-1]:
                        obv -= volumes[i]
                if len(close_prices) >= 10:
                    for i in range(1, min(len(close_prices), len(volumes)) - 9):
                        if close_prices[i] > close_prices[i-1]:
                            prev_obv += volumes[i]
                        elif close_prices[i] < close_prices[i-1]:
                            prev_obv -= volumes[i]
                indicators['obv'] = obv
                if prev_obv != 0:
                    obv_change = (obv - prev_obv) / abs(prev_obv)
                    indicators['obv_trend'] = min(1.0, max(-1.0, obv_change * 5))
                else:
                    indicators['obv_trend'] = 0.0 if obv == 0 else (1.0 if obv > 0 else -1.0)
    if has_sma_data:
        sma_50 = None
        sma_200 = None
        if technical_data.get('sma_50') and len(technical_data['sma_50']) > 0:
            sma_50 = extract_value(technical_data['sma_50'][-1])
            if sma_50 is not None:
                indicators['sma_50'] = sma_50
        if technical_data.get('sma_200') and len(technical_data['sma_200']) > 0:
            sma_200 = extract_value(technical_data['sma_200'][-1])
            if sma_200 is not None:
                indicators['sma_200'] = sma_200
        if sma_50 is not None and sma_200 is not None:
            indicators['sma_cross_signal'] = 1.0 if sma_50 > sma_200 else -1.0
        if current_price is not None:
            if sma_50 is not None:
                indicators['price_rel_sma_50'] = (current_price / sma_50) - 1.0
            if sma_200 is not None:
                indicators['price_rel_sma_200'] = (current_price / sma_200) - 1.0
                indicators['price_rel_sma'] = indicators['price_rel_sma_200']
    if 'sma_cross_signal' not in indicators:
        if 'price_rel_sma_200' in indicators:
            indicators['sma_cross_signal'] = 1.0 if indicators['price_rel_sma_200'] > 0 else -1.0
        elif 'price_performance_3m' in indicators:
            indicators['sma_cross_signal'] = 1.0 if indicators['price_performance_3m'] > 0 else -1.0
        else:
            indicators['sma_cross_signal'] = 0.0
    if has_ema_data:
        ema_12 = None
        ema_26 = None
        if technical_data.get('ema_12') and len(technical_data['ema_12']) > 0:
            ema_12 = extract_value(technical_data['ema_12'][-1])
            if ema_12 is not None:
                indicators['ema_12'] = ema_12
        if technical_data.get('ema_26') and len(technical_data['ema_26']) > 0:
            ema_26 = extract_value(technical_data['ema_26'][-1])
            if ema_26 is not None:
                indicators['ema_26'] = ema_26
        if ema_12 is not None and ema_26 is not None:
            indicators['ema_cross_signal'] = 1.0 if ema_12 > ema_26 else -1.0
        if current_price is not None and ema_26 is not None:
            indicators['price_rel_ema'] = (current_price / ema_26) - 1.0
    if 'ema_cross_signal' not in indicators:
        if 'price_rel_ema' in indicators:
            indicators['ema_cross_signal'] = 1.0 if indicators['price_rel_ema'] > 0 else -1.0
        elif 'sma_cross_signal' in indicators:
            indicators['ema_cross_signal'] = indicators['sma_cross_signal']
        elif 'price_performance_1m' in indicators:
            indicators['ema_cross_signal'] = 1.0 if indicators['price_performance_1m'] > 0 else -1.0
        else:
            indicators['ema_cross_signal'] = 0.0
    if has_macd_data and technical_data.get('macd') and len(technical_data['macd']) > 0:
        macd_data = technical_data['macd'][-1]
        macd_line = extract_value(getattr(macd_data, 'macd', None))
        macd_signal = extract_value(getattr(macd_data, 'signal', None))
        macd_hist = extract_value(getattr(macd_data, 'histogram', None))
        if macd_line is not None:
            indicators['macd_line'] = macd_line
        if macd_signal is not None:
            indicators['macd_signal_line'] = macd_signal
        if macd_hist is not None:
            indicators['macd_hist'] = macd_hist
        if macd_line is not None and macd_signal is not None:
            indicators['macd_signal'] = 1.0 if macd_line > macd_signal else -1.0
    if 'macd_signal' not in indicators:
        if 'ema_cross_signal' in indicators:
            indicators['macd_signal'] = indicators['ema_cross_signal']
        elif 'sma_cross_signal' in indicators:
            indicators['macd_signal'] = indicators['sma_cross_signal']
        elif 'price_performance_1m' in indicators and 'price_performance_3m' in indicators:
            indicators['macd_signal'] = 1.0 if indicators['price_performance_1m'] > indicators['price_performance_3m'] else -1.0
        else:
            indicators['macd_signal'] = 0.0
    if has_rsi_data and technical_data.get('rsi') and len(technical_data['rsi']) > 0:
        rsi_value = extract_value(technical_data['rsi'][-1])
        if rsi_value is not None:
            indicators['rsi'] = rsi_value
            if rsi_value >= 70:
                indicators['rsi_signal'] = -1.0
            elif rsi_value <= 30:
                indicators['rsi_signal'] = 1.0
            else:
                indicators['rsi_signal'] = (50 - rsi_value) / 20
    if 'rsi_signal' not in indicators and 'rsi' in indicators:
        rsi_value = indicators['rsi']
        if rsi_value >= 70:
            indicators['rsi_signal'] = -1.0
        elif rsi_value <= 30:
            indicators['rsi_signal'] = 1.0
        else:
            indicators['rsi_signal'] = (50 - rsi_value) / 20
    elif 'rsi_signal' not in indicators:
        if 'price_performance_1m' in indicators:
            perf_1m = indicators['price_performance_1m']
            if perf_1m > 0.1:
                indicators['rsi_signal'] = -0.8
            elif perf_1m < -0.1:
                indicators['rsi_signal'] = 0.8
            else:
                indicators['rsi_signal'] = -perf_1m * 8
        else:
            indicators['rsi_signal'] = 0.0
    if has_stoch_data and technical_data.get('stoch') and len(technical_data['stoch']) > 0:
        stoch_data = technical_data['stoch'][-1]
        stoch_k = extract_value(getattr(stoch_data, 'k', None))
        stoch_d = extract_value(getattr(stoch_data, 'd', None))
        if stoch_k is not None:
            indicators['stoch_k'] = stoch_k
        if stoch_d is not None:
            indicators['stoch_d'] = stoch_d
        if stoch_k is not None and stoch_d is not None:
            if stoch_d >= 80:
                indicators['stoch_signal'] = -1.0
            elif stoch_d <= 20:
                indicators['stoch_signal'] = 1.0
            else:
                indicators['stoch_signal'] = (50 - stoch_d) / 30
    if 'stoch_signal' not in indicators:
        if 'rsi_signal' in indicators:
            indicators['stoch_signal'] = indicators['rsi_signal']
        else:
            if 'price_performance_1m' in indicators:
                perf_1m = indicators['price_performance_1m']
                if perf_1m > 0.08:
                    indicators['stoch_signal'] = -0.7
                elif perf_1m < -0.08:
                    indicators['stoch_signal'] = 0.7
                else:
                    indicators['stoch_signal'] = -perf_1m * 8
            else:
                indicators['stoch_signal'] = 0.0
    if technical_data.get('bbands') and len(technical_data['bbands']) > 0:
        bb_data = technical_data['bbands'][-1]
        bb_upper = extract_value(getattr(bb_data, 'upper', None))
        bb_middle = extract_value(getattr(bb_data, 'middle', None))
        bb_lower = extract_value(getattr(bb_data, 'lower', None))
        if bb_upper is not None:
            indicators['bb_upper'] = bb_upper
        if bb_middle is not None:
            indicators['bb_middle'] = bb_middle
        if bb_lower is not None:
            indicators['bb_lower'] = bb_lower
        if current_price is not None and bb_upper is not None and bb_lower is not None:
            bb_range = bb_upper - bb_lower
            if bb_range > 0:
                position = (current_price - bb_lower) / bb_range
                indicators['bb_position'] = (position - 0.5) * 2
                if bb_middle is not None and bb_middle > 0:
                    indicators['bb_width'] = bb_range / bb_middle
    if 'bb_position' not in indicators:
        if 'price_rel_sma' in indicators:
            rel_sma = indicators['price_rel_sma']
            indicators['bb_position'] = min(1.0, max(-1.0, rel_sma * 5))
        elif 'price_performance_1m' in indicators:
            indicators['bb_position'] = min(1.0, max(-1.0, indicators['price_performance_1m'] * 10))
        else:
            indicators['bb_position'] = 0.0
    if 'bb_width' not in indicators:
        if 'atr_percent' in indicators:
            indicators['bb_width'] = indicators['atr_percent'] * 4
        else:
            indicators['bb_width'] = 0.05
    if not indicators.get('donchian_width') and has_price_history and len(technical_data['price_history']) >= 20:
        highs = []
        lows = []
        for i in range(min(20, len(technical_data['price_history']))):
            p = technical_data['price_history'][-i-1]
            if hasattr(p, 'high'):
                high_val = extract_value(p.high)
                if high_val is not None:
                    highs.append(high_val)
            if hasattr(p, 'low'):
                low_val = extract_value(p.low)
                if low_val is not None:
                    lows.append(low_val)
        if highs and lows:
            highest_high = max(highs)
            lowest_low = min(lows)
            if highest_high > 0:
                indicators['donchian_width'] = (highest_high - lowest_low) / ((highest_high + lowest_low) / 2)
            else:
                indicators['donchian_width'] = 0.05
        else:
            indicators['donchian_width'] = 0.05
    elif not indicators.get('donchian_width'):
        indicators['donchian_width'] = 0.05
    else:
        indicators['donchian_width'] = indicators.get('donchian_width', 0.05)
    if technical_data.get('cones') and isinstance(technical_data['cones'], dict) and technical_data['cones']:
        cone_values = []
        for model, result in technical_data['cones'].items():
            if isinstance(result, list) and len(result) > 0:
                last_cone = result[-1]
                upper = get_attribute_value(last_cone, 'upper', None)
                lower = get_attribute_value(last_cone, 'lower', None)
                if upper is not None and lower is not None and current_price is not None and current_price != 0:
                    cone_width = (upper - lower) / current_price
                    cone_values.append(cone_width)
                elif isinstance(last_cone, (int, float)):
                    cone_values.append(last_cone)
        if cone_values:
            indicators['volatility_cones'] = (sum(cone_values) / len(cone_values)) * 5
        else:
            indicators['volatility_cones'] = indicators.get('atr_percent', 0.02) * 5
    else:
        indicators['volatility_cones'] = indicators.get('atr_percent', 0.02) * 5
    if technical_data.get('adx') and len(technical_data['adx']) > 0:
        adx_data = technical_data['adx'][-1]
        adx_value = extract_value(getattr(adx_data, 'adx', None))
        if adx_value is not None:
            indicators['adx'] = adx_value
            if adx_value >= 25:
                trend_strength = 0.25 + (adx_value - 25) * 0.75 / 75
                indicators['adx_trend'] = min(1.0, trend_strength)
            else:
                indicators['adx_trend'] = adx_value * 0.25 / 25
    if 'adx_trend' not in indicators:
        if 'price_performance_1m' in indicators and 'price_performance_3m' in indicators:
            perf_1m = indicators['price_performance_1m']
            perf_3m = indicators['price_performance_3m']
            if (perf_1m > 0 and perf_3m > 0) or (perf_1m < 0 and perf_3m < 0):
                indicators['adx_trend'] = min(1.0, (abs(perf_1m) + abs(perf_3m)) * 5)
            else:
                indicators['adx_trend'] = min(0.25, (abs(perf_3m) * 2))
        else:
            indicators['adx_trend'] = 0.3
    if technical_data.get('atr') and len(technical_data['atr']) > 0:
        atr_value = extract_value(technical_data['atr'][-1])
        if atr_value is not None:
            indicators['atr'] = atr_value
            if current_price is not None and current_price > 0:
                indicators['atr_percent'] = atr_value / current_price
    if 'atr_percent' not in indicators:
        if has_price_history and len(technical_data['price_history']) > 10:
            price_history = technical_data['price_history']
            recent_prices = []
            for i in range(min(10, len(price_history))):
                if hasattr(price_history[-i-1], 'close'):
                    close_val = extract_value(price_history[-i-1].close)
                    if close_val is not None:
                        recent_prices.append(close_val)
            if recent_prices and len(recent_prices) > 1:
                daily_changes = []
                for i in range(1, len(recent_prices)):
                    if recent_prices[i-1] > 0:
                        daily_change = abs(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]
                        daily_changes.append(daily_change)
                if daily_changes:
                    indicators['atr_percent'] = sum(daily_changes) / len(daily_changes)
                else:
                    indicators['atr_percent'] = 0.02
            else:
                indicators['atr_percent'] = 0.02
        else:
            indicators['atr_percent'] = 0.02
    if technical_data.get('obv') and len(technical_data['obv']) > 1:
        obv_current = extract_value(technical_data['obv'][-1])
        obv_prev = extract_value(technical_data['obv'][-2])
        if obv_current is not None:
            indicators['obv'] = obv_current
            if obv_prev is not None:
                obv_change = (obv_current - obv_prev) / abs(obv_prev)
                indicators['obv_trend'] = min(1.0, max(-1.0, obv_change * 5))
    if 'obv_trend' not in indicators:
        if 'volume_trend' in indicators:
            vol_trend = indicators['volume_trend']
            price_direction = 1.0 if indicators.get('price_performance_1m', 0) > 0 else -1.0
            indicators['obv_trend'] = vol_trend * price_direction
        elif 'price_performance_1m' in indicators:
            indicators['obv_trend'] = 1.0 if indicators['price_performance_1m'] > 0 else -1.0
        else:
            indicators['obv_trend'] = 0.0
    if technical_data.get('ad') and len(technical_data['ad']) > 1:
        ad_current = extract_value(technical_data['ad'][-1])
        ad_prev = extract_value(technical_data['ad'][-2])
        if ad_current is not None:
            indicators['ad'] = ad_current
            if ad_prev is not None:
                indicators['adl_trend'] = 1.0 if ad_current > ad_prev else -1.0
                if len(technical_data['ad']) >= 10:
                    ad_10_periods_ago = extract_value(technical_data['ad'][-10])
                    if ad_10_periods_ago is not None and ad_10_periods_ago != 0:
                        ad_roc = (ad_current - ad_10_periods_ago) / abs(ad_10_periods_ago)
                        indicators['adl_trend'] = min(1.0, max(-1.0, ad_roc * 5))
    if 'adl_trend' not in indicators:
        if 'obv_trend' in indicators:
            indicators['adl_trend'] = indicators['obv_trend']
        elif 'volume_trend' in indicators and 'price_performance_1m' in indicators:
            vol_trend = indicators['volume_trend']
            price_perf = indicators['price_performance_1m']
            indicators['adl_trend'] = vol_trend * (1.0 if price_perf > 0 else -1.0)
        else:
            indicators['adl_trend'] = 0.0
    if technical_data.get('vwap') and len(technical_data['vwap']) > 0:
        vwap_value = extract_value(technical_data['vwap'][-1])
        if vwap_value is not None:
            indicators['vwap'] = vwap_value
            if current_price is not None and vwap_value > 0:
                vwap_diff = (current_price / vwap_value) - 1.0
                indicators['vwap_position'] = min(1.0, max(-1.0, vwap_diff * 5))
    if 'vwap_position' not in indicators:
        if 'price_rel_sma' in indicators:
            indicators['vwap_position'] = indicators['price_rel_sma']
        elif 'price_performance_1m' in indicators:
            indicators['vwap_position'] = min(1.0, max(-1.0, indicators['price_performance_1m'] * 10))
        else:
            indicators['vwap_position'] = 0.0
    if not indicators.get('keltner_width') and indicators.get('atr_percent'):
        indicators['keltner_width'] = indicators['atr_percent'] * 4
    elif not indicators.get('keltner_width'):
        indicators['keltner_width'] = 0.05

    required_indicators = [
        'sma_cross_signal', 'ema_cross_signal', 'price_rel_sma', 'price_rel_ema',
        'adx_trend', 'ichimoku_signal', 'macd_signal', 'bb_position',
        'rsi_signal', 'stoch_signal', 'cci_signal', 'clenow_momentum',
        'fisher_transform', 'price_performance_1m', 'price_performance_3m',
        'atr_percent', 'bb_width', 'keltner_width', 'volatility_cones',
        'donchian_width', 'price_target_upside',
        'obv_trend', 'adl_trend', 'adosc_signal', 'vwap_position', 'volume_trend'
    ]
    if 'adosc_signal' not in indicators:
        if 'adl_trend' in indicators:
            indicators['adosc_signal'] = indicators['adl_trend']
        else:
            indicators['adosc_signal'] = 0.0
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
    if current_price is not None:
        w_pt = 0.5
        w_m = 0.3
        w_t = 0.2
        combined_factor = (w_pt * indicators.get('price_target_upside', 0) +
                           w_m * indicators.get('momentum_signal', 0) +
                           w_t * indicators.get('trend_signal', 0)) / (w_pt + w_m + w_t)
        expected_price = current_price * (1 + combined_factor)
        vol_components = []
        if 'atr_percent' in indicators:
            vol_components.append(indicators['atr_percent'])
        if 'bb_width' in indicators:
            vol_components.append(indicators['bb_width'])
        if 'volatility_cones' in indicators:
            vol_components.append(indicators['volatility_cones'] / 5)
        if vol_components:
            avg_vol = sum(vol_components) / len(vol_components)
        else:
            avg_vol = 0.02
        vol_range = current_price * avg_vol
        probable_low = expected_price - vol_range
        probable_high = expected_price + vol_range
        indicators['expected_price'] = expected_price
        indicators['probable_low'] = probable_low
        indicators['probable_high'] = probable_high

    return indicators

def normalize_data(data_dict):
    normalized_data = {ticker: {} for ticker in data_dict}
    metrics_data = {}
    for ticker, metrics in data_dict.items():
        for metric, value in metrics.items():
            if metric not in metrics_data:
                metrics_data[metric] = []
            if isinstance(value, (int, float)) and not pd.isna(value) and value is not None:
                metrics_data[metric].append(value)
    for ticker, metrics in data_dict.items():
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)) or pd.isna(value) or value is None:
                continue
            normalized_data[ticker][metric] = value
    return normalized_data

def calculate_z_scores(data_dict):
    z_scores = {ticker: {} for ticker in data_dict}
    metrics_dict = {}
    for ticker, metrics in data_dict.items():
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and not pd.isna(value) and value is not None:
                metrics_dict.setdefault(metric, []).append((ticker, value))
    for metric, ticker_values in metrics_dict.items():
        if len(ticker_values) < 2:
            if len(ticker_values) == 1:
                ticker, value = ticker_values[0]
                z_scores[ticker][metric] = 0.0
            continue
        tickers, values = zip(*ticker_values)
        try:
            values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
            if torch.std(values_tensor).item() < 1e-8:
                for ticker in tickers:
                    z_scores[ticker][metric] = 0.0
                continue
            mean_val = torch.mean(values_tensor)
            std_val = torch.std(values_tensor)
            std_val = std_val if std_val > 1e-10 else torch.tensor(1e-10, device=device)
            if len(values) > 4:
                skewness = torch.abs(torch.mean(((values_tensor - mean_val) / std_val) ** 3)).item()
                if skewness > 2:
                    median_val = torch.median(values_tensor)
                    q1 = torch.quantile(values_tensor, 0.25)
                    q3 = torch.quantile(values_tensor, 0.75)
                    iqr = q3 - q1
                    robust_std = max((iqr / 1.349).item(), 1e-10)
                    metric_z_scores = (values_tensor - median_val) / robust_std
                else:
                    metric_z_scores = (values_tensor - mean_val) / std_val
            else:
                metric_z_scores = (values_tensor - mean_val) / std_val
            metric_z_scores = torch.clamp(metric_z_scores, -3, 3)
            for ticker, z_score in zip(tickers, metric_z_scores):
                z_scores[ticker][metric] = z_score.item()
        except Exception as e:
            logger.warning(f"Error calculating z-scores for {metric}: {e}")
            min_val = min(values)
            max_val = max(values)
            if max_val > min_val:
                range_val = max_val - min_val
                for ticker, value in zip(tickers, values):
                    normalized_val = (value - min_val) / range_val * 2 - 1
                    z_scores[ticker][metric] = normalized_val
            else:
                for ticker in tickers:
                    z_scores[ticker][metric] = 0.0
    derived_metrics = {
        'trend': ['price_rel_sma', 'price_rel_ema', 'macd_signal', 'sma_cross_signal', 'ema_cross_signal'],
        'momentum': ['price_performance_1m', 'price_performance_3m', 'rsi_signal', 'stoch_signal', 'cci_signal'],
        'volatility': ['atr_percent', 'bb_width', 'keltner_width'],
        'volume': ['volume_trend', 'obv_trend', 'adl_trend']
    }
    for ticker in data_dict:
        if 'price_performance_1m' in data_dict[ticker] and 'price_performance_1m' not in z_scores[ticker]:
            raw_value = data_dict[ticker]['price_performance_1m']
            z_scores[ticker]['price_performance_1m'] = min(max(raw_value * 5, -3), 3)
        if 'price_performance_3m' in data_dict[ticker] and 'price_performance_3m' not in z_scores[ticker]:
            raw_value = data_dict[ticker]['price_performance_3m']
            z_scores[ticker]['price_performance_3m'] = min(max(raw_value * 4, -3), 3)
        if 'volume_trend' in data_dict[ticker] and 'volume_trend' not in z_scores[ticker]:
            raw_value = data_dict[ticker]['volume_trend']
            z_scores[ticker]['volume_trend'] = min(max(raw_value * 3, -3), 3)
    return z_scores
