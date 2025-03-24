import asyncio
import time
import threading
import logging

from .utils import get_openbb_client
from .constants import API_CALLS_PER_MINUTE, CACHE_SIZE

logger = logging.getLogger(__name__)

# Semaphore for concurrent API calls.
api_semaphore = asyncio.Semaphore(40)
api_call_timestamps = []
api_lock = threading.RLock()

# Simple cache for API results.
api_cache = {}

async def rate_limited_api_call(func, *args, **kwargs):
    """
    Call an API function with rate limiting and caching.
    Retries up to 3 times if a rate limit error is encountered.
    """
    cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
    if cache_key in api_cache:
        return api_cache[cache_key]
    async with api_semaphore:
        with api_lock:
            current_time = time.time()
            global api_call_timestamps
            api_call_timestamps = [ts for ts in api_call_timestamps if current_time - ts < 60]
            if len(api_call_timestamps) >= API_CALLS_PER_MINUTE:
                sleep_time = 60 - (current_time - api_call_timestamps[0])
                if sleep_time > 0:
                    logger.debug(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                    await asyncio.sleep(sleep_time)
            api_call_timestamps.append(time.time())
        try:
            for attempt in range(3):
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, lambda: func(*args, **kwargs))
                    api_cache[cache_key] = result
                    if len(api_cache) > CACHE_SIZE:
                        oldest_key = next(iter(api_cache))
                        del api_cache[oldest_key]
                    return result
                except Exception as e:
                    if "Limit Reach" in str(e):
                        logger.warning(f"Attempt {attempt+1}: Rate limit error encountered. Retrying after delay...")
                        await asyncio.sleep(60)
                    else:
                        raise
            raise Exception("Max retry attempts reached for API call")
        except Exception as e:
            logger.error(f"API call error: {e}")
            raise

async def get_technical_data_async(ticker: str):
    """
    Fetch technical data for a ticker asynchronously using multiple API calls.
    """
    from datetime import datetime, timedelta
    technical_data = {}
    try:
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        obb_client = get_openbb_client()
        price_history_task = rate_limited_api_call(
            obb_client.equity.price.historical,
            symbol=ticker, start_date=one_year_ago, provider='fmp'
        )
        price_perf_task = rate_limited_api_call(
            obb_client.equity.price.performance,
            symbol=ticker, provider='fmp'
        )
        price_target_task = rate_limited_api_call(
            obb_client.equity.estimates.consensus,
            symbol=ticker, provider='fmp'
        )
        results = await asyncio.gather(
            price_history_task, price_perf_task, price_target_task,
            return_exceptions=True
        )
        price_history_response, price_perf_response, price_target_response = results
        if isinstance(price_history_response, Exception):
            logger.error(f"Essential price history data fetch failed for {ticker}: {price_history_response}")
            return (ticker, technical_data)
        technical_data['price_history'] = price_history_response.results if not isinstance(price_history_response, Exception) else []
        technical_data['price_performance'] = price_perf_response.results if not isinstance(price_perf_response, Exception) else []
        technical_data['price_target'] = price_target_response.results if not isinstance(price_target_response, Exception) else []
        if technical_data['price_history']:
            sma_50_task = rate_limited_api_call(
                obb_client.technical.sma,
                data=technical_data['price_history'], target='close', length=50
            )
            sma_200_task = rate_limited_api_call(
                obb_client.technical.sma,
                data=technical_data['price_history'], target='close', length=200
            )
            ema_12_task = rate_limited_api_call(
                obb_client.technical.ema,
                data=technical_data['price_history'], target='close', length=12
            )
            ema_26_task = rate_limited_api_call(
                obb_client.technical.ema,
                data=technical_data['price_history'], target='close', length=26
            )
            ema_50_task = rate_limited_api_call(
                obb_client.technical.ema,
                data=technical_data['price_history'], target='close', length=50
            )
            bbands_task = rate_limited_api_call(
                obb_client.technical.bbands,
                data=technical_data['price_history'], target='close', length=20, std=2
            )
            keltner_task = rate_limited_api_call(
                obb_client.technical.kc,
                data=technical_data['price_history'], length=20, scalar=2
            )
            results = await asyncio.gather(
                sma_50_task, sma_200_task, ema_12_task, ema_26_task, ema_50_task,
                bbands_task, keltner_task,
                return_exceptions=True
            )
            (sma_50_response, sma_200_response, ema_12_response, ema_26_response, ema_50_response,
             bbands_response, keltner_response) = results
            technical_data['sma_50'] = sma_50_response.results if not isinstance(sma_50_response, Exception) else []
            technical_data['sma_200'] = sma_200_response.results if not isinstance(sma_200_response, Exception) else []
            technical_data['ema_12'] = ema_12_response.results if not isinstance(ema_12_response, Exception) else []
            technical_data['ema_26'] = ema_26_response.results if not isinstance(ema_26_response, Exception) else []
            technical_data['ema_50'] = ema_50_response.results if not isinstance(ema_50_response, Exception) else []
            technical_data['bbands'] = bbands_response.results if not isinstance(bbands_response, Exception) else []
            technical_data['keltner'] = keltner_response.results if not isinstance(keltner_response, Exception) else []
            
            macd_task = rate_limited_api_call(
                obb_client.technical.macd,
                data=technical_data['price_history'], target='close', fast=12, slow=26, signal=9
            )
            rsi_task = rate_limited_api_call(
                obb_client.technical.rsi,
                data=technical_data['price_history'], target='close', length=14
            )
            stoch_task = rate_limited_api_call(
                obb_client.technical.stoch,
                data=technical_data['price_history'], fast_k_period=14, slow_d_period=3
            )
            cci_task = rate_limited_api_call(
                obb_client.technical.cci,
                data=technical_data['price_history'], length=20
            )
            adx_task = rate_limited_api_call(
                obb_client.technical.adx,
                data=technical_data['price_history'], length=14
            )
            obv_task = rate_limited_api_call(
                obb_client.technical.obv,
                data=technical_data['price_history']
            )
            ad_task = rate_limited_api_call(
                obb_client.technical.ad,
                data=technical_data['price_history']
            )
            results = await asyncio.gather(
                macd_task, rsi_task, stoch_task, cci_task, adx_task, obv_task, ad_task,
                return_exceptions=True
            )
            (macd_response, rsi_response, stoch_response, cci_response,
             adx_response, obv_response, ad_response) = results
            technical_data['macd'] = macd_response.results if not isinstance(macd_response, Exception) else []
            technical_data['rsi'] = rsi_response.results if not isinstance(rsi_response, Exception) else []
            technical_data['stoch'] = stoch_response.results if not isinstance(stoch_response, Exception) else []
            technical_data['cci'] = cci_response.results if not isinstance(cci_response, Exception) else []
            technical_data['adx'] = adx_response.results if not isinstance(adx_response, Exception) else []
            technical_data['obv'] = obv_response.results if not isinstance(obv_response, Exception) else []
            technical_data['ad'] = ad_response.results if not isinstance(ad_response, Exception) else []
            
            atr_task = rate_limited_api_call(
                obb_client.technical.atr,
                data=technical_data['price_history'], length=14
            )
            donchian_task = rate_limited_api_call(
                obb_client.technical.donchian,
                data=technical_data['price_history'], lower_length=20, upper_length=20
            )
            fisher_task = rate_limited_api_call(
                obb_client.technical.fisher,
                data=technical_data['price_history'], length=14
            )
            ichimoku_task = rate_limited_api_call(
                obb_client.technical.ichimoku,
                data=technical_data['price_history'], conversion=9, base=26
            )
            adosc_task = rate_limited_api_call(
                obb_client.technical.adosc,
                data=technical_data['price_history'], fast=3, slow=10
            )
            vwap_task = rate_limited_api_call(
                obb_client.technical.vwap,
                data=technical_data['price_history'], anchor='D'
            )
            clenow_task = rate_limited_api_call(
                obb_client.technical.clenow,
                data=technical_data['price_history'], period=90
            )
            results = await asyncio.gather(
                atr_task, donchian_task, fisher_task, ichimoku_task, adosc_task, vwap_task, clenow_task,
                return_exceptions=True
            )
            (atr_response, donchian_response, fisher_response, ichimoku_response,
             adosc_response, vwap_response, clenow_response) = results
            technical_data['atr'] = atr_response.results if not isinstance(atr_response, Exception) else []
            technical_data['donchian'] = donchian_response.results if not isinstance(donchian_response, Exception) else []
            technical_data['fisher'] = fisher_response.results if not isinstance(fisher_response, Exception) else []
            technical_data['ichimoku'] = ichimoku_response.results if not isinstance(ichimoku_response, Exception) else []
            technical_data['adosc'] = adosc_response.results if not isinstance(adosc_response, Exception) else []
            technical_data['vwap'] = vwap_response.results if not isinstance(vwap_response, Exception) else []
            technical_data['clenow'] = clenow_response.results if not isinstance(clenow_response, Exception) else []
            
            try:
                aroon_task = rate_limited_api_call(
                    obb_client.technical.aroon,
                    data=technical_data['price_history'], length=25
                )
                aroon_response = await aroon_task
                technical_data['aroon'] = aroon_response.results if not isinstance(aroon_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching Aroon indicator for {ticker}: {e}")
                technical_data['aroon'] = []
            
            try:
                fib_task = rate_limited_api_call(
                    obb_client.technical.fib,
                    data=technical_data['price_history'], period=120
                )
                fib_response = await fib_task
                technical_data['fib'] = fib_response.results if not isinstance(fib_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching Fibonacci levels for {ticker}: {e}")
                technical_data['fib'] = []
            
            try:
                hma_task = rate_limited_api_call(
                    obb_client.technical.hma,
                    data=technical_data['price_history'], target='close', length=50
                )
                hma_response = await hma_task
                technical_data['hma'] = hma_response.results if not isinstance(hma_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching HMA for {ticker}: {e}")
                technical_data['hma'] = []
            
            try:
                wma_task = rate_limited_api_call(
                    obb_client.technical.wma,
                    data=technical_data['price_history'], target='close', length=50
                )
                wma_response = await wma_task
                technical_data['wma'] = wma_response.results if not isinstance(wma_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching WMA for {ticker}: {e}")
                technical_data['wma'] = []
            
            try:
                zlma_task = rate_limited_api_call(
                    obb_client.technical.zlma,
                    data=technical_data['price_history'], target='close', length=50
                )
                zlma_response = await zlma_task
                technical_data['zlma'] = zlma_response.results if not isinstance(zlma_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching ZLMA for {ticker}: {e}")
                technical_data['zlma'] = []
            
            try:
                demark_task = rate_limited_api_call(
                    obb_client.technical.demark,
                    data=technical_data['price_history']
                )
                demark_response = await demark_task
                technical_data['demark'] = demark_response.results if not isinstance(demark_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching Demark Sequential for {ticker}: {e}")
                technical_data['demark'] = []
            
            cone_models = ['std', 'garman_klass', 'hodges_tompkins', 'rogers_satchell', 'yang_zhang']
            cones_results = {}
            for model in cone_models:
                try:
                    cones_task = rate_limited_api_call(
                        obb_client.technical.cones,
                        data=technical_data['price_history'], lower_q=0.25, upper_q=0.75, model=model
                    )
                    cones_response = await cones_task
                    if not isinstance(cones_response, Exception) and cones_response.results:
                        cones_results[model] = cones_response.results
                except Exception as e:
                    logger.warning(f"Error fetching volatility cones for model {model} for {ticker}: {e}")
            technical_data['cones'] = cones_results
            
            try:
                price_targets_task = rate_limited_api_call(
                    obb_client.equity.estimates.price_target,
                    symbol=ticker, provider='fmp', limit=10
                )
                price_targets_response = await price_targets_task
                technical_data['price_targets'] = price_targets_response.results if not isinstance(price_targets_response, Exception) else []
            except Exception as e:
                logger.warning(f"Error fetching price targets for {ticker}: {e}")
                technical_data['price_targets'] = []
    
    except Exception as e:
        logger.error(f"Error fetching technical data for {ticker}: {e}")
        logger.exception(e)
    
    return (ticker, technical_data)
