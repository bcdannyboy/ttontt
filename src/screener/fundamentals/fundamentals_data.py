import asyncio
import traceback
import logging

from src.screener.fundamentals.fundamentals_core import get_openbb_client, rate_limited_api_call
from src.screener.fundamentals.fundamentals_metrics import construct_earnings_from_income

logger = logging.getLogger(__name__)

async def get_financial_data_async(ticker: str):
    """
    Async version of get_financial_data using OpenBB API.
    Fetches all required financial data for a stock, handling provider fallbacks properly.
    
    Returns:
        tuple: (ticker, financial_data dictionary)
    """
    obb_client = get_openbb_client()
    financial_data = {}
    
    try:
        # Fetch essential financial statements first
        income_task = rate_limited_api_call(
            obb_client.equity.fundamental.income,
            symbol=ticker, period='annual', limit=5, provider='fmp'
        )
        balance_task = rate_limited_api_call(
            obb_client.equity.fundamental.balance,
            symbol=ticker, period='annual', limit=5, provider='fmp'
        )
        cash_task = rate_limited_api_call(
            obb_client.equity.fundamental.cash,
            symbol=ticker, period='annual', limit=5, provider='fmp'
        )
        
        income_response, balance_response, cash_response = await asyncio.gather(
            income_task, balance_task, cash_task, 
            return_exceptions=True
        )
        
        # Check if essential data is available
        if isinstance(income_response, Exception) or isinstance(balance_response, Exception):
            logger.error(f"Essential data fetch failed for {ticker}")
            return (ticker, financial_data)
        
        financial_data['income'] = income_response.results if not isinstance(income_response, Exception) else []
        financial_data['balance'] = balance_response.results if not isinstance(balance_response, Exception) else []
        financial_data['cash'] = cash_response.results if not isinstance(cash_response, Exception) else []
        
        # If essential data is available, fetch additional data
        if financial_data['income'] and financial_data['balance']:
            income_growth_task = rate_limited_api_call(
                obb_client.equity.fundamental.income_growth,
                symbol=ticker, period='annual', limit=5, provider='fmp'
            )
            balance_growth_task = rate_limited_api_call(
                obb_client.equity.fundamental.balance_growth,
                symbol=ticker, period='annual', limit=5, provider='fmp'
            )
            cash_growth_task = rate_limited_api_call(
                obb_client.equity.fundamental.cash_growth,
                symbol=ticker, period='annual', limit=5, provider='fmp'
            )
            ratios_task = rate_limited_api_call(
                obb_client.equity.fundamental.ratios,
                symbol=ticker, period='annual', limit=5, provider='fmp'
            )
            metrics_task = rate_limited_api_call(
                obb_client.equity.fundamental.metrics,
                symbol=ticker, period='annual', limit=5, provider='fmp'
            )
            historical_estimates_task = rate_limited_api_call(
                obb_client.equity.estimates.historical,
                symbol=ticker, provider='fmp'
            )
            
            results = await asyncio.gather(
                income_growth_task, balance_growth_task, cash_growth_task, 
                ratios_task, metrics_task, historical_estimates_task,
                return_exceptions=True
            )
            
            financial_data['income_growth'] = results[0].results if not isinstance(results[0], Exception) else []
            financial_data['balance_growth'] = results[1].results if not isinstance(results[1], Exception) else []
            financial_data['cash_growth'] = results[2].results if not isinstance(results[2], Exception) else []
            financial_data['ratios'] = results[3].results if not isinstance(results[3], Exception) else []
            financial_data['metrics'] = results[4].results if not isinstance(results[4], Exception) else []
            financial_data['historical_estimates'] = results[5].results if not isinstance(results[5], Exception) else []
            
            # Forward sales fallback chain: fmp > intrinio > default
            try:
                forward_sales_response = await rate_limited_api_call(
                    obb_client.equity.estimates.forward_sales,
                    symbol=ticker
                )
                financial_data['forward_sales'] = forward_sales_response.results
            except Exception as e:
                logger.warning(f"Error fetching forward sales for {ticker} from standard: \n{e}")
                try:
                    forward_sales_response = await rate_limited_api_call(
                        obb_client.equity.estimates.forward_sales,
                        symbol=ticker, provider='intrinio'
                    )
                    financial_data['forward_sales'] = forward_sales_response.results
                except Exception as e2:
                    logger.warning(f"Error fetching forward sales for {ticker} from Intrinio: \n{e2}")
                    try:
                        forward_sales_response = await rate_limited_api_call(
                            obb_client.equity.estimates.forward_sales,
                            symbol=ticker
                        )
                        financial_data['forward_sales'] = forward_sales_response.results
                    except Exception as e3:
                        logger.warning(f"Error fetching forward sales for {ticker} with default provider: \n{e3}")
                        financial_data['forward_sales'] = []
            
            # Forward EBITDA fallback chain: fmp > intrinio > default
            try:
                forward_ebitda_response = await rate_limited_api_call(
                    obb_client.equity.estimates.forward_ebitda,
                    symbol=ticker, fiscal_period='annual', provider='fmp'
                )
                financial_data['forward_ebitda'] = forward_ebitda_response.results
            except Exception as e:
                logger.warning(f"Error fetching forward EBITDA for {ticker} from FMP: \n{e}")
                try:
                    forward_ebitda_response = await rate_limited_api_call(
                        obb_client.equity.estimates.forward_ebitda,
                        symbol=ticker, fiscal_period='annual', provider='intrinio'
                    )
                    financial_data['forward_ebitda'] = forward_ebitda_response.results
                except Exception as e2:
                    logger.warning(f"Error fetching forward EBITDA for {ticker} from Intrinio: \n{e2}")
                    try:
                        forward_ebitda_response = await rate_limited_api_call(
                            obb_client.equity.estimates.forward_ebitda,
                            symbol=ticker, fiscal_period='annual'
                        )
                        financial_data['forward_ebitda'] = forward_ebitda_response.results
                    except Exception as e3:
                        logger.warning(f"Error fetching forward EBITDA for {ticker} with default provider: \n{e3}")
                        financial_data['forward_ebitda'] = []
            
            # Dividends
            try:
                dividends_response = await rate_limited_api_call(
                    obb_client.equity.fundamental.dividends,
                    symbol=ticker, provider='fmp'
                )
                financial_data['dividends'] = dividends_response.results
            except Exception as e:
                logger.warning(f"No dividend data for {ticker}: {e}")
                financial_data['dividends'] = []
            
            # Earnings
            try:
                if hasattr(obb_client.equity.fundamental, 'earnings'):
                    earnings_response = await rate_limited_api_call(
                        obb_client.equity.fundamental.earnings,
                        symbol=ticker, provider='fmp'
                    )
                    financial_data['earnings'] = earnings_response.results
                else:
                    logger.warning(f"Earnings endpoint not available for {ticker}, constructing from income data")
                    financial_data['earnings'] = construct_earnings_from_income(financial_data['income'])
            except Exception as e:
                logger.warning(f"Earnings endpoint not available for {ticker}, constructing from income data: {e}")
                financial_data['earnings'] = construct_earnings_from_income(financial_data['income'])
    
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        logger.error(traceback.format_exc())
    
    return (ticker, financial_data)
