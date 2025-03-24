import logging
import traceback
from datetime import datetime
import numpy as np
import torch
from src.screener.fundamentals.fundamentals_core import get_openbb_client
from src.screener.fundamentals.fundamentals_metrics import extract_metrics_from_financial_data, construct_earnings_from_income, calculate_estimate_accuracy, get_attribute_value

logger = logging.getLogger(__name__)

def generate_stock_report(ticker: str) -> dict:
    """
    Generate a comprehensive fundamental analysis report for a stock.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    obb_client = get_openbb_client()
    financial_data = {}
    
    try:
        for data_type in ['income', 'balance', 'cash']:
            try:
                response = getattr(obb_client.equity.fundamental, data_type)(
                    symbol=ticker, period='annual', limit=5, provider='fmp'
                )
                financial_data[data_type] = response.results
            except Exception as e:
                logger.warning(f"Error fetching {data_type} for {ticker}: {e}")
                financial_data[data_type] = []
        for data_type in ['income_growth', 'balance_growth', 'cash_growth']:
            try:
                response = getattr(obb_client.equity.fundamental, data_type)(
                    symbol=ticker, period='annual', limit=5, provider='fmp'
                )
                financial_data[data_type] = response.results
            except Exception as e:
                logger.warning(f"Error fetching {data_type} for {ticker}: {e}")
                financial_data[data_type] = []
        for data_type in ['ratios', 'metrics']:
            try:
                response = getattr(obb_client.equity.fundamental, data_type)(
                    symbol=ticker, period='annual', limit=5, provider='fmp'
                )
                financial_data[data_type] = response.results
            except Exception as e:
                logger.warning(f"Error fetching {data_type} for {ticker}: {e}")
                financial_data[data_type] = []
        try:
            dividends_response = obb_client.equity.fundamental.dividends(
                symbol=ticker, provider='fmp'
            )
            financial_data['dividends'] = dividends_response.results
        except Exception as e:
            logger.warning(f"No dividend data for {ticker}: {e}")
            financial_data['dividends'] = []
        try:
            forward_sales_response = obb_client.equity.estimates.forward_sales(
                symbol=ticker
            )
            financial_data['forward_sales'] = forward_sales_response.results
        except Exception as e:
            logger.warning(f"Error fetching forward sales for {ticker} from standard: {e}")
            try:
                forward_sales_response = obb_client.equity.estimates.forward_sales(
                    symbol=ticker, provider='intrinio'
                )
                financial_data['forward_sales'] = forward_sales_response.results
            except Exception as e2:
                logger.warning(f"Error fetching forward sales for {ticker} from Intrinio: {e2}")
                financial_data['forward_sales'] = []
        try:
            forward_ebitda_response = obb_client.equity.estimates.forward_ebitda(
                symbol=ticker, fiscal_period='annual', provider='fmp'
            )
            financial_data['forward_ebitda'] = forward_ebitda_response.results
        except Exception as e:
            logger.warning(f"Error fetching forward EBITDA for {ticker} from FMP: {e}")
            try:
                forward_ebitda_response = obb_client.equity.estimates.forward_ebitda(
                    symbol=ticker, fiscal_period='annual', provider='intrinio'
                )
                financial_data['forward_ebitda'] = forward_ebitda_response.results
            except Exception as e2:
                logger.warning(f"Error fetching forward EBITDA for {ticker} from Intrinio: {e2}")
                financial_data['forward_ebitda'] = []
        try:
            historical_estimates_response = obb_client.equity.estimates.historical(
                symbol=ticker, fiscal_period='annual', provider='fmp'
            )
            financial_data['historical_estimates'] = historical_estimates_response.results
        except Exception as e:
            logger.warning(f"No historical estimates data for {ticker}: {e}")
            financial_data['historical_estimates'] = []
        try:
            if hasattr(obb_client.equity.fundamental, 'earnings'):
                earnings_response = obb_client.equity.fundamental.earnings(
                    symbol=ticker, provider='fmp'
                )
                financial_data['earnings'] = earnings_response.results
            else:
                logger.warning(f"Earnings endpoint not available for {ticker}, constructing from income data")
                financial_data['earnings'] = construct_earnings_from_income(financial_data['income'])
        except Exception as e:
            logger.warning(f"Error fetching earnings data for {ticker}: {e}")
            financial_data['earnings'] = construct_earnings_from_income(financial_data['income'])
    except Exception as e:
        logger.error(f"Error fetching financial data for {ticker}: {e}")
    
    metrics = extract_metrics_from_financial_data(financial_data)
    estimate_accuracy = {}
    if financial_data.get('historical_estimates') and financial_data.get('earnings'):
        estimate_accuracy = calculate_estimate_accuracy(
            financial_data['historical_estimates'],
            financial_data['earnings']
        )
    ticker_results = None
    try:
        # Import inside function to avoid circular dependency
        from src.screener.fundamentals.fundamentals_screen import screen_stocks
        ticker_results = screen_stocks([ticker], max_workers=1)
    except Exception as e:
        logger.error(f"Error screening ticker {ticker}: {e}")
    category_scores = {}
    composite_score = 0.0
    if ticker_results:
        _, composite_score, detailed_results = ticker_results[0]
        category_scores = detailed_results['category_scores']
    report = {
        'ticker': ticker,
        'composite_score': composite_score,
        'category_scores': category_scores,
        'key_metrics': {
            'profitability': {
                'gross_margin': metrics.get('gross_profit_margin'),
                'operating_margin': metrics.get('operating_income_margin'),
                'net_margin': metrics.get('net_income_margin'),
                'roe': metrics.get('return_on_equity'),
                'roa': metrics.get('return_on_assets')
            },
            'growth': {
                'revenue_growth': metrics.get('growth_revenue'),
                'earnings_growth': metrics.get('growth_net_income'),
                'eps_growth': metrics.get('growth_eps'),
                'forward_sales_growth': metrics.get('forward_sales_growth'),
                'forward_ebitda_growth': metrics.get('forward_ebitda_growth')
            },
            'financial_health': {
                'current_ratio': metrics.get('current_ratio'),
                'debt_to_equity': metrics.get('debt_to_equity'),
                'debt_to_assets': metrics.get('debt_to_assets'),
                'interest_coverage': metrics.get('interest_coverage'),
                'cash_to_debt': metrics.get('cash_to_debt')
            },
            'valuation': {
                'pe_ratio': metrics.get('pe_ratio'),
                'price_to_book': metrics.get('price_to_book'),
                'price_to_sales': metrics.get('price_to_sales'),
                'ev_to_ebitda': metrics.get('ev_to_ebitda'),
                'dividend_yield': metrics.get('dividend_yield'),
                'peg_ratio': metrics.get('peg_ratio')
            },
            'efficiency': {
                'asset_turnover': metrics.get('asset_turnover'),
                'inventory_turnover': metrics.get('inventory_turnover'),
                'receivables_turnover': metrics.get('receivables_turnover'),
                'cash_conversion_cycle': metrics.get('cash_conversion_cycle'),
                'capex_to_revenue': metrics.get('capex_to_revenue')
            },
            'analyst_estimates': {
                'estimate_eps_accuracy': estimate_accuracy.get('estimate_eps_accuracy', 0),
                'estimate_revenue_accuracy': estimate_accuracy.get('estimate_revenue_accuracy', 0),
                'forward_sales_growth': metrics.get('forward_sales_growth'),
                'forward_ebitda_growth': metrics.get('forward_ebitda_growth'),
                'estimate_revision_momentum': metrics.get('estimate_revision_momentum', 0),
                'estimate_consensus_deviation': metrics.get('estimate_consensus_deviation', 0)
            }
        },
        'strengths': [],
        'weaknesses': [],
        'raw_metrics': metrics,
        'z_scores': {}
    }
    if ticker_results:
        report['z_scores'] = ticker_results[0][2]['z_scores']
    profitability_benchmarks = {
        'gross_profit_margin': 0.35,
        'operating_income_margin': 0.15,
        'net_income_margin': 0.10,
        'return_on_equity': 0.15,
        'return_on_assets': 0.05
    }
    financial_health_benchmarks = {
        'current_ratio': 1.5,
        'debt_to_equity': 1.0,
        'interest_coverage': 3.0,
        'cash_to_debt': 0.5
    }
    growth_benchmarks = {
        'growth_revenue': 0.10,
        'growth_net_income': 0.10,
        'growth_eps': 0.10,
        'forward_sales_growth': 0.10,
        'forward_ebitda_growth': 0.10
    }
    estimate_benchmarks = {
        'estimate_eps_accuracy': 0.7,
        'estimate_revenue_accuracy': 0.7,
        'estimate_revision_momentum': 0.05
    }
    valuation_benchmarks = {
        'pe_ratio': 20.0,
        'price_to_book': 3.0,
        'price_to_sales': 2.0,
        'ev_to_ebitda': 12.0,
        'dividend_yield': 0.02
    }
    for metric, benchmark in profitability_benchmarks.items():
        if metric in metrics and metrics.get(metric) is not None and np.isfinite(metrics.get(metric)):
            if metrics.get(metric) > benchmark * 1.5:
                report['strengths'].append(f"Strong {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
            elif metrics.get(metric) < benchmark * 0.5:
                report['weaknesses'].append(f"Low {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
    for metric, benchmark in financial_health_benchmarks.items():
        if metric in metrics and metrics.get(metric) is not None and np.isfinite(metrics.get(metric)):
            if metric == 'debt_to_equity' or metric == 'debt_to_assets':
                if metrics.get(metric) < benchmark * 0.5:
                    report['strengths'].append(f"Low {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
                elif metrics.get(metric) > benchmark * 1.5:
                    report['weaknesses'].append(f"High {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
            else:
                if metrics.get(metric) > benchmark * 1.5:
                    report['strengths'].append(f"Strong {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
                elif metrics.get(metric) < benchmark * 0.5:
                    report['weaknesses'].append(f"Low {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
    for metric, benchmark in growth_benchmarks.items():
        if metric in metrics and metrics.get(metric) is not None and np.isfinite(metrics.get(metric)):
            if metrics.get(metric) > benchmark * 1.5:
                report['strengths'].append(f"Strong {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
            elif metrics.get(metric) < 0:
                report['weaknesses'].append(f"Negative {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
    for metric, benchmark in estimate_benchmarks.items():
        if metric in metrics and metrics.get(metric) is not None and np.isfinite(metrics.get(metric)):
            if metrics.get(metric) > benchmark:
                report['strengths'].append(f"Strong {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
            elif metrics.get(metric) < 0:
                report['weaknesses'].append(f"Poor {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
    for metric, benchmark in valuation_benchmarks.items():
        if metric in metrics and metrics.get(metric) is not None and np.isfinite(metrics.get(metric)):
            if metric == 'dividend_yield':
                if metrics.get(metric) > benchmark * 1.5:
                    report['strengths'].append(f"High {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
            elif metric in ['pe_ratio', 'price_to_book', 'price_to_sales', 'ev_to_ebitda']:
                if metrics.get(metric) < benchmark * 0.6:
                    report['strengths'].append(f"Low {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
                elif metrics.get(metric) > benchmark * 1.5:
                    report['weaknesses'].append(f"High {metric.replace('_', ' ')}: {metrics.get(metric):.2f}")
    return report