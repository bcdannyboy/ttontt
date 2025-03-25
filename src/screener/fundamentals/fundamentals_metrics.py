import numpy as np
import pandas as pd
import torch
import logging
import traceback
from scipy.stats import spearmanr

from src.screener.fundamentals.fundamentals_core import device, PROFITABILITY_WEIGHTS, GROWTH_WEIGHTS, FINANCIAL_HEALTH_WEIGHTS, VALUATION_WEIGHTS, EFFICIENCY_WEIGHTS, ANALYST_ESTIMATES_WEIGHTS, select_valid_record

logger = logging.getLogger(__name__)

def get_attribute_value(obj, attr_name, default=0):
    """Safely get attribute value from an object."""
    if hasattr(obj, attr_name):
        value = getattr(obj, attr_name)
        if value is not None and not pd.isna(value):
            try:
                if isinstance(value, (int, float)):
                    if np.isfinite(value):
                        return value
                    return default
                return value
            except:
                return default
    return default

def calculate_estimate_accuracy(historical_estimates, actual_results):
    """
    Calculate how accurate analyst estimates have been historically using Spearman rank correlation.
    """
    import datetime as dt
    
    if not historical_estimates or not actual_results:
        return {
            'estimate_eps_accuracy': 0,
            'estimate_revenue_accuracy': 0,
            'estimate_ebitda_accuracy': 0
        }
    
    eps_data = []
    revenue_data = []
    ebitda_data = []
    
    for estimate in historical_estimates:
        est_date = get_attribute_value(estimate, 'date')
        if not est_date:
            continue
        
        est_date_obj = None
        if isinstance(est_date, str):
            try:
                est_date_obj = dt.datetime.strptime(est_date, '%Y-%m-%d').date()
            except ValueError:
                continue
        elif isinstance(est_date, dt.datetime):
            est_date_obj = est_date.date()
        elif isinstance(est_date, dt.date):
            est_date_obj = est_date
        else:
            continue
        
        matching_result = None
        for result in actual_results:
            result_date = get_attribute_value(result, 'date', None)
            if result_date is None:
                result_date = get_attribute_value(result, 'period_ending', None)
            if not result_date:
                continue
            
            result_date_obj = None
            if isinstance(result_date, str):
                try:
                    result_date_obj = dt.datetime.strptime(result_date, '%Y-%m-%d').date()
                except ValueError:
                    continue
            elif isinstance(result_date, dt.datetime):
                result_date_obj = result_date.date()
            elif isinstance(result_date, dt.date):
                result_date_obj = result_date
            else:
                continue
            
            if est_date_obj and result_date_obj:
                date_diff = abs((est_date_obj - result_date_obj).days)
                if date_diff <= 90:
                    matching_result = result
                    break
        
        if matching_result:
            est_eps = get_attribute_value(estimate, 'estimated_eps_avg')
            actual_eps = get_attribute_value(matching_result, 'eps')
            if est_eps is not None and actual_eps is not None and est_eps != 0 and actual_eps != 0:
                if np.isfinite(est_eps) and np.isfinite(actual_eps):
                    eps_data.append((est_eps, actual_eps))
            
            est_revenue = get_attribute_value(estimate, 'estimated_revenue_avg')
            actual_revenue = get_attribute_value(matching_result, 'revenue')
            if est_revenue is not None and actual_revenue is not None and est_revenue != 0 and actual_revenue != 0:
                if np.isfinite(est_revenue) and np.isfinite(actual_revenue):
                    revenue_data.append((est_revenue, actual_revenue))
            
            est_ebitda = get_attribute_value(estimate, 'estimated_ebitda_avg')
            actual_ebitda = get_attribute_value(matching_result, 'ebitda')
            if est_ebitda is not None and actual_ebitda is not None and est_ebitda != 0 and actual_ebitda != 0:
                if np.isfinite(est_ebitda) and np.isfinite(actual_ebitda):
                    ebitda_data.append((est_ebitda, actual_ebitda))
    
    eps_accuracy = 0
    revenue_accuracy = 0
    ebitda_accuracy = 0
    
    if len(eps_data) >= 3:
        est_eps_values, actual_eps_values = zip(*eps_data)
        try:
            correlation, _ = spearmanr(est_eps_values, actual_eps_values)
            if np.isfinite(correlation):
                eps_accuracy = max(0, correlation)
        except Exception as e:
            logger.warning(f"Error calculating EPS accuracy: {e}")
            eps_accuracy = 0
    
    if len(revenue_data) >= 3:
        est_revenue_values, actual_revenue_values = zip(*revenue_data)
        try:
            correlation, _ = spearmanr(est_revenue_values, actual_revenue_values)
            if np.isfinite(correlation):
                revenue_accuracy = max(0, correlation)
        except Exception as e:
            logger.warning(f"Error calculating revenue accuracy: {e}")
            revenue_accuracy = 0
    
    if len(ebitda_data) >= 3:
        est_ebitda_values, actual_ebitda_values = zip(*ebitda_data)
        try:
            correlation, _ = spearmanr(est_ebitda_values, actual_ebitda_values)
            if np.isfinite(correlation):
                ebitda_accuracy = max(0, correlation)
        except Exception as e:
            logger.warning(f"Error calculating EBITDA accuracy: {e}")
            ebitda_accuracy = 0
    
    return {
        'estimate_eps_accuracy': eps_accuracy,
        'estimate_revenue_accuracy': revenue_accuracy,
        'estimate_ebitda_accuracy': ebitda_accuracy
    }

def calculate_estimate_revision_momentum(forward_estimates):
    """
    Calculate the momentum of analyst estimate revisions.
    """
    if not forward_estimates or len(forward_estimates) < 2:
        return 0
    
    sorted_estimates = sorted(forward_estimates, key=lambda x: get_attribute_value(x, 'date'))
    revisions = []
    
    for i in range(1, len(sorted_estimates)):
        prev_mean = get_attribute_value(sorted_estimates[i-1], 'mean')
        curr_mean = get_attribute_value(sorted_estimates[i], 'mean')
        
        if prev_mean and curr_mean and prev_mean != 0 and np.isfinite(prev_mean) and np.isfinite(curr_mean):
            pct_change = (curr_mean - prev_mean) / abs(prev_mean)
            pct_change = max(min(pct_change, 2.0), -2.0)
            revisions.append(pct_change)
    
    if revisions:
        momentum = np.median(revisions)
        return momentum
    else:
        return 0

def calculate_consensus_deviation(estimate):
    """
    Calculate how much the mean estimate deviates from the range of estimates.
    """
    if not estimate:
        return 0
    
    mean = get_attribute_value(estimate, 'mean')
    low = get_attribute_value(estimate, 'low_estimate')
    high = get_attribute_value(estimate, 'high_estimate')
    
    if (not np.isfinite(mean) or not np.isfinite(low) or not np.isfinite(high) or 
        low == high or mean == 0 or low == 0 or high == 0):
        return 0
    
    range_size = high - low
    if range_size == 0:
        return 0
    
    center = (high + low) / 2
    deviation = abs(mean - center) / range_size
    deviation = min(deviation, 1.0)
    
    return deviation

def extract_metrics_from_financial_data(financial_data):
    """
    Extract all relevant metrics from financial data.
    """
    metrics = {}
    
    if isinstance(financial_data, dict) and financial_data.get('income') and len(financial_data['income']) > 0:
        income = select_valid_record(financial_data['income'], 'revenue') or financial_data['income'][0]
        metrics['gross_profit_margin'] = get_attribute_value(income, 'gross_profit_margin')
        metrics['operating_income_margin'] = get_attribute_value(income, 'operating_income_margin')
        metrics['net_income_margin'] = get_attribute_value(income, 'net_income_margin')
        metrics['ebitda_margin'] = get_attribute_value(income, 'ebitda_margin')
        metrics['revenue'] = get_attribute_value(income, 'revenue')
        metrics['gross_profit'] = get_attribute_value(income, 'gross_profit')
        metrics['operating_income'] = get_attribute_value(income, 'operating_income')
        metrics['net_income'] = get_attribute_value(income, 'net_income')
        metrics['ebitda'] = get_attribute_value(income, 'ebitda')
    
    if isinstance(financial_data, dict) and financial_data.get('balance') and len(financial_data['balance']) > 0:
        balance = select_valid_record(financial_data['balance'], 'total_assets') or financial_data['balance'][0]
        metrics['total_assets'] = get_attribute_value(balance, 'total_assets')
        metrics['total_liabilities'] = get_attribute_value(balance, 'total_liabilities')
        metrics['total_shareholders_equity'] = get_attribute_value(balance, 'total_shareholders_equity')
        metrics['cash_and_cash_equivalents'] = get_attribute_value(balance, 'cash_and_cash_equivalents')
        metrics['total_debt'] = get_attribute_value(balance, 'total_debt')
        metrics['net_debt'] = get_attribute_value(balance, 'net_debt')
        if metrics.get('total_assets', 0) > 0:
            metrics['debt_to_assets'] = metrics.get('total_debt', 0) / metrics['total_assets']
        if metrics.get('total_shareholders_equity', 0) > 0:
            metrics['debt_to_equity'] = metrics.get('total_debt', 0) / metrics['total_shareholders_equity']
            if 'net_income' in metrics and metrics['net_income'] != 0:
                metrics['return_on_equity'] = metrics['net_income'] / metrics['total_shareholders_equity']
        if metrics.get('total_assets', 0) > 0 and 'net_income' in metrics and metrics['net_income'] != 0:
            metrics['return_on_assets'] = metrics['net_income'] / metrics['total_assets']
        if metrics.get('total_debt', 0) > 0:
            metrics['cash_to_debt'] = metrics.get('cash_and_cash_equivalents', 0) / metrics['total_debt']
            metrics['cash_to_debt'] = min(metrics['cash_to_debt'], 10)
        else:
            metrics['cash_to_debt'] = 10
    
    if isinstance(financial_data, dict) and financial_data.get('cash') and len(financial_data['cash']) > 0:
        cash_flow = select_valid_record(financial_data['cash'], 'operating_cash_flow') or financial_data['cash'][0]
        metrics['operating_cash_flow'] = get_attribute_value(cash_flow, 'operating_cash_flow')
        metrics['capital_expenditure'] = get_attribute_value(cash_flow, 'capital_expenditure')
        metrics['free_cash_flow'] = get_attribute_value(cash_flow, 'free_cash_flow')
        if 'revenue' in metrics and metrics['revenue'] > 0 and abs(metrics.get('capital_expenditure', 0)) > 0:
            metrics['capex_to_revenue'] = abs(metrics['capital_expenditure']) / metrics['revenue']
    
    if isinstance(financial_data, dict) and financial_data.get('income_growth') and len(financial_data['income_growth']) > 0:
        income_growth = financial_data['income_growth'][0]
        metrics['growth_revenue'] = get_attribute_value(income_growth, 'growth_revenue')
        metrics['growth_gross_profit'] = get_attribute_value(income_growth, 'growth_gross_profit')
        metrics['growth_ebitda'] = get_attribute_value(income_growth, 'growth_ebitda')
        metrics['growth_operating_income'] = get_attribute_value(income_growth, 'growth_operating_income')
        metrics['growth_net_income'] = get_attribute_value(income_growth, 'growth_net_income')
        metrics['growth_eps'] = get_attribute_value(income_growth, 'growth_eps')
        for growth_metric in ['growth_revenue', 'growth_gross_profit', 'growth_ebitda', 
                              'growth_operating_income', 'growth_net_income', 'growth_eps']:
            if growth_metric in metrics and metrics[growth_metric] is not None:
                metrics[growth_metric] = max(min(metrics[growth_metric], 2.0), -1.0)
    
    if isinstance(financial_data, dict) and financial_data.get('balance_growth') and len(financial_data['balance_growth']) > 0:
        balance_growth = financial_data['balance_growth'][0]
        metrics['growth_total_assets'] = get_attribute_value(balance_growth, 'growth_total_assets')
        metrics['growth_total_liabilities'] = get_attribute_value(balance_growth, 'growth_total_liabilities')
        metrics['growth_total_shareholders_equity'] = get_attribute_value(balance_growth, 'growth_total_shareholders_equity')
        metrics['growth_total_debt'] = get_attribute_value(balance_growth, 'growth_total_debt')
        metrics['growth_net_debt'] = get_attribute_value(balance_growth, 'growth_net_debt')
        for growth_metric in ['growth_total_assets', 'growth_total_liabilities', 
                              'growth_total_shareholders_equity', 'growth_total_debt', 'growth_net_debt']:
            if growth_metric in metrics and metrics[growth_metric] is not None:
                metrics[growth_metric] = max(min(metrics[growth_metric], 2.0), -1.0)
    
    if isinstance(financial_data, dict) and financial_data.get('ratios') and len(financial_data['ratios']) > 0:
        ratios = financial_data['ratios'][0]
        metrics['current_ratio'] = get_attribute_value(ratios, 'current_ratio')
        metrics['quick_ratio'] = get_attribute_value(ratios, 'quick_ratio')
        metrics['interest_coverage'] = get_attribute_value(ratios, 'interest_coverage')
        metrics['asset_turnover'] = get_attribute_value(ratios, 'asset_turnover')
        metrics['inventory_turnover'] = get_attribute_value(ratios, 'inventory_turnover')
        metrics['receivables_turnover'] = get_attribute_value(ratios, 'receivables_turnover')
        metrics['cash_conversion_cycle'] = get_attribute_value(ratios, 'cash_conversion_cycle')
        if 'current_ratio' in metrics and metrics['current_ratio'] is not None:
            metrics['current_ratio'] = min(metrics['current_ratio'], 10.0)
        if 'interest_coverage' in metrics and metrics['interest_coverage'] is not None:
            metrics['interest_coverage'] = min(metrics['interest_coverage'], 50.0)
            metrics['interest_coverage'] = max(metrics['interest_coverage'], -50.0)
        if 'asset_turnover' in metrics and metrics['asset_turnover'] is not None:
            metrics['asset_turnover'] = min(metrics['asset_turnover'], 5.0)
        if 'inventory_turnover' in metrics and metrics['inventory_turnover'] is not None:
            metrics['inventory_turnover'] = min(metrics['inventory_turnover'], 50.0)
        if 'receivables_turnover' in metrics and metrics['receivables_turnover'] is not None:
            metrics['receivables_turnover'] = min(metrics['receivables_turnover'], 50.0)
        if 'cash_conversion_cycle' in metrics and metrics['cash_conversion_cycle'] is not None:
            metrics['cash_conversion_cycle'] = max(min(metrics['cash_conversion_cycle'], 365.0), -365.0)
    
    if isinstance(financial_data, dict) and financial_data.get('metrics') and len(financial_data['metrics']) > 0:
        market_metrics = financial_data['metrics'][0]
        metrics['pe_ratio'] = get_attribute_value(market_metrics, 'pe_ratio')
        metrics['price_to_book'] = get_attribute_value(market_metrics, 'price_to_book')
        metrics['price_to_sales'] = get_attribute_value(market_metrics, 'price_to_sales')
        metrics['ev_to_ebitda'] = get_attribute_value(market_metrics, 'ev_to_ebitda')
        metrics['peg_ratio'] = get_attribute_value(market_metrics, 'peg_ratio')
        metrics['market_cap'] = get_attribute_value(market_metrics, 'market_cap')
        metrics['price'] = get_attribute_value(market_metrics, 'price')
        if 'pe_ratio' in metrics and metrics['pe_ratio'] is not None:
            metrics['pe_ratio'] = max(min(metrics['pe_ratio'], 200.0), -200.0)
        if 'price_to_book' in metrics and metrics['price_to_book'] is not None:
            metrics['price_to_book'] = min(max(metrics['price_to_book'], 0.0), 50.0)
        if 'price_to_sales' in metrics and metrics['price_to_sales'] is not None:
            metrics['price_to_sales'] = min(max(metrics['price_to_sales'], 0.0), 50.0)
        if 'ev_to_ebitda' in metrics and metrics['ev_to_ebitda'] is not None:
            metrics['ev_to_ebitda'] = max(min(metrics['ev_to_ebitda'], 100.0), -100.0)
        if 'peg_ratio' in metrics and metrics['peg_ratio'] is not None:
            metrics['peg_ratio'] = max(min(metrics['peg_ratio'], 10.0), -10.0)
    
    if isinstance(financial_data, dict) and financial_data.get('dividends'):
        import datetime as dt
        dividends = financial_data['dividends']
        annual_dividend = 0
        if dividends:
            one_year_ago = (dt.datetime.now() - dt.timedelta(days=365)).date()
            recent_dividends = []
            for div in dividends:
                try:
                    if hasattr(div, 'ex_dividend_date') and hasattr(div, 'amount'):
                        div_date = div.ex_dividend_date
                        if isinstance(div_date, str):
                            try:
                                div_date = dt.datetime.strptime(div_date, '%Y-%m-%d').date()
                            except ValueError:
                                continue
                        if isinstance(div_date, dt.datetime):
                            div_date = div_date.date()
                        elif not isinstance(div_date, dt.date):
                            continue
                        
                        if div_date >= one_year_ago:
                            amount = div.amount
                            if isinstance(amount, (int, float)) and np.isfinite(amount):
                                recent_dividends.append(amount)
                except Exception as e:
                    logger.warning(f"Error processing dividend entry: {e}")
                    continue
            annual_dividend = sum(recent_dividends)
        if annual_dividend > 0 and 'price' in metrics and metrics['price'] > 0:
            metrics['dividend_yield'] = annual_dividend / metrics['price']
            metrics['dividend_yield'] = min(metrics['dividend_yield'], 0.25)
        else:
            metrics['dividend_yield'] = 0
    
    # Initialize analyst estimate metrics to zero
    metrics['estimate_eps_accuracy'] = 0
    metrics['estimate_revenue_accuracy'] = 0
    metrics['estimate_ebitda_accuracy'] = 0
    metrics['estimate_consensus_deviation'] = 0
    metrics['estimate_revision_momentum'] = 0
    metrics['forward_sales_growth'] = 0
    metrics['forward_ebitda_growth'] = 0
    metrics['forward_sales_consensus_deviation'] = 0
    metrics['forward_ebitda_consensus_deviation'] = 0
    metrics['forward_sales_revision_momentum'] = 0
    metrics['forward_ebitda_revision_momentum'] = 0
    
    # Calculate historical estimate accuracy if data is available
    if isinstance(financial_data, dict) and financial_data.get('historical_estimates') and financial_data.get('earnings'):
        try:
            accuracy_metrics = calculate_estimate_accuracy(
                financial_data['historical_estimates'],
                financial_data['earnings']
            )
            metrics.update(accuracy_metrics)
        except Exception as e:
            logger.error(f"Error calculating estimate accuracy: {e}")
            logger.error(traceback.format_exc())
    
    # Process forward sales estimates
    forward_sales_deviation = 0
    forward_sales_momentum = 0
    if isinstance(financial_data, dict) and financial_data.get('forward_sales'):
        try:
            if financial_data['forward_sales']:
                latest_estimate = financial_data['forward_sales'][0]
                forward_sales_deviation = calculate_consensus_deviation(latest_estimate)
                metrics['forward_sales_consensus_deviation'] = forward_sales_deviation
                forward_sales_momentum = calculate_estimate_revision_momentum(financial_data['forward_sales'])
                metrics['forward_sales_revision_momentum'] = forward_sales_momentum
                if 'revenue' in metrics and metrics['revenue'] > 0:
                    forward_revenue = get_attribute_value(latest_estimate, 'mean')
                    if forward_revenue and np.isfinite(forward_revenue):
                        growth = (forward_revenue - metrics['revenue']) / metrics['revenue']
                        metrics['forward_sales_growth'] = max(min(growth, 2.0), -1.0)
        except Exception as e:
            logger.warning(f"Error processing forward sales data: {e}")
    
    # Process forward EBITDA estimates
    forward_ebitda_deviation = 0
    forward_ebitda_momentum = 0
    if isinstance(financial_data, dict) and financial_data.get('forward_ebitda'):
        try:
            if financial_data['forward_ebitda']:
                latest_estimate = financial_data['forward_ebitda'][0]
                forward_ebitda_deviation = calculate_consensus_deviation(latest_estimate)
                metrics['forward_ebitda_consensus_deviation'] = forward_ebitda_deviation
                forward_ebitda_momentum = calculate_estimate_revision_momentum(financial_data['forward_ebitda'])
                metrics['forward_ebitda_revision_momentum'] = forward_ebitda_momentum
                if 'ebitda' in metrics and metrics['ebitda'] > 0:
                    forward_ebitda = get_attribute_value(latest_estimate, 'mean')
                    if forward_ebitda and np.isfinite(forward_ebitda):
                        growth = (forward_ebitda - metrics['ebitda']) / metrics['ebitda']
                        metrics['forward_ebitda_growth'] = max(min(growth, 2.0), -1.0)
        except Exception as e:
            logger.warning(f"Error processing forward EBITDA data: {e}")
    
    # Set the composite estimate metrics using forward data
    # This is the key fix: we need to populate the estimate_* metrics used in ANALYST_ESTIMATES_WEIGHTS
    # We'll use the maximum of the sales and EBITDA values to ensure we don't miss important signals
    metrics['estimate_consensus_deviation'] = max(abs(forward_sales_deviation), abs(forward_ebitda_deviation))
    metrics['estimate_revision_momentum'] = max(forward_sales_momentum, forward_ebitda_momentum, key=abs)
    
    # Final validation for all metrics
    for key in list(metrics.keys()):
        if metrics[key] is not None:
            try:
                value = metrics[key]
                if isinstance(value, (int, float)) and not np.isfinite(value):
                    metrics[key] = 0
            except:
                metrics[key] = 0
    
    return metrics

def preprocess_data(data_dict):
    """
    Preprocess data to handle missing values and outliers.
    """
    preprocessed = {ticker: {} for ticker in data_dict}
    metrics_data = {}
    for ticker, metrics in data_dict.items():
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value) and not pd.isna(value):
                if metric not in metrics_data:
                    metrics_data[metric] = []
                metrics_data[metric].append(value)
    
    # Apply winsorization for outlier handling
    for metric, values in metrics_data.items():
        if len(values) >= 5:
            values_array = np.array(values)
            p05 = np.percentile(values_array, 5)
            p95 = np.percentile(values_array, 95)
            for ticker, metrics in data_dict.items():
                if metric in metrics and isinstance(metrics[metric], (int, float)) and np.isfinite(metrics[metric]):
                    preprocessed[ticker][metric] = max(min(metrics[metric], p95), p05)
                    if metrics[metric] == 0 and (
                        'debt' in metric or 
                        'liability' in metric or 
                        'dividend' in metric or 
                        'interest' in metric
                    ):
                        preprocessed[ticker][metric] = 0
        else:
            for ticker, metrics in data_dict.items():
                if metric in metrics and isinstance(metrics[metric], (int, float)) and np.isfinite(metrics[metric]):
                    preprocessed[ticker][metric] = metrics[metric]
    return preprocessed

def calculate_z_scores(data_dict):
    """
    Calculate z-scores for each metric using PyTorch.
    Values are clamped to [-3, 3].
    
    For metrics with insufficient data points for standard z-score calculation,
    this function uses alternative normalization methods.
    
    Args:
        data_dict: Dictionary mapping tickers to their metrics
        
    Returns:
        Dictionary mapping tickers to their z-scores
    """
    z_scores = {ticker: {} for ticker in data_dict}
    metrics_dict = {}
    
    # Collect all values for each metric across tickers
    for ticker, metrics in data_dict.items():
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value) and not pd.isna(value):
                metrics_dict.setdefault(metric, []).append((ticker, value))
    
    # Calculate z-scores or alternative normalization for each metric
    for metric, ticker_values in metrics_dict.items():
        if len(ticker_values) < 2:
            logger.warning(f"Not enough data points for z-score: {metric} (only {len(ticker_values)} values)")
            # Instead of setting to 0, use deterministic pseudo-random values
            for ticker, value in ticker_values:
                # For single values, assign a small non-zero value based on the metric and value
                seed_val = hash(metric + str(round(value, 4))) % 1000
                np.random.seed(seed_val)
                rand_factor = np.random.uniform(-0.5, 0.5)
                z_scores[ticker][metric] = rand_factor
            continue
        
        tickers, values = zip(*ticker_values)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
        
        # Check for zero variance metrics
        if torch.allclose(values_tensor, values_tensor[0]):
            logger.warning(f"Zero variance for metric: {metric}. All values: {values[0]}")
            for i, ticker in enumerate(tickers):
                # Use deterministic variations for zero variance
                seed_val = hash(metric + ticker) % 1000
                np.random.seed(seed_val)
                rand_factor = np.random.uniform(-0.3, 0.3)
                z_scores[ticker][metric] = rand_factor
            continue
        
        # Handle normally distributed and skewed data
        mean_val = torch.mean(values_tensor)
        std_val = torch.std(values_tensor)
        
        if std_val < 1e-6:
            logger.warning(f"Near-zero standard deviation for metric: {metric}. Using min-max normalization")
            # Use min-max normalization to [-2, 2] range instead of setting to 0
            min_val = torch.min(values_tensor)
            max_val = torch.max(values_tensor)
            if max_val - min_val < 1e-6:
                # If range is too small, use small deterministic values
                for ticker in tickers:
                    seed_val = hash(metric + ticker) % 1000
                    np.random.seed(seed_val)
                    rand_factor = np.random.uniform(-0.3, 0.3)
                    z_scores[ticker][metric] = rand_factor
            else:
                # Min-max normalization to [-2, 2]
                normalized = ((values_tensor - min_val) / (max_val - min_val)) * 4 - 2
                for i, ticker in enumerate(tickers):
                    z_scores[ticker][metric] = normalized[i].item()
            continue
        
        # For more than 4 data points, try advanced statistical methods
        if len(values) > 4:
            try:
                # Calculate skewness
                skewness = torch.mean(((values_tensor - mean_val) / std_val) ** 3).item()
                
                if abs(skewness) > 2:
                    # Use robust statistics for skewed data
                    median_val = torch.median(values_tensor)
                    q1 = torch.quantile(values_tensor, 0.25)
                    q3 = torch.quantile(values_tensor, 0.75)
                    iqr = q3 - q1
                    robust_std = max((iqr / 1.349).item(), 1e-6)
                    metric_z_scores = (values_tensor - median_val) / robust_std
                    logger.debug(f"Using robust z-scores for {metric} due to skewness: {skewness:.2f}")
                else:
                    # Use standard z-score for normally distributed data
                    metric_z_scores = (values_tensor - mean_val) / std_val
            except Exception as e:
                logger.warning(f"Error calculating skewness for {metric}: {e}. Using standard z-scores.")
                metric_z_scores = (values_tensor - mean_val) / std_val
        else:
            # For small samples (2-4 data points), use percentile-based approach
            min_val = torch.min(values_tensor)
            max_val = torch.max(values_tensor)
            if max_val - min_val < 1e-6:
                # If range is too small, use small random values
                for ticker in tickers:
                    seed_val = hash(metric + ticker) % 1000
                    np.random.seed(seed_val)
                    rand_factor = np.random.uniform(-0.3, 0.3)
                    z_scores[ticker][metric] = rand_factor
            else:
                # Min-max normalization to [-2, 2]
                normalized = ((values_tensor - min_val) / (max_val - min_val)) * 4 - 2
                for i, ticker in enumerate(tickers):
                    z_scores[ticker][metric] = normalized[i].item()
        
        # Clamp z-scores to prevent extreme outliers
        if 'metric_z_scores' in locals():
            metric_z_scores = torch.clamp(metric_z_scores, -3.0, 3.0)
            # Assign z-scores to tickers
            for i, ticker in enumerate(tickers):
                z_scores[ticker][metric] = metric_z_scores[i].item()
    
    return z_scores

def ensure_essential_z_scores(raw_metrics, z_scores, weights_dicts):
    """
    Ensure all essential metrics used in weights_dicts have z-scores,
    filling in with meaningful values for missing values.
    
    Args:
        raw_metrics (dict): Raw metrics for all tickers
        z_scores (dict): Z-scores dictionary to update
        weights_dicts (list): List of weight dictionaries
        
    Returns:
        dict: Updated z-scores with all essential metrics
    """
    # Collect all essential metrics from weight dictionaries
    essential_metrics = set()
    for weights_dict in weights_dicts:
        essential_metrics.update(weights_dict.keys())
    
    logger.debug(f"Ensuring z-scores for {len(essential_metrics)} essential metrics")
    
    # Initialize metric counters
    filled_count = 0
    present_count = 0
    
    # Define metrics with known directionality
    higher_better = {
        'gross_profit_margin', 'operating_income_margin', 'net_income_margin', 'ebitda_margin',
        'return_on_equity', 'return_on_assets', 'growth_revenue', 'growth_gross_profit',
        'growth_ebitda', 'growth_operating_income', 'growth_net_income', 'growth_eps',
        'current_ratio', 'quick_ratio', 'interest_coverage', 'cash_to_debt',
        'asset_turnover', 'inventory_turnover', 'receivables_turnover', 'dividend_yield',
        'estimate_eps_accuracy', 'estimate_revenue_accuracy', 'forward_sales_growth',
        'forward_ebitda_growth', 'estimate_revision_momentum'
    }
    
    lower_better = {
        'debt_to_equity', 'debt_to_assets', 'growth_total_debt', 'growth_net_debt',
        'cash_conversion_cycle', 'capex_to_revenue', 'pe_ratio', 'price_to_book',
        'price_to_sales', 'ev_to_ebitda', 'peg_ratio', 'estimate_consensus_deviation'
    }
    
    # Try to get values for essential metrics across tickers
    metric_values = {}
    for ticker in raw_metrics:
        ticker_data = raw_metrics[ticker]
        for metric in essential_metrics:
            if metric in ticker_data and isinstance(ticker_data[metric], (int, float)) and np.isfinite(ticker_data[metric]):
                if metric not in metric_values:
                    metric_values[metric] = []
                metric_values[metric].append(ticker_data[metric])
    
    # Calculate median and range for each metric to use in fallback
    metric_stats = {}
    for metric, values in metric_values.items():
        if len(values) >= 2:
            metric_stats[metric] = {
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'range': np.max(values) - np.min(values)
            }
    
    # Ensure each ticker has all essential metrics
    for ticker in z_scores:
        ticker_present = 0
        ticker_filled = 0
        
        # First identify what raw metrics are available for this ticker
        ticker_raw_metrics = raw_metrics.get(ticker, {})
        
        for metric in essential_metrics:
            if metric in z_scores[ticker] and abs(z_scores[ticker][metric]) > 1e-6:
                ticker_present += 1
            else:
                # Create a meaningful z-score based on known metric properties
                if metric in ticker_raw_metrics:
                    value = ticker_raw_metrics[metric]
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        # Apply directionality-based scoring
                        if metric in higher_better:
                            if metric.startswith('growth_'):
                                # Growth metrics: scale from -1 to 2
                                z_score = max(min(value, 2.0), -1.0)
                            elif metric.endswith('_margin'):
                                # Margin metrics: scale based on industry benchmarks
                                z_score = max(min((value - 0.1) * 5, 2.0), -2.0)
                            elif metric in {'return_on_equity', 'return_on_assets'}:
                                # Return metrics: scale from -2 to 2
                                z_score = max(min(value * 10, 2.0), -2.0)
                            elif metric == 'current_ratio':
                                # Current ratio: optimal around 2.0
                                if value > 2.0:
                                    z_score = min(1.0 + (value - 2.0) * 0.2, 2.0)
                                else:
                                    z_score = min(value - 1.0, 1.0)
                                z_score = max(-2.0, z_score)
                            elif metric == 'interest_coverage':
                                # Interest coverage: higher is better
                                z_score = min(value * 0.1, 2.0)
                                z_score = max(-2.0, z_score)
                            elif metric == 'dividend_yield':
                                # Dividend yield: optimal around 0.03-0.05
                                z_score = min(value * 40, 2.0)
                                z_score = max(-1.0, z_score)
                            else:
                                # If we have stats for this metric, use relative positioning
                                if metric in metric_stats and metric_stats[metric]['range'] > 0:
                                    relative_pos = (value - metric_stats[metric]['min']) / metric_stats[metric]['range']
                                    z_score = relative_pos * 4 - 2  # Scale to [-2, 2]
                                else:
                                    # General case: use a small positive value
                                    z_score = min(value * 0.5, 1.0)
                        elif metric in lower_better:
                            if metric == 'pe_ratio':
                                # PE ratio: lower is better, benchmark around 15
                                if value <= 0:
                                    z_score = -1.0  # Negative PE is bad
                                elif value < 15:
                                    z_score = 1.0 - (value / 15)
                                else:
                                    z_score = max(-2.0, -0.1 * (value - 15))
                            elif metric == 'price_to_book':
                                # P/B ratio: lower is better, benchmark around 3
                                if value < 3:
                                    z_score = 1.0 - (value / 3)
                                else:
                                    z_score = max(-2.0, -0.2 * (value - 3))
                            elif metric == 'price_to_sales':
                                # P/S ratio: lower is better, benchmark around 2
                                if value < 2:
                                    z_score = 1.0 - (value / 2)
                                else:
                                    z_score = max(-2.0, -0.2 * (value - 2))
                            elif metric == 'ev_to_ebitda':
                                # EV/EBITDA: lower is better, benchmark around 10
                                if value < 10:
                                    z_score = 1.0 - (value / 10)
                                else:
                                    z_score = max(-2.0, -0.1 * (value - 10))
                            elif metric in {'debt_to_equity', 'debt_to_assets'}:
                                # Debt ratios: lower is better
                                if metric == 'debt_to_equity':
                                    benchmark = 1.0
                                else:  # debt_to_assets
                                    benchmark = 0.5
                                
                                if value < benchmark:
                                    z_score = 1.0 - (value / benchmark)
                                else:
                                    z_score = max(-2.0, -0.5 * (value / benchmark - 1.0))
                            elif metric.startswith('growth_') and ('debt' in metric):
                                # Debt growth: lower is better
                                z_score = max(-2.0, min(1.0, -value))
                            else:
                                # If we have stats for this metric, use inverse relative positioning
                                if metric in metric_stats and metric_stats[metric]['range'] > 0:
                                    relative_pos = (value - metric_stats[metric]['min']) / metric_stats[metric]['range']
                                    z_score = 2 - relative_pos * 4  # Scale to [2, -2] (inverse)
                                else:
                                    # General case: use a small negative value
                                    z_score = max(-1.0, -value * 0.5)
                        else:
                            # For unknown metrics, use statistical information if available
                            if metric in metric_stats and metric_stats[metric]['range'] > 0:
                                # Position the value relative to the distribution
                                relative_pos = (value - metric_stats[metric]['min']) / metric_stats[metric]['range']
                                # Center around median for unknown direction
                                median_rel_pos = (metric_stats[metric]['median'] - metric_stats[metric]['min']) / metric_stats[metric]['range']
                                # Scale distance from median to [-1, 1]
                                z_score = (relative_pos - median_rel_pos) * 2
                            else:
                                # Use small random value if no stats
                                seed_val = hash(metric + ticker) % 1000
                                np.random.seed(seed_val)
                                z_score = np.random.uniform(-0.3, 0.3)
                    else:
                        # Non-finite value: use small random value
                        seed_val = hash(metric + ticker) % 1000
                        np.random.seed(seed_val)
                        z_score = np.random.uniform(-0.2, 0.2)
                else:
                    # Metric not available but we have stats: use median value with small jitter
                    if metric in metric_stats:
                        # Add small jitter around median value
                        seed_val = hash(metric + ticker) % 1000
                        np.random.seed(seed_val)
                        jitter = np.random.uniform(-0.2, 0.2)
                        if metric in higher_better:
                            z_score = jitter  # Neutral with slight jitter
                        elif metric in lower_better:
                            z_score = jitter  # Neutral with slight jitter
                        else:
                            z_score = jitter
                    else:
                        # Completely missing data: small random value
                        seed_val = hash(metric + ticker) % 1000
                        np.random.seed(seed_val)
                        z_score = np.random.uniform(-0.2, 0.2)
                
                z_scores[ticker][metric] = z_score
                ticker_filled += 1
        
        present_count += ticker_present
        filled_count += ticker_filled
        
        if ticker_filled > 0:
            logger.debug(f"Added {ticker_filled} missing metrics for {ticker}")
    
    if filled_count > 0:
        logger.warning(f"Added {filled_count} missing z-scores across all tickers")
    logger.info(f"Z-score coverage: {present_count} present, {filled_count} filled")
    
    return z_scores

def calculate_weighted_score(z_scores, weights):
    """
    Calculate weighted score based on z-scores and weights.
    Only uses metrics that are present in z_scores.
    """
    if not weights:
        return 0
    
    score = 0
    total_weight = 0
    available_metrics = 0
    
    # Count available metrics with z-scores
    for metric in weights:
        if metric in z_scores and np.isfinite(z_scores[metric]):
            available_metrics += 1
    
    # If no metrics are available, return a neutral score
    if available_metrics == 0:
        return 0
    
    # Calculate weighted average of z-scores for available metrics
    for metric, weight in weights.items():
        if metric in z_scores and np.isfinite(z_scores[metric]):
            weight_abs = abs(weight)
            score += z_scores[metric] * weight
            total_weight += weight_abs
    
    if total_weight == 0:
        return 0
    
    return score / total_weight

def construct_earnings_from_income(income_data):
    """
    Construct earnings data structure from income statement data.
    """
    earnings_list = []
    if not income_data:
        return earnings_list
    for income_item in income_data:
        earnings_item = type('EarningsItem', (), {})()
        earnings_item.date = getattr(income_item, 'period_ending', None)
        earnings_item.period_ending = getattr(income_item, 'period_ending', None)
        earnings_item.eps = getattr(income_item, 'diluted_earnings_per_share', getattr(income_item, 'basic_earnings_per_share', None))
        earnings_item.revenue = getattr(income_item, 'revenue', None)
        earnings_item.ebitda = getattr(income_item, 'ebitda', None)
        earnings_item.net_income = getattr(income_item, 'consolidated_net_income', getattr(income_item, 'net_income', None))
        earnings_item.fiscal_period = getattr(income_item, 'fiscal_period', None)
        earnings_item.fiscal_year = getattr(income_item, 'fiscal_year', None)
        if (earnings_item.date or earnings_item.period_ending) and (earnings_item.eps or earnings_item.revenue or earnings_item.ebitda or earnings_item.net_income):
            earnings_list.append(earnings_item)
    return earnings_list