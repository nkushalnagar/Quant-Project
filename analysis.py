"""Robustness checks and sensitivity analysis."""

import pandas as pd
import numpy as np
from config import MIN_PRICE, MIN_ADV_USD, HOLD_DAYS, TRANSACTION_COST_BPS_PER_SIDE
from signal_generation import apply_filters
from portfolio_optimization import form_mv_positions, estimate_covariance_matrix, mean_variance_optimize
from backtesting import backtest, summarize_performance

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def rolling_window_analysis(ret_p, sig_z, prc_p, adv_p, mapping, 
                           window_size=52, step_size=13,
                           risk_aversion=2.0, top_q=0.3, lookback_cov=20,
                           frequency='weekly'):
    """
    Rolling window robustness check: test strategy on multiple time windows.
    
    Args:
        ret_p: Returns panel (weekly)
        sig_z: Standardized signals
        prc_p: Price panel
        adv_p: ADV panel
        mapping: Leader-follower mapping
        window_size: Size of each test window (weeks, default 52 = 1 year)
        step_size: Step size between windows (weeks, default 13 = 1 quarter)
        risk_aversion: Risk aversion parameter
        top_q: Quantile threshold
        lookback_cov: Covariance lookback (weeks)
        frequency: 'daily' or 'weekly'
    
    Returns:
        DataFrame with performance metrics for each window
    """
    print("\n" + "="*80)
    print("ROLLING WINDOW ROBUSTNESS CHECK")
    print("="*80)
    
    results = []
    dates = sorted(ret_p.index)
    n_windows = len(range(0, len(dates) - window_size, step_size))
    
    print(f"Testing {n_windows} rolling windows (window_size={window_size} weeks, step={step_size} weeks)...")
    for start_idx in tqdm(range(0, len(dates) - window_size, step_size), desc="Rolling windows"):
        end_idx = start_idx + window_size
        window_dates = dates[start_idx:end_idx]
        
        if len(window_dates) < window_size:
            break
        
        window_start = window_dates[0]
        window_end = window_dates[-1]
        
        # Get data for this window
        ret_p_window = ret_p.loc[window_dates]
        sig_z_window = sig_z.loc[window_dates]
        prc_p_window = prc_p.loc[window_dates]
        adv_p_window = adv_p.loc[window_dates]
        
        # Apply filters
        sig_masked = apply_filters(sig_z_window, prc_p_window, adv_p_window, 
                                   MIN_PRICE, MIN_ADV_USD)
        
        # Form positions
        positions = form_mv_positions(
            sig_masked, ret_p_window,
            top_q=top_q,
            risk_aversion=risk_aversion,
            lookback_cov=min(lookback_cov, len(window_dates) // 2),
            vol_target=0.10,
            max_weight=0.05,
            min_weight=0.01
        )
        
        # Backtest
        pnl_net, pnl_gross, turnover = backtest(
            ret_p_window, positions, hold_days=HOLD_DAYS, 
            cost_bps_per_side=TRANSACTION_COST_BPS_PER_SIDE
        )
        
        # Compute stats
        stats = summarize_performance(pnl_net, frequency=frequency)
        
        results.append({
            'start_date': window_start,
            'end_date': window_end,
            'n_periods': len(window_dates),
            'sharpe': stats.get('Sharpe', np.nan),
            'return': stats.get('Annualized Return', np.nan),
            'vol': stats.get('Annualized Vol', np.nan),
            'hit_rate': stats.get(f'{frequency.capitalize()} Hit Rate', np.nan),
            'cumulative': stats.get('Cumulative Return', np.nan)
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"\nRolling window results ({len(results)} windows):")
    print(f"  Average Sharpe: {results_df['sharpe'].mean():.4f} ± {results_df['sharpe'].std():.4f}")
    print(f"  Sharpe range: [{results_df['sharpe'].min():.4f}, {results_df['sharpe'].max():.4f}]")
    print(f"  Positive Sharpe windows: {(results_df['sharpe'] > 0).sum()}/{len(results_df)}")
    print(f"  Average Return: {results_df['return'].mean():.4f}")
    print(f"  Average Vol: {results_df['vol'].mean():.4f}")
    
    return results_df


def parameter_sensitivity_analysis(ret_p_train, sig_z_train, prc_p_train, adv_p_train,
                                   base_risk_aversion=2.0, base_top_q=0.3, frequency='weekly'):
    """
    Test sensitivity to key parameters.
    
    Args:
        ret_p_train: Training returns
        sig_z_train: Training signals
        prc_p_train: Training prices
        adv_p_train: Training ADV
        base_risk_aversion: Base risk aversion
        base_top_q: Base quantile threshold
        frequency: 'daily' or 'weekly'
    
    Returns:
        Dictionary with sensitivity results
    """
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    sensitivity_results = {}
    
    # Pre-compute covariance once
    ann_factor = 252 if frequency == 'daily' else 52
    cov_dict = estimate_covariance_matrix(ret_p_train, lookback=20, shrinkage=0.1)
    
    # Test risk_aversion sensitivity
    print("\nTesting risk_aversion sensitivity...")
    risk_aversion_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    ra_results = []
    
    for ra in tqdm(risk_aversion_values, desc="Risk aversion sensitivity"):
        positions = pd.DataFrame(0.0, index=sig_z_train.index, columns=sig_z_train.columns, dtype=float)
        dates_list = sorted(sig_z_train.index)
        for date in dates_list:
            sig_row = sig_z_train.loc[date]
            cov_matrix = cov_dict.get(date)
            if cov_matrix is None:
                continue
            weights = mean_variance_optimize(
                sig_row, None, cov_matrix,
                risk_aversion=ra,
                top_q=base_top_q,
                max_weight=0.05,
                min_weight=0.01
            )
            positions.loc[date] = weights
        
        # Apply vol targeting
        pnl_gross = (positions.shift(1) * ret_p_train).sum(axis=1)
        realized_vol = pnl_gross.rolling(20, min_periods=20).std() * np.sqrt(ann_factor)
        scale = 0.10 / (realized_vol.shift(1) + 1e-6)
        scale = scale.clip(upper=5.0).fillna(1.0)
        positions = positions.multiply(scale, axis=0)
        
        pnl_net, _, _ = backtest(ret_p_train, positions, hold_days=HOLD_DAYS,
                                cost_bps_per_side=TRANSACTION_COST_BPS_PER_SIDE)
        stats = summarize_performance(pnl_net, frequency=frequency)
        ra_results.append({
            'risk_aversion': ra,
            'sharpe': stats.get('Sharpe', np.nan),
            'return': stats.get('Annualized Return', np.nan),
            'vol': stats.get('Annualized Vol', np.nan)
        })
    
    sensitivity_results['risk_aversion'] = pd.DataFrame(ra_results)
    print("\nRisk aversion results:")
    for r in ra_results:
        print(f"  risk_aversion={r['risk_aversion']}: Sharpe={r['sharpe']:.4f}, Return={r['return']:.4f}")
    
    # Test top_q sensitivity
    print("\nTesting top_q sensitivity...")
    top_q_values = [0.2, 0.3, 0.4, 0.5]
    top_q_results = []
    
    for tq in tqdm(top_q_values, desc="Top_q sensitivity"):
        positions = pd.DataFrame(0.0, index=sig_z_train.index, columns=sig_z_train.columns, dtype=float)
        dates_list = sorted(sig_z_train.index)
        for date in dates_list:
            sig_row = sig_z_train.loc[date]
            cov_matrix = cov_dict.get(date)
            if cov_matrix is None:
                continue
            weights = mean_variance_optimize(
                sig_row, None, cov_matrix,
                risk_aversion=base_risk_aversion,
                top_q=tq,
                max_weight=0.05,
                min_weight=0.01
            )
            positions.loc[date] = weights
        
        # Apply vol targeting
        pnl_gross = (positions.shift(1) * ret_p_train).sum(axis=1)
        realized_vol = pnl_gross.rolling(20, min_periods=20).std() * np.sqrt(ann_factor)
        scale = 0.10 / (realized_vol.shift(1) + 1e-6)
        scale = scale.clip(upper=5.0).fillna(1.0)
        positions = positions.multiply(scale, axis=0)
        
        pnl_net, _, _ = backtest(ret_p_train, positions, hold_days=HOLD_DAYS,
                                cost_bps_per_side=TRANSACTION_COST_BPS_PER_SIDE)
        stats = summarize_performance(pnl_net, frequency=frequency)
        top_q_results.append({
            'top_q': tq,
            'sharpe': stats.get('Sharpe', np.nan),
            'return': stats.get('Annualized Return', np.nan),
            'vol': stats.get('Annualized Vol', np.nan)
        })
    
    sensitivity_results['top_q'] = pd.DataFrame(top_q_results)
    print("\nTop_q results:")
    for r in top_q_results:
        print(f"  top_q={r['top_q']}: Sharpe={r['sharpe']:.4f}, Return={r['return']:.4f}")
    
    print("="*80 + "\n")
    
    return sensitivity_results
