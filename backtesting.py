"""Backtesting engine and performance metrics."""

import pandas as pd
import numpy as np
from scipy import stats


def backtest(ret_p, positions, hold_days=1, cost_bps_per_side=5.0):
    """Simple backtest with transaction costs."""
    positions = positions.reindex(ret_p.index).fillna(0.0)
    
    if hold_days == 1:
        pos_hold = positions
    else:
        weights = np.ones(hold_days) / hold_days
        pos_hold = pd.DataFrame(index=positions.index, columns=positions.columns, dtype=float)
        for i in range(hold_days):
            pos_hold = pos_hold.add(positions.shift(i) * weights[i], fill_value=0.0)

    pnl_gross = (pos_hold.shift(1) * ret_p).sum(axis=1)

    pos_prev = pos_hold.shift(1).fillna(0.0)
    pos_curr = pos_hold.fillna(0.0)
    daily_turnover = (pos_curr - pos_prev).abs().sum(axis=1)
    per_side = cost_bps_per_side / 10000.0
    pnl_costs = daily_turnover * per_side
    pnl_net = pnl_gross - pnl_costs

    return pnl_net, pnl_gross, daily_turnover


def summarize_performance(r, frequency='daily'):
    """
    Compute performance stats.
    
    Args:
        r: Return series
        frequency: 'daily' or 'weekly'
    """
    r = r.dropna()
    if r.empty:
        return {}
    
    # Annualization factor
    ann_factor = 252.0 if frequency == 'daily' else 52.0
    period_name = 'Daily' if frequency == 'daily' else 'Weekly'
    
    mu = r.mean() * ann_factor
    sd = r.std(ddof=0) * np.sqrt(ann_factor)
    sharpe = mu / sd if pd.notna(sd) and sd > 0 else np.nan
    cum = (1 + r).prod() - 1
    hit = (r > 0).mean()
    return {
        "Annualized Return": mu,
        "Annualized Vol": sd,
        "Sharpe": sharpe,
        "Cumulative Return": cum,
        f"{period_name} Hit Rate": hit,
        f"Average {period_name} Return (bps)": r.mean() * 1e4,
        "Average Turnover": r.abs().mean(),
        f"N {period_name.lower()}": int(r.shape[0]),
    }


def test_alpha(returns, alpha_name="Strategy", frequency='daily'):
    """
    Test if alpha is significantly greater than 0.
    
    Args:
        returns: Return series
        alpha_name: Name for display
        frequency: 'daily' or 'weekly'
    """
    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        return None
    
    alpha = returns_clean.mean()
    n = len(returns_clean)
    se = returns_clean.std(ddof=1) / np.sqrt(n)
    t_stat = alpha / se if se > 0 else np.nan
    
    if pd.notna(t_stat):
        p_value = 1 - stats.t.cdf(t_stat, df=n-1)
    else:
        p_value = np.nan
    
    ann_factor = 252.0 if frequency == 'daily' else 52.0
    alpha_ann = alpha * ann_factor
    
    return {
        "Alpha (daily)": alpha,
        "Alpha (annualized)": alpha_ann,
        "t-statistic": t_stat,
        "p-value": p_value,
        "N observations": n,
        "Significant (5%)": p_value < 0.05 if pd.notna(p_value) else False,
        "Significant (1%)": p_value < 0.01 if pd.notna(p_value) else False,
    }
