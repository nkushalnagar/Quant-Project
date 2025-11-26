"""Portfolio optimization using mean-variance framework."""

import pandas as pd
import numpy as np
from scipy.optimize import minimize

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def add_leader_hedges(positions, mapping):
    """
    Add leader hedges to follower positions for pairs trade strategy.
    
    For each follower position, short its leaders proportionally.
    Example: If long NVDA with weight 0.05, short its leaders (ASML, AMAT, etc.)
    
    Args:
        positions: DataFrame of follower positions (date x ticker)
        mapping: Leader-follower mapping dict
    
    Returns:
        DataFrame with both follower and leader positions
    """
    # Create expanded positions DataFrame with all tickers (followers + leaders)
    all_tickers = set(positions.columns)
    for follower, leader_pairs in mapping.items():
        for leader, _ in leader_pairs:
            all_tickers.add(leader)
    
    positions_hedged = pd.DataFrame(0.0, index=positions.index, columns=sorted(all_tickers))
    
    # Copy follower positions
    for col in positions.columns:
        if col in positions_hedged.columns:
            positions_hedged[col] = positions[col]
    
    # Add leader hedges
    for date in positions.index:
        for follower in positions.columns:
            follower_weight = positions.loc[date, follower]
            
            if abs(follower_weight) < 1e-10:  # Skip if position is zero
                continue
            
            # Get leaders for this follower
            if follower in mapping:
                leader_pairs = mapping[follower]
                n_leaders = len(leader_pairs)
                
                if n_leaders > 0:
                    # Distribute hedge equally across leaders
                    # If long follower, short leaders; if short follower, long leaders
                    for leader, weight in leader_pairs:
                        if leader in positions_hedged.columns:
                            # Hedge amount = follower position * leader weight / total weight
                            hedge_weight = -follower_weight * weight
                            positions_hedged.loc[date, leader] += hedge_weight
    
    return positions_hedged


def add_market_hedge(positions, direction=-1, weight=1.0, market_ticker='SPY'):
    """
    Add market hedge (long or short SPY) to portfolio.
    
    Args:
        positions: DataFrame of positions (date x ticker)
        direction: -1 for short (market neutral), +1 for long (add market beta)
        weight: Weight of market hedge relative to portfolio (1.0 = dollar-neutral)
        market_ticker: Ticker to use for market hedge (default: SPY)
    
    Returns:
        DataFrame with market hedge added
    """
    # Add SPY column if not present
    if market_ticker not in positions.columns:
        positions[market_ticker] = 0.0
    
    # Calculate total gross exposure for each date
    for date in positions.index:
        gross_exposure = positions.loc[date].abs().sum()
        
        # Add market hedge: direction * weight * gross_exposure
        # If short SPY (direction=-1), this creates market-neutral portfolio
        # If long SPY (direction=+1), this adds market beta
        positions.loc[date, market_ticker] = direction * weight * gross_exposure
    
    return positions


def estimate_covariance_matrix(ret_p, lookback=60, shrinkage=0.1):
    """Estimate covariance matrix with shrinkage."""
    cov_dict = {}
    for date in tqdm(ret_p.index, desc="Covariance estimation"):
        hist_ret = ret_p.loc[ret_p.index <= date].tail(lookback)
        
        if len(hist_ret) < 20 or len(hist_ret.columns) == 0:
            n = len(ret_p.columns)
            if n > 0:
                cov_dict[date] = pd.DataFrame(
                    np.eye(n) * 0.0004,
                    index=ret_p.columns,
                    columns=ret_p.columns
                )
            continue
        
        cov_sample = hist_ret.cov().values
        diag_cov = np.diag(np.diag(cov_sample))
        cov_shrunk = (1 - shrinkage) * cov_sample + shrinkage * diag_cov
        
        cov_dict[date] = pd.DataFrame(
            cov_shrunk,
            index=hist_ret.columns,
            columns=hist_ret.columns
        )
    
    return cov_dict


def mean_variance_optimize(sig_z_row, ret_p_hist, cov_matrix, risk_aversion=1.0,
                           top_q=0.3, max_weight=0.05, min_weight=0.01):
    """Mean-variance optimization for a single date."""
    sig_clean = sig_z_row.dropna()
    if len(sig_clean) < 2:
        return pd.Series(0.0, index=sig_z_row.index)
    
    q_top = sig_clean.quantile(1 - top_q)
    q_bottom = sig_clean.quantile(top_q)
    
    longs = sig_clean[sig_clean >= q_top].index.tolist()
    shorts = sig_clean[sig_clean <= q_bottom].index.tolist()
    
    if not longs or not shorts:
        return pd.Series(0.0, index=sig_z_row.index)
    
    selected = longs + shorts
    n = len(selected)
    
    # Fast path for small problems
    if n <= 4:
        w_series = pd.Series(0.0, index=sig_z_row.index)
        if longs:
            for ticker in longs:
                w_series[ticker] = 0.5 / len(longs)
        if shorts:
            for ticker in shorts:
                w_series[ticker] = -0.5 / len(shorts)
        return w_series
    
    cov_subset = cov_matrix.loc[selected, selected].values
    mu = sig_clean.loc[selected].values
    risk_penalty = risk_aversion / 2.0
    
    def objective(w):
        portfolio_return = np.dot(w, mu)
        cov_w = np.dot(cov_subset, w)
        portfolio_variance = np.dot(w, cov_w)
        return -(portfolio_return - risk_penalty * portfolio_variance)
    
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w)}]
    
    bounds = []
    for ticker in selected:
        if ticker in longs:
            bounds.append((min_weight, max_weight))
        else:
            bounds.append((-max_weight, -min_weight))
    
    w0 = np.zeros(n)
    n_long, n_short = len(longs), len(shorts)
    if n_long > 0:
        w0[:n_long] = 0.5 / n_long
    if n_short > 0:
        w0[n_long:] = -0.5 / n_short
    
    try:
        result = minimize(
            objective, w0, method='SLSQP', bounds=bounds, constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-5, 'disp': False}
        )
        weights = result.x if result.success else w0
    except:
        weights = w0
    
    w_series = pd.Series(0.0, index=sig_z_row.index)
    for i, ticker in enumerate(selected):
        w_series[ticker] = weights[i]
    
    return w_series


def form_mv_positions(sig_z, ret_p, top_q=0.3, risk_aversion=1.0,
                     lookback_cov=60, cov_shrinkage=0.1, vol_target=0.10,
                     max_weight=0.05, min_weight=0.01, lookback_vol=20, max_scale=5.0):
    """Form positions using mean-variance optimization."""
    positions = pd.DataFrame(0.0, index=sig_z.index, columns=sig_z.columns, dtype=float)
    
    print("Estimating covariance matrices...")
    cov_dict = estimate_covariance_matrix(ret_p, lookback=lookback_cov, shrinkage=cov_shrinkage)
    
    print("Optimizing portfolio weights...")
    dates_list = sorted(sig_z.index)
    for i, date in enumerate(tqdm(dates_list, desc="MV optimization")):
        if i % 100 == 0:
            print(f"  Processing date {i+1}/{len(sig_z.index)}: {date}")
        sig_row = sig_z.loc[date]
        cov_matrix = cov_dict.get(date)
        
        if cov_matrix is None:
            continue
        
        weights = mean_variance_optimize(
            sig_row, None, cov_matrix,
            risk_aversion=risk_aversion,
            top_q=top_q,
            max_weight=max_weight,
            min_weight=min_weight
        )
        
        positions.loc[date] = weights
    
    # Volatility targeting
    if vol_target is not None and ret_p is not None:
        pnl_gross = (positions.shift(1) * ret_p).sum(axis=1)
        realized_vol = pnl_gross.rolling(lookback_vol, min_periods=lookback_vol).std() * np.sqrt(252)
        scale = vol_target / (realized_vol.shift(1) + 1e-6)
        scale = scale.clip(upper=max_scale).fillna(1.0)
        positions = positions.multiply(scale, axis=0)
    
    return positions
