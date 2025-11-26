"""Signal generation logic."""

import pandas as pd
import numpy as np
from collections import defaultdict


def convert_to_weekly_returns(ret_p):
    """
    Convert daily returns to weekly cumulative returns.
    Returns a DataFrame indexed by week-end dates.
    """
    # Add week identifier (ISO week)
    ret_weekly = ret_p.copy()
    ret_weekly['year'] = ret_weekly.index.year
    ret_weekly['week'] = ret_weekly.index.isocalendar().week
    
    # Group by year-week and compute cumulative return: (1+r1)*(1+r2)*...*(1+rn) - 1
    weekly_data = {}
    for ticker in ret_p.columns:
        weekly_returns = ret_weekly.groupby(['year', 'week'])[ticker].apply(
            lambda x: (1 + x).prod() - 1
        )
        weekly_data[ticker] = weekly_returns
    
    ret_weekly_df = pd.DataFrame(weekly_data)
    
    # Create a proper date index (use the last date of each week)
    week_end_dates = ret_p.groupby([ret_p.index.year, ret_p.index.isocalendar().week]).apply(
        lambda x: x.index.max()
    )
    ret_weekly_df.index = week_end_dates.values
    
    return ret_weekly_df


def build_multi_lag_signal(ret_p, mapping, lags=(1,), lag_decay=0.7, winsor=(0.01, 0.99)):
    """
    Build multi-lag signals with winsorization and exponential decay.
    For weekly data: lag=1 means leader returns from week x-1 predict follower in week x.
    """
    followers = sorted(mapping.keys())
    leaders_needed = sorted({ldr for pairs in mapping.values() for (ldr, _) in pairs})
    missing_leaders = [t for t in leaders_needed if t not in ret_p.columns]
    missing_followers = [t for t in followers if t not in ret_p.columns]
    if missing_leaders:
        print(f"Warning: leaders missing: {missing_leaders}")
    if missing_followers:
        print(f"Warning: followers missing: {missing_followers}")
    
    # Winsorize returns cross-sectionally
    ret_winsor = ret_p.copy()
    q_low = ret_p.quantile(winsor[0], axis=1)
    q_high = ret_p.quantile(winsor[1], axis=1)
    for date in ret_p.index:
        if pd.notna(q_low[date]) and pd.notna(q_high[date]):
            ret_winsor.loc[date] = ret_p.loc[date].clip(lower=q_low[date], upper=q_high[date])
    
    # Build multi-lag signals
    sig = pd.DataFrame(index=ret_p.index, columns=followers, dtype=float)
    sig[:] = np.nan
    
    for fol, pairs in mapping.items():
        if fol not in ret_p.columns:
            continue
        weights, cols = [], []
        for ldr, w in pairs:
            if ldr in ret_winsor.columns:
                cols.append(ldr)
                weights.append(w)
        if not cols:
            continue
        w_arr = np.array(weights, dtype=float)
        w_arr = w_arr / w_arr.sum()
        
        s_agg = pd.Series(0.0, index=ret_p.index)
        for k, lag in enumerate(lags, start=1):
            decay_weight = lag_decay ** (k - 1)
            ret_lag_k = ret_winsor[cols].shift(lag)
            s_k = ret_lag_k.dot(w_arr)
            s_agg = s_agg + decay_weight * s_k
        sig[fol] = s_agg
    return sig


def build_groups_from_mapping(mapping):
    """Construct follower clusters by connecting followers that share leaders."""
    leader_to_followers = defaultdict(set)
    for fol, pairs in mapping.items():
        for ldr, _ in pairs:
            leader_to_followers[ldr].add(fol)
    
    visited, groups, group_id = set(), {}, 0
    
    def dfs(fol, current_group):
        if fol in visited:
            return
        visited.add(fol)
        groups[fol] = current_group
        if fol in mapping:
            for ldr, _ in mapping[fol]:
                for connected_fol in leader_to_followers[ldr]:
                    if connected_fol != fol:
                        dfs(connected_fol, current_group)
    
    for fol in mapping.keys():
        if fol not in visited:
            dfs(fol, group_id)
            group_id += 1
    
    return groups


def apply_filters(sig, prc_p, adv_p, min_price, min_adv_usd):
    """Mask signals where follower fails liquidity/price filters."""
    prc_lag = prc_p.shift(1)
    adv_lag = adv_p.shift(1)
    mask = (prc_lag >= min_price) & (adv_lag >= min_adv_usd)
    return sig.where(mask, other=np.nan)


def cross_sectional_standardize(sig, groups=None):
    """Demean within groups, then z-score across all followers per day."""
    sig_z = sig.copy()
    
    if groups is not None:
        group_dict = {}
        for group_id in set(groups.values()):
            group_followers = [f for f, g in groups.items() if g == group_id and f in sig.columns]
            if group_followers:
                group_dict[group_id] = group_followers
        
        for group_id, group_followers in group_dict.items():
            group_sig = sig_z[group_followers]
            group_means = group_sig.mean(axis=1)
            sig_z[group_followers] = group_sig.subtract(group_means, axis=0)
    
    row_means = sig_z.mean(axis=1)
    row_stds = sig_z.std(axis=1).replace(0, np.nan)
    sig_z = sig_z.subtract(row_means, axis=0).divide(row_stds, axis=0).fillna(0.0)
    
    return sig_z


def random_week_split(ret_weekly, train_pct=0.3, random_seed=42):
    """
    Randomly split weeks into train and test sets.
    
    Args:
        ret_weekly: DataFrame with weekly returns
        train_pct: Fraction of weeks for training
        random_seed: Random seed for reproducibility
    
    Returns:
        ret_train, ret_test: DataFrames with train and test weeks
    """
    np.random.seed(random_seed)
    
    all_weeks = ret_weekly.index.tolist()
    n_weeks = len(all_weeks)
    n_train = int(n_weeks * train_pct)
    
    # Randomly select train weeks
    train_weeks = np.random.choice(all_weeks, size=n_train, replace=False)
    train_weeks = sorted(train_weeks)
    
    # Test weeks are the remaining weeks
    test_weeks = [w for w in all_weeks if w not in train_weeks]
    test_weeks = sorted(test_weeks)
    
    ret_train = ret_weekly.loc[train_weeks].copy()
    ret_test = ret_weekly.loc[test_weeks].copy()
    
    print(f"\nRandom week split (seed={random_seed}):")
    print(f"  Total weeks: {n_weeks}")
    print(f"  Train weeks: {len(train_weeks)} ({len(train_weeks)/n_weeks*100:.1f}%)")
    print(f"  Test weeks: {len(test_weeks)} ({len(test_weeks)/n_weeks*100:.1f}%)")
    print(f"  Train date range: {ret_train.index.min()} to {ret_train.index.max()}")
    print(f"  Test date range: {ret_test.index.min()} to {ret_test.index.max()}")
    
    return ret_train, ret_test
