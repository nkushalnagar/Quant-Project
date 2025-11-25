#!/usr/bin/env python3
"""
Raw leader->follower lead-lag strategy using CRSP daily returns from WRDS.
- Hand-written leader/follower mapping (no SIC/FF49, no hedges).
- Builds daily signals from leaders' t-1 returns and trades followers on t+1.

Requirements:
  pip install wrds pandas numpy pyarrow

WRDS access:
  - Ensure you have WRDS credentials. wrds.Connection() will prompt or use env vars.
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np

try:
    import wrds
except ImportError:
    wrds = None
    print("Warning: wrds package not found. First-run will fail. Install with: pip install wrds")


def fetch_risk_free_rate(start_date, end_date):
    """
    Fetch daily risk-free rate from WRDS Fama-French factors.
    Returns Series with date index and daily risk-free rate.
    """
    if wrds is None:
        # Fallback: use constant 2% annualized (approx 0.0079% daily)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.Series(0.000079, index=dates, name='rf')
    
    try:
        db = wrds.Connection()
        sd = pd.to_datetime(start_date).date().isoformat()
        ed = pd.to_datetime(end_date).date().isoformat()
        sql = f"""
            select date, rf
            from ff.factors_daily
            where date between '{sd}' and '{ed}'
            order by date
        """
        df = db.raw_sql(sql, date_cols=['date'])
        rf_series = df.set_index('date')['rf'] / 100.0  # Convert from percentage
        return rf_series
    except Exception as e:
        print(f"Warning: Could not fetch risk-free rate from WRDS: {e}")
        print("Using constant 2% annualized rate")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.Series(0.000079, index=dates, name='rf')

# ----------------------------
# User-configurable parameters
# ----------------------------

START_DATE = "2000-01-01"
END_DATE   = "2025-11-01"

# Trading configuration
MIN_PRICE = 3.0            # skip stocks with previous close < $3
MIN_ADV_USD = 2_000_000    # skip stocks with 20-day ADV (USD) below this
SIGNAL_ABS_THRESHOLD = 0.0 # e.g., 0.01 to require |signal| >= 1%; 0.0 = no threshold
TRANSACTION_COST_BPS_PER_SIDE = 0.0  # set to 0 to disable costs
HOLD_DAYS = 1              # 1-day hold; raw spec

# Train/Test split configuration
TRAIN_END_PCT = 0.4        # Fraction of data for training (0.4 = 40% train, 60% test)
                            # Common values: 0.5 (50/50), 0.6 (60/40), 0.7 (70/30)

# ----------------------------
# Hand-written mapping
# follower -> list of (leader, weight)
# Weights will be normalized per follower to sum to 1.
# ----------------------------

def get_leader_follower_mapping():
    """
    Returns dict: { follower: [(leader, weight), ...], ... }
    No SIC/FF49 used. Edit to your preference.
    """
    mapping = {
        # Semicap -> Chips
        "NVDA": [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],
        "AMD":  [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],
        "AVGO": [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],
        "MU":   [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],
        "TXN":  [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],
        "MRVL": [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],
        "ADI":  [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],

        # Memory -> Hardware OEMs
        "DELL": [("MU", 1), ("WDC", 1), ("STX", 1)],
        "HPQ":  [("MU", 1), ("WDC", 1), ("STX", 1)],
        "HPE":  [("MU", 1), ("WDC", 1), ("STX", 1)],
        "SMCI": [("MU", 1)],

        # Foundries -> Chip designers
        "QCOM": [("TSM", 1), ("GFS", 1), ("INTC", 1)],
        "MPWR": [("TSM", 1), ("GFS", 1)],
        "MCHP": [("TSM", 1), ("GFS", 1), ("INTC", 1)],
        "NXPI": [("TSM", 1), ("GFS", 1)],

        # Lithium/materials -> EV OEMs
        "TSLA": [("ALB", 1), ("SQM", 1), ("LTHM", 1)],
        "GM":   [("ALB", 1), ("SQM", 1)],
        "F":    [("ALB", 1), ("SQM", 1)],

        # Steel -> OEMs/Industrial consumers
        "CAT":  [("NUE", 1), ("STLD", 1), ("X", 1), ("CLF", 1)],
        "DE":   [("NUE", 1), ("STLD", 1), ("X", 1), ("CLF", 1)],
        "PCAR": [("NUE", 1), ("STLD", 1)],
        "GM":   [("NUE", 1), ("STLD", 1), ("X", 1)],

        # Refiners -> Airlines
        "DAL":  [("VLO", 1), ("MPC", 1), ("PSX", 1)],
        "UAL":  [("VLO", 1), ("MPC", 1), ("PSX", 1)],
        "AAL":  [("VLO", 1), ("MPC", 1), ("PSX", 1)],
        "LUV":  [("VLO", 1), ("MPC", 1), ("PSX", 1)],

        # EMS -> OEMs/components
        "AAPL": [("FLEX", 1), ("JBL", 1), ("SANM", 1)],
        "CSCO": [("FLEX", 1), ("JBL", 1)],
        "DELL": [("FLEX", 1), ("JBL", 1), ("SANM", 1)],

        # Big box retail -> Branded CPG
        "PG":   [("WMT", 1), ("COST", 1), ("TGT", 1)],
        "KMB":  [("WMT", 1), ("COST", 1), ("TGT", 1)],
        "CL":   [("WMT", 1), ("COST", 1), ("TGT", 1)],

        # Hyperscalers/infra -> Cloud software
        "SNOW": [("AMZN", 1), ("MSFT", 1), ("GOOGL", 1)],
        "MDB":  [("AMZN", 1), ("MSFT", 1), ("GOOGL", 1)],
        "DDOG": [("AMZN", 1), ("MSFT", 1), ("GOOGL", 1)],
        "CRWD": [("AMZN", 1), ("MSFT", 1), ("GOOGL", 1)],

        # Banks (mortgage proxies) -> Homebuilders
        "DHI":  [("WFC", 1), ("JPM", 1), ("BAC", 1)],
        "LEN":  [("WFC", 1), ("JPM", 1), ("BAC", 1)],
        "PHM":  [("WFC", 1), ("JPM", 1), ("BAC", 1)],

        # Homebuilders -> Building products
        "TREX": [("DHI", 1), ("LEN", 1), ("PHM", 1)],
        "AZEK": [("DHI", 1), ("LEN", 1), ("PHM", 1)],
        "OC":   [("DHI", 1), ("LEN", 1), ("PHM", 1)],

        # Airlines -> Travel platforms/hotels
        "BKNG": [("DAL", 1), ("UAL", 1), ("AAL", 1)],
        "EXPE": [("DAL", 1), ("UAL", 1), ("AAL", 1)],
        "ABNB": [("DAL", 1), ("UAL", 1), ("AAL", 1)],

        # PC core vendors -> PC ecosystem
        "LOGI": [("INTC", 1), ("AMD", 1), ("MSFT", 1)],
        "STX":  [("INTC", 1), ("AMD", 1), ("MSFT", 1)],
        "WDC":  [("INTC", 1), ("AMD", 1), ("MSFT", 1)],
    }

    # Normalize weights per follower
    normed = {}
    for fol, pairs in mapping.items():
        total = sum(w for _, w in pairs)
        if total == 0:
            continue
        normed[fol] = [(ldr, w / total) for ldr, w in pairs]
    return normed

# ----------------------------
# Data access
# ----------------------------

def tickers_in_mapping(mapping):
    leaders = {ldr for pairs in mapping.values() for (ldr, _) in pairs}
    followers = set(mapping.keys())
    return sorted(list(leaders.union(followers)))

def fetch_crsp_daily_from_wrds(tickers, start_date, end_date):
    """
    Pull CRSP daily for a set of tickers (using CRSP PERMNO join via CCM names).
    We use CRSP daily stock file (dsf) joined to stocknames (dsn) for tickers.
    """
    if wrds is None:
        raise RuntimeError("wrds package not available; cannot fetch from WRDS.")

    db = wrds.Connection()
    # Normalize dates for SQL
    sd = pd.to_datetime(start_date).date().isoformat()
    ed = pd.to_datetime(end_date).date().isoformat()



    # Query: map tickers to permno via stocknames table, then pull dsf
    # Join with dse to filter by share code and exchange code
    tickers_list = "', '".join([t.replace("'", "") for t in tickers])
    sql = f"""
        select
            d.date,
            n.ticker,
            d.permno,
            d.prc,
            d.ret,
            d.vol,
            d.shrout
        from crsp.dsf d
        inner join crsp.stocknames n 
            on d.permno = n.permno
            and d.date >= n.namedt
            and d.date <= n.nameenddt
        left join crsp.dse e on d.permno = e.permno and d.date = e.date
        where d.date between '{sd}' and '{ed}'
          and n.ticker in ('{tickers_list}')
          and (e.shrcd in (10,11) or e.shrcd is null)
          and (e.exchcd in (1,2,3) or e.exchcd is null)
        order by d.date, n.ticker
    """
    df = db.raw_sql(sql, date_cols=['date'])
    # Clean: CRSP returns are strings sometimes; convert, handle missing
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    df['prc'] = pd.to_numeric(df['prc'], errors='coerce')
    df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
    df['shrout'] = pd.to_numeric(df['shrout'], errors='coerce')
    # Adjust price to positive (CRSP stores negative for bid/ask)
    df['prc'] = df['prc'].abs()
    return df

# ----------------------------
# Strategy construction
# ----------------------------

def compute_adv_usd(df, window=20):
    """
    Compute 20-day ADV in USD per ticker: rolling mean of (PRC * VOL).
    """
    df = df.sort_values(['ticker', 'date']).copy()
    df['dollar_vol'] = df['prc'] * df['vol']
    df['adv_usd'] = df.groupby('ticker')['dollar_vol'].transform(lambda x: x.rolling(window, min_periods=window).mean())
    return df

def prepare_returns_panel(df):
    """
    Pivot returns by date x ticker; also returns price and ADV panels for filters.
    """
    # Remove duplicates: if same (date, ticker) appears multiple times, keep first
    # This can happen if a PERMNO had multiple tickers or if join created duplicates
    df = df.drop_duplicates(subset=['date', 'ticker'], keep='first')
    ret_p = df.pivot(index='date', columns='ticker', values='ret').sort_index()
    prc_p = df.pivot(index='date', columns='ticker', values='prc').sort_index()
    adv_p = df.pivot(index='date', columns='ticker', values='adv_usd').sort_index()
    return ret_p, prc_p, adv_p

def adjust_returns_for_rf(ret_p, rf_series):
    """
    Subtract risk-free rate from returns panel.
    Returns adjusted returns panel.
    """
    ret_excess = ret_p.copy()
    # Align rf_series with ret_p index
    rf_aligned = rf_series.reindex(ret_p.index, method='ffill').fillna(0.0)
    # Subtract risk-free rate from each column
    for col in ret_excess.columns:
        ret_excess[col] = ret_excess[col] - rf_aligned
    return ret_excess


def build_multi_lag_signal(ret_p, mapping, lags=(1,2,3), lag_decay=0.7, winsor=(0.01, 0.99)):
    """
    Build multi-lag signals with winsorization and exponential decay.
    For each follower i and day t: s_i,t = sum_k lambda^(k-1) * sum_j w_ij * r_leader_j,t-k
    Returns a DataFrame of signals aligned with ret_p index and followers as columns.
    """
    followers = sorted(mapping.keys())
    leaders_needed = sorted({ldr for pairs in mapping.values() for (ldr, _) in pairs})
    missing_leaders = [t for t in leaders_needed if t not in ret_p.columns]
    missing_followers = [t for t in followers if t not in ret_p.columns]
    if missing_leaders:
        print(f"Warning: leaders missing from price panel (no data): {missing_leaders}")
    if missing_followers:
        print(f"Warning: followers missing from price panel (no data): {missing_followers}")
    
    # Winsorize returns cross-sectionally per day (vectorized)
    ret_winsor = ret_p.copy()
    # Compute quantiles per row (date) efficiently
    q_low = ret_p.quantile(winsor[0], axis=1)
    q_high = ret_p.quantile(winsor[1], axis=1)
    # Clip each row using its quantiles
    for date in ret_p.index:
        if pd.notna(q_low[date]) and pd.notna(q_high[date]):
            ret_winsor.loc[date] = ret_p.loc[date].clip(lower=q_low[date], upper=q_high[date])
    
    # Build multi-lag signals with exponential decay
    sig = pd.DataFrame(index=ret_p.index, columns=followers, dtype=float)
    sig[:] = np.nan
    
    for fol, pairs in mapping.items():
        if fol not in ret_p.columns:
            continue
        weights = []
        cols = []
        for ldr, w in pairs:
            if ldr in ret_winsor.columns:
                cols.append(ldr)
                weights.append(w)
        if not cols:
            continue
        w_arr = np.array(weights, dtype=float)
        w_arr = w_arr / w_arr.sum()
        
        # Multi-lag aggregation with decay
        s_agg = pd.Series(0.0, index=ret_p.index)
        for k, lag in enumerate(lags, start=1):
            decay_weight = lag_decay ** (k - 1)
            ret_lag_k = ret_winsor[cols].shift(lag)
            s_k = ret_lag_k.dot(w_arr)
            s_agg = s_agg + decay_weight * s_k
        sig[fol] = s_agg
    return sig

def build_groups_from_mapping(mapping):
    """
    Construct follower clusters by connecting followers that share any leader.
    Returns dict: follower -> group_id
    """
    from collections import defaultdict
    
    # Build leader -> followers mapping
    leader_to_followers = defaultdict(set)
    for fol, pairs in mapping.items():
        for ldr, _ in pairs:
            leader_to_followers[ldr].add(fol)
    
    # Find connected components (followers that share leaders)
    visited = set()
    groups = {}
    group_id = 0
    
    def dfs(fol, current_group):
        if fol in visited:
            return
        visited.add(fol)
        groups[fol] = current_group
        # Find all followers connected through shared leaders
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
    """
    Mask signals where follower fails liquidity/price filters based on t-1 data.
    """
    prc_lag = prc_p.shift(1)
    adv_lag = adv_p.shift(1)
    mask = (prc_lag >= min_price) & (adv_lag >= min_adv_usd)
    sig_f = sig.where(mask, other=np.nan)
    return sig_f

def cross_sectional_standardize(sig, groups=None):
    """
    First demean sig within each group by date, then z-score across all followers per day.
    Returns standardized signal DataFrame. (Vectorized for performance)
    """
    sig_z = sig.copy()
    
    # Group-demean if groups provided (vectorized)
    if groups is not None:
        # Build group mapping for efficient lookup
        group_dict = {}
        for group_id in set(groups.values()):
            group_followers = [f for f, g in groups.items() if g == group_id and f in sig.columns]
            if len(group_followers) > 0:
                group_dict[group_id] = group_followers
        
        # Demean by group per date (vectorized)
        for group_id, group_followers in group_dict.items():
            group_sig = sig_z[group_followers]
            group_means = group_sig.mean(axis=1)
            sig_z[group_followers] = group_sig.subtract(group_means, axis=0)
    
    # Cross-sectional z-score per day (vectorized)
    row_means = sig_z.mean(axis=1)
    row_stds = sig_z.std(axis=1)
    # Avoid division by zero
    row_stds = row_stds.replace(0, np.nan)
    sig_z = sig_z.subtract(row_means, axis=0).divide(row_stds, axis=0)
    sig_z = sig_z.fillna(0.0)
    
    return sig_z

def form_cs_positions(sig_z, top_q=0.3, vol_target=0.10, ret_p=None, lookback_vol=20, max_scale=5.0):
    """
    Form cross-sectional positions using quantile-based long/short.
    Top quantile goes long, bottom quantile goes short, dollar-neutral.
    Optionally apply volatility targeting.
    Returns positions DataFrame.
    """
    sig_z = sig_z.apply(pd.to_numeric, errors="coerce")
    positions = pd.DataFrame(0.0, index=sig_z.index, columns=sig_z.columns, dtype=float)
    
    # Compute quantiles per row (vectorized)
    q_top = sig_z.quantile(1 - top_q, axis=1)
    q_bottom = sig_z.quantile(top_q, axis=1)
    
    # Apply weights
    for date in sig_z.index:
        row = sig_z.loc[date].dropna()
        if len(row) < 2:
            continue
        
        q_top_val = q_top[date]
        q_bottom_val = q_bottom[date]
        
        longs = row[row >= q_top_val].index
        shorts = row[row <= q_bottom_val].index
        
        if len(longs) > 0 and len(shorts) > 0:
            w_long = 0.5 / len(longs)
            w_short = -0.5 / len(shorts)
            positions.loc[date, longs] = w_long
            positions.loc[date, shorts] = w_short
    
    # Optional volatility targeting
    if vol_target is not None and ret_p is not None:
        # Compute trailing realized portfolio vol from unscaled positions
        pnl_gross = (positions.shift(1) * ret_p).sum(axis=1)
        realized_vol = pnl_gross.rolling(lookback_vol, min_periods=lookback_vol).std() * np.sqrt(252)
        
        # Scale positions forward: use yesterday's vol to scale today
        scale = vol_target / (realized_vol.shift(1) + 1e-6)
        scale = scale.clip(upper=max_scale)
        scale = scale.fillna(1.0)
        
        # Apply scaling
        positions = positions.multiply(scale, axis=0)
    
    return positions

def backtest(ret_p, positions, hold_days=1, cost_bps_per_side=5.0):
    """
    Simple backtest:
    - Enter positions at close of t based on signals computed with t-1 leader returns.
    - Realize returns over next day(s).
    - For hold_days=1, portfolio return on t+1 is sum_i w_i,t * r_i,t+1.
    - Costs: per-entry and per-exit cost applied when positions change.
      Here we approximate a daily turnover-based cost.

    Returns a Series of daily portfolio returns net of costs.
    """
    # Align
    positions = positions.reindex(ret_p.index).fillna(0.0)
    # Shift positions to represent positions held over next day(s)
    # We construct a simple 1-day hold; for >1, we average overlapping positions
    if hold_days == 1:
        pos_hold = positions
    else:
        # Build decay = equal weight over hold_days
        weights = np.ones(hold_days) / hold_days
        # rolling sum of lagged positions:
        # pos_hold[t] = average of positions from t, t-1, ..., t-hold_days+1
        pos_hold = pd.DataFrame(index=positions.index, columns=positions.columns, dtype=float)
        for i in range(hold_days):
            pos_hold = pos_hold.add(positions.shift(i) * weights[i], fill_value=0.0)

    # Daily portfolio returns before costs: use next-day returns
    # If signals formed at close t (using t-1 data), positions applied starting t+1 close-to-close
    pnl_gross = (pos_hold.shift(1) * ret_p).sum(axis=1)

    # Turnover and costs:
    # Approximate daily turnover as sum of |pos_t - pos_{t-1}| across names.
    pos_prev = pos_hold.shift(1).fillna(0.0)
    pos_curr = pos_hold.fillna(0.0)
    daily_turnover = (pos_curr - pos_prev).abs().sum(axis=1)
    # Cost per round-trip per unit weight ~ 2 * cost_bps_per_side; apply on turnover
    # Here, we apply cost = cost_bps_per_side/10000 per side for changes in weight
    per_side = cost_bps_per_side / 10000.0
    pnl_costs = daily_turnover * per_side
    pnl_net = pnl_gross - pnl_costs

    return pnl_net, pnl_gross, daily_turnover

def summarize_performance(r):
    """
    Compute simple performance stats.
    """
    r = r.dropna()
    if r.empty:
        return {}
    ann_factor = 252.0
    mu = r.mean() * ann_factor
    sd = r.std(ddof=0) * np.sqrt(ann_factor)
    sharpe = mu / sd if pd.notna(sd) and sd > 0 else np.nan
    cum = (1 + r).prod() - 1
    hit = (r > 0).mean()
    stats = {
        "Annualized Return": mu,
        "Annualized Vol": sd,
        "Sharpe": sharpe,
        "Cumulative Return": cum,
        "Daily Hit Rate": hit,
        "Average Daily Return (bps)": r.mean() * 1e4,
        "Average Daily Turnover": r.abs().mean(),  # not exact turnover; kept for brevity
        "N days": int(r.shape[0]),
    }
    return stats

def test_alpha(returns, alpha_name="Strategy"):
    """
    Test if alpha (mean excess return) is significantly greater than 0.
    Performs one-sided t-test: H0: alpha <= 0, H1: alpha > 0
    Returns dict with alpha, t-statistic, p-value, and significance.
    """
    from scipy import stats
    
    returns_clean = returns.dropna()
    if len(returns_clean) < 2:
        return None
    
    # Compute alpha (mean excess return)
    alpha = returns_clean.mean()
    
    # Standard error of the mean
    n = len(returns_clean)
    se = returns_clean.std(ddof=1) / np.sqrt(n)
    
    # t-statistic for one-sided test (H0: alpha <= 0, H1: alpha > 0)
    t_stat = alpha / se if se > 0 else np.nan
    
    # p-value for one-sided test (degrees of freedom = n-1)
    if pd.notna(t_stat):
        p_value = 1 - stats.t.cdf(t_stat, df=n-1)
    else:
        p_value = np.nan
    
    # Annualized alpha
    alpha_ann = alpha * 252.0
    
    result = {
        "Alpha (daily)": alpha,
        "Alpha (annualized)": alpha_ann,
        "t-statistic": t_stat,
        "p-value": p_value,
        "N observations": n,
        "Significant (5%)": p_value < 0.05 if pd.notna(p_value) else False,
        "Significant (1%)": p_value < 0.01 if pd.notna(p_value) else False,
    }
    
    return result

def main():
    mapping = get_leader_follower_mapping()
    all_tickers = tickers_in_mapping(mapping)
    # Add CRSP Large Cap Index ETFs for benchmark (VV tracks CRSP U.S. Large Cap Index)
    benchmark_candidates = ["VV", "VONE"]  # VV tracks CRSP U.S. Large Cap Index
    all_tickers.extend(benchmark_candidates)
    all_tickers = list(set(all_tickers))  # Remove duplicates
    print(f"Total unique tickers (leaders + followers): {len(all_tickers)}")
    print(", ".join(sorted(all_tickers)))

    # Fetch data from WRDS
    print("Fetching CRSP daily data from WRDS...")
    df = fetch_crsp_daily_from_wrds(all_tickers, START_DATE, END_DATE)

    # Compute ADV and build panels
    df = compute_adv_usd(df, window=20)
    ret_p, prc_p, adv_p = prepare_returns_panel(df)
    
    # Fetch and adjust returns for risk-free rate
    rf_series = fetch_risk_free_rate(START_DATE, END_DATE)
    
    # Print head of risk-free rate data for verification
    print("\nRisk-free rate data (first 10 rows):")
    print(rf_series.head(10))
    print(f"\nRisk-free rate stats:")
    print(f"  Date range: {rf_series.index.min()} to {rf_series.index.max()}")
    print(f"  Mean daily rate: {rf_series.mean():.6f} ({rf_series.mean()*252*100:.2f}% annualized)")
    print(f"  Min: {rf_series.min():.6f}, Max: {rf_series.max():.6f}")
    print()
    ret_p = adjust_returns_for_rf(ret_p, rf_series)
    
    
    # Train-test split (configurable via TRAIN_END_PCT)
    dates_sorted = sorted(ret_p.index)
    split_idx = int(len(dates_sorted) * TRAIN_END_PCT)
    train_end_date = dates_sorted[split_idx]
    
    train_pct = TRAIN_END_PCT * 100
    test_pct = (1 - TRAIN_END_PCT) * 100
    print(f"Train/Test split: {train_pct:.0f}% train / {test_pct:.0f}% test")
    
    # Split data
    ret_p_train = ret_p[ret_p.index <= train_end_date].copy()
    ret_p_test = ret_p[ret_p.index > train_end_date].copy()
    prc_p_train = prc_p[prc_p.index <= train_end_date].copy()
    prc_p_test = prc_p[prc_p.index > train_end_date].copy()
    adv_p_train = adv_p[adv_p.index <= train_end_date].copy()
    adv_p_test = adv_p[adv_p.index > train_end_date].copy()
    
    print(f"\nTrain period: {ret_p_train.index.min()} to {ret_p_train.index.max()} ({len(ret_p_train)} days)")
    print(f"Test period: {ret_p_test.index.min()} to {ret_p_test.index.max()} ({len(ret_p_test)} days)")
    print()

    # Build signals for train and test separately
    sig_raw_train = build_multi_lag_signal(ret_p_train, mapping, lags=(1,2,3), lag_decay=0.7, winsor=(0.01, 0.99))
    sig_raw_test = build_multi_lag_signal(ret_p_test, mapping, lags=(1,2,3), lag_decay=0.7, winsor=(0.01, 0.99))
    
    # Apply filters
    sig_masked_train = apply_filters(sig_raw_train, prc_p_train, adv_p_train, MIN_PRICE, MIN_ADV_USD)
    sig_masked_test = apply_filters(sig_raw_test, prc_p_test, adv_p_test, MIN_PRICE, MIN_ADV_USD)
    
    # Standardize
    groups = build_groups_from_mapping(mapping)
    sig_z_train = cross_sectional_standardize(sig_masked_train, groups)
    sig_z_test = cross_sectional_standardize(sig_masked_test, groups)
    
    # Form positions
    positions_train = form_cs_positions(sig_z_train, top_q=0.3, vol_target=0.10, ret_p=ret_p_train, lookback_vol=20)
    positions_test = form_cs_positions(sig_z_test, top_q=0.3, vol_target=0.10, ret_p=ret_p_test, lookback_vol=20)

    
    # Backtest train set
    print("="*80)
    print("IN-SAMPLE (TRAIN) RESULTS")
    print("="*80)
    pnl_net_train, pnl_gross_train, turnover_train = backtest(
        ret_p_train, positions_train, hold_days=HOLD_DAYS, cost_bps_per_side=TRANSACTION_COST_BPS_PER_SIDE
    )
    
    # Backtest test set
    print("="*80)
    print("OUT-OF-SAMPLE (TEST) RESULTS")
    print("="*80)
    pnl_net_test, pnl_gross_test, turnover_test = backtest(
        ret_p_test, positions_test, hold_days=HOLD_DAYS, cost_bps_per_side=TRANSACTION_COST_BPS_PER_SIDE
    )

    
    # Summaries for train set
    stats_net_train = summarize_performance(pnl_net_train)
    stats_gross_train = summarize_performance(pnl_gross_train)
    print("\nPerformance (Net of costs) - TRAIN:")
    for k, v in stats_net_train.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\nPerformance (Gross, before costs) - TRAIN:")
    for k, v in stats_gross_train.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    
    # Alpha tests for train set
    print("\nAlpha Test (Net of costs) - TRAIN:")
    alpha_test_net_train = test_alpha(pnl_net_train, "Strategy (Net) - Train")
    if alpha_test_net_train:
        print(f"  Alpha (annualized): {alpha_test_net_train['Alpha (annualized)']:.4f}")
        print(f"  t-statistic: {alpha_test_net_train['t-statistic']:.4f}")
        print(f"  p-value: {alpha_test_net_train['p-value']:.4f}")
        print(f"  Significant at 5%: {alpha_test_net_train['Significant (5%)']}")
        print(f"  Significant at 1%: {alpha_test_net_train['Significant (1%)']}")
    
    print("\nAlpha Test (Gross, before costs) - TRAIN:")
    alpha_test_gross_train = test_alpha(pnl_gross_train, "Strategy (Gross) - Train")
    if alpha_test_gross_train:
        print(f"  Alpha (annualized): {alpha_test_gross_train['Alpha (annualized)']:.4f}")
        print(f"  t-statistic: {alpha_test_gross_train['t-statistic']:.4f}")
        print(f"  p-value: {alpha_test_gross_train['p-value']:.4f}")
        print(f"  Significant at 5%: {alpha_test_gross_train['Significant (5%)']}")
        print(f"  Significant at 1%: {alpha_test_gross_train['Significant (1%)']}")

    
    # Summaries for test set
    stats_net_test = summarize_performance(pnl_net_test)
    stats_gross_test = summarize_performance(pnl_gross_test)
    print("\nPerformance (Net of costs) - TEST:")
    for k, v in stats_net_test.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    
    print("\nPerformance (Gross, before costs) - TEST:")
    for k, v in stats_gross_test.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    
    # Alpha tests for test set
    print("\nAlpha Test (Net of costs) - TEST:")
    alpha_test_net_test = test_alpha(pnl_net_test, "Strategy (Net) - Test")
    if alpha_test_net_test:
        print(f"  Alpha (annualized): {alpha_test_net_test['Alpha (annualized)']:.4f}")
        print(f"  t-statistic: {alpha_test_net_test['t-statistic']:.4f}")
        print(f"  p-value: {alpha_test_net_test['p-value']:.4f}")
        print(f"  Significant at 5%: {alpha_test_net_test['Significant (5%)']}")
        print(f"  Significant at 1%: {alpha_test_net_test['Significant (1%)']}")
    
    print("\nAlpha Test (Gross, before costs) - TEST:")
    alpha_test_gross_test = test_alpha(pnl_gross_test, "Strategy (Gross) - Test")
    if alpha_test_gross_test:
        print(f"  Alpha (annualized): {alpha_test_gross_test['Alpha (annualized)']:.4f}")
        print(f"  t-statistic: {alpha_test_gross_test['t-statistic']:.4f}")
        print(f"  p-value: {alpha_test_gross_test['p-value']:.4f}")
        print(f"  Significant at 5%: {alpha_test_gross_test['Significant (5%)']}")
        print(f"  Significant at 1%: {alpha_test_gross_test['Significant (1%)']}")

    # Benchmark comparison for TRAIN set

    # Compute benchmark (SPY excess returns)
    # Find which benchmark ETF exists in the data
    benchmark_ticker = None
    for candidate in benchmark_candidates:
        if candidate in ret_p.columns:
            benchmark_ticker = candidate
            print(f"Using {candidate} as market benchmark (VV tracks CRSP U.S. Large Cap Index)")
            break
    
    if benchmark_ticker is not None and benchmark_ticker in ret_p.columns:
        benchmark_returns_train = ret_p_train[benchmark_ticker].dropna()
        benchmark_cumulative_train = (1 + benchmark_returns_train).cumprod() - 1
        
        # Strategy cumulative returns (net and gross)
        strat_net_cum_train = (1 + pnl_net_train).cumprod() - 1
        strat_gross_cum_train = (1 + pnl_gross_train).cumprod() - 1
        
        # Align dates
        common_dates_train = strat_net_cum_train.index.intersection(benchmark_cumulative_train.index)
        if len(common_dates_train) > 0:
            strat_net_cum_train = strat_net_cum_train.loc[common_dates_train]
            strat_gross_cum_train = strat_gross_cum_train.loc[common_dates_train]
            benchmark_cumulative_train = benchmark_cumulative_train.loc[common_dates_train]
            
            # Plot cumulative returns
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(strat_net_cum_train.index, strat_net_cum_train.values * 100, label="Strategy (Net)", linewidth=2)
            plt.plot(strat_gross_cum_train.index, strat_gross_cum_train.values * 100, label="Strategy (Gross)", linewidth=2)
            benchmark_label = f"{benchmark_ticker} (Excess)" if benchmark_ticker else "Market (Excess)"
            plt.plot(benchmark_cumulative_train.index, benchmark_cumulative_train.values * 100, label=benchmark_label, linewidth=2, linestyle="--")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return (%)")
            plt.title("Cumulative Returns: Strategy vs Benchmark (TRAIN) (Excess Returns)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_dir = "./output"
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, "cumulative_returns_comparison_train.png"), dpi=150)
            print(f"Saved cumulative returns plot to {out_dir}/cumulative_returns_comparison_train.png")
            
            # Sharpe ratio comparison
            ann_factor = 252.0
            strat_net_sharpe_train = (pnl_net_train.loc[common_dates_train].mean() * ann_factor) / (pnl_net_train.loc[common_dates_train].std() * np.sqrt(ann_factor)) if pnl_net_train.loc[common_dates_train].std() > 0 else np.nan
            strat_gross_sharpe_train = (pnl_gross_train.loc[common_dates_train].mean() * ann_factor) / (pnl_gross_train.loc[common_dates_train].std() * np.sqrt(ann_factor)) if pnl_gross_train.loc[common_dates_train].std() > 0 else np.nan
            benchmark_sharpe_train = (benchmark_returns_train.loc[common_dates_train].mean() * ann_factor) / (benchmark_returns_train.loc[common_dates_train].std() * np.sqrt(ann_factor)) if benchmark_returns_train.loc[common_dates_train].std() > 0 else np.nan
            
            print("\nSharpe Ratio Comparison:")
            print(f"  Strategy (Net):  {strat_net_sharpe_train:.4f}")
            print(f"  Strategy (Gross): {strat_gross_sharpe_train:.4f}")
            print(f"  Benchmark (Excess):    {benchmark_sharpe_train:.4f}")
    
    # Benchmark comparison for TEST set

    # Compute benchmark (SPY excess returns)
    # Find which benchmark ETF exists in the data
    benchmark_ticker = None
    for candidate in benchmark_candidates:
        if candidate in ret_p.columns:
            benchmark_ticker = candidate
            print(f"Using {candidate} as market benchmark (VV tracks CRSP U.S. Large Cap Index)")
            break
    
    if benchmark_ticker is not None and benchmark_ticker in ret_p.columns:
        benchmark_returns_test = ret_p_test[benchmark_ticker].dropna()
        benchmark_cumulative_test = (1 + benchmark_returns_test).cumprod() - 1
        
        # Strategy cumulative returns (net and gross)
        strat_net_cum_test = (1 + pnl_net_test).cumprod() - 1
        strat_gross_cum_test = (1 + pnl_gross_test).cumprod() - 1
        
        # Align dates
        common_dates_test = strat_net_cum_test.index.intersection(benchmark_cumulative_test.index)
        if len(common_dates_test) > 0:
            strat_net_cum_test = strat_net_cum_test.loc[common_dates_test]
            strat_gross_cum_test = strat_gross_cum_test.loc[common_dates_test]
            benchmark_cumulative_test = benchmark_cumulative_test.loc[common_dates_test]
            
            # Plot cumulative returns
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(strat_net_cum_test.index, strat_net_cum_test.values * 100, label="Strategy (Net)", linewidth=2)
            plt.plot(strat_gross_cum_test.index, strat_gross_cum_test.values * 100, label="Strategy (Gross)", linewidth=2)
            benchmark_label = f"{benchmark_ticker} (Excess)" if benchmark_ticker else "Market (Excess)"
            plt.plot(benchmark_cumulative_test.index, benchmark_cumulative_test.values * 100, label=benchmark_label, linewidth=2, linestyle="--")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return (%)")
            plt.title("Cumulative Returns: Strategy vs Benchmark (TEST) (Excess Returns)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_dir = "./output"
            os.makedirs(out_dir, exist_ok=True)
            plt.savefig(os.path.join(out_dir, "cumulative_returns_comparison_test.png"), dpi=150)
            print(f"Saved cumulative returns plot to {out_dir}/cumulative_returns_comparison_test.png")
            
            # Sharpe ratio comparison
            ann_factor = 252.0
            strat_net_sharpe_test = (pnl_net_test.loc[common_dates_test].mean() * ann_factor) / (pnl_net_test.loc[common_dates_test].std() * np.sqrt(ann_factor)) if pnl_net_test.loc[common_dates_test].std() > 0 else np.nan
            strat_gross_sharpe_test = (pnl_gross_test.loc[common_dates_test].mean() * ann_factor) / (pnl_gross_test.loc[common_dates_test].std() * np.sqrt(ann_factor)) if pnl_gross_test.loc[common_dates_test].std() > 0 else np.nan
            benchmark_sharpe_test = (benchmark_returns_test.loc[common_dates_test].mean() * ann_factor) / (benchmark_returns_test.loc[common_dates_test].std() * np.sqrt(ann_factor)) if benchmark_returns_test.loc[common_dates_test].std() > 0 else np.nan
            
            print("\nSharpe Ratio Comparison:")
            print(f"  Strategy (Net):  {strat_net_sharpe_test:.4f}")
            print(f"  Strategy (Gross): {strat_gross_sharpe_test:.4f}")
            print(f"  Benchmark (Excess):    {benchmark_sharpe_test:.4f}")
    
    
    # Save train results
    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)
    pnl_df_train = pd.DataFrame({
        "pnl_net": pnl_net_train,
        "pnl_gross": pnl_gross_train,
        "turnover": turnover_train
    })
    pnl_df_train.to_csv(os.path.join(out_dir, "daily_pnl_train.csv"))
    positions_train.to_parquet(os.path.join(out_dir, "positions_train.parquet"))
    sig_z_train.to_parquet(os.path.join(out_dir, "signals_train.parquet"))
    
    # Save test results
    pnl_df_test = pd.DataFrame({
        "pnl_net": pnl_net_test,
        "pnl_gross": pnl_gross_test,
        "turnover": turnover_test
    })
    pnl_df_test.to_csv(os.path.join(out_dir, "daily_pnl_test.csv"))
    positions_test.to_parquet(os.path.join(out_dir, "positions_test.parquet"))
    sig_z_test.to_parquet(os.path.join(out_dir, "signals_test.parquet"))
    print(f"\nSaved train outputs to {out_dir}/daily_pnl_train.csv and Parquet files.")
    print(f"Saved test outputs to {out_dir}/daily_pnl_test.csv and Parquet files.")

if __name__ == "__main__":
    main()
