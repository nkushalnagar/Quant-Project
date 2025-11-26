"""WRDS data fetching and preprocessing."""

import pandas as pd
import numpy as np

try:
    import wrds
except ImportError:
    wrds = None
    print("Warning: wrds package not found. Install with: pip install wrds")


def fetch_risk_free_rate(start_date, end_date):
    """Fetch daily risk-free rate from WRDS Fama-French factors."""
    if wrds is None:
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
        rf_series = df.set_index('date')['rf'] / 100.0
        return rf_series
    except Exception as e:
        print(f"Warning: Could not fetch risk-free rate: {e}")
        print("Using constant 2% annualized rate")
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        return pd.Series(0.000079, index=dates, name='rf')


def tickers_in_mapping(mapping):
    """Extract all unique tickers from mapping."""
    leaders = {ldr for pairs in mapping.values() for (ldr, _) in pairs}
    followers = set(mapping.keys())
    return sorted(list(leaders.union(followers)))


def fetch_crsp_daily_from_wrds(tickers, start_date, end_date):
    """Pull CRSP daily data for tickers."""
    if wrds is None:
        raise RuntimeError("wrds package not available")

    db = wrds.Connection()
    sd = pd.to_datetime(start_date).date().isoformat()
    ed = pd.to_datetime(end_date).date().isoformat()

    tickers_list = "', '".join([t.replace("'", "") for t in tickers])
    sql = f"""
        select
            d.date, n.ticker, d.permno, d.prc, d.ret, d.vol, d.shrout
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
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce')
    df['prc'] = pd.to_numeric(df['prc'], errors='coerce').abs()
    df['vol'] = pd.to_numeric(df['vol'], errors='coerce')
    df['shrout'] = pd.to_numeric(df['shrout'], errors='coerce')
    return df


def compute_adv_usd(df, window=20):
    """Compute 20-day ADV in USD."""
    df = df.sort_values(['ticker', 'date']).copy()
    df['dollar_vol'] = df['prc'] * df['vol']
    df['adv_usd'] = df.groupby('ticker')['dollar_vol'].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    return df


def prepare_returns_panel(df):
    """Pivot returns by date x ticker."""
    df = df.drop_duplicates(subset=['date', 'ticker'], keep='first')
    ret_p = df.pivot(index='date', columns='ticker', values='ret').sort_index()
    prc_p = df.pivot(index='date', columns='ticker', values='prc').sort_index()
    adv_p = df.pivot(index='date', columns='ticker', values='adv_usd').sort_index()
    return ret_p, prc_p, adv_p


def adjust_returns_for_rf(ret_p, rf_series):
    """Subtract risk-free rate from returns."""
    ret_excess = ret_p.copy()
    rf_aligned = rf_series.reindex(ret_p.index, method='ffill').fillna(0.0)
    for col in ret_excess.columns:
        ret_excess[col] = ret_excess[col] - rf_aligned
    return ret_excess
