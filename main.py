#!/usr/bin/env python3
"""
Lead-lag trading strategy - Main entry point.
Runs the full backtest pipeline with train/test split.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import from our modules
from config import *
from data_fetcher import *
from signal_generation import *
from portfolio_optimization import *
from backtesting import *
from analysis import *


def main():
    # Get mapping and tickers
    mapping = get_leader_follower_mapping()
    all_tickers = tickers_in_mapping(mapping)
    all_tickers.extend(BENCHMARK_CANDIDATES)
    all_tickers = list(set(all_tickers))
    print(f"Total unique tickers: {len(all_tickers)}")
    print(", ".join(sorted(all_tickers)))

    # Fetch data from WRDS
    print("\nFetching CRSP daily data from WRDS...")
    df = fetch_crsp_daily_from_wrds(all_tickers, START_DATE, END_DATE)

    # Compute ADV and build panels
    df = compute_adv_usd(df, window=20)
    ret_p, prc_p, adv_p = prepare_returns_panel(df)
    
    # Fetch and adjust returns for risk-free rate
    rf_series = fetch_risk_free_rate(START_DATE, END_DATE)
    print("\nRisk-free rate data (first 10 rows):")
    print(rf_series.head(10))
    print(f"\nRisk-free rate stats:")
    print(f"  Date range: {rf_series.index.min()} to {rf_series.index.max()}")
    print(f"  Mean daily rate: {rf_series.mean():.6f} ({rf_series.mean()*252*100:.2f}% annualized)")
    print(f"  Min: {rf_series.min():.6f}, Max: {rf_series.max():.6f}\n")
    ret_p = adjust_returns_for_rf(ret_p, rf_series)
    
    # Convert to weekly cumulative returns
    print("\nConverting daily returns to weekly cumulative returns...")
    ret_weekly = convert_to_weekly_returns(ret_p)
    prc_weekly = convert_to_weekly_returns(prc_p)  # Use last price of week
    adv_weekly = convert_to_weekly_returns(adv_p)  # Average ADV for week
    
    print(f"Daily data: {len(ret_p)} days")
    print(f"Weekly data: {len(ret_weekly)} weeks")
    print(f"Date range: {ret_weekly.index.min()} to {ret_weekly.index.max()}")
    
    # Random week split (instead of chronological)
    ret_p_train, ret_p_test = random_week_split(ret_weekly, train_pct=TRAIN_END_PCT, random_seed=42)
    prc_p_train, prc_p_test = random_week_split(prc_weekly, train_pct=TRAIN_END_PCT, random_seed=42)
    adv_p_train, adv_p_test = random_week_split(adv_weekly, train_pct=TRAIN_END_PCT, random_seed=42)

    # Build signals (weekly: lag=1 means previous week's leader returns predict this week's follower)
    print("\nBuilding weekly lead-lag signals...")
    sig_raw_train = build_multi_lag_signal(ret_p_train, mapping, lags=(1,), lag_decay=1.0, winsor=(0.01, 0.99))
    sig_raw_test = build_multi_lag_signal(ret_p_test, mapping, lags=(1,), lag_decay=1.0, winsor=(0.01, 0.99))
    
    # Apply filters
    sig_masked_train = apply_filters(sig_raw_train, prc_p_train, adv_p_train, MIN_PRICE, MIN_ADV_USD)
    sig_masked_test = apply_filters(sig_raw_test, prc_p_test, adv_p_test, MIN_PRICE, MIN_ADV_USD)
    
    # Standardize
    groups = build_groups_from_mapping(mapping)
    sig_z_train = cross_sectional_standardize(sig_masked_train, groups)
    sig_z_test = cross_sectional_standardize(sig_masked_test, groups)
    
    # Form positions with mean-variance optimization
    print("\n" + "="*80)
    print("IN-SAMPLE OPTIMIZATION (TRAINING PERIOD)")
    print("="*80)
    
    best_risk_aversion = 2.0
    print(f"\nUsing risk_aversion = {best_risk_aversion} for mean-variance optimization")
    
    positions_train = form_mv_positions(
        sig_z_train, ret_p_train,
        top_q=0.3,
        risk_aversion=best_risk_aversion,
        lookback_cov=60,
        cov_shrinkage=0.1,
        vol_target=0.10,
        max_weight=0.05,
        min_weight=0.01,
        lookback_vol=20,
        max_scale=5.0
    )
    
    # Add leader hedges if enabled
    if USE_LEADER_HEDGE:
        print("\nAdding leader hedges (pairs trade strategy)...")
        positions_train = add_leader_hedges(positions_train, mapping)
        print(f"Positions expanded from {len(sig_z_train.columns)} to {len(positions_train.columns)} tickers")
    
    print("\n" + "="*80)
    print("OUT-OF-SAMPLE TESTING (TEST PERIOD)")
    print("="*80)
    print(f"Using risk_aversion={best_risk_aversion} (fixed parameter)")
    
    positions_test = form_mv_positions(
        sig_z_test, ret_p_test,
        top_q=0.3,
        risk_aversion=best_risk_aversion,
        lookback_cov=60,
        cov_shrinkage=0.1,
        vol_target=0.10,
        max_weight=0.05,
        min_weight=0.01,
        lookback_vol=20,
        max_scale=5.0
    )
    
    # Add leader hedges if enabled
    if USE_LEADER_HEDGE:
        print("\nAdding leader hedges (pairs trade strategy)...")
        positions_test = add_leader_hedges(positions_test, mapping)
        print(f"Positions expanded from {len(sig_z_test.columns)} to {len(positions_test.columns)} tickers")

    # Backtest
    print("\n" + "="*80)
    print("IN-SAMPLE (TRAIN) RESULTS")
    print("="*80)
    pnl_net_train, pnl_gross_train, turnover_train = backtest(
        ret_p_train, positions_train, hold_days=HOLD_DAYS, cost_bps_per_side=TRANSACTION_COST_BPS_PER_SIDE
    )
    
    print("="*80)
    print("OUT-OF-SAMPLE (TEST) RESULTS")
    print("="*80)
    pnl_net_test, pnl_gross_test, turnover_test = backtest(
        ret_p_test, positions_test, hold_days=HOLD_DAYS, cost_bps_per_side=TRANSACTION_COST_BPS_PER_SIDE
    )

    # Performance summaries (weekly frequency)
    stats_net_train = summarize_performance(pnl_net_train, frequency='weekly')
    stats_gross_train = summarize_performance(pnl_gross_train, frequency='weekly')
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

    # Alpha tests (weekly frequency)
    print("\nAlpha Test (Net of costs) - TRAIN:")
    alpha_test_net_train = test_alpha(pnl_net_train, "Strategy (Net) - Train", frequency='weekly')
    if alpha_test_net_train:
        print(f"  Alpha (annualized): {alpha_test_net_train['Alpha (annualized)']:.4f}")
        print(f"  t-statistic: {alpha_test_net_train['t-statistic']:.4f}")
        print(f"  p-value: {alpha_test_net_train['p-value']:.4f}")
        print(f"  Significant at 5%: {alpha_test_net_train['Significant (5%)']}")
        print(f"  Significant at 1%: {alpha_test_net_train['Significant (1%)']}")
    
    print("\nAlpha Test (Gross, before costs) - TRAIN:")
    alpha_test_gross_train = test_alpha(pnl_gross_train, "Strategy (Gross) - Train", frequency='weekly')
    if alpha_test_gross_train:
        print(f"  Alpha (annualized): {alpha_test_gross_train['Alpha (annualized)']:.4f}")
        print(f"  t-statistic: {alpha_test_gross_train['t-statistic']:.4f}")
        print(f"  p-value: {alpha_test_gross_train['p-value']:.4f}")
        print(f"  Significant at 5%: {alpha_test_gross_train['Significant (5%)']}")
        print(f"  Significant at 1%: {alpha_test_gross_train['Significant (1%)']}")

    # Test set summaries (weekly frequency)
    stats_net_test = summarize_performance(pnl_net_test, frequency='weekly')
    stats_gross_test = summarize_performance(pnl_gross_test, frequency='weekly')
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

    # Alpha tests for test set (weekly frequency)
    print("\nAlpha Test (Net of costs) - TEST:")
    alpha_test_net_test = test_alpha(pnl_net_test, "Strategy (Net) - Test", frequency='weekly')
    if alpha_test_net_test:
        print(f"  Alpha (annualized): {alpha_test_net_test['Alpha (annualized)']:.4f}")
        print(f"  t-statistic: {alpha_test_net_test['t-statistic']:.4f}")
        print(f"  p-value: {alpha_test_net_test['p-value']:.4f}")
        print(f"  Significant at 5%: {alpha_test_net_test['Significant (5%)']}")
        print(f"  Significant at 1%: {alpha_test_net_test['Significant (1%)']}")
    
    print("\nAlpha Test (Gross, before costs) - TEST:")
    alpha_test_gross_test = test_alpha(pnl_gross_test, "Strategy (Gross) - Test", frequency='weekly')
    if alpha_test_gross_test:
        print(f"  Alpha (annualized): {alpha_test_gross_test['Alpha (annualized)']:.4f}")
        print(f"  t-statistic: {alpha_test_gross_test['t-statistic']:.4f}")
        print(f"  p-value: {alpha_test_gross_test['p-value']:.4f}")
        print(f"  Significant at 5%: {alpha_test_gross_test['Significant (5%)']}")
        print(f"  Significant at 1%: {alpha_test_gross_test['Significant (1%)']}")

    # Benchmark comparison
    benchmark_ticker = None
    for candidate in BENCHMARK_CANDIDATES:
        if candidate in ret_p.columns:
            benchmark_ticker = candidate
            print(f"\nUsing {candidate} as market benchmark")
            break
    
    if benchmark_ticker and benchmark_ticker in ret_p.columns:
        # Train benchmark
        benchmark_returns_train = ret_p_train[benchmark_ticker].dropna()
        benchmark_cumulative_train = (1 + benchmark_returns_train).cumprod() - 1
        strat_net_cum_train = (1 + pnl_net_train).cumprod() - 1
        strat_gross_cum_train = (1 + pnl_gross_train).cumprod() - 1
        
        common_dates_train = strat_net_cum_train.index.intersection(benchmark_cumulative_train.index)
        if len(common_dates_train) > 0:
            strat_net_cum_train = strat_net_cum_train.loc[common_dates_train]
            strat_gross_cum_train = strat_gross_cum_train.loc[common_dates_train]
            benchmark_cumulative_train = benchmark_cumulative_train.loc[common_dates_train]
            
            plt.figure(figsize=(12, 6))
            plt.plot(strat_net_cum_train.index, strat_net_cum_train.values * 100, label="Strategy (Net)", linewidth=2)
            plt.plot(strat_gross_cum_train.index, strat_gross_cum_train.values * 100, label="Strategy (Gross)", linewidth=2)
            plt.plot(benchmark_cumulative_train.index, benchmark_cumulative_train.values * 100, 
                    label=f"{benchmark_ticker} (Excess)", linewidth=2, linestyle="--")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return (%)")
            plt.title("Cumulative Returns: Strategy vs Benchmark (TRAIN)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            plt.savefig(os.path.join(OUTPUT_DIR, "cumulative_returns_comparison_train.png"), dpi=150)
            print(f"\nSaved train plot to {OUTPUT_DIR}/cumulative_returns_comparison_train.png")
        
        # Test benchmark
        benchmark_returns_test = ret_p_test[benchmark_ticker].dropna()
        benchmark_cumulative_test = (1 + benchmark_returns_test).cumprod() - 1
        strat_net_cum_test = (1 + pnl_net_test).cumprod() - 1
        strat_gross_cum_test = (1 + pnl_gross_test).cumprod() - 1
        
        common_dates_test = strat_net_cum_test.index.intersection(benchmark_cumulative_test.index)
        if len(common_dates_test) > 0:
            strat_net_cum_test = strat_net_cum_test.loc[common_dates_test]
            strat_gross_cum_test = strat_gross_cum_test.loc[common_dates_test]
            benchmark_cumulative_test = benchmark_cumulative_test.loc[common_dates_test]
            
            plt.figure(figsize=(12, 6))
            plt.plot(strat_net_cum_test.index, strat_net_cum_test.values * 100, label="Strategy (Net)", linewidth=2)
            plt.plot(strat_gross_cum_test.index, strat_gross_cum_test.values * 100, label="Strategy (Gross)", linewidth=2)
            plt.plot(benchmark_cumulative_test.index, benchmark_cumulative_test.values * 100,
                    label=f"{benchmark_ticker} (Excess)", linewidth=2, linestyle="--")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Return (%)")
            plt.title("Cumulative Returns: Strategy vs Benchmark (TEST)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "cumulative_returns_comparison_test.png"), dpi=150)
            print(f"Saved test plot to {OUTPUT_DIR}/cumulative_returns_comparison_test.png")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pnl_df_train = pd.DataFrame({
        "pnl_net": pnl_net_train,
        "pnl_gross": pnl_gross_train,
        "turnover": turnover_train
    })
    pnl_df_train.to_csv(os.path.join(OUTPUT_DIR, "daily_pnl_train.csv"))
    positions_train.to_parquet(os.path.join(OUTPUT_DIR, "positions_train.parquet"))
    sig_z_train.to_parquet(os.path.join(OUTPUT_DIR, "signals_train.parquet"))
    
    pnl_df_test = pd.DataFrame({
        "pnl_net": pnl_net_test,
        "pnl_gross": pnl_gross_test,
        "turnover": turnover_test
    })
    pnl_df_test.to_csv(os.path.join(OUTPUT_DIR, "daily_pnl_test.csv"))
    positions_test.to_parquet(os.path.join(OUTPUT_DIR, "positions_test.parquet"))
    sig_z_test.to_parquet(os.path.join(OUTPUT_DIR, "signals_test.parquet"))
    
    strategy_type = "Pairs Trade (with leader hedges)" if USE_LEADER_HEDGE else "Long/Short Followers Only"
    print(f"\n{'='*80}")
    print(f"Strategy: {strategy_type}")
    print(f"All results saved to: {OUTPUT_DIR}/")
    print("="*80)


if __name__ == "__main__":
    main()
