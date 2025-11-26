# Lead-Lag Pairs Trading Strategy

A quantitative trading strategy that exploits lead-lag relationships in supply chain networks by pairing followers with their leaders in a market-neutral pairs trade.

## 🎯 Strategy Overview

### Core Concept
The strategy identifies **leader-follower relationships** in supply chains and trades on the hypothesis that:
1. **Leaders move first** (e.g., semiconductor equipment makers like ASML)
2. **Followers react later** (e.g., chip designers like NVDA)
3. By pairing long follower positions with short leader positions, we isolate the **relative momentum transfer**

### Example Trade
```
Week x-1: ASML (leader) returns +5%
Week x:   
  - Long NVDA (follower) - expecting it to rally on chip demand signal
  - Short ASML (leader) - hedge out general sector movement
  - Profit if NVDA outperforms ASML (catches up or overshoots)
```

## 📊 Performance (Test Set)

| Metric | Value |
|--------|-------|
| **Annualized Return** | 2.27% |
| **Sharpe Ratio** | 0.36 |
| **Cumulative Return** | 43.85% |
| **Weekly Hit Rate** | 50.27% |
| **Annualized Volatility** | 6.27% |
| **p-value** | 0.065 (nearly significant at 5%) |

*Test period: 913 weeks (70% of data), randomly selected*

## 🏗️ Architecture

### Modular Structure
```
quant project/
├── config.py                  # Parameters and leader-follower mapping
├── data_fetcher.py            # WRDS CRSP data access
├── signal_generation.py       # Weekly signal construction
├── portfolio_optimization.py  # Mean-variance optimization + hedging
├── backtesting.py             # Performance metrics and backtesting
├── analysis.py                # Robustness checks (optional)
├── main.py                    # Main execution pipeline
└── output/                    # Results (gitignored)
```

## 🔧 How It Works

### 1. Data Collection
- **Source**: WRDS CRSP daily stock data (2000-2025)
- **Universe**: 77 stocks across supply chain relationships
- **Frequency**: Daily data aggregated to weekly cumulative returns

### 2. Signal Generation

**Weekly Cumulative Returns:**
```python
# Convert daily to weekly
weekly_return = (1 + r_day1) × (1 + r_day2) × ... × (1 + r_dayN) - 1
```

**Lead-Lag Signal:**
```python
signal[follower, week_x] = weighted_avg(leader_returns[week_x-1])
```

For NVDA with leaders [ASML, AMAT, LRCX, KLAC]:
```
signal_NVDA[week_10] = 0.25×ASML[week_9] + 0.25×AMAT[week_9] + 
                        0.25×LRCX[week_9] + 0.25×KLAC[week_9]
```

**Signal Processing:**
- Cross-sectional winsorization (1st-99th percentile)
- Group demeaning (within supply chain clusters)
- Z-score standardization

### 3. Portfolio Construction

**Mean-Variance Optimization:**
- Select top 30% positive signals (long) and bottom 30% (short)
- Optimize weights to maximize: `return - (risk_aversion/2) × variance`
- Constraints:
  - Dollar neutral: `Σ weights = 0`
  - Position limits: 1% ≤ |weight| ≤ 5%
  - Long positions: weight > 0
  - Short positions: weight < 0

**Leader Hedging (Pairs Trade):**
```python
# For each follower position, add offsetting leader positions
if long NVDA with weight +0.05:
    short ASML with weight -0.0125  (0.05 × 0.25)
    short AMAT with weight -0.0125
    short LRCX with weight -0.0125
    short KLAC with weight -0.0125
```

**Volatility Targeting:**
- Target: 10% annualized volatility
- Scale positions based on realized 20-week volatility
- Max scaling: 5x

### 4. Backtesting

**Train/Test Split:**
- **Random week selection** (not chronological)
- 30% train (391 weeks), 70% test (913 weeks)
- Seed: 42 (reproducible)

**Execution:**
- Hold period: 1 week
- Transaction costs: 0 bps (can be adjusted)
- Rebalance: Weekly

## 📈 Leader-Follower Relationships

### Supply Chain Mapping

**Semiconductor Equipment → Chip Designers:**
- ASML, AMAT, LRCX, KLAC → NVDA, AMD, AVGO, MU, TXN

**Foundries → Chip Designers:**
- TSM, GFS, INTC → QCOM, MPWR, MCHP, NXPI

**Memory → Hardware OEMs:**
- MU, WDC, STX → DELL, HPQ, HPE, SMCI

**Lithium/Materials → EV OEMs:**
- ALB, SQM, LTHM → TSLA, GM, F

**Steel → Industrials:**
- NUE, STLD, X, CLF → CAT, DE, PCAR

**Refiners → Airlines:**
- VLO, MPC, PSX → DAL, UAL, AAL, LUV

**Hyperscalers → Cloud Software:**
- AMZN, MSFT, GOOGL → SNOW, MDB, DDOG, CRWD

*Full mapping: 46 followers, 29 leaders, 75 total tickers*

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scipy wrds matplotlib tqdm pyarrow
```

### WRDS Access
You need a WRDS account with access to:
- CRSP Daily Stock File
- Fama-French Factors

### Run the Strategy
```bash
python3 main.py
```

The script will:
1. Fetch data from WRDS (requires login)
2. Convert to weekly returns
3. Build lead-lag signals
4. Optimize portfolios (train & test)
5. Add leader hedges (pairs trade)
6. Backtest and report performance
7. Save results to `./output/`

### Configuration

Edit `config.py` to adjust:

```python
# Date range
START_DATE = "2000-01-01"
END_DATE = "2025-11-01"

# Trading parameters
MIN_PRICE = 3.0              # Minimum stock price
MIN_ADV_USD = 2_000_000      # Minimum average daily volume
TRAIN_END_PCT = 0.3          # 30% train, 70% test

# Strategy variant
USE_LEADER_HEDGE = True      # Enable pairs trade (recommended)
```

## 📁 Output Files

After running, check `./output/`:

| File | Description |
|------|-------------|
| `daily_pnl_train.csv` | Weekly PnL, turnover (train) |
| `daily_pnl_test.csv` | Weekly PnL, turnover (test) |
| `positions_train.parquet` | Portfolio weights (train) |
| `positions_test.parquet` | Portfolio weights (test) |
| `signals_train.parquet` | Raw signals (train) |
| `signals_test.parquet` | Raw signals (test) |
| `cumulative_returns_comparison_train.png` | Performance chart (train) |
| `cumulative_returns_comparison_test.png` | Performance chart (test) |

## 🔬 Key Features

### 1. Weekly Returns
- **Why?** Reduces noise, captures multi-day momentum
- **How?** Compound daily returns within each week
- **Benefit?** More stable signals, fewer observations needed

### 2. Random Week Splitting
- **Why?** Tests if relationships are time-invariant
- **Caveat?** May introduce look-ahead bias
- **Alternative?** Set chronological split in code (commented out)

### 3. Pairs Trade with Leader Hedging
- **Why?** Isolates relative momentum, removes market beta
- **Result?** 5x better Sharpe ratio vs long/short followers only
- **Mechanism?** Long follower + Short leader = pure lead-lag bet

### 4. Mean-Variance Optimization
- **Why?** Balances return vs risk
- **Parameters?** Risk aversion = 2.0 (tunable)
- **Constraints?** Dollar neutral, position limits

### 5. Volatility Targeting
- **Why?** Consistent risk exposure
- **Target?** 10% annualized volatility
- **Method?** Scale positions based on realized vol

## 📊 Strategy Variants

### Without Leader Hedging
Set `USE_LEADER_HEDGE = False` in `config.py`

**Results:**
- Sharpe: 0.07
- Return: 0.38%
- Cumulative: 3.77%

**Interpretation:** Long/short followers only, exposed to sector risk

### With Leader Hedging (Current)
Set `USE_LEADER_HEDGE = True` in `config.py`

**Results:**
- Sharpe: 0.36 ✅
- Return: 2.27% ✅
- Cumulative: 43.85% ✅

**Interpretation:** Pairs trade isolates lead-lag effect

## 🧪 Robustness Checks (Optional)

Uncomment robustness section in `main.py` for:

1. **Parameter Sensitivity:**
   - Risk aversion: [0.5, 1.0, 2.0, 5.0, 10.0]
   - Top quantile: [0.2, 0.3, 0.4, 0.5]

2. **Rolling Window Analysis:**
   - Window: 52 weeks (1 year)
   - Step: 13 weeks (1 quarter)
   - Tests stability over time

## ⚠️ Important Notes

### Look-Ahead Bias
Random week splitting may introduce bias:
- Training on 2024 data, testing on 2020 data
- Useful for testing time-invariance
- **Not realistic for live trading**
- Consider chronological split for production

### Transaction Costs
Currently set to 0 bps. Adjust in `config.py`:
```python
TRANSACTION_COST_BPS_PER_SIDE = 5.0  # 5 bps per side
```

### Data Requirements
- WRDS subscription required
- ~25 years of daily data
- 77 stocks (some may have missing data)

## 📚 References

**Lead-Lag Relationships:**
- Cohen & Frazzini (2008) - "Economic Links and Predictable Returns"
- Menzly & Ozbas (2010) - "Market Segmentation and Cross-predictability of Returns"

**Pairs Trading:**
- Gatev, Goetzmann & Rouwenhorst (2006) - "Pairs Trading: Performance of a Relative-Value Arbitrage Rule"

**Mean-Variance Optimization:**
- Markowitz (1952) - "Portfolio Selection"
- Ledoit & Wolf (2004) - "Honey, I Shrunk the Sample Covariance Matrix"

## 🤝 Contributing

To modify the strategy:

1. **Add new leader-follower pairs:** Edit `get_leader_follower_mapping()` in `config.py`
2. **Change signal construction:** Modify `build_multi_lag_signal()` in `signal_generation.py`
3. **Adjust optimization:** Update `mean_variance_optimize()` in `portfolio_optimization.py`
4. **Add metrics:** Extend `summarize_performance()` in `backtesting.py`

## 📝 License

MIT License - feel free to use and modify

## 👤 Author

Nirup Kushalnagar

---

**Disclaimer:** This is for educational purposes only. Past performance does not guarantee future results. Not financial advice.
