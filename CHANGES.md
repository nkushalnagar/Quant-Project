# Changes Made to Lead-Lag Strategy

## Summary of Updates (Nov 25, 2024)

### 1. Train/Test Split: 70/30 (Changed from 30/70)
- **File**: `config.py`
- **Change**: `TRAIN_END_PCT = 0.7` (70% train, 30% test)

### 2. Weekly Cumulative Returns (Changed from Daily)
- **Files**: `signal_generation.py`, `main.py`, `backtesting.py`
- **Logic**: 
  - Daily returns are now aggregated into weekly cumulative returns
  - Formula: `(1 + r_day1) * (1 + r_day2) * ... * (1 + r_dayN) - 1`
  - Week identifier uses ISO week numbers
  - Week-end date is the last trading day of each week

### 3. Lead-Lag Signal with Weekly Data
- **Signal Logic**: Leader returns from week `x-1` predict follower returns in week `x`
- **Implementation**: `lags=(1,)` with `lag_decay=1.0` in weekly context
- **Example**: 
  - If ASML (leader) has cumulative return of +5% in week 10
  - This predicts NVDA (follower) performance in week 11

### 4. Chronological Split (FIXED - was random, now chronological)
- **File**: `main.py`
- **Logic**:
  - First 70% of weeks used for training
  - Last 30% of weeks used for testing
  - **Avoids look-ahead bias** (no future data in training)
  - Standard practice in quantitative finance
  - Cumulative return plots are now meaningful (chronological order)

### 5. Performance Metrics Updated
- **File**: `backtesting.py`
- **Changes**:
  - Annualization factor: 52 weeks (changed from 252 days)
  - Metrics now show "Weekly Hit Rate" instead of "Daily Hit Rate"
  - Alpha tests use weekly annualization (×52 instead of ×252)

## New Functions Added

### `convert_to_weekly_returns(ret_p)` in `signal_generation.py`
Converts daily return panel to weekly cumulative returns.

### `random_week_split(ret_weekly, train_pct, random_seed)` in `signal_generation.py`
Randomly splits weeks into train/test sets instead of chronological split.

## Files Modified
1. `config.py` - Train/test split percentage
2. `signal_generation.py` - Weekly conversion and random splitting
3. `backtesting.py` - Weekly performance metrics
4. `main.py` - Updated workflow for weekly data

## How to Run
```bash
python3 main.py
```

## Important Notes

### ⚠️ Random Week Splitting Caveat
Random week selection violates the time-series nature of financial data:
- **Pro**: Tests if lead-lag relationships are stable across time
- **Con**: May introduce look-ahead bias (using future data to predict past)
- **Recommendation**: Also test with chronological split for comparison

### Weekly vs Daily Returns
- Weekly returns capture multi-day momentum/reversal effects
- Reduces noise from daily volatility
- Better for testing supply chain relationships that take days to materialize
- Fewer observations (52 weeks/year vs 252 days/year)

## Example Output
```
Converting daily returns to weekly cumulative returns...
Daily data: 6500 days
Weekly data: 1300 weeks
Date range: 2000-01-07 to 2025-10-31

Random week split (seed=42):
  Total weeks: 1300
  Train weeks: 390 (30.0%)
  Test weeks: 910 (70.0%)
  Train date range: 2000-01-14 to 2025-10-24
  Test date range: 2000-01-07 to 2025-10-31
```
