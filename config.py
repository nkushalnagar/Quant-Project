"""Configuration parameters and leader-follower mapping."""

# Date range
START_DATE = "2000-01-01"
END_DATE = "2025-11-01"

# Trading configuration
MIN_PRICE = 3.0
MIN_ADV_USD = 2_000_000
SIGNAL_ABS_THRESHOLD = 0.0
TRANSACTION_COST_BPS_PER_SIDE = 0.0
HOLD_DAYS = 1

# Train/Test split
TRAIN_END_PCT = 0.7  # 70% train, 30% test (chronological split)

# Benchmark tickers (in order of preference)
BENCHMARK_CANDIDATES = ["SPY", "VV", "VONE"]  # SPY has longest history (1993+)

# Output
OUTPUT_DIR = "./output"

# Strategy variant
USE_LEADER_HEDGE = True  # If True, hedge followers by shorting their leaders (pairs trade)
USE_MARKET_HEDGE = False  # If True, add market hedge (SPY)
MARKET_HEDGE_DIRECTION = 1  # -1 for short SPY (market neutral), +1 for long SPY (market beta)
MARKET_HEDGE_WEIGHT = 1.0  # Weight of market hedge relative to portfolio (1.0 = dollar-neutral)


def get_leader_follower_mapping():
    """Returns dict: { follower: [(leader, weight), ...], ... }"""
    mapping = {
        # Semicap -> Chips
        "NVDA": [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],
        "AMD": [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],
        "AVGO": [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],
        "MU": [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],
        "TXN": [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],
        "MRVL": [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],
        "ADI": [("ASML", 1), ("AMAT", 1), ("LRCX", 1), ("KLAC", 1)],
        # Memory -> Hardware OEMs
        "DELL": [("MU", 1), ("WDC", 1), ("STX", 1)],
        "HPQ": [("MU", 1), ("WDC", 1), ("STX", 1)],
        "HPE": [("MU", 1), ("WDC", 1), ("STX", 1)],
        "SMCI": [("MU", 1)],
        # Foundries -> Chip designers
        "QCOM": [("TSM", 1), ("GFS", 1), ("INTC", 1)],
        "MPWR": [("TSM", 1), ("GFS", 1)],
        "MCHP": [("TSM", 1), ("GFS", 1), ("INTC", 1)],
        "NXPI": [("TSM", 1), ("GFS", 1)],
        # Lithium/materials -> EV OEMs
        "TSLA": [("ALB", 1), ("SQM", 1), ("LTHM", 1)],
        "GM": [("ALB", 1), ("SQM", 1)],
        "F": [("ALB", 1), ("SQM", 1)],
        # Steel -> OEMs/Industrial consumers
        "CAT": [("NUE", 1), ("STLD", 1), ("X", 1), ("CLF", 1)],
        "DE": [("NUE", 1), ("STLD", 1), ("X", 1), ("CLF", 1)],
        "PCAR": [("NUE", 1), ("STLD", 1)],
        # Refiners -> Airlines
        "DAL": [("VLO", 1), ("MPC", 1), ("PSX", 1)],
        "UAL": [("VLO", 1), ("MPC", 1), ("PSX", 1)],
        "AAL": [("VLO", 1), ("MPC", 1), ("PSX", 1)],
        "LUV": [("VLO", 1), ("MPC", 1), ("PSX", 1)],
        # EMS -> OEMs/components
        "AAPL": [("FLEX", 1), ("JBL", 1), ("SANM", 1)],
        "CSCO": [("FLEX", 1), ("JBL", 1)],
        # Big box retail -> Branded CPG
        "PG": [("WMT", 1), ("COST", 1), ("TGT", 1)],
        "KMB": [("WMT", 1), ("COST", 1), ("TGT", 1)],
        "CL": [("WMT", 1), ("COST", 1), ("TGT", 1)],
        # Hyperscalers/infra -> Cloud software
        "SNOW": [("AMZN", 1), ("MSFT", 1), ("GOOGL", 1)],
        "MDB": [("AMZN", 1), ("MSFT", 1), ("GOOGL", 1)],
        "DDOG": [("AMZN", 1), ("MSFT", 1), ("GOOGL", 1)],
        "CRWD": [("AMZN", 1), ("MSFT", 1), ("GOOGL", 1)],
        # Banks -> Homebuilders
        "DHI": [("WFC", 1), ("JPM", 1), ("BAC", 1)],
        "LEN": [("WFC", 1), ("JPM", 1), ("BAC", 1)],
        "PHM": [("WFC", 1), ("JPM", 1), ("BAC", 1)],
        # Homebuilders -> Building products
        "TREX": [("DHI", 1), ("LEN", 1), ("PHM", 1)],
        "AZEK": [("DHI", 1), ("LEN", 1), ("PHM", 1)],
        "OC": [("DHI", 1), ("LEN", 1), ("PHM", 1)],
        # Airlines -> Travel platforms
        "BKNG": [("DAL", 1), ("UAL", 1), ("AAL", 1)],
        "EXPE": [("DAL", 1), ("UAL", 1), ("AAL", 1)],
        "ABNB": [("DAL", 1), ("UAL", 1), ("AAL", 1)],
        # PC core vendors -> PC ecosystem
        "LOGI": [("INTC", 1), ("AMD", 1), ("MSFT", 1)],
        "STX": [("INTC", 1), ("AMD", 1), ("MSFT", 1)],
        "WDC": [("INTC", 1), ("AMD", 1), ("MSFT", 1)],
    }

    # Normalize weights per follower
    normed = {}
    for fol, pairs in mapping.items():
        total = sum(w for _, w in pairs)
        if total > 0:
            normed[fol] = [(ldr, w / total) for ldr, w in pairs]
    return normed
