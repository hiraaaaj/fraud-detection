import numpy as np
import pandas as pd

MERCHANTS = ["grocery", "food", "fuel", "travel", "electronics", "gaming", "jewellery", "utilities", "crypto"]
WEEK = 7 * 24 * 3600
RISKY = {"electronics", "gaming", "jewellery", "crypto", "travel"}

FEATURES = [
    "log_amount", "amount_ratio", "hour", "is_night", "city_mismatch", "device_new",
    "international", "risky_merchant", "secs_since_last_tx", "tx_count_24h",
] + [f"merchant_{m}" for m in MERCHANTS]


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-user behavioural features (sirf past data use hota hai -> no leakage)."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["user_id", "timestamp"])
    df["secs_since_last_tx"] = (df.groupby("user_id")["timestamp"].diff().dt.total_seconds()
                                .fillna(WEEK).clip(upper=WEEK))
    # pichle 24h me is user ke kitne tx (current tx ko chhodkar)
    cnt = df.set_index("timestamp").groupby("user_id")["amount"].rolling("24h").count()
    df["tx_count_24h"] = (cnt.values - 1).clip(min=0)
    return df.sort_values("timestamp").reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["log_amount"] = np.log1p(df["amount"])
    X["amount_ratio"] = df["amount"] / df["avg_amount"].clip(lower=1)
    X["hour"] = df["hour"]
    X["is_night"] = ((df["hour"] <= 5) | (df["hour"] >= 22)).astype(int)
    X["city_mismatch"] = (df["city"] != df["home_city"]).astype(int)
    X["device_new"] = df["device_new"]
    X["international"] = df["international"]
    X["risky_merchant"] = df["merchant"].isin(RISKY).astype(int)
    X["secs_since_last_tx"] = df["secs_since_last_tx"]
    X["tx_count_24h"] = df["tx_count_24h"]
    for m in MERCHANTS:
        X[f"merchant_{m}"] = (df["merchant"] == m).astype(int)
    return X[FEATURES]
