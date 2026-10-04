"""
Synthetic UPI/Card transaction generator with realistic fraud patterns.
Real Kaggle dataset use karna ho to: data/transactions.csv me same columns rakh do
(ya train.py me load_data() badal do).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N_USERS, N_TX, FRAUD_RATE = 3000, 120_000, 0.015
CITIES = ["Delhi", "Mumbai", "Kanpur", "Lucknow", "Bengaluru", "Pune", "Kolkata", "Chennai"]
MERCHANTS = ["grocery", "food", "fuel", "travel", "electronics", "gaming", "jewellery", "utilities", "crypto"]
RISKY = {"electronics", "gaming", "jewellery", "crypto", "travel"}


def make():
    users = pd.DataFrame({
        "user_id": np.arange(N_USERS),
        "home_city": rng.choice(CITIES, N_USERS),
        "avg_amount": rng.lognormal(6.2, 0.6, N_USERS),   # ~Rs 500 typical
    })
    start = pd.Timestamp("2026-01-01")
    uid = rng.integers(0, N_USERS, N_TX)
    ts = start + pd.to_timedelta(np.sort(rng.integers(0, 180 * 24 * 3600, N_TX)), unit="s")
    df = pd.DataFrame({"user_id": uid, "timestamp": ts}).merge(users, on="user_id")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["is_fraud"] = (rng.random(N_TX) < FRAUD_RATE).astype(int)
    f = df["is_fraud"] == 1
    n_f = f.sum()

    # --- legit behaviour ---
    df["amount"] = df["avg_amount"] * rng.lognormal(0, 0.5, N_TX)
    df["merchant"] = rng.choice(MERCHANTS[:8], N_TX, p=[.25, .25, .12, .08, .07, .05, .03, .15])
    df["city"] = np.where(rng.random(N_TX) < 0.92, df["home_city"], rng.choice(CITIES, N_TX))
    hour = np.clip(rng.normal(15, 4.5, N_TX), 0, 23).astype(int)
    df["device_new"] = (rng.random(N_TX) < 0.04).astype(int)
    df["international"] = (rng.random(N_TX) < 0.02).astype(int)

    # --- fraud behaviour (patterns the model should discover) ---
    df.loc[f, "amount"] = df.loc[f, "avg_amount"] * rng.lognormal(1.6, 0.7, n_f)
    df.loc[f, "merchant"] = rng.choice(MERCHANTS, n_f, p=[.03, .04, .03, .12, .25, .17, .16, .02, .18])
    df.loc[f, "city"] = rng.choice(CITIES, n_f)
    hour[f.values] = rng.choice([0, 1, 2, 3, 4, 23, 22, 14, 15], n_f, p=[.14, .14, .14, .12, .1, .08, .08, .1, .1])
    df.loc[f, "device_new"] = (rng.random(n_f) < 0.70).astype(int)
    df.loc[f, "international"] = (rng.random(n_f) < 0.35).astype(int)
    df["hour"] = hour
    # timestamp ka hour = hour column (consistency)
    df["timestamp"] = (df["timestamp"].dt.normalize() + pd.to_timedelta(df["hour"], unit="h")
                       + pd.to_timedelta(rng.integers(0, 3600, N_TX), unit="s"))
    # fraud bursts: 40% fraud tx kisi pichle fraud ke 10-300 sec baad, SAME user (card testing)
    fi = np.where(df["is_fraud"] == 1)[0]
    prev, cur = fi[:-1], fi[1:]
    pick = rng.random(len(cur)) < 0.40
    c, p = cur[pick], prev[pick]
    df.loc[c, "user_id"] = df.loc[p, "user_id"].values
    df.loc[c, "home_city"] = df.loc[p, "home_city"].values
    df.loc[c, "avg_amount"] = df.loc[p, "avg_amount"].values
    df.loc[c, "timestamp"] = df.loc[p, "timestamp"].values + pd.to_timedelta(rng.integers(10, 300, len(c)), unit="s")
    df.loc[c, "hour"] = df.loc[c, "timestamp"].dt.hour
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["amount"] = df["amount"].round(2)
    df["tx_id"] = np.arange(len(df))
    return df[["tx_id", "user_id", "timestamp", "amount", "merchant", "city", "home_city",
               "avg_amount", "device_new", "international", "hour", "is_fraud"]]


if __name__ == "__main__":
    d = make()
    d.to_csv("data/transactions.csv", index=False)
    print(d.shape, "| fraud %:", round(d.is_fraud.mean() * 100, 2))
    print(d.head())
