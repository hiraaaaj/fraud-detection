"""Single-transaction scoring + explanation. SHAP mile to wo, warna what-if (occlusion) explanation."""
import json, joblib, numpy as np, pandas as pd
from features import FEATURES, MERCHANTS, RISKY

_model = joblib.load("models/model.joblib")
META = json.load(open("models/meta.json"))
THRESHOLD = META["threshold"]

# "normal behaviour" baseline: ek typical legit transaction
BASELINE = {f: 0 for f in FEATURES}
BASELINE.update(log_amount=np.log1p(500), amount_ratio=1.0, hour=15, secs_since_last_tx=3 * 24 * 3600, merchant_grocery=1)

NICE = {
    "log_amount": "Amount", "amount_ratio": "Amount vs user's average", "hour": "Hour of day",
    "is_night": "Night-time transaction", "city_mismatch": "City different from home city",
    "device_new": "New / unseen device", "international": "International transaction",
    "risky_merchant": "High-risk merchant category", "secs_since_last_tx": "Time since previous transaction",
    "tx_count_24h": "Transactions in last 24h",
}


def make_row(amount, avg_amount, hour, merchant, city, home_city, device_new, international,
             secs_since_last_tx, tx_count_24h) -> pd.DataFrame:
    r = dict(
        log_amount=np.log1p(amount), amount_ratio=amount / max(avg_amount, 1), hour=hour,
        is_night=int(hour <= 5 or hour >= 22), city_mismatch=int(city != home_city),
        device_new=int(device_new), international=int(international), risky_merchant=int(merchant in RISKY),
        secs_since_last_tx=secs_since_last_tx, tx_count_24h=tx_count_24h,
    )
    for m in MERCHANTS:
        r[f"merchant_{m}"] = int(merchant == m)
    return pd.DataFrame([r])[FEATURES]


def score(row: pd.DataFrame) -> dict:
    p = float(_model.predict_proba(row)[0, 1])
    level = "HIGH" if p >= max(THRESHOLD, .5) else "MEDIUM" if p >= THRESHOLD else "LOW"
    return {"probability": p, "is_fraud": p >= THRESHOLD, "risk": level,
            "action": {"HIGH": "BLOCK + OTP/call verification", "MEDIUM": "Hold for review / step-up auth", "LOW": "Approve"}[level]}


def explain(row: pd.DataFrame, top=5, n_perm=60, seed=0) -> pd.DataFrame:
    """Shapley-sampling explanation (SHAP jaisa idea): random order me features ko
    'normal' se 'actual' value pe badlo aur har step pe prob ka marginal change attribute karo.
    Redundant signals (night + new device + intl) ko bhi fair credit milta hai."""
    rng = np.random.default_rng(seed)
    groups = {k: [k] for k in NICE if k in FEATURES}
    groups["merchant_category"] = [f for f in FEATURES if f.startswith("merchant_")]
    names = list(groups)
    base = pd.DataFrame([BASELINE])[FEATURES].astype(float)
    actual = row.astype(float).reset_index(drop=True)
    frames, index = [base], []
    for _ in range(n_perm):
        order = rng.permutation(len(names)); cur = base.copy()
        for gi in order:
            for c in groups[names[gi]]:
                cur[c] = actual[c].values[0]
            frames.append(cur.copy()); index.append(gi)
    P = _model.predict_proba(pd.concat(frames, ignore_index=True))[:, 1]
    pos = 1; contrib = np.zeros(len(names))
    for k in range(n_perm):
        prev = P[0]
        for step in range(len(names)):
            p = P[pos]; contrib[index[pos - 1]] += p - prev; prev = p; pos += 1
    contrib /= n_perm
    df = pd.DataFrame({"factor": [NICE.get(n, "Merchant category") for n in names], "impact": contrib})
    return df[df.impact > 0.005].sort_values("impact", ascending=False).head(top).reset_index(drop=True)


if __name__ == "__main__":
    tests = {
        "Normal grocery buy (Kanpur, Rs 450)": dict(amount=450, avg_amount=500, hour=18, merchant="grocery", city="Kanpur", home_city="Kanpur", device_new=0, international=0, secs_since_last_tx=200000, tx_count_24h=0),
        "Rs 45,000 crypto @3AM, new device, intl": dict(amount=45000, avg_amount=500, hour=3, merchant="crypto", city="Mumbai", home_city="Kanpur", device_new=1, international=1, secs_since_last_tx=90, tx_count_24h=2),
        "Rs 2,500 electronics, day, same city": dict(amount=2500, avg_amount=600, hour=14, merchant="electronics", city="Kanpur", home_city="Kanpur", device_new=0, international=0, secs_since_last_tx=100000, tx_count_24h=0),
    }
    for k, v in tests.items():
        row = make_row(**v); s = score(row)
        print(f"\n{k}\n  fraud prob={s['probability']:.3f} risk={s['risk']} -> {s['action']}")
        if s["probability"] > .05:
            print(explain(row).to_string(index=False))
