"""Train + compare models, tune threshold by business cost, save best model + plots."""
import json, warnings
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from features import add_history_features, build_features, FEATURES

warnings.filterwarnings("ignore")
REVIEW_COST = 50.0   # Rs: ek false alarm ko manually review karne ka cost


def load_data():
    df = pd.read_csv("data/transactions.csv", parse_dates=["timestamp"])
    df = add_history_features(df)
    return df, build_features(df), df["is_fraud"]


def models():
    m = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(learning_rate=0.08, max_iter=300, class_weight="balanced", random_state=42),
    }
    try:
        from xgboost import XGBClassifier  # optional
        m["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=5, eval_metric="aucpr", n_jobs=-1)
    except ImportError:
        print("[info] xgboost nahi mila -> skip (pip install xgboost se add ho jayega)")
    return m


def cost_at(th, y, p, amt):
    pred = p >= th
    fn = ((~pred) & (y == 1)) * amt          # fraud miss -> poora amount loss
    fp = (pred & (y == 0)) * REVIEW_COST     # false alarm -> review cost
    return float(fn.sum() + fp.sum())


def main():
    df, X, y = load_data()
    # TIME-BASED split: past se seekho, future pe test (real life jaisa)
    n = len(df); i1, i2 = int(n * .70), int(n * .85)
    Xtr, ytr = X.iloc[:i1], y.iloc[:i1]
    Xva, yva, ava = X.iloc[i1:i2], y.iloc[i1:i2], df["amount"].iloc[i1:i2].values
    Xte, yte, ate = X.iloc[i2:], y.iloc[i2:], df["amount"].iloc[i2:].values
    print(f"train {len(Xtr)} | val {len(Xva)} | test {len(Xte)} | fraud% train {ytr.mean()*100:.2f}")

    rows, fitted, probs = [], {}, {}
    for name, mdl in models().items():
        mdl.fit(Xtr, ytr)
        fitted[name] = mdl
        pv, pt = mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1]
        ths = np.linspace(0.02, 0.98, 97)
        th = ths[int(np.argmin([cost_at(t, yva.values, pv, ava) for t in ths]))]  # threshold VALIDATION se
        pred = pt >= th
        rows.append(dict(Model=name, PR_AUC=average_precision_score(yte, pt), ROC_AUC=roc_auc_score(yte, pt),
                         Precision=precision_score(yte, pred), Recall=recall_score(yte, pred),
                         F1=f1_score(yte, pred), Threshold=th,
                         Cost_Rs=cost_at(th, yte.values, pt, ate)))
        probs[name] = (pt, th)
        print(f"  {name:22s} PR-AUC={rows[-1]['PR_AUC']:.3f}  P={rows[-1]['Precision']:.2f}  R={rows[-1]['Recall']:.2f}  th={th:.2f}")

    res = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False).round(4)
    res.to_csv("reports/model_comparison.csv", index=False)
    best = res.iloc[0]["Model"]; pt, th = probs[best]
    no_model_cost = float((yte.values * ate).sum())     # kuch nahi kiya to saara fraud amount loss
    print("\n", res.to_string(index=False))
    print(f"\nBEST: {best} | cost with model Rs {res.iloc[0]['Cost_Rs']:,.0f} vs without model Rs {no_model_cost:,.0f}")

    # ---------- plots ----------
    plt.figure(figsize=(7, 5))
    for name, (p, _) in probs.items():
        pr, rc, _ = precision_recall_curve(yte, p); plt.plot(rc, pr, label=f"{name} ({average_precision_score(yte, p):.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curves (test set)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("reports/pr_curves.png", dpi=130); plt.close()

    cm = confusion_matrix(yte, pt >= th)
    ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"]).plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {best} (th={th:.2f})"); plt.tight_layout(); plt.savefig("reports/confusion_matrix.png", dpi=130); plt.close()

    ths = np.linspace(0.02, 0.98, 97)
    plt.figure(figsize=(7, 4)); plt.plot(ths, [cost_at(t, yte.values, pt, ate) for t in ths]); plt.axvline(th, color="r", ls="--", label="chosen threshold")
    plt.xlabel("Threshold"); plt.ylabel("Total cost (Rs)"); plt.title("Business cost vs threshold"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("reports/cost_vs_threshold.png", dpi=130); plt.close()

    imp = permutation_importance(fitted[best], Xte.iloc[:5000], yte.iloc[:5000], scoring="average_precision", n_repeats=5, random_state=0, n_jobs=-1)
    s = pd.Series(imp.importances_mean, index=FEATURES).sort_values().tail(12)
    plt.figure(figsize=(7, 5)); s.plot.barh(color="teal"); plt.title("Top features (permutation importance)")
    plt.tight_layout(); plt.savefig("reports/feature_importance.png", dpi=130); plt.close()
    pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).to_csv("reports/feature_importance.csv")

    joblib.dump(fitted[best], "models/model.joblib")
    json.dump({"best_model": best, "threshold": float(th), "features": FEATURES,
               "metrics": res.iloc[0].to_dict(), "no_model_cost": no_model_cost}, open("models/meta.json", "w"), indent=2, default=float)
    print("saved -> models/model.joblib, reports/*.png")


if __name__ == "__main__":
    main()
