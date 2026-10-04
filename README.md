# 🛡️ Real-Time Fraud Detection System (UPI / Card)

End-to-end ML project: data → features → model comparison → business-cost threshold → explainable predictions → dashboard.

## Highlights
- **Imbalanced data** (1.5% fraud) handled with class weights; **PR-AUC / Precision / Recall** (accuracy nahi)
- **Time-based split** (past pe train, future pe test) – leakage se bachav
- **Behavioural features**: amount vs user average, velocity (secs since last tx, tx in 24h), city mismatch, new device, night, risky merchant
- **Cost-based threshold**: fraud miss = poora amount loss, false alarm = Rs 50 review cost
- **Explainability**: per-transaction Shapley-sampling reasons
- **Streamlit dashboard**: single check, batch CSV, model report

## Run
```bash
pip install -r requirements.txt
python generate_data.py     # synthetic data (ya apna Kaggle data data/transactions.csv me rakho)
python train.py             # models compare + best save + plots
python predict.py           # CLI demo
python server.py          # FULL-STACK app -> http://localhost:8000
streamlit run app.py        # (optional) Streamlit dashboard
```

## Structure
generate_data.py · features.py · train.py · predict.py · app.py · data/ · models/ · reports/

## Note
Results synthetic data pe hain (patterns saaf hain, isliye score high). Real dataset (Kaggle ULB / IEEE-CIS / PaySim) pe re-run karke metrics update karo.
