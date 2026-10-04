"""Streamlit dashboard:  streamlit run app.py"""
import json
import pandas as pd
import streamlit as st
from features import MERCHANTS
from predict import make_row, score, explain, META

CITIES = ["Delhi", "Mumbai", "Kanpur", "Lucknow", "Bengaluru", "Pune", "Kolkata", "Chennai"]
st.set_page_config(page_title="Fraud Detection System", page_icon="🛡️", layout="wide")
st.title("🛡️ Real-Time Fraud Detection System")
st.caption(f"Model: {META['best_model']}  |  Decision threshold: {META['threshold']:.2f} (business-cost optimised)")

tab1, tab2, tab3 = st.tabs(["🔍 Check a transaction", "📂 Batch CSV scoring", "📊 Model report"])

with tab1:
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Amount (Rs)", 1.0, 1_000_000.0, 450.0, step=50.0)
    avg_amount = c1.number_input("User's usual avg amount (Rs)", 1.0, 100_000.0, 500.0)
    merchant = c1.selectbox("Merchant category", MERCHANTS)
    hour = c2.slider("Hour of day", 0, 23, 15)
    city = c2.selectbox("Transaction city", CITIES, index=2)
    home_city = c2.selectbox("User's home city", CITIES, index=2)
    gap_min = c3.number_input("Minutes since previous tx", 0.0, 10_000.0, 2000.0)
    cnt = c3.number_input("Transactions in last 24h", 0, 100, 0)
    device_new = c3.checkbox("New / unseen device")
    intl = c3.checkbox("International transaction")

    if st.button("Analyse", type="primary"):
        row = make_row(amount, avg_amount, hour, merchant, city, home_city, device_new, intl, gap_min * 60, cnt)
        r = score(row)
        colour = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[r["risk"]]
        st.markdown(f"### Fraud probability: **{r['probability']*100:.1f}%**  ·  Risk: :{colour}[{r['risk']}]")
        st.progress(min(r["probability"], 1.0))
        st.info(f"Recommended action: **{r['action']}**")
        ex = explain(row)
        if len(ex):
            st.subheader("Why this score?")
            st.bar_chart(ex.set_index("factor")["impact"])

with tab2:
    st.write("CSV upload karo (columns: amount, avg_amount, hour, merchant, city, home_city, device_new, international, secs_since_last_tx, tx_count_24h)")
    up = st.file_uploader("Transactions CSV", type="csv")
    if up:
        d = pd.read_csv(up)
        probs = [score(make_row(r.amount, r.avg_amount, int(r.hour), r.merchant, r.city, r.home_city, r.device_new,
                                r.international, r.secs_since_last_tx, r.tx_count_24h))["probability"] for r in d.itertuples()]
        d["fraud_probability"] = probs
        d["flagged"] = d["fraud_probability"] >= META["threshold"]
        st.write(f"Flagged {int(d.flagged.sum())} of {len(d)} transactions")
        st.dataframe(d.sort_values("fraud_probability", ascending=False))
        st.download_button("Download results", d.to_csv(index=False), "scored.csv")

with tab3:
    st.dataframe(pd.read_csv("reports/model_comparison.csv"))
    cols = st.columns(2)
    for i, img in enumerate(["pr_curves", "confusion_matrix", "cost_vs_threshold", "feature_importance"]):
        cols[i % 2].image(f"reports/{img}.png")
