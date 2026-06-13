import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="PV Signal Intelligence",
    page_icon="🔬",
    layout="wide"
)

# ── Styling ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #F8FAFC; }

.metric-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
}
.metric-num { font-size: 2rem; font-weight: 700; color: #1E3A5F; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 0.8rem; color: #64748B; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }

.signal-badge-ACCEPTED  { background:#DCFCE7; color:#166534; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.signal-badge-REJECTED  { background:#FEE2E2; color:#991B1B; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.signal-badge-DEFERRED  { background:#FEF3C7; color:#92400E; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }
.signal-badge-PENDING_REVIEW { background:#EFF6FF; color:#1D4ED8; padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:600; }

.section-header {
    font-size: 1.1rem; font-weight: 700; color: #1E3A5F;
    border-left: 4px solid #2563EB; padding-left: 12px;
    margin: 24px 0 14px 0;
}
.rationale-box {
    background: #F0F7FF; border-left: 3px solid #2563EB;
    padding: 12px 16px; border-radius: 0 8px 8px 0;
    font-size: 0.88rem; color: #1E3A5F; line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────
BASE = "/Users/manasauppuluri/Desktop/FAERS/faers_ascii_2026q1"

@st.cache_data
def load_data():
    signals_path = f"{BASE}/signals_all_drugs.csv"
    review_path  = f"{BASE}/review_log.csv"
    rxnorm_path  = f"{BASE}/rxnorm_cache.csv"

    signals = pd.read_csv(signals_path) if os.path.exists(signals_path) else pd.DataFrame()
    review  = pd.read_csv(review_path)  if os.path.exists(review_path)  else pd.DataFrame()
    rxnorm  = pd.read_csv(rxnorm_path)  if os.path.exists(rxnorm_path)  else pd.DataFrame()
    return signals, review, rxnorm

signals_df, review_df, rxnorm_df = load_data()

# ── Header ────────────────────────────────────────────────
st.markdown("## 🔬 Pharmacovigilance Signal Intelligence")
st.markdown("<span style='color:#64748B;font-size:0.9rem'>FAERS Q1 2026 · AI-enabled signal detection · RxNorm-mapped · Human-in-the-loop review</span>", unsafe_allow_html=True)
st.divider()

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔎 Signal Explorer", "✅ Review Log", "💊 RxNorm Mapping"])

# ════════════════════════════════════════════════════════
# TAB 1 — Overview
# ════════════════════════════════════════════════════════
with tab1:
    if not signals_df.empty:
        total   = len(signals_df)
        flagged = signals_df["signal_flagged"].sum() if "signal_flagged" in signals_df.columns else 0
        drugs   = signals_df["drug"].nunique()

        reviewed   = len(review_df) if not review_df.empty else 0
        accepted   = (review_df["review_status"] == "ACCEPTED").sum()  if not review_df.empty else 0
        rejected   = (review_df["review_status"] == "REJECTED").sum()  if not review_df.empty else 0
        deferred   = (review_df["review_status"] == "DEFERRED").sum()  if not review_df.empty else 0
        pending    = (review_df["review_status"] == "PENDING_REVIEW").sum() if not review_df.empty else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        for col, num, label in [
            (c1, total,    "Drug-Event Pairs"),
            (c2, drugs,    "Drugs Analysed"),
            (c3, accepted, "Accepted Signals"),
            (c4, rejected, "Rejected"),
            (c5, deferred, "Deferred"),
        ]:
            col.markdown(f"""<div class="metric-card">
                <div class="metric-num">{num:,}</div>
                <div class="metric-label">{label}</div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Signals per Drug</div>', unsafe_allow_html=True)
        drug_counts = signals_df[signals_df["signal_flagged"] == True]["drug"].value_counts().reset_index()
        drug_counts.columns = ["Drug", "Flagged Signals"]
        fig = px.bar(drug_counts, x="Drug", y="Flagged Signals",
                     color="Flagged Signals", color_continuous_scale="Blues",
                     template="plotly_white")
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          margin=dict(t=20, b=20), height=340)
        st.plotly_chart(fig, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-header">Review Decision Breakdown</div>', unsafe_allow_html=True)
            if not review_df.empty:
                status_counts = review_df["review_status"].value_counts().reset_index()
                status_counts.columns = ["Status", "Count"]
                colors = {"ACCEPTED":"#22C55E","REJECTED":"#EF4444","DEFERRED":"#F59E0B","PENDING_REVIEW":"#3B82F6"}
                fig2 = px.pie(status_counts, names="Status", values="Count",
                              color="Status", color_discrete_map=colors,
                              template="plotly_white", hole=0.45)
                fig2.update_layout(margin=dict(t=10, b=10), height=300, legend=dict(orientation="h"))
                st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-header">Top 10 Signals by ROR</div>', unsafe_allow_html=True)
            top = signals_df[signals_df["signal_flagged"] == True].nlargest(10, "ror")[["drug","event","ror","count"]]
            fig3 = px.bar(top, x="ror", y="event", orientation="h", color="drug",
                          template="plotly_white", labels={"ror":"ROR","event":""})
            fig3.update_layout(height=300, margin=dict(t=10, b=10), showlegend=True,
                               legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("signals_all_drugs.csv not found. Run agent_pipeline.py first.")

# ════════════════════════════════════════════════════════
# TAB 2 — Signal Explorer
# ════════════════════════════════════════════════════════
with tab2:
    if not signals_df.empty:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            drug_options = ["All"] + sorted(signals_df["drug"].unique().tolist())
            selected_drug = st.selectbox("Filter by Drug", drug_options)
        with col2:
            min_ror = st.slider("Minimum ROR", 1.0, 50.0, 2.0, 0.5)
        with col3:
            flagged_only = st.checkbox("Flagged only", value=True)

        df = signals_df.copy()
        if selected_drug != "All":
            df = df[df["drug"] == selected_drug]
        if flagged_only:
            df = df[df["signal_flagged"] == True]
        df = df[df["ror"] >= min_ror].sort_values("ror", ascending=False)

        st.markdown(f'<div class="section-header">{len(df):,} signals shown</div>', unsafe_allow_html=True)

        if not df.empty:
            fig4 = px.scatter(df, x="count", y="ror", size="chi2",
                              color="drug", hover_name="event",
                              log_x=True, template="plotly_white",
                              labels={"count":"Report Count (log)", "ror":"ROR"},
                              size_max=30)
            fig4.update_layout(height=380, margin=dict(t=20, b=20))
            st.plotly_chart(fig4, use_container_width=True)

            display_cols = ["drug","event","count","ror","ror_ci_lower","ror_ci_upper","prr","chi2"]
            display_cols = [c for c in display_cols if c in df.columns]
            st.dataframe(df[display_cols].reset_index(drop=True),
                         use_container_width=True, height=360)
    else:
        st.warning("No signal data available.")

# ════════════════════════════════════════════════════════
# TAB 3 — Review Log
# ════════════════════════════════════════════════════════
with tab3:
    if not review_df.empty:
        status_filter = st.multiselect(
            "Filter by Review Status",
            options=["ACCEPTED","REJECTED","DEFERRED","PENDING_REVIEW"],
            default=["ACCEPTED","REJECTED","DEFERRED"]
        )
        filtered = review_df[review_df["review_status"].isin(status_filter)] if status_filter else review_df

        st.markdown(f'<div class="section-header">{len(filtered):,} records</div>', unsafe_allow_html=True)

        for _, row in filtered.head(200).iterrows():
            with st.expander(f"**{row.get('drug','?')}** — {row.get('event','?')}  |  ROR={row.get('ror','?')}  |  n={row.get('count','?')}"):
                badge_class = f"signal-badge-{row.get('review_status','PENDING_REVIEW')}"
                st.markdown(f'<span class="{badge_class}">{row.get("review_status","?")}</span>', unsafe_allow_html=True)
                if pd.notna(row.get("rationale","")):
                    st.markdown(f'<div class="rationale-box">{row.get("rationale","")}</div>', unsafe_allow_html=True)
                if pd.notna(row.get("reviewer_notes","")) and str(row.get("reviewer_notes","")).strip():
                    st.markdown(f"**Reviewer note:** {row.get('reviewer_notes','')}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ROR", row.get("ror","—"))
                c2.metric("PRR", row.get("prr","—"))
                c3.metric("Chi²", row.get("chi2","—"))
                c4.metric("Count", row.get("count","—"))
    else:
        st.warning("review_log.csv not found. Run agent_pipeline.py first.")

# ════════════════════════════════════════════════════════
# TAB 4 — RxNorm Mapping
# ════════════════════════════════════════════════════════
with tab4:
    if not rxnorm_df.empty:
        mapped   = (rxnorm_df["status"] == "MAPPED").sum()
        unmapped = (rxnorm_df["status"] == "NEEDS_REVIEW").sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Drug Names", len(rxnorm_df))
        c2.metric("✅ Mapped to RxNorm", mapped)
        c3.metric("⚠️ Needs Review", unmapped)

        st.markdown('<div class="section-header">RxNorm Mapping Results</div>', unsafe_allow_html=True)
        status_filter2 = st.radio("Show", ["All","MAPPED","NEEDS_REVIEW"], horizontal=True)
        df2 = rxnorm_df if status_filter2 == "All" else rxnorm_df[rxnorm_df["status"] == status_filter2]
        st.dataframe(df2[["original_name","normalized_name","rxcui","status"]].reset_index(drop=True),
                     use_container_width=True)
    else:
        st.warning("rxnorm_cache.csv not found. Run agent_pipeline.py first.")
