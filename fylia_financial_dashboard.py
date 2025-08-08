# fylia_financial_dashboard.py
# Streamlit app — Fylia Financial Dashboard with animations & upload
# Run: streamlit run fylia_financial_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Fylia Financial Dashboard", layout="wide")
st.title("📊 Fylia Financial Dashboard — Animated Projections & Upload")

# ----------------------
# Helper functions
# ----------------------
def csv_excel_to_df(uploaded_file):
    """Read uploaded CSV/XLSX into DataFrame; expects Month, Caregivers, Revenue, Expenses optional fields."""
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Couldn't read uploaded file: {e}")
        return None

def df_to_excel_bytes(df_dict):
    """Return an in-memory Excel file (bytes) from a dict of dataframes."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()

def prepare_projection_from_inputs(start_caregivers, monthly_growth_rate, avg_bookings_per_caregiver,
                                   avg_booking_value, commission_rate, marketing_spend, months, fixed_costs):
    months_list = list(range(1, months + 1))
    caregivers = []
    c = start_caregivers
    for m in months_list:
        if m == 1:
            caregivers.append(c)
        else:
            c = c * (1 + monthly_growth_rate)
            caregivers.append(c)
    bookings = [c * avg_bookings_per_caregiver for c in caregivers]
    revenue = [b * avg_booking_value for b in bookings]
    commission = [r * commission_rate for r in revenue]
    # Optional: marketing gives a small bookings uplift over time; simple model: every 3 months add 2% uplift * (marketing_spend/5000)
    uplift = [(1 + 0.02 * ((marketing_spend / 5000) * (m//3))) for m in months_list]
    commission_uplift = [commission[i] * uplift[i] for i in range(len(commission))]
    net_profit = [commission_uplift[i] - fixed_costs - marketing_spend for i in range(len(commission_uplift))]
    cumulative = np.cumsum(net_profit)
    df = pd.DataFrame({
        "MonthIndex": months_list,
        "MonthLabel": [f"Month {m}" for m in months_list],
        "Caregivers": np.round(caregivers).astype(int),
        "Bookings": np.round(bookings).astype(int),
        "Revenue": np.round(revenue, 2),
        "Commission": np.round(commission_uplift, 2),
        "MarketingSpend": marketing_spend,
        "FixedCosts": fixed_costs,
        "NetProfit": np.round(net_profit, 2),
        "CumulativeProfit": np.round(cumulative, 2)
    })
    return df

# ----------------------
# Sidebar — Inputs or upload
# ----------------------
st.sidebar.header("Input Options — defaults preloaded for Accra launch")

upload = st.sidebar.file_uploader("Upload monthly data (CSV or XLSX). Required columns if uploading: MonthIndex (1..N) or MonthLabel, Caregivers, Revenue, Expenses or Commission (you can provide raw fields)", type=["csv", "xlsx"])

st.sidebar.markdown("---")
st.sidebar.subheader("Or use model inputs")

start_caregivers = st.sidebar.number_input("Starting caregivers", min_value=1, max_value=5000, value=50, step=1)
monthly_growth_pct = st.sidebar.slider("Monthly caregiver growth (%)", 0.0, 30.0, 5.0, step=0.5)
avg_bookings_per_caregiver = st.sidebar.number_input("Avg bookings per caregiver / month", min_value=1, max_value=200, value=20, step=1)
avg_booking_value = st.sidebar.number_input("Avg booking value (GHS)", min_value=20, max_value=5000, value=150, step=5)
commission_pct = st.sidebar.slider("Platform commission (%)", 1, 30, 15, step=1)
marketing_spend = st.sidebar.number_input("Monthly marketing spend (GHS)", min_value=0, max_value=100000, value=3000, step=100)
fixed_costs = st.sidebar.number_input("Monthly fixed costs (GHS)", min_value=0, max_value=200000, value=20000, step=500)
months = st.sidebar.slider("Projection months", 6, 60, 24, step=1)

st.sidebar.markdown("**Advanced:** animation speed")
anim_speed = st.sidebar.slider("Animation frame duration (ms)", 200, 2000, 700, step=50)

# ----------------------
# Data: use uploaded file if provided, else simulated projection
# ----------------------
if upload:
    user_df = csv_excel_to_df(upload)
    if user_df is None:
        st.stop()
    # Try to normalize columns
    df = user_df.copy()
    # If user provided MonthIndex numeric, use it; else create MonthIndex from row order
    if "MonthIndex" not in df.columns:
        df.insert(0, "MonthIndex", range(1, len(df) + 1))
    if "MonthLabel" not in df.columns:
        df["MonthLabel"] = df["MonthIndex"].apply(lambda x: f"Month {int(x)}")
    # Ensure required columns exist
    if "Caregivers" not in df.columns:
        st.error("Uploaded file must include a 'Caregivers' column (or provide data compatible).")
        st.stop()
    # If Commission not present but Revenue present, compute Commission using commission_pct
    if "Commission" not in df.columns:
        if "Revenue" in df.columns:
            df["Commission"] = df["Revenue"] * (commission_pct / 100.0)
        else:
            # attempt estimate from Caregivers * avg bookings * booking value
            df["Bookings"] = df["Caregivers"] * avg_bookings_per_caregiver
            df["Revenue"] = df["Bookings"] * avg_booking_value
            df["Commission"] = df["Revenue"] * (commission_pct / 100.0)
    # Compute NetProfit and CumulativeProfit if missing
    if "NetProfit" not in df.columns:
        df["NetProfit"] = df["Commission"] - fixed_costs - marketing_spend
    if "CumulativeProfit" not in df.columns:
        df["CumulativeProfit"] = df["NetProfit"].cumsum()
    # finalize df with required columns
    df = df.rename(columns={c: c for c in df.columns})
    # Ensure MonthIndex integer
    df["MonthIndex"] = df["MonthIndex"].astype(int)
    df = df.sort_values("MonthIndex").reset_index(drop=True)
else:
    df = prepare_projection_from_inputs(
        start_caregivers,
        monthly_growth_pct / 100.0,
        avg_bookings_per_caregiver,
        avg_booking_value,
        commission_pct / 100.0,
        marketing_spend,
        months,
        fixed_costs
    )

# ----------------------
# Key metrics & ROI alert
# ----------------------
st.markdown("## Key Metrics")
col1, col2, col3, col4 = st.columns(4)
initial_investment = st.number_input("Initial investment (GHS) — affects break-even", min_value=0, value=200000, step=1000)

total_commission_sum = df["Commission"].sum() if "Commission" in df.columns else df["Commission"].sum()
total_net = df["NetProfit"].sum()
latest_cumulative = df["CumulativeProfit"].iloc[-1] if "CumulativeProfit" in df.columns else df["NetProfit"].cumsum().iloc[-1]
# break-even detection
be_month = next((int(r["MonthIndex"]) for _, r in df.iterrows() if r["CumulativeProfit"] >= initial_investment), None)

col1.metric("Projection months", f"{df['MonthIndex'].max()}")
col2.metric("Projected total commission (GHS)", f"{df['Commission'].sum():,.0f}")
col3.metric("Projected total net profit (GHS)", f"{df['NetProfit'].sum():,.0f}")
col4.metric("Cumulative at end (GHS)", f"{df['CumulativeProfit'].iloc[-1] - initial_investment:,.0f}")

if be_month:
    st.success(f"✅ Break-even at month {be_month}.")
else:
    st.info("⚠ Break-even not reached within this projection.")

# ----------------------
# Animated charts — three small charts side-by-side
# ----------------------
st.markdown("## Animated Story — Growth, Commission, Cumulative Profit")
c1, c2, c3 = st.columns(3)

# Prepare data for animation: Plotly needs a frame column
anim_df = df.copy()
# For safety, ensure columns used exist
if "Caregivers" not in anim_df.columns:
    anim_df["Caregivers"] = np.round(anim_df["Bookings"] / max(1, avg_bookings_per_caregiver)).astype(int)
if "Commission" not in anim_df.columns:
    anim_df["Commission"] = anim_df["Revenue"] * (commission_pct / 100.0)
if "CumulativeProfit" not in anim_df.columns:
    anim_df["CumulativeProfit"] = anim_df["NetProfit"].cumsum()

# Chart 1: Caregiver growth (animated)
with c1:
    fig_cg = px.line(anim_df, x="MonthIndex", y="Caregivers", title="Caregiver Growth",
                     labels={"MonthIndex": "Month", "Caregivers": "Caregivers"})
    fig_cg.update_traces(mode="lines+markers")
    fig_cg.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=10))
    # add animation frames by plotting as scatter with animation_frame
    scatter_cg = px.scatter(anim_df, x="MonthIndex", y="Caregivers",
                            animation_frame="MonthIndex", size="Caregivers",
                            range_x=[anim_df["MonthIndex"].min()-0.5, anim_df["MonthIndex"].max()+0.5],
                            range_y=[0, max(anim_df["Caregivers"].max()*1.2, 5)])
    scatter_cg.update_layout(height=300, showlegend=False)
    # sync frame duration
    scatter_cg.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = anim_speed
    st.plotly_chart(scatter_cg, use_container_width=True)

# Chart 2: Commission income (animated)
with c2:
    scatter_comm = px.bar(anim_df, x="MonthIndex", y="Commission", animation_frame="MonthIndex",
                         range_y=[0, max(anim_df["Commission"].max()*1.2, 100)], labels={"Commission": "Commission (GHS)"})
    scatter_comm.update_layout(title="Commission Income (animated)", height=300, margin=dict(t=40, b=20, l=20, r=10))
    scatter_comm.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = anim_speed
    st.plotly_chart(scatter_comm, use_container_width=True)

# Chart 3: Cumulative profit (animated) with break-even marker
with c3:
    anim_df["CumulativeForPlot"] = anim_df["CumulativeProfit"] - initial_investment
    scatter_cumu = px.line(anim_df, x="MonthIndex", y="CumulativeForPlot", labels={"CumulativeForPlot": "Cumulative (GHS)"},
                           title="Cumulative Profit (after initial investment)", markers=True)
    # Create frames by plotting scatter with animation_frame
    scatter_cumu2 = px.scatter(anim_df, x="MonthIndex", y="CumulativeForPlot", animation_frame="MonthIndex",
                               range_y=[min(anim_df["CumulativeForPlot"].min()*1.2, -1000),
                                        max(anim_df["CumulativeForPlot"].max()*1.2, 1000)])
    scatter_cumu2.update_layout(height=300, showlegend=False)
    # add horizontal zero line annotation: can't animate annotation, but we'll show line in layout
    scatter_cumu2.add_hline(y=0, line_dash="dash", line_color="red")
    scatter_cumu2.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = anim_speed
    st.plotly_chart(scatter_cumu2, use_container_width=True)

# ----------------------
# Table & download
# ----------------------
st.markdown("## Projection Table")
display_df = df[["MonthIndex", "MonthLabel", "Caregivers", "Bookings", "Revenue", "Commission", "NetProfit", "CumulativeProfit"]].copy()
display_df = display_df.rename(columns={
    "MonthIndex": "Month",
    "MonthLabel": "Label",
    "NetProfit": "NetProfit (GHS)",
    "CumulativeProfit": "CumulativeProfit (GHS)",
    "Commission": "Commission (GHS)",
    "Revenue": "Revenue (GHS)"
})
st.dataframe(display_df.style.format({
    "Revenue (GHS)": "{:,.0f}",
    "Commission (GHS)": "{:,.0f}",
    "NetProfit (GHS)": "{:,.0f}",
    "CumulativeProfit (GHS)": "{:,.0f}"
}), height=300)

# Download Excel
excel_bytes = df_to_excel_bytes({"Projection": display_df})
st.download_button("📥 Download Excel report", excel_bytes, file_name="fylia_projection_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.markdown("---")
st.markdown("**Notes:** Upload a CSV/Excel with MonthIndex (1..N) and Caregivers/Revenue if you want to visualize your real dataset. Animation uses MonthIndex as frames. The model uses a simple marketing uplift heuristic — you can replace with real campaign conversion lifts for more accuracy.")
