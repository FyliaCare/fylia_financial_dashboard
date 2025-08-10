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
    """Read uploaded CSV/XLSX into DataFrame; expects MonthIndex, Caregivers, Revenue, Expenses or Commission fields."""
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

upload = st.sidebar.file_uploader("Upload monthly data (CSV or XLSX). Required columns: MonthIndex, Caregivers, Revenue, Expenses or Commission", type=["csv", "xlsx"])

st.sidebar.markdown("---")
st.sidebar.subheader("Or use model inputs")

start_caregivers = st.sidebar.number_input("Starting caregivers", min_value=1, max_value=5000, value=50, step=1)
monthly_growth_pct = st.sidebar.slider("Monthly caregiver growth (%)", 0.0, 30.0, 5.0, step=0.5)
avg_bookings_per_caregiver = st.sidebar.number_input("Avg bookings per caregiver / month", min_value=1, max_value=200, value=20, step=1)
avg_booking_value = st.sidebar.number_input("Avg booking value (GHS)", min_value=20, max_value=5000, value=150, step=5)
commission_pct = st.sidebar.slider("Platform commission (%)", 1, 30, 15, step=1)
marketing_spend = st.sidebar.number_input("Monthly marketing spend (GHS)", min_value=0, max_value=100000, value=3000, step=100)
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
    df = user_df.copy()
    if "MonthIndex" not in df.columns:
        df.insert(0, "MonthIndex", range(1, len(df) + 1))
    if "MonthLabel" not in df.columns:
        df["MonthLabel"] = df["MonthIndex"].apply(lambda x: f"Month {int(x)}")
    if "Caregivers" not in df.columns:
        st.error("Uploaded file must include a 'Caregivers' column.")
        st.stop()
    if "Commission" not in df.columns:
        if "Revenue" in df.columns:
            df["Commission"] = df["Revenue"] * (commission_pct / 100.0)
        else:
            df["Bookings"] = df["Caregivers"] * avg_bookings_per_caregiver
            df["Revenue"] = df["Bookings"] * avg_booking_value
            df["Commission"] = df["Revenue"] * (commission_pct / 100.0)
    if "NetProfit" not in df.columns:
        # We'll get fixed_costs from below input, but to not break flow, initialize with zero here.
        df["NetProfit"] = df["Commission"] - marketing_spend
    if "CumulativeProfit" not in df.columns:
        df["CumulativeProfit"] = df["NetProfit"].cumsum()
    df = df.rename(columns={c: c for c in df.columns})
    df["MonthIndex"] = df["MonthIndex"].astype(int)
    df = df.sort_values("MonthIndex").reset_index(drop=True)
else:
    # Temporarily fixed_costs = 0 here; will be updated after user inputs below
    fixed_costs = 0
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
# Animated charts — three small charts side-by-side
# ----------------------
st.markdown("## Animated Story — Growth, Commission, Cumulative Profit")
c1, c2, c3 = st.columns(3)

anim_df = df.copy()
if "Caregivers" not in anim_df.columns:
    anim_df["Caregivers"] = np.round(anim_df["Bookings"] / max(1, avg_bookings_per_caregiver)).astype(int)
if "Commission" not in anim_df.columns:
    anim_df["Commission"] = anim_df["Revenue"] * (commission_pct / 100.0)
if "CumulativeProfit" not in anim_df.columns:
    anim_df["CumulativeProfit"] = anim_df["NetProfit"].cumsum()

with c1:
    fig_cg = px.line(anim_df, x="MonthIndex", y="Caregivers", title="Caregiver Growth",
                     labels={"MonthIndex": "Month", "Caregivers": "Caregivers"})
    fig_cg.update_traces(mode="lines+markers")
    fig_cg.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=10))
    scatter_cg = px.scatter(anim_df, x="MonthIndex", y="Caregivers",
                            animation_frame="MonthIndex", size="Caregivers",
                            range_x=[anim_df["MonthIndex"].min()-0.5, anim_df["MonthIndex"].max()+0.5],
                            range_y=[0, max(anim_df["Caregivers"].max()*1.2, 5)])
    scatter_cg.update_layout(height=300, showlegend=False)
    scatter_cg.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = anim_speed
    st.plotly_chart(scatter_cg, use_container_width=True)

with c2:
    scatter_comm = px.bar(anim_df, x="MonthIndex", y="Commission", animation_frame="MonthIndex",
                         range_y=[0, max(anim_df["Commission"].max()*1.2, 100)], labels={"Commission": "Commission (GHS)"})
    scatter_comm.update_layout(title="Commission Income (animated)", height=300, margin=dict(t=40, b=20, l=20, r=10))
    scatter_comm.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = anim_speed
    st.plotly_chart(scatter_comm, use_container_width=True)

with c3:
    anim_df["CumulativeForPlot"] = anim_df["CumulativeProfit"] - 0  # will update initial_investment below
    scatter_cumu2 = px.scatter(anim_df, x="MonthIndex", y="CumulativeForPlot", animation_frame="MonthIndex",
                               range_y=[min(anim_df["CumulativeForPlot"].min()*1.2, -1000),
                                        max(anim_df["CumulativeForPlot"].max()*1.2, 1000)])
    scatter_cumu2.update_layout(title="Cumulative Profit (after initial investment)", height=300, showlegend=False)
    scatter_cumu2.add_hline(y=0, line_dash="dash", line_color="red")
    scatter_cumu2.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = anim_speed
    st.plotly_chart(scatter_cumu2, use_container_width=True)

# ----------------------
# DETAILED COST INPUTS moved from sidebar to here, with flexible entry methods
# ----------------------
st.markdown("---")
st.markdown("## Detailed Cost Breakdown — Initial Investment & Monthly Fixed Costs")

# ----------------------
# INITIAL INVESTMENT ENTRY MODE
# ----------------------
st.subheader("Initial Investment — choose input method")

init_mode = st.radio(
    "How do you want to enter Initial Investment?",
    ("Enter full initial cost", "Use Fylia predefined breakdown", "Enter custom breakdown"),
    index=1,
    key="init_mode"
)

# Fylia predefined breakdown (as provided)
fylia_defaults = {
    "Product & engineering (MVP + 9 months enhancements)": 240000,
    "Ops & salaries (core team for 12 months)": 420000,
    "Marketing & CAC (launch + growth)": 240000,
    "Caregiver onboarding & training, QA": 96000,
    "Legal / regulatory / insurance": 48000,
    "Working capital / contingency": 156000
}

if init_mode == "Enter full initial cost":
    init_full = st.number_input("Enter total initial investment (GHS)", min_value=0, value=183700, step=1000, key="init_full")
    initial_investment = init_full

elif init_mode == "Use Fylia predefined breakdown":
    st.markdown("**Fylia default allocation — edit if you want**")
    col_a, col_b = st.columns(2)
    init_vals = {}
    with col_a:
        init_vals["Product & engineering (MVP + 9 months enhancements)"] = st.number_input(
            "Product & engineering (GHS)", min_value=0, value=fylia_defaults["Product & engineering (MVP + 9 months enhancements)"], step=1000, key="init_prod"
        )
        init_vals["Marketing & CAC (GHS)"] = st.number_input(
            "Marketing & CAC (GHS)", min_value=0, value=fylia_defaults["Marketing & CAC (launch + growth)"], step=1000, key="init_marketing_main"
        )
        init_vals["Legal / regulatory / insurance (GHS)"] = st.number_input(
            "Legal / regulatory / insurance (GHS)", min_value=0, value=fylia_defaults["Legal / regulatory / insurance"], step=1000, key="init_legal"
        )
    with col_b:
        init_vals["Ops & salaries (GHS)"] = st.number_input(
            "Ops & salaries (GHS)", min_value=0, value=fylia_defaults["Ops & salaries (core team for 12 months)"], step=1000, key="init_ops"
        )
        init_vals["Caregiver onboarding & training (GHS)"] = st.number_input(
            "Caregiver onboarding & training (GHS)", min_value=0, value=fylia_defaults["Caregiver onboarding & training, QA"], step=1000, key="init_cg"
        )
        init_vals["Working capital / contingency (GHS)"] = st.number_input(
            "Working capital / contingency (GHS)", min_value=0, value=fylia_defaults["Working capital / contingency"], step=1000, key="init_wc"
        )
    initial_investment = sum(init_vals.values())

else:  # custom breakdown
    st.markdown("**Enter your custom initial cost categories**")
    custom_initial = {}
    # provide 6 editable lines but user can set to 0 if unused
    for i in range(1, 7):
        label = st.text_input(f"Custom initial category {i} name", value=f"Initial item {i}", key=f"cust_init_name_{i}")
        amount = st.number_input(f"Cost for '{label}' (GHS)", min_value=0, value=0, step=1000, key=f"cust_init_val_{i}")
        custom_initial[label] = amount
    initial_investment = sum(custom_initial.values())

st.markdown(f"### Total Initial Investment: GHS {initial_investment:,.0f}")

# ----------------------
# MONTHLY FIXED COSTS ENTRY MODE
# ----------------------
st.subheader("Monthly Fixed Costs — choose input method")

month_mode = st.radio(
    "How do you want to enter Monthly Fixed Costs?",
    ("Enter full monthly fixed cost", "Use detailed monthly inputs", "Enter custom monthly breakdown"),
    index=1,
    key="month_mode"
)

if month_mode == "Enter full monthly fixed cost":
    monthly_full = st.number_input("Enter total monthly fixed costs (GHS)", min_value=0, value=18000, step=500, key="monthly_full")
    # map to original variable names for compatibility
    monthly_salaries = monthly_full * 0.6  # heuristic split for display
    monthly_rent = monthly_full * 0.2
    monthly_maintenance = monthly_full * 0.1
    monthly_insurance = monthly_full * 0.05
    monthly_other = monthly_full * 0.05
    fixed_costs = monthly_full

elif month_mode == "Use detailed monthly inputs":
    col_month_1, col_month_2, col_month_3 = st.columns(3)
    with col_month_1:
        monthly_salaries = st.number_input("Salaries & Wages", min_value=0, value=10000, step=500, key="monthly_salaries")
        monthly_rent = st.number_input("Rent & Utilities", min_value=0, value=5000, step=500, key="monthly_rent")
    with col_month_2:
        monthly_maintenance = st.number_input("Maintenance & Supplies", min_value=0, value=3000, step=500, key="monthly_maintenance")
        monthly_insurance = st.number_input("Insurance & Fees", min_value=0, value=1000, step=500, key="monthly_insurance")
    with col_month_3:
        monthly_other = st.number_input("Other Monthly Costs", min_value=0, value=1000, step=500, key="monthly_other")
    fixed_costs = monthly_salaries + monthly_rent + monthly_maintenance + monthly_insurance + monthly_other

else:  # custom monthly breakdown
    st.markdown("**Enter your custom monthly categories**")
    custom_monthly = {}
    for i in range(1, 7):
        label = st.text_input(f"Custom monthly category {i} name", value=f"Monthly item {i}", key=f"cust_month_name_{i}")
        amount = st.number_input(f"Amount for '{label}' (GHS)", min_value=0, value=0, step=100, key=f"cust_month_val_{i}")
        custom_monthly[label] = amount
    fixed_costs = sum(custom_monthly.values())
    # assign friendly names for display (may be 0)
    monthly_salaries = custom_monthly.get(list(custom_monthly.keys())[0], 0)
    monthly_rent = custom_monthly.get(list(custom_monthly.keys())[1], 0) if len(custom_monthly) > 1 else 0
    monthly_maintenance = custom_monthly.get(list(custom_monthly.keys())[2], 0) if len(custom_monthly) > 2 else 0
    monthly_insurance = custom_monthly.get(list(custom_monthly.keys())[3], 0) if len(custom_monthly) > 3 else 0
    monthly_other = sum(list(custom_monthly.values())[4:]) if len(custom_monthly) > 4 else 0

st.markdown(f"### Total Monthly Fixed Costs: GHS {fixed_costs:,.0f}")

# ----------------------
# Now update projections with real fixed_costs and initial_investment (only if no upload)
# ----------------------
if not upload:
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

# Update cumulative profit to subtract initial investment for charts & alerts
df["CumulativeProfit"] = df["NetProfit"].cumsum() - initial_investment

# ----------------------
# Key metrics & ROI alert updated with new initial_investment and fixed_costs
# ----------------------
st.markdown("---")
st.markdown("## Key Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Projection months", f"{df['MonthIndex'].max()}")
col2.metric("Projected total commission (GHS)", f"{df['Commission'].sum():,.0f}")
col3.metric("Projected total net profit (GHS)", f"{df['NetProfit'].sum():,.0f}")
col4.metric("Cumulative at end (GHS)", f"{df['CumulativeProfit'].iloc[-1]:,.0f}")

# Break-even detection
be_month = next((int(r["MonthIndex"]) for _, r in df.iterrows() if r["CumulativeProfit"] >= 0), None)
if be_month:
    st.success(f"✅ Break-even at month {be_month}.")
else:
    st.info("⚠ Break-even not reached within this projection.")

# ROI detection (100% return = cumulative profit = initial investment)
roi_month = next((int(r["MonthIndex"]) for _, r in df.iterrows() if r["CumulativeProfit"] >= initial_investment), None)
if roi_month:
    st.success(f"🎉 ROI (100%) achieved at month {roi_month}.")
else:
    st.info("⚠ ROI (100%) not reached within this projection.")

# Warning on negative cash flow months
negative_cashflow_months = df[df["NetProfit"] < 0]["MonthIndex"].tolist()
if negative_cashflow_months:
    st.warning(f"⚠ Negative net profit projected in months: {', '.join(map(str, negative_cashflow_months))}")

# ----------------------
# Projection Table
# ----------------------
st.markdown("---")
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

# ----------------------
# Download Excel
# ----------------------
excel_bytes = df_to_excel_bytes({"Projection": display_df})
st.download_button("📥 Download Excel report", excel_bytes, file_name="fylia_projection_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ----------------------
# Notes
# ----------------------
st.markdown("---")
st.markdown("""
**Notes:**

- Upload a CSV/Excel with columns: MonthIndex (1..N), Caregivers, Revenue, Expenses or Commission to visualize your real dataset.
- Animation uses MonthIndex as frames.
- Marketing uplift is a simple heuristic; replace with real campaign data for more accuracy.
- Initial and monthly costs breakdown inputs are above the projection table for detailed cost control.
""")
