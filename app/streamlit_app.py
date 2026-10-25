import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MAROON = "#7A1F2E"
GOLD = "#C9A227"
TEAL = "#2F6F73"
TAUPE = "#8C7B6B"
TEXT = "#2B1A1A"
GOOD = "#2E7D5B"
WARN = "#D99A00"
BAD = "#D62828"
BAND_COLORS = {"Low": GOOD, "Medium": WARN, "High": BAD}

st.set_page_config(page_title="ChurnShield", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1300px;}
    div[data-testid="stMetric"] {
        background: #FFFDF8; border: 1px solid #E3D5BE;
        border-left: 5px solid #7A1F2E;
        padding: 16px 18px; border-radius: 12px;
    }
    div[data-testid="stMetricLabel"] {opacity: 0.8;}
    .hero {
        background: linear-gradient(120deg, #4A0E1C 0%, #7A1F2E 55%, #A63D4F 100%);
        padding: 28px 32px; border-radius: 16px; margin-bottom: 1.2rem;
        border-bottom: 5px solid #C9A227;
    }
    .hero-title {
        color: #FAF5EC; font-size: 2.4rem; font-weight: 700; margin: 0;
        font-family: Georgia, 'Times New Roman', serif;
    }
    .hero-sub {color: #F1E7D6; font-size: 1.05rem; margin: 0.4rem 0 0 0;}
    .profile-card {
        background: #FFFDF8; border: 1px solid #E3D5BE;
        border-left: 5px solid #7A1F2E; border-radius: 12px;
        padding: 8px 18px; margin-bottom: 0.5rem;
    }
    .profile-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 9px 0; border-bottom: 1px solid #EFE4D0; font-size: 1rem;
    }
    .profile-row:last-child {border-bottom: none;}
    .profile-row span {opacity: 0.75;}
    button[data-baseweb="tab"] {font-size: 1rem; padding: 10px 18px;}
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4A0E1C 0%, #6B1A2B 100%);
    }
    section[data-testid="stSidebar"] * {color: #F5EBDD !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_risk_table():
    df = pd.read_csv(ROOT / "data" / "risk_table.csv")
    df["Risk Band"] = pd.cut(
        df["Churn Probability"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)
    return df


@st.cache_data
def load_features():
    return pd.read_csv(ROOT / "data" / "test_features.csv")


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "churn_model.joblib")


def style(fig, height=380):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(size=13, color=TEXT),
    )
    return fig


risk = load_risk_table()

# ---------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🛡️ ChurnShield")
    st.caption("Churn prediction and retention decision support")
    st.divider()
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Data:** IBM Telco, 1,409 held-out customers")
    st.markdown("**Risk bands**")
    st.markdown("🟢 Low: under 30%\n\n🟡 Medium: 30 to 60%\n\n🔴 High: over 60%")
    st.divider()
    st.caption("Built with scikit-learn, SHAP, Streamlit and Plotly.")

# ------------------------------------------------------------------------- hero
st.markdown(
    """
    <div class="hero">
        <p class="hero-title">🛡️ ChurnShield</p>
        <p class="hero-sub">Find the customers who are about to leave, understand why,
        and decide where retention money is best spent.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_risk, tab_sim, tab_models, tab_explain = st.tabs(
    ["📊 Customer Risk", "💰 Retention Simulator", "🧪 Model Comparison", "🔍 Explain a Customer"]
)

# ---------------------------------------------------------------- Customer Risk
with tab_risk:
    high = risk[risk["Risk Band"] == "High"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(risk):,}")
    k2.metric("Average churn risk", f"{risk['Churn Probability'].mean():.1%}")
    k3.metric("High-risk customers", f"{len(high):,}")
    k4.metric("CLTV at high risk", f"${high['CLTV'].sum():,.0f}")

    st.write("")
    ch1, ch2 = st.columns([3, 2])
    with ch1:
        fig_hist = go.Figure(
            go.Histogram(x=risk["Churn Probability"] * 100, nbinsx=25, marker_color=MAROON)
        )
        fig_hist.update_layout(
            title="Distribution of churn risk",
            xaxis_title="Churn probability (%)",
            yaxis_title="Customers",
        )
        st.plotly_chart(style(fig_hist), width="stretch")
    with ch2:
        counts = (
            risk["Risk Band"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        )
        fig_pie = go.Figure(
            go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.6,
                sort=False,
                marker=dict(colors=[BAND_COLORS[b] for b in counts.index]),
            )
        )
        fig_pie.update_layout(title="Customers by risk band")
        st.plotly_chart(style(fig_pie), width="stretch")

    st.subheader("Customer risk table")
    f1, f2 = st.columns([1, 2])
    bands = f1.multiselect(
        "Risk band", ["High", "Medium", "Low"], default=["High", "Medium", "Low"]
    )
    search = f2.text_input("Search customer ID", placeholder="e.g. 3668-QPYBK")

    filtered = risk[risk["Risk Band"].isin(bands)]
    if search:
        filtered = filtered[
            filtered["CustomerID"].str.contains(search.strip(), case=False, na=False)
        ]

    view = filtered[
        ["CustomerID", "Risk Band", "Churn Probability", "CLTV", "Expected Benefit", "Target"]
    ].copy()
    view["Churn Probability"] = view["Churn Probability"] * 100
    view = view.rename(
        columns={"Churn Probability": "Churn Risk (%)", "Expected Benefit": "Expected Benefit ($)"}
    ).sort_values("Churn Risk (%)", ascending=False)

    st.dataframe(
        view,
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "Churn Risk (%)": st.column_config.ProgressColumn(
                "Churn Risk (%)", min_value=0, max_value=100, format="%.1f"
            ),
            "CLTV": st.column_config.NumberColumn("CLTV ($)", format="%d"),
            "Expected Benefit ($)": st.column_config.NumberColumn(format="%.0f"),
        },
    )
    st.caption(
        f"Showing {len(view):,} customers. Expected benefit uses the default assumptions "
        "(offer $300, 30% success). Change them in the Retention Simulator tab."
    )
    st.download_button(
        "⬇️ Download this list (CSV)",
        view.to_csv(index=False).encode("utf-8"),
        file_name="churnshield_risk_list.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------- Retention Simulator
with tab_sim:
    st.subheader("Retention cost-benefit simulator")
    st.write(
        "A customer is worth targeting when: churn probability × offer success rate × "
        "customer lifetime value − offer cost is above zero."
    )

    s1, s2 = st.columns(2)
    cost = s1.slider("Cost per retention offer ($)", 50, 1000, 300, step=10)
    success = s2.slider("Offer success rate", 0.05, 0.90, 0.30, step=0.05)

    sim = risk.copy()
    sim["Expected Benefit"] = sim["Churn Probability"] * success * sim["CLTV"] - cost
    sim["Target"] = sim["Expected Benefit"] > 0
    targeted = sim[sim["Target"]]

    total_cost = len(targeted) * cost
    net = targeted["Expected Benefit"].sum()
    roi = net / total_cost if total_cost > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers to target", f"{len(targeted):,}")
    m2.metric("Total offer cost", f"${total_cost:,.0f}")
    m3.metric("Expected net benefit", f"${net:,.0f}")
    m4.metric("Return on spend", f"{roi:.1f}x")

    if targeted.empty:
        st.warning("No customer clears the bar at these settings. Try a lower cost or a higher success rate.")
    else:
        ordered = sim.sort_values("Expected Benefit", ascending=False).reset_index(drop=True)
        ordered["Cumulative Net Benefit"] = ordered["Expected Benefit"].cumsum()
        peak = int(ordered["Cumulative Net Benefit"].idxmax())

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ordered.index + 1,
                y=ordered["Cumulative Net Benefit"],
                mode="lines",
                line=dict(color=MAROON, width=3),
                fill="tozeroy",
                fillcolor="rgba(122,31,46,0.12)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[peak + 1],
                y=[ordered.loc[peak, "Cumulative Net Benefit"]],
                mode="markers+text",
                marker=dict(size=13, color=GOLD, line=dict(color=MAROON, width=2)),
                text=[f"Best: {peak + 1:,} customers"],
                textposition="top center",
            )
        )
        fig.update_layout(
            title="Net benefit as you target more customers (highest benefit first)",
            xaxis_title="Customers targeted",
            yaxis_title="Cumulative expected net benefit ($)",
            showlegend=False,
        )
        st.plotly_chart(style(fig, 420), width="stretch")

        st.subheader("Customers worth targeting")
        out = targeted[["CustomerID", "Risk Band", "Churn Probability", "CLTV", "Expected Benefit"]].copy()
        out["Churn Probability"] = out["Churn Probability"] * 100
        out = out.rename(columns={"Churn Probability": "Churn Risk (%)"}).sort_values(
            "Expected Benefit", ascending=False
        )
        st.dataframe(
            out,
            hide_index=True,
            width="stretch",
            height=380,
            column_config={
                "Churn Risk (%)": st.column_config.ProgressColumn(
                    "Churn Risk (%)", min_value=0, max_value=100, format="%.1f"
                ),
                "Expected Benefit": st.column_config.NumberColumn("Expected Benefit ($)", format="%.0f"),
            },
        )

# ------------------------------------------------------------ Model Comparison
with tab_models:
    st.subheader("Model comparison (5-fold cross-validation)")
    comp = pd.read_csv(ROOT / "models" / "model_comparison.csv")

    st.dataframe(comp.style.format(precision=3), hide_index=True, width="stretch")

    fig_models = go.Figure()
    metric_colors = {"Recall": MAROON, "Precision": GOLD, "F1": TEAL, "ROC-AUC": TAUPE}
    for metric, color in metric_colors.items():
        fig_models.add_bar(name=metric, x=comp["Model"], y=comp[metric], marker_color=color)
    fig_models.update_layout(
        barmode="group",
        title="Scores by model",
        yaxis_title="Score",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(style(fig_models, 420), width="stretch")

    st.success(
        "Logistic Regression was selected as the final model: best recall, F1 and "
        "ROC-AUC, and its coefficients are easy to explain."
    )

# ----------------------------------------------------------- Explain a Customer
with tab_explain:
    st.subheader("Explain a customer's churn risk")
    features = load_features()
    model = load_model()

    customer_id = st.selectbox(
        "Choose a customer (highest risk first)",
        risk.sort_values("Churn Probability", ascending=False)["CustomerID"],
    )
    cust = risk.loc[risk["CustomerID"] == customer_id].iloc[0]
    prob = float(cust["Churn Probability"])
    band = cust["Risk Band"]

    row = features[features["CustomerID"] == customer_id].drop(columns="CustomerID")
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["model"]
    scaled = scaler.transform(row)[0]
    contrib = pd.Series(clf.coef_[0] * scaled, index=row.columns)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)
    top = top.sort_values()

    g1, g2 = st.columns([1, 2])
    with g1:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 44}},
                title={"text": f"Risk band: {band}"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": BAND_COLORS[band]},
                    "steps": [
                        {"range": [0, 30], "color": "#D5EADF"},
                        {"range": [30, 60], "color": "#F8E7B5"},
                        {"range": [60, 100], "color": "#F4C9C9"},
                    ],
                },
            )
        )
        st.plotly_chart(style(fig_gauge, 300), width="stretch")

        st.markdown("**Customer profile**")
        st.markdown(
            f"""
            <div class="profile-card">
                <div class="profile-row"><span>Tenure</span><b>{int(cust['Tenure Months'])} months</b></div>
                <div class="profile-row"><span>Monthly bill</span><b>${cust['Monthly Charges']:.2f}</b></div>
                <div class="profile-row"><span>CLTV</span><b>${cust['CLTV']:,.0f}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("CLTV = estimated lifetime value of this customer.")

    with g2:
        colors = [BAD if v > 0 else TEAL for v in top]
        fig_exp = go.Figure(
            go.Bar(x=top.values, y=top.index, orientation="h", marker_color=colors)
        )
        fig_exp.update_layout(
            title="Top factors (red raises risk, teal lowers it)",
            xaxis_title="Effect on churn score",
        )
        st.plotly_chart(style(fig_exp, 420), width="stretch")

    st.caption("Factor bars come from the logistic regression weights for this customer.")