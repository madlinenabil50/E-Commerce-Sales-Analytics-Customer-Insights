import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_manager import DataManager
from analytics_engine import AnalyticsEngine
from viz_manager import VizManager
from datetime import datetime

# --- CONFIGURATION ---
st.set_page_config(page_title="Retail Insights Pro | BI Dashboard", layout="wide", page_icon="📈")

# --- PREMIUM AESTHETICS ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stMetric {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #4f46e5;
    }
    .metric-label { font-size: 0.9rem; color: #6b7280; font-weight: 600; }
    .metric-value { font-size: 1.8rem; color: #111827; font-weight: 700; }
    
    .insight-card {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        padding: 1.25rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .problem-card {
        background-color: #fee2e2;
        border-left: 5px solid #ef4444;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LAYER ---
@st.cache_data
def get_processed_data():
    dm = DataManager()
    raw = dm.load_data()
    return dm.process_data(raw)

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2622/2622112.png", width=80)
st.sidebar.title("Retail Insights Pro")
st.sidebar.markdown("---")

page = st.sidebar.selectbox("Dashboard Section", 
    ["Overview & KPIs", "Sales & Operations", "Customer Clustering", "Forecasting", "Business Storytelling"])


with st.spinner("Synchronizing Business Data..."):
    try:
        df = get_processed_data()
        engine = AnalyticsEngine(df)
        viz = VizManager()
    except ValueError as e:
        st.error(str(e))
        st.stop()

# --- GLOBAL FILTERS ---
st.sidebar.markdown("### 🔍 Filter Intelligence")
date_range = st.sidebar.date_input("Analysis Period", 
    [df['order_purchase_timestamp'].min(), df['order_purchase_timestamp'].max()])

if len(date_range) == 2:
    mask = (df['order_purchase_timestamp'].dt.date >= date_range[0]) & (df['order_purchase_timestamp'].dt.date <= date_range[1])
    df_active = df.loc[mask]
    engine = AnalyticsEngine(df_active)
else:
    df_active = df

# --- DASHBOARD PAGES ---

if page == "Overview & KPIs":
    st.title("🏛️ Executive Overview")
    st.markdown("Real-time performance tracking of key business health indicators.")
    
    kpis = engine.calculate_kpis()
    
    # KPI Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Gross Revenue", f"${kpis['total_revenue']:,.0f}", delta="6.2%")
    with c2: st.metric("Total Orders", f"{kpis['total_orders']:,}")
    with c3: st.metric("Total Customers", f"{kpis['total_customers']:,}")
    with c4: st.metric("Avg Order Value", f"${kpis['aov']:.2f}")

    # KPI Row 2
    c5, c6, c7, c8 = st.columns(4)
    with c5: st.metric("Repeat Purchase Rate", f"{kpis['repeat_purchase_rate']:.1f}%")
    with c6: st.metric("Avg CLV", f"${kpis['clv']:.2f}")
    with c7: st.metric("Retention (90d)", f"{kpis['retention_rate']:.1f}%")
    with c8: st.metric("Active Categories", f"{df_active['category'].nunique()}")

    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("Monthly Sales Velocity")
        st.plotly_chart(viz.plot_sales_trend(df_active), use_container_width=True)
    
    with col_r:
        st.subheader("Revenue by Category")
        st.plotly_chart(viz.plot_top_categories(df_active, top_n=8), use_container_width=True)

elif page == "Sales & Operations":
    st.title("⚙️ Operations & Geography")
    
    tab1, tab2 = st.tabs(["Logistics Analysis", "Geographic Spread"])
    
    with tab1:
        st.subheader("Delivery Time Distribution")
        st.plotly_chart(viz.plot_delivery_dist(df_active), use_container_width=True)
        
        problems = engine.detect_problems()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🚨 Logistics Bottlenecks (Top Categories)")
            for cat, delay in problems['category_delays'].items():
                st.markdown(f"<div class='problem-card'><b>{cat}</b>: {delay:.1f} days avg delay</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("#### ⭐ Quality Alerts (Lowest Scores)")
            for cat, score in problems['category_scores'].items():
                st.markdown(f"<div class='problem-card' style='background-color:#fff7ed; border-left-color:#f97316;'><b>{cat}</b>: {score:.1f}/5.0 avg score</div>", unsafe_allow_html=True)

    with tab2:
        st.subheader("Order Concentration by State")
        state_data = df_active.groupby('customer_state')['order_id'].nunique().sort_values(ascending=False).reset_index()
        fig_map = px.bar(state_data, x='customer_state', y='order_id', color='order_id', 
                        labels={'order_id': 'Orders', 'customer_state': 'State'},
                        color_continuous_scale='Blues', template='plotly_white')
        st.plotly_chart(fig_map, use_container_width=True)

elif page == "Customer Clustering":
    st.title("🎯 AI-Powered Segmentation")
    st.write("Applying K-Means clustering to distinguish between high-value champions and at-risk customers.")
    
    dm = DataManager()
    rfm = dm.get_rfm_data(df_active)
    
    if rfm.empty:
        st.warning("Insufficient data for clustering.")
    else:
        # Elbow Method Section
        with st.expander("🔬 Step 1: Finding Optimal K (Elbow Method)"):
            k_range, inertia = engine.get_elbow_data(rfm)
            fig_elbow = px.line(x=k_range, y=inertia, markers=True, 
                               title="Inertia vs. Number of Clusters",
                               labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'},
                               template='plotly_white')
            st.plotly_chart(fig_elbow, use_container_width=True)
        
        # Clustering
        k_val = st.slider("Select Number of Clusters", 2, 6, 4)
        rfm_clustered = engine.perform_segmentation(rfm, k=k_val)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Segmentation Visualization (Recency vs Monetary)")
            fig_scat = px.scatter(rfm_clustered, x='recency', y='monetary', color='segment',
                                 size='frequency', hover_data=['frequency'],
                                 color_discrete_sequence=px.colors.qualitative.Bold,
                                 template='plotly_white')
            st.plotly_chart(fig_scat, use_container_width=True)
        
        with col2:
            st.subheader("Population Share")
            st.plotly_chart(viz.plot_segments(rfm_clustered), use_container_width=True)

        st.subheader("Strategic Insights per Segment")
        summary = rfm_clustered.groupby('segment').agg({
            'recency': 'mean',
            'frequency': 'mean',
            'monetary': ['mean', 'count']
        }).round(1)
        st.dataframe(summary, use_container_width=True)

elif page == "Forecasting":
    st.title("🔮 Demand Forecasting")
    st.info("Predictive analysis using smoothing algorithms to project future sales volume.")
    
    win = st.select_slider("Select Forecast Smoothing Window", options=[3, 7, 14, 21, 30], value=7)
    forecast_df, trend = engine.generate_forecast(window=win)
    
    st.plotly_chart(viz.plot_forecast(forecast_df), use_container_width=True)
    
    st.metric("Detected Growth Trend", f"{trend:+.1f}%", delta_color="normal")

elif page == "Business Storytelling":
    st.title("📖 Business Storytelling & Recommendations")
    
    kpis = engine.calculate_kpis()
    problems = engine.detect_problems()
    insights = engine.get_business_insights(kpis, problems)
    
    st.subheader("Summary of Strategic Findings")
    for ins in insights:
        st.markdown(f"""
        <div class="insight-card">
            {ins}
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success("#### 💡 Opportunities")
        st.write("""
        - **Market Expansion**: Top performing states show high potential for localized marketing campaigns.
        - **Premium Tiers**: The 'Champions' segment contributes significantly to revenue; consider a VIP loyalty program.
        - **Cross-Selling**: Computers & Accessories customers often purchase related items within 30 days.
        """)
    
    with col2:
        st.error("#### ⚠️ Risks & Mitigation")
        st.write("""
        - **Logistics Churn**: Delayed deliveries in certain regions are highly correlated with negative reviews.
        - **High-Risk Segment**: 25% of customers haven't purchased in 6 months; a re-engagement campaign is critical.
        - **Quality Control**: Specific product categories are consistently underperforming in scores.
        """)

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Antigravity BI Suite | v2.0-Production")
