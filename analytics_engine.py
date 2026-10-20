import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

class AnalyticsEngine:
    """
    Core engine for retail data analytics, KPIs, and machine learning models.
    Separates computation from UI for modularity and testing.
    """

    def __init__(self, dataframe):
        self.df = dataframe
        # Ensure we work with copies to avoid SettingWithCopy warnings
        self.df = self.df.copy()

    # --- KPI CALCULATIONS ---
    def calculate_kpis(self):
        """Calculates essential and advanced business metrics."""
        # Basic KPIs
        total_revenue = self.df['payment_value'].sum()
        total_orders = self.df['order_id'].nunique()
        total_customers = self.df['customer_unique_id'].nunique()
        aov = self.df.groupby('order_id')['payment_value'].sum().mean()

        # Advanced KPIs
        # 1. Repeat Purchase Rate: customers with more than 1 unique order
        orders_per_customer = self.df.groupby('customer_unique_id')['order_id'].nunique()
        repeat_customers = (orders_per_customer > 1).sum()
        repeat_purchase_rate = (repeat_customers / total_customers) * 100 if total_customers > 0 else 0

        # 2. Customer Lifetime Value (CLV) - Simple approximation
        # CLV = (Avg Order Value * Purchase Frequency)
        # Here we use: Total Revenue / Total Unique Customers
        clv = total_revenue / total_customers if total_customers > 0 else 0

        # 3. Retention Rate (approximate for snapshot data)
        # Using "active" customers in the last 3 months vs total
        max_date = self.df['order_purchase_timestamp'].max()
        cutoff_date = max_date - timedelta(days=90)
        recent_customers = self.df[self.df['order_purchase_timestamp'] >= cutoff_date]['customer_unique_id'].nunique()
        retention_rate = (recent_customers / total_customers) * 100 if total_customers > 0 else 0

        return {
            "total_revenue": total_revenue,
            "total_orders": total_orders,
            "total_customers": total_customers,
            "aov": aov,
            "repeat_purchase_rate": repeat_purchase_rate,
            "clv": clv,
            "retention_rate": retention_rate
        }

    # --- SEGMENTATION ---
    def get_elbow_data(self, rfm_df, max_k=10):
        """Calculates inertia for the Elbow Method."""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(rfm_df[['recency', 'frequency', 'monetary']])
        
        inertia = []
        k_range = range(1, max_k + 1)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            inertia.append(kmeans.inertia_)
            
        return k_range, inertia

    def perform_segmentation(self, rfm_df, k=4):
        """Performs KMeans clustering and labels segments logically."""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(rfm_df[['recency', 'frequency', 'monetary']])
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        rfm_df['cluster'] = kmeans.fit_predict(scaled_data)
        
        # Robust labeling based on cluster means
        # We calculate the mean 'monetary' for each cluster to rank them
        cluster_ranking = rfm_df.groupby('cluster')['monetary'].mean().sort_values(ascending=False).index
        
        # Mapping labels based on rank (High Monetary = Champions)
        label_map = {
            cluster_ranking[0]: "Champions (High Value)",
            cluster_ranking[1]: "Loyal Customers",
            cluster_ranking[2]: "At Risk",
            cluster_ranking[3] if len(cluster_ranking) > 3 else cluster_ranking[-1]: "Hibernating"
        }
        
        # Fill remaining clusters if K > 4
        for i in range(k):
            if i not in label_map:
                label_map[i] = f"Segment {i}"
                
        rfm_df['segment'] = rfm_df['cluster'].map(label_map)
        return rfm_df

    # --- FORECASTING ---
    def generate_forecast(self, window=7):
        """Generates simple moving average forecast with trend detection."""
        daily_sales = self.df.set_index('order_purchase_timestamp').resample('D')['payment_value'].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
        
        daily_sales['y_ma'] = daily_sales['y'].rolling(window=window).mean()
        daily_sales['y_std'] = daily_sales['y'].rolling(window=window).std()
        
        # Trend detection
        last_val = daily_sales['y_ma'].iloc[-1]
        prev_val = daily_sales['y_ma'].iloc[-(window+1)] if len(daily_sales) > window else daily_sales['y_ma'].iloc[0]
        trend_pct = ((last_val - prev_val) / prev_val * 100) if prev_val != 0 else 0
        
        return daily_sales, trend_pct

    # --- PROBLEM DETECTION ---
    def detect_problems(self):
        """Identifies operational bottlenecks and quality issues."""
        # 1. Delayed Deliveries (Top 5 categories with highest average delivery time)
        delays = self.df.groupby('category')['delivery_time'].mean().sort_values(ascending=False).head(5).to_dict()
        
        # 2. Quality Issues (Top 5 categories with lowest average review score)
        scores = self.df.groupby('category')['review_score'].mean().sort_values(ascending=True).head(5).to_dict()
        
        # 3. Geography issues (Top 5 states with most delays)
        state_delays = self.df.groupby('customer_state')['delivery_time'].mean().sort_values(ascending=False).head(5).to_dict()
        
        return {
            "category_delays": delays,
            "category_scores": scores,
            "state_delays": state_delays
        }

    # --- STORYTELLING & INSIGHTS ---
    def get_business_insights(self, kpis, problems):
        """Translates data into actionable business language."""
        insights = []
        
        # Revenue insight
        if kpis['total_revenue'] > 50000:
            insights.append("Revenue performance is robust. Focus on high-value segment retention.")
        else:
            insights.append("Revenue shows room for growth. Consider cross-selling strategies in top categories.")
            
        # Retention insight
        if kpis['repeat_purchase_rate'] < 10:
            insights.append("Low repeat purchase rate detected. Loyalty programs or email marketing could boost retention.")
            
        # Quality/Operations insight
        worst_cat = list(problems['category_scores'].keys())[0] if problems['category_scores'] else "N/A"
        insights.append(f"Operational Alert: '{worst_cat}' has the lowest customer satisfaction. Investigation into suppliers recommended.")
        
        # Delay insight
        worst_state = list(problems['state_delays'].keys())[0] if problems['state_delays'] else "N/A"
        insights.append(f"Logistics Alert: Significant delivery delays observed in {worst_state}. Reviewing regional carrier performance is advised.")
        
        return insights
