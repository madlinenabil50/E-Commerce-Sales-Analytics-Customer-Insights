import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class VizManager:
    """Manager for generating Plotly visualizations to separate UI logic from business logic."""
    
    def __init__(self):
        self.template = 'plotly_white'
        self.color_sequence = px.colors.qualitative.Bold

    def plot_sales_trend(self, df):
        # Monthly Sales Velocity
        # Use ME for Month End to avoid warnings in newer pandas
        monthly_sales = df.set_index('order_purchase_timestamp').resample('ME')['payment_value'].sum().reset_index()
        fig = px.area(monthly_sales, x='order_purchase_timestamp', y='payment_value',
                      title="Revenue Over Time",
                      labels={'order_purchase_timestamp': 'Date', 'payment_value': 'Revenue ($)'},
                      template=self.template)
        fig.update_traces(line_color='#4f46e5', fillcolor='rgba(79, 70, 229, 0.2)')
        return fig

    def plot_top_categories(self, df, top_n=8):
        cat_revenue = df.groupby('category')['payment_value'].sum().sort_values(ascending=False).head(top_n).reset_index()
        fig = px.bar(cat_revenue, x='payment_value', y='category', orientation='h',
                     title=f"Top {top_n} Categories by Revenue",
                     labels={'payment_value': 'Revenue ($)', 'category': 'Category'},
                     template=self.template,
                     color='payment_value', color_continuous_scale='Blues')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        return fig

    def plot_delivery_dist(self, df):
        fig = px.histogram(df, x='delivery_time', nbins=50,
                           title="Delivery Time Distribution (Days)",
                           labels={'delivery_time': 'Days', 'count': 'Number of Orders'},
                           template=self.template,
                           color_discrete_sequence=['#10b981'])
        return fig

    def plot_segments(self, rfm_clustered):
        seg_counts = rfm_clustered['segment'].value_counts().reset_index()
        seg_counts.columns = ['segment', 'count']
        fig = px.pie(seg_counts, names='segment', values='count', hole=0.4,
                     title="Customer Segments Distribution",
                     color_discrete_sequence=self.color_sequence)
        return fig

    def plot_forecast(self, forecast_df):
        fig = go.Figure()
        
        # Actual sales
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['y'], 
                                 mode='lines', name='Actual Daily Sales',
                                 line=dict(color='lightgray', width=1)))
        
        # Moving Average
        fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['y_ma'], 
                                 mode='lines', name='Trend (Moving Average)',
                                 line=dict(color='#4f46e5', width=3)))
                                 
        fig.update_layout(title="Sales Demand Forecast (Moving Average)",
                          xaxis_title="Date",
                          yaxis_title="Revenue ($)",
                          template=self.template,
                          hovermode="x unified")
        return fig
