import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataManager:
    """Handles data ingestion, synthetic data generation, and feature engineering."""
    
    def __init__(self, data_path='.', sheet_mapping=None):
        self.data_path = data_path
        self.sheet_mapping = sheet_mapping or {}
        self.datasets = [
            'olist_customers_dataset', 
            'olist_geolocation_dataset',
            'olist_orders_dataset', 
            'olist_order_items_dataset', 
            'olist_products_dataset',
            'olist_order_payments_dataset', 
            'olist_order_reviews_dataset',
            'olist_sellers_dataset', 
            'product_category_name_translation'
        ]

    def load_data(self, sheet_name=None):
        """
        Loads all datasets. 
        Prioritizes real CSV/Excel files from the specified data_path. 
        """
        data_frames = {}
        found_files = []
        missing_files = []
        
        # 1. Check if data_path exists, fallback to current dir if not
        search_path = self.data_path
        if not os.path.exists(search_path):
            logger.warning(f"Specified path '{search_path}' not found. Searching in current directory instead.")
            search_path = '.'

        # 2. Attempt to load real data
        abs_search_path = os.path.abspath(search_path)
        logger.info(f"Scanning for real CSV/Excel data in: {abs_search_path}")
        
        for ds in self.datasets:
                loaded = False
                # Prioritize .csv, then .xlsx
                for ext in ['.csv', '.xlsx']:
                    path = os.path.join(search_path, f"{ds}{ext}")
                    if os.path.exists(path):
                        try:
                            if ext == '.csv':
                                # Multi-encoding attempt for robust CSV reading
                                for enc in ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1']:
                                    try:
                                        df = pd.read_csv(path, encoding=enc, low_memory=False)
                                        data_frames[ds] = df
                                        found_files.append(f"{ds}{ext}")
                                        logger.info(f"Loaded: {ds}{ext}")
                                        print(f"SUCCESS: Loaded {ds}{ext}")
                                        loaded = True
                                        break
                                    except (UnicodeDecodeError, pd.errors.ParserError):
                                        continue
                            else:
                                # Excel support
                                target_sheet = sheet_name if sheet_name is not None else self.sheet_mapping.get(ds, 0)
                                df = pd.read_excel(path, sheet_name=target_sheet)
                                data_frames[ds] = df
                                found_files.append(f"{ds}{ext}")
                                logger.info(f"Loaded: {ds}{ext} (Sheet: {target_sheet})")
                                print(f"SUCCESS: Loaded {ds}{ext}")
                                loaded = True
                            
                            if loaded:
                                break
                        except Exception as e:
                            logger.error(f"Error reading {path}: {str(e)}")
                
                if not loaded:
                    missing_files.append(ds)

        # 3. Final Check
        if not data_frames:
            error_msg = "No real datasets found. Please ensure the Olist CSV files are in the data directory."
            logger.error(error_msg)
            print(f"CRITICAL ERROR: {error_msg}")
            raise ValueError(error_msg)
        
        if missing_files:
            logger.warning(f"Some files were missing: {missing_files}. Pipeline may be incomplete.")
            print(f"WARNING: Missing {len(missing_files)} datasets.")
        else:
            logger.info("All real datasets loaded successfully.")
            print("SUCCESS: All datasets loaded from CSV/Excel.")

        return data_frames


    def process_data(self, dfs):
        """Cleans, merges, and engineer features."""
        # Safety check for required datasets to prevent pipeline crash
        required = ['olist_orders_dataset', 'olist_customers_dataset', 'olist_order_items_dataset', 
                    'olist_products_dataset', 'product_category_name_translation', 
                    'olist_order_payments_dataset', 'olist_order_reviews_dataset']
        
        missing = [r for r in required if r not in dfs]
        if missing:
            error_msg = f"CRITICAL ERROR: Missing required datasets for processing: {missing}. Please ensure all CSV files are present in the project directory."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Merge hierarchy
        orders = dfs['olist_orders_dataset'].copy()
        customers = dfs['olist_customers_dataset']
        items = dfs['olist_order_items_dataset']
        products = dfs['olist_products_dataset']
        translation = dfs['product_category_name_translation']
        payments = dfs['olist_order_payments_dataset']
        reviews = dfs['olist_order_reviews_dataset']
        
        # Convert dates
        date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date']
        for col in date_cols:
            orders[col] = pd.to_datetime(orders[col], errors='coerce')
        
        # 1. Merge Orders + Customers
        df = pd.merge(orders, customers, on='customer_id', how='left')
        
        # 2. Merge Items + Products + Translation
        items_prod = pd.merge(items, products, on='product_id', how='left')
        items_prod = pd.merge(items_prod, translation, on='product_category_name', how='left')
        
        # 3. Merge main with items
        df = pd.merge(df, items_prod, on='order_id', how='inner')
        
        # 4. Merge Payments & Reviews
        df = pd.merge(df, payments[['order_id', 'payment_value', 'payment_type']], on='order_id', how='left')
        df = pd.merge(df, reviews[['order_id', 'review_score']], on='order_id', how='left')
        
        # Feature Engineering: Delivery Time
        df['delivery_time'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
        
        # Handle outliers and negatives in delivery_time (vectorized)
        df.loc[df['delivery_time'] < 0, 'delivery_time'] = np.nan
        
        # Handle NAs in review_score
        df['review_score'] = df['review_score'].fillna(df['review_score'].mean())
        
        # Handle NAs in translation
        df['category'] = df['product_category_name_english'].fillna(df['product_category_name']).fillna('Unknown')
        
        # Ensure monetary values are positive (vectorized)
        df['payment_value'] = df['payment_value'].clip(lower=0)

        return df

    def get_rfm_data(self, df):
        """Calculates RFM metrics for segmentation."""
        now = df['order_purchase_timestamp'].max() + timedelta(days=1)
        
        rfm = df.groupby('customer_unique_id').agg({
            'order_purchase_timestamp': lambda x: (now - x.max()).days,
            'order_id': 'nunique',
            'payment_value': 'sum'
        }).rename(columns={
            'order_purchase_timestamp': 'recency',
            'order_id': 'frequency',
            'payment_value': 'monetary'
        })
        
        # Enhancement: Add Average Order Value (AOV)
        rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
        
        return rfm
