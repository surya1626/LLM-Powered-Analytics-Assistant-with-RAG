import logging
import pandas as pd
import sqlite3
import os
import numpy as np
import sys
from config import DATA_DIR, DB_DIR

logger = logging.getLogger(__name__)

# Google Drive folder ID
GOOGLE_DRIVE_FOLDER_ID = "1OyiJwJGIIwB6Dh2JGAjBLPHkNGI32OOX"

CSV_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]

TABLE_MAP = {
    "olist_customers_dataset": "customers",
    "olist_geolocation_dataset": "geolocation",
    "olist_order_items_dataset": "order_items",
    "olist_order_payments_dataset": "payments",
    "olist_order_reviews_dataset": "order_reviews",
    "olist_orders_dataset": "orders",
    "olist_products_dataset": "products",
    "olist_sellers_dataset": "sellers",
    "product_category_name_translation": "category_translation",
}


def download_csv_files_from_gdrive(output_dir: str = None) -> bool:
    """
    Download CSV files from Google Drive folder.
    
    Returns:
        bool: True if files were downloaded or already exist, False if download failed
    """
    if output_dir is None:
        output_dir = str(DATA_DIR)
    
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if files already exist
    csv_files = list(output_path.glob("*.csv"))
    if len(csv_files) >= len(CSV_FILES):
        logger.info(f"✅ CSV files already exist in {output_dir}")
        return True
    
    logger.info(f"📁 Data directory: {output_path}")
    logger.info(f"⏳ Downloading {len(CSV_FILES)} CSV files from Google Drive...")
    
    try:
        import gdown
    except ImportError:
        logger.error("❌ gdown library not installed.")
        logger.error("   Install it with: pip install gdown")
        return False
    
    try:
        gdown.download_folder(
            id=GOOGLE_DRIVE_FOLDER_ID,
            output=output_dir,
            quiet=False,
            use_cookies=False
        )
        
        # Verify download
        csv_files = list(output_path.glob("*.csv"))
        logger.info(f"✅ Download completed! Found {len(csv_files)} CSV files:")
        for f in sorted(csv_files):
            logger.info(f"   • {f.name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        return False


class OlistDataLoader:
    def __init__(self, data_dir: str, db_path: str):
        self.data_dir = data_dir
        self.db_path = db_path
        os.makedirs(self.data_dir, exist_ok=True)
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)

    def load_csv(self, file_name: str) -> pd.DataFrame:
        path = os.path.join(self.data_dir, file_name)
        df = pd.read_csv(path, low_memory=False)
        logger.info(f"Loaded {file_name} | Shape: {df.shape}")
        return df

    def _parse_dates(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        return df

    def _preprocess_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        n0 = len(df)
        df = df.copy()
        df = df.dropna(subset=["customer_id", "customer_unique_id"])
        df = df.drop_duplicates(subset=["customer_id"])
        df["customer_state"] = df["customer_state"].str.strip().str.upper()
        logger.info(f"customers: {n0} → {len(df)} rows | city/state normalised")
        return df

    def _preprocess_sellers(self, df: pd.DataFrame) -> pd.DataFrame:
        n0 = len(df)
        df = df.copy()
        df = df.dropna(subset=["seller_id"])
        df = df.drop_duplicates(subset=["seller_id"])
        df["seller_state"] = df["seller_state"].str.strip().str.upper()
        logger.info(f"sellers: {n0} → {len(df)} rows | city/state normalised")
        return df

    def _preprocess_payments(self, df: pd.DataFrame) -> pd.DataFrame:
        n0 = len(df)
        df = df.copy()

        df = df.dropna(subset=["order_id"])
        df = df.drop_duplicates()

        df["payment_value"] = pd.to_numeric(df["payment_value"], errors="coerce").clip(lower=0)

        df["payment_installments"] = (
            pd.to_numeric(df["payment_installments"], errors="coerce")
            .fillna(1).clip(lower=1).astype(int)
        )

        df["payment_type"] = df["payment_type"].str.strip().str.lower()

        agg = df.groupby("order_id").agg(
            payment_type=("payment_type", "first"),
            payment_installments=("payment_installments", "max"),
            payment_value=("payment_value", "sum"),
            payment_sequential=("payment_sequential", "max"),
        ).reset_index()

        logger.info(f"payments: {n0} → {len(agg)} rows | values clipped, aggregated per order")
        return agg

    def clean_reviews(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(subset=["review_comment_message"])

        df["review_comment_message"] = (
            df["review_comment_message"].astype(str).str.strip()
        )
        df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
        df = df[df["review_score"].between(1, 5)]
        df["review_score"] = df["review_score"].astype(int)

        def _sentiment(score: int) -> str:
            if score >= 4:
                return "positive"
            if score == 3:
                return "neutral"
            return "negative"

        df["sentiment_label"] = df["review_score"].map(_sentiment)
        df["review_comment_title"] = (
            df["review_comment_title"]
            .astype(str)
            .str.strip()
            .replace({"nan": None, "": None})
        )

        df["has_review_text"] = df["review_comment_message"].notna()
        logger.info(f"Cleaned reviews | Shape: {df.shape}")
        return df

    def _preprocess_order_items(self, df: pd.DataFrame) -> pd.DataFrame:
        n0 = len(df)
        df = df.copy()

        df = df.dropna(subset=["order_id", "product_id", "seller_id"])
        df = df.drop_duplicates()

        df["price"] = df["price"].clip(lower=0)
        df["freight_value"] = df["freight_value"].clip(lower=0)
        df["item_total"] = (df["price"] + df["freight_value"]).round(2)

        df = self._parse_dates(df, ["shipping_limit_date"])

        logger.info(f"order_items: {n0} → {len(df)} rows | item_total added, negatives clipped")
        return df

    def _preprocess_orders(self, df: pd.DataFrame) -> pd.DataFrame:
        n0 = len(df)
        df = df.copy()

        date_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
        df = self._parse_dates(df, date_cols)

        df = df.dropna(subset=["order_id", "order_purchase_timestamp"])
        df = df.drop_duplicates(subset=["order_id"])

        df["order_year"] = df["order_purchase_timestamp"].dt.year
        df["order_month"] = df["order_purchase_timestamp"].dt.month
        df["order_weekday"] = df["order_purchase_timestamp"].dt.day_name()

        df["delivery_days"] = (
            (df["order_delivered_customer_date"] - df["order_purchase_timestamp"])
            .dt.total_seconds() / 86400
        ).round(1)

        df["estimated_days"] = (
            (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"])
            .dt.total_seconds() / 86400
        ).round(1)

        df["is_late_delivery"] = (
            df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
        ).where(df["order_delivered_customer_date"].notna(), other=None)

        mask_bad = df["delivery_days"].notna() & (
            (df["delivery_days"] < 0) | (df["delivery_days"] > 365)
        )
        df.loc[mask_bad, "delivery_days"] = np.nan

        df["order_status"] = df["order_status"].str.strip().str.lower()

        logger.info(f"orders: {n0} → {len(df)} rows | dates parsed, delivery_days + late_flag added")
        return df

    def load_all_to_sqlite(self):
        for fname in CSV_FILES:
            path = os.path.join(self.data_dir, fname)

            if not os.path.exists(path):
                logger.warning(f"Skipping missing file: {fname}")
                continue

            table_name = TABLE_MAP.get(fname.replace(".csv", ""), fname.replace(".csv", ""))
            df = self.load_csv(fname)

            # FIX 3: Apply all preprocessors, not just reviews
            if table_name == "orders":
                df = self._preprocess_orders(df)
            elif table_name == "order_items":
                df = self._preprocess_order_items(df)
            elif table_name == "order_reviews":
                df = self.clean_reviews(df)
            elif table_name == "customers":
                df = self._preprocess_customers(df)
            elif table_name == "sellers":
                df = self._preprocess_sellers(df)
            elif table_name == "payments":
                df = self._preprocess_payments(df)

            # FIX 4: Convert datetime columns to strings before writing to SQLite
            for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
                df[col] = df[col].astype(str).replace("NaT", None)

            # FIX 5: Convert boolean columns to int (SQLite has no bool type)
            for col in df.select_dtypes(include=["bool"]).columns:
                df[col] = df[col].astype("Int64")

            df.to_sql(table_name, self.conn, if_exists="replace", index=False)
            logger.info(f"Saved → {table_name} ({len(df)} rows)")

        self.create_analytics_view()

    def create_analytics_view(self):
        self.conn.execute("DROP VIEW IF EXISTS analytics_view")

        self.conn.execute("""
            CREATE VIEW analytics_view AS
            SELECT
                o.order_id,
                o.customer_id,
                o.order_status,
                o.order_purchase_timestamp,
                o.order_delivered_customer_date,
                o.order_estimated_delivery_date,
                o.order_year,
                o.order_month,
                o.order_weekday,
                o.delivery_days,
                o.estimated_days,
                o.is_late_delivery,
                oi.product_id,
                oi.seller_id,
                oi.price,
                oi.freight_value,
                oi.item_total,
                p.product_category_name,
                COALESCE(ct.product_category_name_english, p.product_category_name) AS category_english,
                s.seller_city,
                s.seller_state,
                c.customer_city,
                c.customer_state,
                pay.payment_type,
                pay.payment_value,
                pay.payment_installments,
                r.review_score,
                r.sentiment_label,
                r.review_comment_message,
                r.has_review_text
            FROM orders o
            LEFT JOIN order_items oi ON o.order_id = oi.order_id
            LEFT JOIN products p ON oi.product_id = p.product_id
            LEFT JOIN category_translation ct ON p.product_category_name = ct.product_category_name
            LEFT JOIN sellers s ON oi.seller_id = s.seller_id
            LEFT JOIN customers c ON o.customer_id = c.customer_id
            LEFT JOIN payments pay ON o.order_id = pay.order_id
            LEFT JOIN order_reviews r ON o.order_id = r.order_id
        """)

        self.conn.commit()
        logger.info("Analytics view created")

    def run(self):
        # Download CSV files first
        if not download_csv_files_from_gdrive():
            logger.info("Skipping data loader - CSV files not available")
            exit(1)
        
        if self.db_path.exists():
            self.db_path.unlink()
            logger.info(f"Removed existing database: {self.db_path}")
        
        # Load CSV files into database
        self.load_all_to_sqlite()
        logger.info("🎉 Data ingestion completed successfully!")
        self.conn.close()
        return self.db_path


if __name__ == "__main__":
    # Load CSV files into database
    loader = OlistDataLoader(
        data_dir = str(DATA_DIR),
        db_path = str(DB_DIR)
    )
    loader.run()
