import os
import sqlite3
import pandas as pd

DB_FILE = os.path.join('instance', 'customer_churn.db')

def get_db_connection():
    """Establish and return an SQLite database connection."""
    os.makedirs('instance', exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    """Initialize SQLite tables and create indexes for query optimization."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Drop existing tables to ensure clean schema
    cursor.execute("DROP TABLE IF EXISTS usage_metrics;")
    cursor.execute("DROP TABLE IF EXISTS customers;")
    
    # Create customers table (dimension table)
    cursor.execute("""
    CREATE TABLE customers (
        customer_id TEXT PRIMARY KEY,
        tenure_months INTEGER NOT NULL,
        billing_type TEXT NOT NULL
    );
    """)
    
    # Create usage_metrics table (fact table)
    cursor.execute("""
    CREATE TABLE usage_metrics (
        customer_id TEXT PRIMARY KEY,
        monthly_charges REAL NOT NULL,
        total_charges REAL NOT NULL,
        num_complaints INTEGER NOT NULL,
        usage_gb REAL NOT NULL,
        churn_label INTEGER NOT NULL,
        churn_prob REAL DEFAULT NULL,
        norm_tenure_months REAL,
        norm_monthly_charges REAL,
        norm_total_charges REAL,
        norm_usage_gb REAL,
        norm_num_complaints REAL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
    """)
    
    # Create Indexes to optimize joins and segmentation queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_billing ON customers(billing_type);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_churn ON usage_metrics(churn_label);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_gb ON usage_metrics(usage_gb);")
    
    conn.commit()
    conn.close()
    print("Database schema initialized and indexes created.")

def load_data_to_db(df):
    """
    Split the cleaned DataFrame into the normalization tables schema 
    and load them into the SQLite database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Extract records for customers table
    customers_data = df[['customer_id', 'tenure_months', 'billing_type']].values.tolist()
    
    # Extract records for usage_metrics table
    usage_data = df[[
        'customer_id', 'monthly_charges', 'total_charges', 'num_complaints', 'usage_gb', 'churn_label',
        'norm_tenure_months', 'norm_monthly_charges', 'norm_total_charges', 'norm_usage_gb', 'norm_num_complaints'
    ]].values.tolist()
    
    # Use executemany for fast batch insertion
    cursor.execute("BEGIN TRANSACTION;")
    
    cursor.executemany("""
        INSERT OR REPLACE INTO customers (customer_id, tenure_months, billing_type)
        VALUES (?, ?, ?);
    """, customers_data)
    
    cursor.executemany("""
        INSERT OR REPLACE INTO usage_metrics (
            customer_id, monthly_charges, total_charges, num_complaints, usage_gb, churn_label,
            norm_tenure_months, norm_monthly_charges, norm_total_charges, norm_usage_gb, norm_num_complaints
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, usage_data)
    
    conn.commit()
    
    # Verify records loaded
    cursor.execute("SELECT COUNT(*) FROM customers;")
    cust_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM usage_metrics;")
    metrics_count = cursor.fetchone()[0]
    
    conn.close()
    return cust_count, metrics_count

def get_segmented_customers(billing_type, usage_tier, page=1, per_page=15):
    """
    Queries customer records matching filters.
    Segments by usage_tier (Low <150, Medium 150-350, High >350) and joins with customers.
    Uses indexed queries. Supports pagination.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Build query condition based on usage tier
    # SQLite does not allow WHERE aliases, so we filter by usage_gb ranges.
    usage_condition = ""
    params = []
    
    if billing_type and billing_type != "All":
        billing_cond = "c.billing_type = ?"
        params.append(billing_type)
    else:
        billing_cond = "1=1"
        
    if usage_tier == "Low":
        usage_cond = "u.usage_gb < 150"
    elif usage_tier == "Medium":
        usage_cond = "u.usage_gb BETWEEN 150 AND 350"
    elif usage_tier == "High":
        usage_cond = "u.usage_gb > 350"
    else:
        usage_cond = "1=1"
        
    offset = (page - 1) * per_page
    
    # Get total count of records matching conditions
    count_query = f"""
    SELECT COUNT(*) 
    FROM customers c
    JOIN usage_metrics u ON c.customer_id = u.customer_id
    WHERE {billing_cond} AND {usage_cond}
    """
    cursor.execute(count_query, params)
    total_records = cursor.fetchone()[0]
    
    # Fetch paginated records
    data_query = f"""
    SELECT c.customer_id, c.tenure_months, c.billing_type, 
           u.usage_gb, u.monthly_charges, u.total_charges, u.num_complaints, u.churn_label, u.churn_prob,
           CASE 
               WHEN u.usage_gb < 150 THEN 'Low'
               WHEN u.usage_gb BETWEEN 150 AND 350 THEN 'Medium'
               ELSE 'High'
           END as calculated_usage_tier
    FROM customers c
    JOIN usage_metrics u ON c.customer_id = u.customer_id
    WHERE {billing_cond} AND {usage_cond}
    ORDER BY c.customer_id ASC
    LIMIT ? OFFSET ?
    """
    
    query_params = params + [per_page, offset]
    cursor.execute(data_query, query_params)
    rows = [dict(row) for row in cursor.fetchall()]
    
    # Explain query plan for presentation
    explain_query = f"EXPLAIN QUERY PLAN {data_query}"
    cursor.execute(explain_query, query_params)
    explain_plan = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return rows, total_records, explain_plan, data_query

def update_churn_probabilities(probabilities_dict):
    """
    Updates the usage_metrics table with churn prediction probability scores.
    probabilities_dict: Dictionary mapping customer_id to float probability
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("BEGIN TRANSACTION;")
    data = [(prob, cust_id) for cust_id, prob in probabilities_dict.items()]
    cursor.executemany("""
        UPDATE usage_metrics 
        SET churn_prob = ? 
        WHERE customer_id = ?;
    """, data)
    conn.commit()
    conn.close()
    print(f"Updated churn probabilities for {len(probabilities_dict)} customer records.")
