import os
import pandas as pd
import numpy as np
import time

# Ensure data directory exists
os.makedirs('data', exist_ok=True)
LOG_FILE = os.path.join('data', 'etl.log')

def log_message(message):
    """Log helper to print to console and append to etl.log."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + '\n')

def clear_logs():
    """Clear the ETL log file."""
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    log_message("Starting ETL Logging Session...")

def generate_raw_data(num_records=51000):
    """
    Generates raw telecom customer dataset with anomalies:
    - Missing values (nulls) in numeric columns (~2% each)
    - Duplicate customer IDs (~150 records)
    - Formatting issues (spaces in string representations, e.g. total_charges)
    - Outliers or invalid values
    """
    clear_logs()
    log_message(f"Initiating synthetic data generation for {num_records} records...")
    
    np.random.seed(42)
    
    # Generate unique customer IDs
    cust_ids = [f"CUST-{i:05d}" for i in range(1, num_records + 1)]
    
    # Tenure months: 1 to 72 months
    tenure = np.random.randint(1, 73, size=num_records)
    
    # Billing types
    billing_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_records, p=[0.55, 0.20, 0.25])
    
    # Monthly charges: 20 to 120
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=num_records), 2)
    
    # Total charges: roughly tenure * monthly_charges
    total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 5, size=num_records), 2)
    # Ensure no negative total charges
    total_charges = np.clip(total_charges, a_min=20.0, a_max=None)
    
    # Number of complaints: 0 to 10
    num_complaints = np.random.poisson(lam=1.5, size=num_records)
    num_complaints = np.clip(num_complaints, a_min=0, a_max=10)
    
    # Usage in GB: 5 to 800 GB
    usage_gb = np.round(np.random.exponential(scale=150.0, size=num_records) + 5.0, 1)
    usage_gb = np.clip(usage_gb, a_min=5.0, a_max=900.0)
    
    # Determine Churn Label synthetically based on factors (logistic regression-like probability)
    # Risk factors: high monthly charges, high complaints, month-to-month billing, low tenure, high usage
    billing_risk = np.where(billing_types == 'Month-to-month', 1.5, np.where(billing_types == 'One year', -0.5, -1.5))
    complaint_risk = num_complaints * 0.5
    tenure_risk = (36 - tenure) * 0.05
    charge_risk = (monthly_charges - 70) * 0.02
    usage_risk = (usage_gb - 200) * 0.001
    
    logits = billing_risk + complaint_risk + tenure_risk + charge_risk + usage_risk - 1.2
    probabilities = 1 / (1 + np.exp(-logits))
    churn_labels = np.random.binomial(1, probabilities)
    
    df = pd.DataFrame({
        'customer_id': cust_ids,
        'tenure_months': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'num_complaints': num_complaints,
        'usage_gb': usage_gb,
        'billing_type': billing_types,
        'churn_label': churn_labels
    })
    
    log_message("Base dataset created successfully.")

    # Inject duplicate customer_id records (around 150)
    log_message("Injecting duplicate records for ETL processing...")
    dup_indices = np.random.choice(num_records, size=150, replace=False)
    dups = df.iloc[dup_indices].copy()
    # Add a bit of jitter or keep identical
    df = pd.concat([df, dups], ignore_index=True)
    
    # Inject Null values in numeric columns (approx 2% missing each)
    log_message("Injecting null values into numeric columns...")
    for col in ['tenure_months', 'monthly_charges', 'total_charges', 'num_complaints', 'usage_gb']:
        null_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
        df.loc[null_indices, col] = np.nan
        
    # Inject string formatting anomalies (e.g. convert some numeric total_charges to strings with space)
    log_message("Injecting string-formatting anomalies in total_charges...")
    str_indices = np.random.choice(df.index, size=50, replace=False)
    # Set to strings with leading/trailing spaces
    for idx in str_indices:
        val = df.loc[idx, 'total_charges']
        if not pd.isna(val):
            df.loc[idx, 'total_charges'] = f"  {val}  "
            
    raw_path = os.path.join('data', 'raw_customers.csv')
    df.to_csv(raw_path, index=False)
    log_message(f"Raw dataset generated and saved to '{raw_path}' (Total rows: {len(df)})")
    return raw_path

def run_etl(raw_path=None):
    """
    ETL Pipeline:
    - E (Extract): Load raw_customers.csv
    - T (Transform):
      - Standardize data types (e.g. strip whitespace, convert total_charges to float)
      - Resolve nulls via median imputation
      - Remove duplicates
      - Normalize numeric features (Min-Max Scaling)
    - L (Load): Returns cleaned DataFrame and saves it. (Loaded into SQLite in db_manager)
    """
    if raw_path is None:
        raw_path = os.path.join('data', 'raw_customers.csv')
        if not os.path.exists(raw_path):
            log_message("Raw data file not found! Generating raw data first...")
            generate_raw_data()
            
    log_message("Starting ETL execution pipeline...")
    
    # 1. Extract
    log_message(f"Extracting raw data from: {raw_path}")
    df = pd.read_csv(raw_path)
    initial_row_count = len(df)
    log_message(f"Extracted {initial_row_count} raw customer records.")
    
    # 2. Transform: Clean and Parse Types
    log_message("Standardizing data types and cleaning text fields...")
    # Clean whitespace and force total_charges to numeric
    if df['total_charges'].dtype == object:
        df['total_charges'] = df['total_charges'].astype(str).str.strip()
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
        log_message("Handled spaces and string formats in 'total_charges' column.")
    else:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
        
    # 3. Transform: Remove Duplicates
    log_message("Checking for duplicate customer records...")
    dup_mask = df.duplicated(subset=['customer_id'], keep='first')
    num_dups = dup_mask.sum()
    df_dedup = df[~dup_mask].copy()
    log_message(f"Found and removed {num_dups} duplicate records based on 'customer_id'.")
    
    # 4. Transform: Resolve Nulls with Median Imputation
    log_message("Imputing missing values with median values...")
    numeric_cols = ['tenure_months', 'monthly_charges', 'total_charges', 'num_complaints', 'usage_gb']
    imputation_stats = {}
    
    for col in numeric_cols:
        null_count = df_dedup[col].isna().sum()
        if null_count > 0:
            median_val = df_dedup[col].median()
            df_dedup[col] = df_dedup[col].fillna(median_val)
            imputation_stats[col] = (int(null_count), float(median_val))
            log_message(f"Column '{col}': Imputed {null_count} nulls with median value: {median_val:.2f}")
        else:
            log_message(f"Column '{col}': No null values found.")
            
    # Ensure types are correct post-imputation
    df_dedup['tenure_months'] = df_dedup['tenure_months'].astype(int)
    df_dedup['num_complaints'] = df_dedup['num_complaints'].astype(int)
    
    # 5. Transform: Normalize Numeric Features
    log_message("Normalizing numeric features for Machine Learning model...")
    # Add normalized columns: value scaled between 0 and 1
    for col in numeric_cols:
        min_val = df_dedup[col].min()
        max_val = df_dedup[col].max()
        # Prevent division by zero
        diff = max_val - min_val if max_val != min_val else 1.0
        df_dedup[f'norm_{col}'] = (df_dedup[col] - min_val) / diff
        log_message(f"Column '{col}': Normalized from range [{min_val:.2f}, {max_val:.2f}] to [0.0, 1.0]")
        
    cleaned_row_count = len(df_dedup)
    log_message(f"Transformations complete. Cleaned row count: {cleaned_row_count} (Discarded {initial_row_count - cleaned_row_count} records).")
    
    # Save Cleaned CSV
    cleaned_path = os.path.join('data', 'cleaned_customers.csv')
    df_dedup.to_csv(cleaned_path, index=False)
    log_message(f"Saved cleaned and normalized data to '{cleaned_path}'. ETL pipeline complete!")
    
    return df_dedup, {
        'initial_count': initial_row_count,
        'cleaned_count': cleaned_row_count,
        'duplicates_removed': int(num_dups),
        'imputation_stats': imputation_stats
    }

if __name__ == "__main__":
    raw_csv = generate_raw_data()
    df_clean, stats = run_etl(raw_csv)
    print("ETL complete. Summary:", stats)
