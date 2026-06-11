import os
import sqlite3
import pandas as pd
import numpy as np

import etl_pipeline
import db_manager
import ml_model

def run_step_wise_tests():
    print("\n" + "="*60)
    print("STEP-WISE SYSTEM TESTING RUNNER")
    print("="*60)

    # -------------------------------------------------------------
    # STEP 1: TEST DATA GENERATION
    # -------------------------------------------------------------
    print("\n--- [STEP 1/4] Testing Data Generation ---")
    raw_path = os.path.join('data', 'raw_customers.csv')
    
    # Remove existing raw file to test fresh generation
    if os.path.exists(raw_path):
        os.remove(raw_path)
        print("Removed existing raw data file.")
        
    generated_path = etl_pipeline.generate_raw_data(1000) # Test with smaller set for quick execution
    
    assert os.path.exists(generated_path), "FAILED: Raw CSV file was not created!"
    print(f"SUCCESS: Raw CSV file created at: {generated_path}")
    
    df_raw = pd.read_csv(generated_path)
    print(f"SUCCESS: Loaded raw dataset containing {len(df_raw)} records.")
    
    # Check for anomalies injected
    null_counts = df_raw.isnull().sum().sum()
    print(f"SUCCESS: Injected null counts: {null_counts}")
    assert null_counts > 0, "FAILED: No null values were injected into the raw dataset!"
    
    print("STEP 1 PASSED: Data generation verified.")

    # -------------------------------------------------------------
    # STEP 2: TEST ETL PROCESSING
    # -------------------------------------------------------------
    print("\n--- [STEP 2/4] Testing ETL Cleaning & Normalization ---")
    cleaned_path = os.path.join('data', 'cleaned_customers.csv')
    if os.path.exists(cleaned_path):
        os.remove(cleaned_path)
        
    df_cleaned, stats = etl_pipeline.run_etl(generated_path)
    
    assert os.path.exists(cleaned_path), "FAILED: Cleaned CSV file was not created!"
    print(f"SUCCESS: Cleaned CSV saved at: {cleaned_path}")
    print(f"SUCCESS: Rows pre-clean: {stats['initial_count']} -> Post-clean: {stats['cleaned_count']}")
    print(f"SUCCESS: Duplicates removed: {stats['duplicates_removed']}")
    
    # Verify no nulls remain in numeric columns
    numeric_cols = ['tenure_months', 'monthly_charges', 'total_charges', 'num_complaints', 'usage_gb']
    for col in numeric_cols:
        null_count = df_cleaned[col].isnull().sum()
        assert null_count == 0, f"FAILED: Cleaned column {col} contains null values!"
        
    # Verify normalized columns are in [0, 1] range
    for col in numeric_cols:
        norm_col = f'norm_{col}'
        min_val = df_cleaned[norm_col].min()
        max_val = df_cleaned[norm_col].max()
        assert 0.0 <= min_val <= 1.0 and 0.0 <= max_val <= 1.0, f"FAILED: Normalized column {norm_col} out of range [0, 1]!"
        
    print("SUCCESS: Checked data clean validations. No nulls or out-of-bounds metrics found.")
    print("STEP 2 PASSED: ETL processing verified.")

    # -------------------------------------------------------------
    # STEP 3: TEST SQL DATABASE LOADER & INDEXES
    # -------------------------------------------------------------
    print("\n--- [STEP 3/4] Testing SQLite Ingestion & Querying ---")
    db_manager.init_db()
    c_count, m_count = db_manager.load_data_to_db(df_cleaned)
    
    assert c_count == stats['cleaned_count'], "FAILED: Mismatched row counts in customers table!"
    assert m_count == stats['cleaned_count'], "FAILED: Mismatched row counts in usage_metrics table!"
    print(f"SUCCESS: Database tables loaded. Rows in customers: {c_count}, usage_metrics: {m_count}")
    
    # Verify indexes and primary keys are working (Explain Query Plan test)
    # Filter by Billing Type = 'Month-to-month' and Usage = Medium
    rows, total_records, explain_plan, query = db_manager.get_segmented_customers('Month-to-month', 'Medium', page=1, per_page=5)
    
    print(f"SUCCESS: Segmentation query executed. Total records matching criteria: {total_records}")
    print("SUCCESS: SQLite Query Plan:")
    for step in explain_plan:
        print(f"  -> {step['detail']}")
        
    # Check if explain plan mentions index lookups
    has_index = any('INDEX' in step['detail'].upper() or 'PRIMARY KEY' in step['detail'].upper() for step in explain_plan)
    assert has_index, "FAILED: SQL query executed without index utilization!"
    
    print("STEP 3 PASSED: SQLite storage and query optimizations verified.")

    # -------------------------------------------------------------
    # STEP 4: TEST MACHINE LEARNING MODEL PIPELINE
    # -------------------------------------------------------------
    print("\n--- [STEP 4/4] Testing Scikit-Learn Model & Simulator ---")
    # Clean model files
    if os.path.exists(ml_model.MODEL_FILE):
        os.remove(ml_model.MODEL_FILE)
    if os.path.exists(ml_model.METADATA_FILE):
        os.remove(ml_model.METADATA_FILE)
        
    metadata = ml_model.train_churn_model()
    
    assert os.path.exists(ml_model.MODEL_FILE), "FAILED: Classifier model file (.pkl) not saved!"
    assert os.path.exists(ml_model.METADATA_FILE), "FAILED: Classifier metadata (.json) not saved!"
    
    print("SUCCESS: Trained Random Forest model.")
    print(f"SUCCESS: Test holdout score -> Accuracy: {metadata['accuracy']*100:.2f}%")
    print(f"SUCCESS: Evaluation -> Precision: {metadata['precision']*100:.2f}%, Recall: {metadata['recall']*100:.2f}%, F1: {metadata['f1_score']*100:.2f}%")
    
    print("SUCCESS: Main Churn Drivers:")
    for item in metadata['feature_importances'][:3]:
        print(f"  - {item['feature']}: {item['importance']*100:.2f}% weight")
        
    # Check if predictions were written back to DB
    conn = db_manager.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM usage_metrics WHERE churn_prob IS NOT NULL;")
    scored_count = cursor.fetchone()[0]
    conn.close()
    
    assert scored_count == m_count, "FAILED: Some records missing back-scored churn probabilities!"
    print(f"SUCCESS: Database back-scored. Scored rows: {scored_count}/{m_count}")
    
    # Test individual simulator predictor
    # Tenure 6m, Billing Month-to-month, complaints 4, monthly charges 95, usage 400 GB
    pred = ml_model.predict_single_customer(
        tenure=6,
        monthly=95.0,
        total=570.0,
        complaints=4,
        usage=400.0,
        billing_type='Month-to-month'
    )
    
    print(f"SUCCESS: Single customer risk simulation score:")
    print(f"  - Inputs: tenure=6m, billing=Month-to-month, complaints=4, charges=$95.00, usage=400GB")
    print(f"  - Output Risk Probability: {pred['churn_probability']*100:.2f}%")
    
    assert pred['churn_probability'] >= 0.5, "FAILED: Risk scoring was not calculated correctly!"
    
    print("STEP 4 PASSED: Random Forest model training, back-scoring, and simulator verified.")
    print("\n" + "="*60)
    print("ALL STEPS COMPLETED AND VERIFIED SUCCESSFULLY!")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_step_wise_tests()
