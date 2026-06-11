import os
import sqlite3
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import db_manager

MODEL_FILE = os.path.join('instance', 'random_forest_model.pkl')
METADATA_FILE = os.path.join('instance', 'model_metadata.json')

def train_churn_model():
    """
    Loads normalized customer data from the SQLite database,
    prepares features (handling billing_type dummies),
    trains a Random Forest Classifier, evaluates performance,
    saves the model, and updates customer churn risk probabilities.
    """
    print("Fetching training data from SQLite database...")
    conn = db_manager.get_db_connection()
    
    # Query database to get features and labels
    query = """
    SELECT c.customer_id, c.billing_type,
           u.norm_tenure_months, u.norm_monthly_charges, u.norm_total_charges, 
           u.norm_usage_gb, u.norm_num_complaints, u.churn_label
    FROM customers c
    JOIN usage_metrics u ON c.customer_id = u.customer_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(df) == 0:
        return {"error": "No data found in database. Run ETL pipeline first."}
        
    print(f"Loaded {len(df)} records for training.")
    
    # Features lists
    feature_cols = [
        'norm_tenure_months', 
        'norm_monthly_charges', 
        'norm_total_charges', 
        'norm_usage_gb', 
        'norm_num_complaints'
    ]
    
    # One-hot encode billing_type
    # Ensure all three possible categories are present and handled correctly
    df_encoded = pd.get_dummies(df, columns=['billing_type'], dtype=int)
    
    # Add dummy column names to feature list
    billing_cols = [col for col in df_encoded.columns if col.startswith('billing_type_')]
    all_features = feature_cols + billing_cols
    
    X = df_encoded[all_features]
    y = df_encoded['churn_label']
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest Classifier model...")
    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Predictions and evaluation
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
    # Feature Importance
    importances = rf.feature_importances_
    # Map feature names to importance values
    raw_feature_names = {
        'norm_tenure_months': 'Tenure Months',
        'norm_monthly_charges': 'Monthly Charges',
        'norm_total_charges': 'Total Charges',
        'norm_usage_gb': 'Usage (GB)',
        'norm_num_complaints': 'Number of Complaints',
        'billing_type_Month-to-month': 'Billing: Month-to-month',
        'billing_type_One year': 'Billing: One Year',
        'billing_type_Two year': 'Billing: Two Year'
    }
    
    feature_importance_list = []
    for col, imp in zip(all_features, importances):
        display_name = raw_feature_names.get(col, col)
        feature_importance_list.append({"feature": display_name, "importance": float(imp)})
        
    feature_importance_list = sorted(feature_importance_list, key=lambda x: x['importance'], reverse=True)
    
    # Save Model to pickle
    os.makedirs('instance', exist_ok=True)
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model': rf,
            'features': all_features
        }, f)
        
    # Save Metadata to JSON
    metadata = {
        'status': 'trained',
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'feature_importances': feature_importance_list,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_records': len(df)
    }
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Predict probabilities for ALL customers in the dataset
    print("Scoring all customers for churn probability...")
    # Predict probabilities
    X_all = df_encoded[all_features]
    probs = rf.predict_proba(X_all)[:, 1] # Probability of Churn (Class 1)
    
    # Map customer_ids to their predictions
    probabilities_dict = dict(zip(df_encoded['customer_id'], [float(p) for p in probs]))
    
    # Write probabilities back to DB
    db_manager.update_churn_probabilities(probabilities_dict)
    
    print("Model training and scoring phase successfully finished.")
    return metadata

def predict_single_customer(tenure, monthly, total, complaints, usage, billing_type):
    """
    Predicts churn risk for a single customer manually input from the UI.
    Requires raw inputs, normalizes them, sets up one-hot columns, and outputs probability.
    """
    if not os.path.exists(MODEL_FILE):
        return {"error": "Model has not been trained yet. Please train the model first."}
        
    with open(MODEL_FILE, 'rb') as f:
        saved_data = pickle.load(f)
        rf = saved_data['model']
        features = saved_data['features']
        
    # Standard values used for normalization (need min and max from raw dataset to replicate scaling)
    # We can load the limits from raw_customers.csv or just read them from database. Let's do database queries.
    conn = db_manager.get_db_connection()
    cursor = conn.cursor()
    
    # Find min and max of each feature to compute Min-Max scaling
    # We query the usage_metrics table join customers to find ranges.
    cursor.execute("""
        SELECT 
            MIN(tenure_months), MAX(tenure_months),
            MIN(monthly_charges), MAX(monthly_charges),
            MIN(total_charges), MAX(total_charges),
            MIN(usage_gb), MAX(usage_gb),
            MIN(num_complaints), MAX(num_complaints)
        FROM customers c
        JOIN usage_metrics u ON c.customer_id = u.customer_id
    """)
    stats = cursor.fetchone()
    conn.close()
    
    if not stats or stats[0] is None:
        return {"error": "No baseline data found in database. Please run ETL."}
        
    min_tenure, max_tenure = stats[0], stats[1]
    min_monthly, max_monthly = stats[2], stats[3]
    min_total, max_total = stats[4], stats[5]
    min_usage, max_usage = stats[6], stats[7]
    min_complaints, max_complaints = stats[8], stats[9]
    
    # Apply Min-Max normalization
    norm_tenure = (tenure - min_tenure) / (max_tenure - min_tenure) if max_tenure != min_tenure else 0.0
    norm_monthly = (monthly - min_monthly) / (max_monthly - min_monthly) if max_monthly != min_monthly else 0.0
    norm_total = (total - min_total) / (max_total - min_total) if max_total != min_total else 0.0
    norm_usage = (usage - min_usage) / (max_usage - min_usage) if max_usage != min_usage else 0.0
    norm_complaints = (complaints - min_complaints) / (max_complaints - min_complaints) if max_complaints != min_complaints else 0.0
    
    # Map billing type to dummies
    input_data = {
        'norm_tenure_months': norm_tenure,
        'norm_monthly_charges': norm_monthly,
        'norm_total_charges': norm_total,
        'norm_usage_gb': norm_usage,
        'norm_num_complaints': norm_complaints,
        'billing_type_Month-to-month': 1 if billing_type == 'Month-to-month' else 0,
        'billing_type_One year': 1 if billing_type == 'One year' else 0,
        'billing_type_Two year': 1 if billing_type == 'Two year' else 0
    }
    
    # Align to model features
    features_vector = [input_data.get(feat, 0) for feat in features]
    
    # Predict
    prob = rf.predict_proba([features_vector])[0][1]
    label = int(rf.predict([features_vector])[0])
    
    return {
        "churn_probability": float(prob),
        "churn_prediction": label
    }

def get_model_metadata():
    """Load model performance parameters if trained."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return None
