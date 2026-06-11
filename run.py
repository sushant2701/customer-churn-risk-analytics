import os
import etl_pipeline
import db_manager
import ml_model
import app

def bootstrap_application():
    """
    Checks if raw files, database, and models exist.
    If not, seeds them so the app loads with data ready to go.
    """
    # Prevent running bootstrap twice in Flask debug mode reloader
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        return
        
    print("====================================================")
    print("CUSTOMER CHURN RISK ANALYTICS - STARTUP BOOTSTRAP")
    print("====================================================")
    
    db_exists = os.path.exists(db_manager.DB_FILE)
    model_exists = os.path.exists(ml_model.MODEL_FILE)
    
    if not db_exists:
        print("[Bootstrap] Database not found. Initiating automated data seeder...")
        try:
            # 1. Generate Raw Data
            print("[Bootstrap] Step 1: Generating 50,000 subscriber records...")
            raw_path = etl_pipeline.generate_raw_data(50000)
            
            # 2. Run ETL pipeline
            print("[Bootstrap] Step 2: Executing ETL Cleaning & Imputation...")
            df_cleaned, stats = etl_pipeline.run_etl(raw_path)
            
            # 3. Load database schema
            print("[Bootstrap] Step 3: Initializing SQLite database schema...")
            db_manager.init_db()
            cust_cnt, metrics_cnt = db_manager.load_data_to_db(df_cleaned)
            print(f"[Bootstrap] Loaded {cust_cnt} records into customer profiles table.")
            
            # 4. Train model
            print("[Bootstrap] Step 4: Training baseline Random Forest classifier...")
            ml_model.train_churn_model()
            print("[Bootstrap] Model training finished, predictions back-scored.")
            print("[Bootstrap] Seeding completed successfully.")
        except Exception as e:
            print(f"[Bootstrap] Critical Error during startup bootstrap seeding: {e}")
    else:
        print("[Bootstrap] Found existing customer SQLite database.")
        if not model_exists:
            print("[Bootstrap] Model weights missing. Training baseline model...")
            try:
                ml_model.train_churn_model()
                print("[Bootstrap] Model training finished, predictions back-scored.")
            except Exception as e:
                print(f"[Bootstrap] Error training model during boot: {e}")
        else:
            print("[Bootstrap] Model and Database are ready.")
            
    print("====================================================")

if __name__ == '__main__':
    bootstrap_application()
    print("Launching Flask Web Analytics Platform...")
    app.app.run(debug=True, host='127.0.0.1', port=5050)
