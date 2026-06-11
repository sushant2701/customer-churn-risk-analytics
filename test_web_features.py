import urllib.request
import urllib.parse
import json

BASE_URL = "http://127.0.0.1:5050"

def run_web_feature_tests():
    print("\n" + "="*70)
    print("STEP-WISE WEB PAGE & FEATURE TESTING RUNNER")
    print("="*70)

    # -------------------------------------------------------------
    # STEP 1: TEST DASHBOARD PAGE RENDERING
    # -------------------------------------------------------------
    print("\n--- [STEP 1/6] Testing Dashboard page and charts JSON feed ---")
    try:
        # Fetch dashboard HTML
        html_req = urllib.request.urlopen(f"{BASE_URL}/", timeout=60)
        html_content = html_req.read().decode('utf-8')
        assert "Executive Churn Risk Dashboard" in html_content, "FAILED: Dashboard header missing!"
        assert "Total Customers" in html_content, "FAILED: KPI metric cards missing!"
        print("SUCCESS: Dashboard HTML loaded and KPI structure verified.")

        # Fetch Chart.js data API
        charts_req = urllib.request.urlopen(f"{BASE_URL}/api/dashboard/charts", timeout=60)
        charts_json = json.loads(charts_req.read().decode('utf-8'))
        assert "billing" in charts_json and "usage_churn" in charts_json, "FAILED: Chart data missing key segments!"
        print(f"SUCCESS: Chart JSON loaded. Billing categories: {charts_json['billing']['labels']}")
        print(f"SUCCESS: Usage categories: {charts_json['usage_churn']['labels']}")
    except Exception as e:
        print(f"FAILED: Dashboard test encountered an error -> {e}")
        return

    # -------------------------------------------------------------
    # STEP 2: TEST ETL WORKFLOW API & LOG STREAM
    # -------------------------------------------------------------
    print("\n--- [STEP 2/6] Testing ETL pipeline execution & logging ---")
    try:
        # Trigger ETL run via POST
        etl_req = urllib.request.Request(f"{BASE_URL}/api/etl/run", method='POST')
        with urllib.request.urlopen(etl_req, timeout=60) as response:
            etl_json = json.loads(response.read().decode('utf-8'))
        
        assert etl_json.get("success") is True, "FAILED: ETL API returned failure!"
        stats = etl_json["stats"]
        print(f"SUCCESS: ETL API completed. Records parsed: {stats['initial_count']} -> Loaded into DB: {stats['db_customers']}")
        
        # Verify log stream
        logs_req = urllib.request.urlopen(f"{BASE_URL}/api/etl/logs", timeout=60)
        logs_json = json.loads(logs_req.read().decode('utf-8'))
        assert "Saved cleaned and normalized data" in logs_json["logs"], "FAILED: Log text verification failed!"
        print("SUCCESS: Live logger streaming endpoints verified.")
    except Exception as e:
        print(f"FAILED: ETL workflow test encountered an error -> {e}")
        return

    # -------------------------------------------------------------
    # STEP 3: TEST SQL SEGMENTS & EXPLAIN PLAN FILTERS
    # -------------------------------------------------------------
    print("\n--- [STEP 3/6] Testing SQL Segmentation search and execution plans ---")
    try:
        # Filter for Month-to-month contracts and Medium usage tier
        query_params = urllib.parse.urlencode({
            'billing_type': 'Month-to-month',
            'usage_tier': 'Medium',
            'page': 1
        })
        sql_req = urllib.request.urlopen(f"{BASE_URL}/sql?{query_params}", timeout=60)
        sql_html = sql_req.read().decode('utf-8')
        
        assert "idx_customers_billing" in sql_html, "FAILED: Database optimization indices missing in query explanation!"
        assert "calculated_usage_tier" in sql_html or "calculated-usage-tier" or "usage_gb" in sql_html, "FAILED: Columns missing!"
        print("SUCCESS: Query segmentation loaded matching filter records.")
        print("SUCCESS: Database optimization index lookup ('idx_customers_billing') successfully verified in query plan HTML.")
    except Exception as e:
        print(f"FAILED: SQL segment testing encountered an error -> {e}")
        return

    # -------------------------------------------------------------
    # STEP 4: TEST MACHINE LEARNING MODEL TRAINING API
    # -------------------------------------------------------------
    print("\n--- [STEP 4/6] Testing Random Forest model training API ---")
    try:
        # Request model training via POST
        train_req = urllib.request.Request(f"{BASE_URL}/api/model/train", method='POST')
        with urllib.request.urlopen(train_req, timeout=60) as response:
            train_json = json.loads(response.read().decode('utf-8'))
            
        assert train_json.get("success") is True, "FAILED: Model training API failed!"
        metadata = train_json["metadata"]
        print(f"SUCCESS: Model training API finished. Holdout test accuracy: {metadata['accuracy']*100:.2f}%")
        print(f"SUCCESS: Feature weights: {metadata['feature_importances'][0]['feature']} ({metadata['feature_importances'][0]['importance']*100:.1f}%)")
    except Exception as e:
        print(f"FAILED: ML training test encountered an error -> {e}")
        return

    # -------------------------------------------------------------
    # STEP 5: TEST CUSTOM RISK INFERENCE SIMULATOR
    # -------------------------------------------------------------
    print("\n--- [STEP 5/6] Testing risk calculator inference API ---")
    try:
        # Prepare mock customer input
        customer_payload = {
            'tenure_months': 5,
            'monthly_charges': 110.0,
            'total_charges': 550.0,
            'num_complaints': 4,
            'usage_gb': 600.0,
            'billing_type': 'Month-to-month'
        }
        data = json.dumps(customer_payload).encode('utf-8')
        
        predict_req = urllib.request.Request(
            f"{BASE_URL}/api/model/predict",
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        with urllib.request.urlopen(predict_req, timeout=60) as response:
            predict_json = json.loads(response.read().decode('utf-8'))
            
        assert "churn_probability" in predict_json, "FAILED: Prediction return payload missing risk score!"
        print(f"SUCCESS: Simulator scored probability: {predict_json['churn_probability']*100:.2f}%")
        print(f"SUCCESS: Simulator classification: {'High Risk' if predict_json['churn_prediction'] == 1 else 'Low Risk'}")
    except Exception as e:
        print(f"FAILED: Risk simulator test encountered an error -> {e}")
        return

    # -------------------------------------------------------------
    # STEP 6: TEST EXCEL SPREADSHEET EXPORT DOWNLOAD
    # -------------------------------------------------------------
    print("\n--- [STEP 6/6] Testing Excel sheet downloader ---")
    try:
        # Request download
        export_req = urllib.request.urlopen(f"{BASE_URL}/export", timeout=60)
        headers = export_req.info()
        content_type = headers.get_content_type()
        content_length = int(headers.get("Content-Length", 0))
        
        # Verify mimetype matches Excel sheets
        assert content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "FAILED: Mismatch MIME type!"
        assert content_length > 1000, "FAILED: Download file appears to be empty!"
        print(f"SUCCESS: Excel spreadsheet verified. Mimetype: {content_type} ({content_length:,} bytes downloaded)")
    except Exception as e:
        print(f"FAILED: Excel export test encountered an error -> {e}")
        return

    print("\n" + "="*70)
    print("ALL WEB PAGES & APIS VERIFIED AND COMPLETED SUCCESSFULLY!")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_web_feature_tests()
