/**
 * Customer Churn Risk Analytics - Frontend Script
 */

document.addEventListener('DOMContentLoaded', function() {
    
    // Check which page we are currently on and initialize components
    if (document.getElementById('churnTierChart') && document.getElementById('billingDistChart')) {
        initDashboardCharts();
    }

    if (document.getElementById('runEtlBtn')) {
        initEtlPage();
    }

    if (document.getElementById('trainModelBtn')) {
        initModelPage();
    }
});

/**
 * -------------------------------------------------------------
 * DASHBOARD PAGE LOGIC
 * -------------------------------------------------------------
 */
function initDashboardCharts() {
    fetch('/api/dashboard/charts')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error("Failed to load chart data:", data.error);
                return;
            }

            // 1. Bar Chart: Churn Rate by Usage Tier
            const ctxBar = document.getElementById('churnTierChart').getContext('2d');
            new Chart(ctxBar, {
                type: 'bar',
                data: {
                    labels: data.usage_churn.labels,
                    datasets: [{
                        label: 'Churn Rate (%)',
                        data: data.usage_churn.rates,
                        backgroundColor: 'rgba(13, 148, 136, 0.75)', // Teal
                        borderColor: '#0d9488',
                        borderWidth: 1.5,
                        borderRadius: 6,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Churn Percentage (%)',
                                font: { weight: 'bold' }
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Usage Tier',
                                font: { weight: 'bold' }
                            }
                        }
                    }
                }
            });

            // 2. Pie Chart: Billing Type Distribution
            const ctxPie = document.getElementById('billingDistChart').getContext('2d');
            new Chart(ctxPie, {
                type: 'doughnut',
                data: {
                    labels: data.billing.labels,
                    datasets: [{
                        data: data.billing.values,
                        backgroundColor: [
                            '#1e3a8a', // Dark Navy
                            '#0d9488', // Dark Teal
                            '#f97316'  // Orange
                        ],
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                boxWidth: 12,
                                padding: 15
                            }
                        }
                    },
                    cutout: '60%'
                }
            });
        })
        .catch(err => console.error("Error fetching dashboard chart data:", err));
}

/**
 * -------------------------------------------------------------
 * ETL PIPELINE PAGE LOGIC
 * -------------------------------------------------------------
 */
function initEtlPage() {
    const runBtn = document.getElementById('runEtlBtn');
    const consoleBox = document.getElementById('consoleBox');
    const logOutput = document.getElementById('logOutput');
    let logInterval = null;

    // Load logs initially
    fetchLogs();

    function fetchLogs() {
        fetch('/api/etl/logs')
            .then(res => res.json())
            .then(data => {
                logOutput.textContent = data.logs;
                // Auto scroll console to bottom
                consoleBox.scrollTop = consoleBox.scrollHeight;
            });
    }

    runBtn.addEventListener('click', function() {
        // Disable UI
        runBtn.disabled = true;
        runBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Executing ETL...';
        
        // Start polling logs
        logInterval = setInterval(fetchLogs, 1000);

        // Run ETL via POST
        fetch('/api/etl/run', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                clearInterval(logInterval);
                fetchLogs(); // Final log update
                
                runBtn.disabled = false;
                runBtn.innerHTML = '<i class="fas fa-play me-2"></i>Run ETL Pipeline';

                if (data.success) {
                    alert("ETL Pipeline ran successfully! Data loaded into database.");
                    window.location.reload(); // Reload to show new stats cards
                } else {
                    alert("ETL Pipeline error: " + data.error);
                }
            })
            .catch(err => {
                clearInterval(logInterval);
                runBtn.disabled = false;
                runBtn.innerHTML = '<i class="fas fa-play me-2"></i>Run ETL Pipeline';
                console.error("ETL request failure:", err);
                alert("An error occurred while running the ETL pipeline.");
            });
    });
}

/**
 * -------------------------------------------------------------
 * ML CHURN MODEL TRAINING & TEST LOGIC
 * -------------------------------------------------------------
 */
function initModelPage() {
    const trainBtn = document.getElementById('trainModelBtn');
    const statusContainer = document.getElementById('trainStatusContainer');
    
    // Model training request
    trainBtn.addEventListener('click', function() {
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Training Model...';
        statusContainer.innerHTML = '<div class="alert alert-info">Training Scikit-learn Random Forest model. This processes 50,000+ rows and computes scores. Please wait...</div>';

        fetch('/api/model/train', { method: 'POST' })
            .then(res => res.json())
            .then(data => {
                trainBtn.disabled = false;
                trainBtn.innerHTML = '<i class="fas fa-cog me-2"></i>Train Churn Classifier';

                if (data.success) {
                    statusContainer.innerHTML = '<div class="alert alert-success">Model trained successfully and Database predictions updated!</div>';
                    // Wait 1.5 seconds and reload
                    setTimeout(() => {
                        window.location.reload();
                    }, 1500);
                } else {
                    statusContainer.innerHTML = `<div class="alert alert-danger">Training failed: ${data.error}</div>`;
                }
            })
            .catch(err => {
                trainBtn.disabled = false;
                trainBtn.innerHTML = '<i class="fas fa-cog me-2"></i>Train Churn Classifier';
                statusContainer.innerHTML = '<div class="alert alert-danger">An unexpected network error occurred during training.</div>';
                console.error("Training request failure:", err);
            });
    });

    // Real-time Single Customer Tester Form
    const predictForm = document.getElementById('predictCustomerForm');
    if (predictForm) {
        predictForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const tenure = document.getElementById('inputTenure').value;
            const monthly = document.getElementById('inputMonthly').value;
            const total = document.getElementById('inputTotal').value;
            const complaints = document.getElementById('inputComplaints').value;
            const usage = document.getElementById('inputUsage').value;
            const billing = document.getElementById('inputBilling').value;

            const requestData = {
                tenure_months: parseFloat(tenure),
                monthly_charges: parseFloat(monthly),
                total_charges: parseFloat(total),
                num_complaints: parseInt(complaints),
                usage_gb: parseFloat(usage),
                billing_type: billing
            };

            const resultDiv = document.getElementById('predictionResult');
            resultDiv.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div>';
            resultDiv.classList.remove('d-none');

            fetch('/api/model/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            })
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    resultDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
                    return;
                }

                const prob = (data.churn_probability * 100).toFixed(2);
                let riskClass = "Low Risk";
                let alertColor = "success";
                
                if (data.churn_probability >= 0.7) {
                    riskClass = "Critical High Churn Risk";
                    alertColor = "danger";
                } else if (data.churn_probability >= 0.4) {
                    riskClass = "Medium Churn Risk";
                    alertColor = "warning";
                }

                resultDiv.innerHTML = `
                    <div class="card border-${alertColor} bg-light">
                        <div class="card-body text-center">
                            <h6 class="text-uppercase text-muted mb-1">Prediction Risk Assessment</h6>
                            <h4 class="text-${alertColor} fw-bold mb-2">${riskClass}</h4>
                            <div class="progress mb-2" style="height: 10px;">
                                <div class="progress-bar bg-${alertColor}" role="progressbar" style="width: ${prob}%" aria-valuenow="${prob}" aria-valuemin="0" aria-valuemax="100"></div>
                            </div>
                            <p class="mb-0">Estimated Churn Probability: <strong>${prob}%</strong></p>
                        </div>
                    </div>
                `;
            })
            .catch(err => {
                resultDiv.innerHTML = '<div class="alert alert-danger">Failed to process prediction. Check connection.</div>';
                console.error("Prediction request failure:", err);
            });
        });
    }
}
