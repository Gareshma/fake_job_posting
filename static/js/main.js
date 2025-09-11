$(document).ready(function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Enable analyze button when job is selected
    $('#jobSelect').on('change', function() {
        const selectedValue = $(this).val();
        $('#analyzeBtn').prop('disabled', !selectedValue);
        
        // Hide previous results
        hideAllSections();
    });

    // Handle analyze button click
    $('#analyzeBtn').on('click', function() {
        const selectedJobId = $('#jobSelect').val();
        if (selectedJobId) {
            analyzeJob(selectedJobId);
        }
    });
}

function analyzeJob(jobId) {
    // Show loading section
    showLoadingSection();
    hideResultsSection();
    hideErrorSection();

    // Make API call
    $.ajax({
        url: '/api/predict',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
            job_id: parseInt(jobId),
            model_type: 'both'
        }),
        success: function(response) {
            hideLoadingSection();
            displayResults(response);
        },
        error: function(xhr, status, error) {
            hideLoadingSection();
            let errorMessage = 'An error occurred while analyzing the job posting.';
            
            if (xhr.responseJSON && xhr.responseJSON.error) {
                errorMessage = xhr.responseJSON.error;
            }
            
            displayError(errorMessage);
        }
    });
}

function displayResults(data) {
    const { job_info, predictions, model_metrics } = data;
    
    // Display job details
    displayJobDetails(job_info);
    
    // Display XGBoost results
    if (predictions.xgboost) {
        displayXGBoostResults(predictions.xgboost);
    }
    
    // Display One-Class SVM results
    if (predictions.ocsvm) {
        displayOCSVMResults(predictions.ocsvm);
    }
    
    // Display model metrics
    displayModelMetrics(model_metrics);
    
    // Show results section
    showResultsSection();
}

function displayJobDetails(jobInfo) {
    const actualLabel = jobInfo.label;
    const labelBadge = actualLabel === 'real' 
        ? '<span class="badge bg-success">REAL JOB</span>' 
        : '<span class="badge bg-danger">FAKE JOB</span>';
    
    const detailsHtml = `
        <div class="job-detail-item">
            <div class="job-detail-label">
                <i class="fas fa-briefcase me-2"></i>Job Title
                ${labelBadge}
            </div>
            <div class="job-detail-content">${jobInfo.title}</div>
        </div>
        <div class="job-detail-item">
            <div class="job-detail-label">
                <i class="fas fa-building me-2"></i>Company
            </div>
            <div class="job-detail-content">${jobInfo.company}</div>
        </div>
        <div class="job-detail-item">
            <div class="job-detail-label">
                <i class="fas fa-align-left me-2"></i>Description
            </div>
            <div class="job-detail-content">${jobInfo.description}</div>
        </div>
        <div class="job-detail-item">
            <div class="job-detail-label">
                <i class="fas fa-list-ul me-2"></i>Requirements
            </div>
            <div class="job-detail-content">${jobInfo.requirements}</div>
        </div>
    `;
    
    $('#jobDetails').html(detailsHtml);
}

function displayXGBoostResults(results) {
    const prediction = results.prediction;
    const confidence = (results.confidence * 100).toFixed(1);
    const probabilities = results.probabilities;
    
    const isReal = prediction === 'real';
    const resultClass = isReal ? 'prediction-real' : 'prediction-fake';
    const icon = isReal ? 'fas fa-check-circle' : 'fas fa-times-circle';
    const confidenceClass = getConfidenceClass(results.confidence);
    
    const resultsHtml = `
        <div class="prediction-result ${resultClass}">
            <div class="prediction-icon">
                <i class="${icon}"></i>
            </div>
            <h5 class="mb-2">Prediction: ${prediction.toUpperCase()}</h5>
            <p class="mb-2">Confidence: ${confidence}%</p>
            <div class="confidence-bar">
                <div class="confidence-fill ${confidenceClass}" style="width: ${confidence}%"></div>
            </div>
        </div>
        
        <div class="mt-3">
            <h6>Probability Breakdown:</h6>
            <div class="probability-container">
                <div class="probability-label">
                    <span><i class="fas fa-check-circle text-success me-1"></i>Real Job</span>
                    <span>${(probabilities.real * 100).toFixed(1)}%</span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill-real" style="width: ${probabilities.real * 100}%"></div>
                </div>
            </div>
            <div class="probability-container">
                <div class="probability-label">
                    <span><i class="fas fa-times-circle text-danger me-1"></i>Fake Job</span>
                    <span>${(probabilities.fake * 100).toFixed(1)}%</span>
                </div>
                <div class="probability-bar">
                    <div class="probability-fill-fake" style="width: ${probabilities.fake * 100}%"></div>
                </div>
            </div>
        </div>
    `;
    
    $('#xgboostResults').html(resultsHtml);
}

function displayOCSVMResults(results) {
    const prediction = results.prediction;
    const confidence = (results.confidence * 100).toFixed(1);
    const anomalyScore = results.anomaly_score.toFixed(4);
    
    const isReal = prediction === 'real';
    const resultClass = isReal ? 'prediction-real' : 'prediction-fake';
    const icon = isReal ? 'fas fa-check-circle' : 'fas fa-exclamation-triangle';
    const confidenceClass = getConfidenceClass(results.confidence);
    
    const resultsHtml = `
        <div class="prediction-result ${resultClass}">
            <div class="prediction-icon">
                <i class="${icon}"></i>
            </div>
            <h5 class="mb-2">Prediction: ${prediction.toUpperCase()}</h5>
            <p class="mb-2">Confidence: ${confidence}%</p>
            <div class="confidence-bar">
                <div class="confidence-fill ${confidenceClass}" style="width: ${confidence}%"></div>
            </div>
        </div>
        
        <div class="mt-3">
            <h6>Anomaly Analysis:</h6>
            <div class="metric-item">
                <div class="metric-label">Anomaly Score</div>
                <div class="metric-value">${anomalyScore}</div>
                <small class="text-muted">
                    ${isReal ? 'Normal pattern detected' : 'Anomalous pattern detected'}
                </small>
            </div>
            <div class="alert ${isReal ? 'alert-success' : 'alert-warning'} mt-2">
                <small>
                    One-Class SVM detects anomalies by comparing against patterns learned from legitimate job postings.
                    ${isReal ? 'This posting matches normal job patterns.' : 'This posting shows unusual characteristics.'}
                </small>
            </div>
        </div>
    `;
    
    $('#ocsvmResults').html(resultsHtml);
}

function displayModelMetrics(metrics) {
    const xgbMetrics = metrics.xgboost;
    const ocsvmMetrics = metrics.ocsvm;
    
    const xgbHtml = `
        <div class="metric-item">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">${(xgbMetrics.f1_score * 100).toFixed(1)}%</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">${(xgbMetrics.accuracy * 100).toFixed(1)}%</div>
        </div>
    `;
    
    const ocsvmMetricsHtml = `
        <div class="metric-item">
            <div class="metric-label">F1 Score</div>
            <div class="metric-value">${(ocsvmMetrics.f1_score * 100).toFixed(1)}%</div>
        </div>
        <div class="metric-item">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">${(ocsvmMetrics.accuracy * 100).toFixed(1)}%</div>
        </div>
    `;
    
    $('#xgboostMetrics').html(xgbHtml);
    $('#ocsvmMetrics').html(ocsvmMetricsHtml);
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'confidence-high';
    if (confidence >= 0.6) return 'confidence-medium';
    return 'confidence-low';
}

function showLoadingSection() {
    $('#loadingSection').slideDown();
}

function hideLoadingSection() {
    $('#loadingSection').slideUp();
}

function showResultsSection() {
    $('#resultsSection').slideDown();
}

function hideResultsSection() {
    $('#resultsSection').slideUp();
}

function showErrorSection() {
    $('#errorSection').slideDown();
}

function hideErrorSection() {
    $('#errorSection').slideUp();
}

function hideAllSections() {
    hideLoadingSection();
    hideResultsSection();
    hideErrorSection();
}

function displayError(message) {
    $('#errorMessage').text(message);
    showErrorSection();
}

// Add smooth scrolling to results
function scrollToResults() {
    $('html, body').animate({
        scrollTop: $('#resultsSection').offset().top - 20
    }, 500);
}

// Initialize tooltips if Bootstrap is available
$(function () {
    if (typeof bootstrap !== 'undefined') {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});