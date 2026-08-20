/* ============================================
   Chart.js Visualization Functions
   ============================================ */

/* --- Color Palette --- */
var chartColors = {
    primary: '#0891b2',
    primaryLight: '#22d3ee',
    secondary: '#6366f1',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    teal: '#14b8a6',
    gray: '#94a3b8',
    grayLight: '#e2e8f0',
    text: '#1e293b',
    textSecondary: '#64748b',
};

var gradientColors = [
    'rgba(8, 145, 178, 0.85)',
    'rgba(20, 184, 166, 0.85)',
    'rgba(99, 102, 241, 0.85)',
    'rgba(16, 185, 129, 0.85)',
    'rgba(245, 158, 11, 0.85)',
    'rgba(239, 68, 68, 0.85)',
];

var gradientBg = [
    'rgba(8, 145, 178, 0.15)',
    'rgba(20, 184, 166, 0.15)',
    'rgba(99, 102, 241, 0.15)',
    'rgba(16, 185, 129, 0.15)',
    'rgba(245, 158, 11, 0.15)',
    'rgba(239, 68, 68, 0.15)',
];

/* --- Feature Importance Horizontal Bar Chart --- */
function renderFeatureImportanceChart(containerId, features, importances) {
    var ctx = document.getElementById(containerId);
    if (!ctx) return;

    var colors = features.map(function(_, i) {
        return gradientColors[i % gradientColors.length];
    });

    var bgColors = features.map(function(_, i) {
        return gradientBg[i % gradientBg.length];
    });

    new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: features,
            datasets: [{
                label: 'Importance',
                data: importances.map(function(v) { return Math.abs(v); }),
                backgroundColor: colors,
                borderColor: colors.map(function(c) { return c.replace('0.85', '1'); }),
                borderWidth: 1,
                borderRadius: 6,
                barPercentage: 0.65,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1000,
                easing: 'easeOutQuart',
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleFont: { size: 13, weight: '600' },
                    bodyFont: { size: 12 },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            var value = context.parsed.x;
                            return 'Contribution: ' + (value * 100).toFixed(1) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                    },
                    ticks: {
                        font: { size: 11 },
                        color: chartColors.textSecondary,
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Relative Importance',
                        font: { size: 12, weight: '600' },
                        color: chartColors.text,
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        font: { size: 12, weight: '500' },
                        color: chartColors.text,
                    }
                }
            }
        }
    });
}

/* --- SHAP Waterfall / Diverging Bar Chart --- */
function renderShapChart(containerId, featureNames, shapValues) {
    var ctx = document.getElementById(containerId);
    if (!ctx) return;

    var colors = shapValues.map(function(v) {
        return v >= 0
            ? 'rgba(239, 68, 68, 0.8)'
            : 'rgba(8, 145, 178, 0.8)';
    });

    var borderColors = shapValues.map(function(v) {
        return v >= 0
            ? 'rgba(239, 68, 68, 1)'
            : 'rgba(8, 145, 178, 1)';
    });

    new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels: featureNames,
            datasets: [{
                label: 'SHAP Value',
                data: shapValues,
                backgroundColor: colors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4,
                barPercentage: 0.6,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 1200,
                easing: 'easeOutQuart',
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleFont: { size: 13, weight: '600' },
                    bodyFont: { size: 12 },
                    padding: 12,
                    cornerRadius: 8,
                    callbacks: {
                        label: function(context) {
                            var val = context.parsed.x;
                            var direction = val >= 0 ? 'increases risk' : 'decreases risk';
                            return 'SHAP: ' + val.toFixed(3) + ' (' + direction + ')';
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        color: function(context) {
                            if (context.tick && context.tick.value === 0) {
                                return 'rgba(0, 0, 0, 0.3)';
                            }
                            return 'rgba(0, 0, 0, 0.05)';
                        },
                        lineWidth: function(context) {
                            if (context.tick && context.tick.value === 0) {
                                return 2;
                            }
                            return 1;
                        }
                    },
                    ticks: {
                        font: { size: 11 },
                        color: chartColors.textSecondary,
                    },
                    title: {
                        display: true,
                        text: 'SHAP Value (← Lower Risk | Higher Risk →)',
                        font: { size: 12, weight: '600' },
                        color: chartColors.text,
                    }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        font: { size: 12, weight: '500' },
                        color: chartColors.text,
                    }
                }
            }
        }
    });
}

/* --- Confidence Gauge (Doughnut) --- */
function renderConfidenceGauge(containerId, confidence) {
    var ctx = document.getElementById(containerId);
    if (!ctx) return;

    var remaining = 100 - confidence;

    var gaugeColor;
    if (confidence >= 70) {
        gaugeColor = chartColors.primary;
    } else if (confidence >= 50) {
        gaugeColor = chartColors.warning;
    } else {
        gaugeColor = chartColors.danger;
    }

    new Chart(ctx.getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: ['Confidence', 'Remaining'],
            datasets: [{
                data: [confidence, remaining],
                backgroundColor: [gaugeColor, chartColors.grayLight],
                borderWidth: 0,
                cutout: '75%',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            animation: {
                animateRotate: true,
                duration: 1500,
                easing: 'easeOutQuart',
            },
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false },
            }
        },
        plugins: [{
            id: 'gaugeCenter',
            afterDraw: function(chart) {
                var width = chart.width;
                var height = chart.height;
                var context = chart.ctx;

                context.restore();

                var fontSize = Math.min(width, height) * 0.15;
                context.font = 'bold ' + fontSize + 'px Inter, sans-serif';
                context.textBaseline = 'middle';
                context.textAlign = 'center';
                context.fillStyle = chartColors.text;
                context.fillText(confidence.toFixed(1) + '%', width / 2, height / 2);

                context.save();
            }
        }]
    });
}

/* --- Risk Level Visual Indicator --- */
function renderRiskLevelChart(containerId, riskLevel) {
    var ctx = document.getElementById(containerId);
    if (!ctx) return;

    var colorMap = {
        'Low': chartColors.success,
        'Medium': chartColors.warning,
        'High': chartColors.danger,
    };

    var valueMap = {
        'Low': 30,
        'Medium': 65,
        'High': 90,
    };

    var color = colorMap[riskLevel] || chartColors.gray;
    var value = valueMap[riskLevel] || 50;

    new Chart(ctx.getContext('2d'), {
        type: 'doughnut',
        data: {
            labels: [riskLevel, 'Other'],
            datasets: [{
                data: [value, 100 - value],
                backgroundColor: [color, chartColors.grayLight],
                borderWidth: 0,
                cutout: '70%',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            animation: {
                animateRotate: true,
                duration: 1200,
            },
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false },
            }
        },
        plugins: [{
            id: 'riskCenter',
            afterDraw: function(chart) {
                var width = chart.width;
                var height = chart.height;
                var context = chart.ctx;

                context.restore();

                var fontSize = Math.min(width, height) * 0.12;
                context.font = 'bold ' + fontSize + 'px Inter, sans-serif';
                context.textBaseline = 'middle';
                context.textAlign = 'center';
                context.fillStyle = color;
                context.fillText(riskLevel, width / 2, height / 2);

                context.save();
            }
        }]
    });
}
