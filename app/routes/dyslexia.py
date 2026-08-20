import os
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app

bp = Blueprint('dyslexia', __name__)

FEATURE_NAMES = [
    'Words Read Per Minute',
    'Reading Errors',
    'Letter Reversals',
    'Comprehension Score',
    'Spelling Errors',
]


def screen_dyslexia(data):
    """Placeholder: replace with actual model inference."""
    try:
        import joblib

        model_path = os.path.join(current_app.config['MODEL_FOLDER'], 'dyslexia_model.pkl')
        if not os.path.exists(model_path):
            return generate_mock_result(data)

        model = joblib.load(model_path)
        features = np.array([[
            float(data['wpm']),
            float(data['errors']),
            float(data['reversals']),
            float(data['comprehension']),
            float(data['spelling_errors']),
        ]])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        risk_level = 'Low' if prediction == 0 else ('Medium' if prediction == 1 else 'High')
        confidence = round(max(probability) * 100, 1)

        return {
            'risk_level': risk_level,
            'confidence': confidence,
            'is_positive': prediction != 0,
            'child_name': data.get('child_name', 'Not provided'),
            'child_age': data.get('age', 'N/A'),
            'grade_level': data.get('grade', 'N/A'),
            'feature_contributions': compute_mock_contributions(data),
            'shap_values': generate_mock_shap(data),
            'key_indicators': generate_key_indicators(data),
        }
    except Exception:
        return generate_mock_result(data)


def generate_mock_result(data):
    wpm = float(data.get('wpm', 60))
    errors = float(data.get('errors', 5))
    reversals = float(data.get('reversals', 3))
    comprehension = float(data.get('comprehension', 70))
    spelling_errors = float(data.get('spelling_errors', 4))

    score = (wpm / 150) * 0.25 + (1 - errors / 20) * 0.2 + (1 - reversals / 10) * 0.25 + \
            (comprehension / 100) * 0.15 + (1 - spelling_errors / 15) * 0.15

    if score > 0.65:
        risk = 'Low'
    elif score > 0.4:
        risk = 'Medium'
    else:
        risk = 'High'

    confidence = round(abs(score - 0.5) * 200, 1)
    confidence = max(confidence, 55.0)
    confidence = min(confidence, 98.0)

    return {
        'risk_level': risk,
        'confidence': confidence,
        'is_positive': risk != 'Low',
        'child_name': data.get('child_name', 'Not provided'),
        'child_age': data.get('age', 'N/A'),
        'grade_level': data.get('grade', 'N/A'),
        'feature_contributions': compute_mock_contributions(data),
        'shap_values': generate_mock_shap(data),
        'key_indicators': generate_key_indicators(data),
    }


def compute_mock_contributions(data):
    contributions = {}
    for name in FEATURE_NAMES:
        contributions[name] = round(np.random.uniform(-5, 5), 2)
    return contributions


def generate_mock_shap(data):
    return {name: round(np.random.uniform(-3, 3), 3) for name in FEATURE_NAMES}


def generate_key_indicators(data):
    indicators = []
    wpm = float(data.get('wpm', 60))
    errors = float(data.get('errors', 5))
    reversals = float(data.get('reversals', 3))
    comprehension = float(data.get('comprehension', 70))
    spelling_errors = float(data.get('spelling_errors', 4))

    if wpm < 80:
        indicators.append({
            'feature': 'Reading Speed',
            'value': f'{int(wpm)} WPM',
            'description': 'Reading speed is below expected range for the age group.',
            'severity': 'high' if wpm < 50 else 'medium',
        })
    if errors > 5:
        indicators.append({
            'feature': 'Reading Errors',
            'value': f'{int(errors)} errors',
            'description': 'Elevated number of errors during oral reading.',
            'severity': 'high' if errors > 10 else 'medium',
        })
    if reversals > 3:
        indicators.append({
            'feature': 'Letter Reversals',
            'value': f'{int(reversals)} reversals',
            'description': 'Frequent letter/number reversals detected (b/d, p/q confusion).',
            'severity': 'high' if reversals > 6 else 'medium',
        })
    if comprehension < 60:
        indicators.append({
            'feature': 'Comprehension',
            'value': f'{int(comprehension)}%',
            'description': 'Reading comprehension below expected level.',
            'severity': 'high' if comprehension < 40 else 'medium',
        })
    if spelling_errors > 5:
        indicators.append({
            'feature': 'Spelling',
            'value': f'{int(spelling_errors)} errors',
            'description': 'Significant spelling difficulties observed.',
            'severity': 'high' if spelling_errors > 10 else 'medium',
        })

    if not indicators:
        indicators.append({
            'feature': 'Overall',
            'value': 'Within normal range',
            'description': 'No significant behavioral indicators of dyslexia detected.',
            'severity': 'low',
        })

    return indicators


@bp.route('/')
def test_form():
    return render_template('dyslexia/test.html')


@bp.route('/screen', methods=['POST'])
def screen():
    required_fields = ['wpm', 'errors', 'reversals', 'comprehension', 'spelling_errors']
    missing = [f for f in required_fields if not request.form.get(f)]

    if missing:
        flash(f'Missing required fields: {", ".join(missing)}', 'error')
        return redirect(url_for('dyslexia.test_form'))

    data = {
        'child_name': request.form.get('child_name', ''),
        'age': request.form.get('age', ''),
        'grade': request.form.get('grade', ''),
        'wpm': request.form.get('wpm'),
        'errors': request.form.get('errors'),
        'reversals': request.form.get('reversals'),
        'comprehension': request.form.get('comprehension'),
        'spelling_errors': request.form.get('spelling_errors'),
    }

    results = screen_dyslexia(data)
    current_app.config['LAST_DYSLEXIA_RESULT'] = results

    return redirect(url_for('dyslexia.results'))


@bp.route('/results')
def results():
    data = current_app.config.get('LAST_DYSLEXIA_RESULT')
    if not data:
        flash('No screening results found. Please complete the test first.', 'error')
        return redirect(url_for('dyslexia.test_form'))

    return render_template('dyslexia/results.html', results=data)
