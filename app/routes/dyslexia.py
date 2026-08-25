import os
import glob
import sys
import time
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
import joblib

bp = Blueprint('dyslexia', __name__)

FORM_FEATURE_NAMES = [
    'wpm', 'errors', 'reversals', 'comprehension', 'spelling_errors',
]

AGGREGATE_FEATURE_NAMES = [
    'age', 'gender', 'total_clicks', 'total_hits',
    'total_misses', 'total_score', 'mean_accuracy', 'mean_missrate'
]

_MODEL_CACHE = {}
_FORM_MODEL_PATH = None
_AGG_MODEL_PATH = None
_IMPUTER_EXISTS = {}
_SCALER_EXISTS = {}


def _get_cached_model(path):
    if path not in _MODEL_CACHE:
        _MODEL_CACHE[path] = joblib.load(path)
    return _MODEL_CACHE[path]


def _find_form_model():
    global _FORM_MODEL_PATH
    if _FORM_MODEL_PATH is not None:
        return _FORM_MODEL_PATH

    model_dir = current_app.config['MODEL_FOLDER']
    form_model_files = [
        f for f in glob.glob(os.path.join(model_dir, 'dyslexia_form_*.pkl'))
        if 'feature_names' not in f and 'imputer' not in f and 'scaler' not in f
    ]

    if not form_model_files:
        tabular_files = [
            f for f in glob.glob(os.path.join(model_dir, 'dyslexia_tabular_*.pkl'))
            if 'feature_names' not in f and 'imputer' not in f and 'scaler' not in f
        ]
        for tf in tabular_files:
            try:
                m = _get_cached_model(tf)
                if hasattr(m, 'n_features_in_') and m.n_features_in_ == 5:
                    form_model_files.append(tf)
                    break
            except Exception:
                continue

    if form_model_files:
        _FORM_MODEL_PATH = sorted(form_model_files)[0]
    return _FORM_MODEL_PATH


def _find_agg_model():
    global _AGG_MODEL_PATH
    if _AGG_MODEL_PATH is not None:
        return _AGG_MODEL_PATH

    model_dir = current_app.config['MODEL_FOLDER']
    agg_model_files = [
        f for f in glob.glob(os.path.join(model_dir, 'dyslexia_aggregate_*.pkl'))
        if 'feature_names' not in f and 'imputer' not in f and 'scaler' not in f
    ]

    if agg_model_files:
        _AGG_MODEL_PATH = sorted(agg_model_files)[0]
    return _AGG_MODEL_PATH


def _path_exists_cached(path):
    if path not in _IMPUTER_EXISTS:
        _IMPUTER_EXISTS[path] = os.path.exists(path)
    return _IMPUTER_EXISTS[path]


def _log(msg):
    sys.stderr.write(f"[DYSLEXIA] {msg}\n")
    sys.stderr.flush()


def screen_dyslexia(data):
    """Reading-metrics form -> tries 5-feature model -> heuristic fallback."""
    t0 = time.time()
    try:
        model_path = _find_form_model()

        if model_path:
            model = _get_cached_model(model_path)
            raw_features = np.array([[
                float(data.get('wpm', 0)),
                float(data.get('errors', 0)),
                float(data.get('reversals', 0)),
                float(data.get('comprehension', 0)),
                float(data.get('spelling_errors', 0)),
            ]])

            if hasattr(model, 'n_features_in_') and model.n_features_in_ == 5:
                prediction = model.predict(raw_features)[0]
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(raw_features)[0]
                    confidence = round(float(max(proba)) * 100, 1)
                else:
                    confidence = 75.0
                risk = 'Low' if prediction == 0 else 'Medium' if prediction == 1 else 'High'
                _log(f"REAL ML (5-feature form model) -> Risk: {risk}, Conf: {confidence}%")
                print(f"[TIMING] screen_dyslexia (ML): {time.time() - t0:.4f}s", flush=True)
                return _build_result(data, risk, confidence, raw_features[0], model=model, method='ml_model')

        _log("HEURISTIC FALLBACK (no compatible 5-feature model)")
        print(f"[TIMING] screen_dyslexia (heuristic): {time.time() - t0:.4f}s", flush=True)
        return _heuristic_screen(data)

    except Exception as e:
        _log(f"ERROR: {str(e)}")
        import traceback
        current_app.logger.error(traceback.format_exc())
        return {
            'error': True,
            'message': f'Screening failed: {str(e)}',
            'risk_level': 'Error',
            'confidence': 0.0,
            'is_positive': False,
        }


def screen_dyslexia_aggregate(data):
    """Visual-search aggregate form -> 8-feature aggregate model."""
    t0 = time.time()
    try:
        model_path = _find_agg_model()

        if not model_path:
            _log("ERROR: No aggregate model found.")
            return {
                'error': True,
                'message': 'No aggregate model found. Run: python -m scripts.training.train --task dyslexia_aggregate',
                'risk_level': 'Unknown',
                'confidence': 0.0,
                'is_positive': False,
            }

        model = _get_cached_model(model_path)

        agg_features = np.array([[
            float(data.get('age', 0)),
            float(data.get('gender', 0)),
            float(data.get('total_clicks', 0)),
            float(data.get('total_hits', 0)),
            float(data.get('total_misses', 0)),
            float(data.get('total_score', 0)),
            float(data.get('mean_accuracy', 0)),
            float(data.get('mean_missrate', 0)),
        ]])

        model_dir = current_app.config['MODEL_FOLDER']
        imputer_path = os.path.join(model_dir, 'dyslexia_aggregate_imputer.pkl')
        scaler_path = os.path.join(model_dir, 'dyslexia_aggregate_scaler.pkl')

        X = agg_features
        if _path_exists_cached(imputer_path):
            imputer = _get_cached_model(imputer_path)
            X = imputer.transform(X)
        if _path_exists_cached(scaler_path):
            scaler = _get_cached_model(scaler_path)
            X = scaler.transform(X)

        prediction = model.predict(X)[0]
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            confidence = round(float(max(proba)) * 100, 1)
        else:
            confidence = 75.0

        risk = 'Low' if prediction == 0 else 'Medium' if prediction == 1 else 'High'
        _log(f"REAL ML (8-feature aggregate model) -> Risk: {risk}, Conf: {confidence}%")

        feature_names = ['Age', 'Gender', 'Total Clicks', 'Total Hits', 'Total Misses',
                        'Total Score', 'Mean Accuracy', 'Mean Miss Rate']

        contributions = _compute_model_contributions(model, X[0], feature_names)
        shap_vals = _compute_model_shap(model, X, feature_names)

        indicators = []
        acc = float(data.get('mean_accuracy', 70))
        miss = float(data.get('mean_missrate', 20))
        hits = float(data.get('total_hits', 50))
        misses = float(data.get('total_misses', 20))

        if acc < 60:
            indicators.append({
                'feature': 'Accuracy', 'value': f'{acc:.1f}%',
                'description': 'Visual search accuracy is below expected level.',
                'severity': 'high' if acc < 40 else 'medium',
            })
        if miss > 30:
            indicators.append({
                'feature': 'Miss Rate', 'value': f'{miss:.1f}%',
                'description': 'High rate of incorrect clicks in visual search task.',
                'severity': 'high' if miss > 50 else 'medium',
            })
        if misses > hits * 0.5:
            indicators.append({
                'feature': 'Error Pattern', 'value': f'{misses} misses vs {hits} hits',
                'description': 'More errors than expected relative to correct responses.',
                'severity': 'high',
            })

        if not indicators:
            indicators.append({
                'feature': 'Overall', 'value': 'Within normal range',
                'description': 'Visual search performance is within expected parameters.',
                'severity': 'low',
            })

        return {
            'risk_level': risk,
            'confidence': confidence,
            'is_positive': risk != 'Low',
            'child_name': data.get('child_name', 'Not provided'),
            'child_age': data.get('age', 'N/A'),
            'grade_level': data.get('grade', 'N/A'),
            'feature_contributions': contributions,
            'shap_values': shap_vals,
            'key_indicators': indicators,
            'method': 'ml_model',
            'method_note': 'ML Model (Random Forest on Rello Visual Search Aggregates)',
        }

        print(f"[TIMING] screen_dyslexia_aggregate: {time.time() - t0:.4f}s", flush=True)
        return result

    except Exception as e:
        _log(f"ERROR in aggregate screening: {str(e)}")
        import traceback
        current_app.logger.error(traceback.format_exc())
        return {
            'error': True,
            'message': f'Aggregate screening failed: {str(e)}',
            'risk_level': 'Error',
            'confidence': 0.0,
            'is_positive': False,
        }


def _compute_model_contributions(model, x, feature_names):
    """Compute real feature contributions from the model itself."""
    contributions = {}
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        contributions = {name: round(float(imp) * 100, 2) for name, imp in zip(feature_names, importances)}
    elif hasattr(model, 'coef_'):
        coefs = np.abs(np.asarray(model.coef_)).flatten()
        contributions = {name: round(float(c) * 100, 2) for name, c in zip(feature_names, coefs)}
    else:
        contributions = {name: 0.0 for name in feature_names}
    return contributions


def _compute_model_shap(model, X, feature_names):
    """Compute SHAP values using the model's own explanation mechanism."""
    try:
        from app.ml.explainability import get_shap_explanation, generate_shap_plot_data
        model_type = "tree" if hasattr(model, "estimators_") else "kernel"
        shap_vals = get_shap_explanation(model, X, feature_names, model_type=model_type)
        if shap_vals is not None:
            return generate_shap_plot_data(shap_vals, feature_names, instance_idx=0)
    except Exception:
        pass
    return {name: 0.0 for name in feature_names}


def _heuristic_screen(data):
    wpm = float(data.get('wpm', 60))
    errors = float(data.get('errors', 5))
    reversals = float(data.get('reversals', 3))
    comprehension = float(data.get('comprehension', 70))
    spelling_errors = float(data.get('spelling_errors', 4))

    score = (
        (wpm / 150) * 0.25 +
        (1 - errors / 20) * 0.20 +
        (1 - reversals / 10) * 0.25 +
        (comprehension / 100) * 0.15 +
        (1 - spelling_errors / 15) * 0.15
    )

    if score > 0.65:
        risk = 'Low'
    elif score > 0.40:
        risk = 'Medium'
    else:
        risk = 'High'

    confidence = round(abs(score - 0.5) * 200, 1)
    confidence = max(confidence, 55.0)
    confidence = min(confidence, 98.0)

    return _build_result(data, risk, confidence,
                         np.array([wpm, errors, reversals, comprehension, spelling_errors]),
                         model=None, method='heuristic')


def _build_result(data, risk_level, confidence, feature_values, model=None, method='heuristic'):
    wpm, errors, reversals, comprehension, spelling = feature_values[:5]

    form_names = ['Words Read Per Minute', 'Reading Errors', 'Letter Reversals',
                  'Comprehension Score', 'Spelling Errors']

    if model is not None and hasattr(model, 'coef_'):
        coefs = np.abs(np.asarray(model.coef_)).flatten()
        contributions = {name: round(float(c) * 100, 2) for name, c in zip(form_names, coefs)}
    elif model is not None and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        contributions = {name: round(float(imp) * 100, 2) for name, imp in zip(form_names, importances)}
    else:
        contributions = {
            'Words Read Per Minute': round((wpm - 100) / 20, 2),
            'Reading Errors': round(errors * 0.4, 2),
            'Letter Reversals': round(reversals * 0.6, 2),
            'Comprehension Score': round((60 - comprehension) * 0.08, 2),
            'Spelling Errors': round(spelling * 0.4, 2),
        }

    shap_vals = {}
    if model is not None:
        try:
            X = np.array([[wpm, errors, reversals, comprehension, spelling]], dtype=np.float64)
            shap_vals = _compute_model_shap(model, X, form_names)
        except Exception:
            shap_vals = {name: 0.0 for name in form_names}
    else:
        shap_vals = {name: 0.0 for name in form_names}

    indicators = []
    if wpm < 80:
        indicators.append({
            'feature': 'Reading Speed', 'value': f'{int(wpm)} WPM',
            'description': 'Reading speed is below expected range for the age group.',
            'severity': 'high' if wpm < 50 else 'medium',
        })
    if errors > 5:
        indicators.append({
            'feature': 'Reading Errors', 'value': f'{int(errors)} errors',
            'description': 'Elevated number of errors during oral reading.',
            'severity': 'high' if errors > 10 else 'medium',
        })
    if reversals > 3:
        indicators.append({
            'feature': 'Letter Reversals', 'value': f'{int(reversals)} reversals',
            'description': 'Frequent letter/number reversals detected (b/d, p/q confusion).',
            'severity': 'high' if reversals > 6 else 'medium',
        })
    if comprehension < 60:
        indicators.append({
            'feature': 'Comprehension', 'value': f'{int(comprehension)}%',
            'description': 'Reading comprehension below expected level.',
            'severity': 'high' if comprehension < 40 else 'medium',
        })
    if spelling > 5:
        indicators.append({
            'feature': 'Spelling', 'value': f'{int(spelling)} errors',
            'description': 'Significant spelling difficulties observed.',
            'severity': 'high' if spelling > 10 else 'medium',
        })

    if not indicators:
        indicators.append({
            'feature': 'Overall', 'value': 'Within normal range',
            'description': 'No significant behavioral indicators of dyslexia detected.',
            'severity': 'low',
        })

    return {
        'risk_level': risk_level,
        'confidence': confidence,
        'is_positive': risk_level != 'Low',
        'child_name': data.get('child_name', 'Not provided'),
        'child_age': data.get('age', 'N/A'),
        'grade_level': data.get('grade', 'N/A'),
        'feature_contributions': contributions,
        'shap_values': shap_vals,
        'key_indicators': indicators,
        'method': method,
        'method_note': 'ML Model' if method == 'ml_model' else 'Rule-Based Heuristic (Educational Psychology Benchmarks)',
    }


@bp.route('/')
def test_form():
    return render_template('dyslexia/test.html')


@bp.route('/screen', methods=['POST'])
def screen():
    t0 = time.time()
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
    print(f"[TIMING] /screen route: {time.time() - t0:.4f}s total", flush=True)
    return redirect(url_for('dyslexia.results'))


@bp.route('/aggregate')
def aggregate_form():
    return render_template('dyslexia/test_aggregate.html')


@bp.route('/screen_aggregate', methods=['POST'])
def screen_aggregate():
    t0 = time.time()
    required_fields = ['age', 'gender', 'total_clicks', 'total_hits', 'total_misses',
                       'total_score', 'mean_accuracy', 'mean_missrate']
    missing = [f for f in required_fields if not request.form.get(f)]

    if missing:
        flash(f'Missing required fields: {", ".join(missing)}', 'error')
        return redirect(url_for('dyslexia.aggregate_form'))

    data = {
        'child_name': request.form.get('child_name', ''),
        'age': request.form.get('age'),
        'grade': request.form.get('grade', ''),
        'gender': request.form.get('gender'),
        'total_clicks': request.form.get('total_clicks'),
        'total_hits': request.form.get('total_hits'),
        'total_misses': request.form.get('total_misses'),
        'total_score': request.form.get('total_score'),
        'mean_accuracy': request.form.get('mean_accuracy'),
        'mean_missrate': request.form.get('mean_missrate'),
    }

    results = screen_dyslexia_aggregate(data)
    current_app.config['LAST_DYSLEXIA_RESULT'] = results
    print(f"[TIMING] /screen_aggregate route: {time.time() - t0:.4f}s total", flush=True)
    return redirect(url_for('dyslexia.results'))


@bp.route('/results')
def results():
    data = current_app.config.get('LAST_DYSLEXIA_RESULT')
    if not data:
        flash('No screening results found. Please complete the test first.', 'error')
        return redirect(url_for('dyslexia.test_form'))

    if data.get('error'):
        flash(data['message'], 'error')
        return redirect(url_for('dyslexia.test_form'))

    return render_template('dyslexia/results.html', results=data)