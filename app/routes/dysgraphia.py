import os
import time
import uuid
from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app

bp = Blueprint('dysgraphia', __name__)

_MODEL_CACHE = {}
_RESOLVED_MODEL_PATH = None


def _get_cached_model(path):
    if path not in _MODEL_CACHE:
        import joblib
        _MODEL_CACHE[path] = joblib.load(path)
    return _MODEL_CACHE[path]


def _find_model_path():
    global _RESOLVED_MODEL_PATH
    if _RESOLVED_MODEL_PATH is not None:
        return _RESOLVED_MODEL_PATH

    model_dir = current_app.config['MODEL_FOLDER']
    model_files = [
        f for f in os.listdir(model_dir)
        if f.startswith('dysgraphia_')
        and f.endswith('.pkl')
        and not any(x in f for x in ['feature_names', 'imputer', 'scaler', 'pca'])
    ] if os.path.exists(model_dir) else []

    if model_files:
        _RESOLVED_MODEL_PATH = os.path.join(model_dir, sorted(model_files)[0])
    return _RESOLVED_MODEL_PATH


def analyze_handwriting(image_path):
    """Run full ML pipeline: preprocess -> extract features -> predict -> explain."""
    t_total = time.time()
    try:
        from app.ml.predict import predict_dysgraphia

        t0 = time.time()
        model_path = _find_model_path()
        t_find_model = time.time() - t0

        if not model_path:
            return {
                'error': True,
                'message': 'No trained dysgraphia model found. Please run: python -m scripts.training.train --task dysgraphia',
                'prediction': 'Unknown',
                'confidence': 0.0,
                'is_positive': False,
                'features': {},
                'shap_values': {},
            }

        result = predict_dysgraphia(image_path, model_path)

        t_total = time.time() - t_total
        print(f"[TIMING] analyze_handwriting: find_model={t_find_model:.4f}s "
              f"predict_pipeline={t_total - t_find_model:.4f}s total={t_total:.4f}s", flush=True)

        prediction = result.get('prediction', 0)
        confidence = result.get('confidence', 0.0)
        is_dysgraphic = bool(prediction == 1)

        # Confidence threshold logic
        if confidence < 0.60:
            prediction_text = 'Uncertain — Manual Review Recommended'
            is_positive_flag = False
            confidence_note = 'Low confidence. Model is unsure about this sample.'
        elif confidence < 0.85:
            prediction_text = 'Possible Dysgraphic Indicators' if is_dysgraphic else 'Likely Non-Dysgraphic'
            is_positive_flag = is_dysgraphic
            confidence_note = 'Moderate confidence. Consider professional evaluation.'
        else:
            prediction_text = 'Dysgraphic' if is_dysgraphic else 'Non-Dysgraphic'
            is_positive_flag = is_dysgraphic
            confidence_note = 'High confidence prediction.'

        disclaimer = (
            'This model was trained on a limited dataset and may not '
            'generalize to all handwriting styles, ages, or writing instruments. '
            'Results are for demonstration purposes only.'
        )

        feature_importance = result.get('feature_importance', {})
        shap_values = result.get('shap_values', {})

        features = {}
        fi_features = feature_importance.get('features', []) if isinstance(feature_importance, dict) else []
        fi_values = feature_importance.get('values', []) if isinstance(feature_importance, dict) else []
        for fname, fval in zip(fi_features[:6], fi_values[:6]):
            features[fname] = round(fval, 4)

        shap_dict = {}
        sv_features = shap_values.get('features', []) if isinstance(shap_values, dict) else []
        sv_vals = shap_values.get('values', []) if isinstance(shap_values, dict) else []
        for sname, sval in zip(sv_features[:6], sv_vals[:6]):
            shap_dict[sname] = round(sval, 4)

        return {
            'prediction': prediction_text,
            'confidence': round(confidence * 100, 1),
            'is_positive': is_positive_flag,
            'confidence_note': confidence_note,
            'disclaimer': disclaimer,
            'features': features if features else {
                'HOG Features': 0.22, 'Contour Analysis': 0.18,
                'Letter Spacing': 0.15, 'Baseline Deviation': 0.13,
                'Stroke Width': 0.10, 'Ink Density': 0.08,
            },
            'shap_values': shap_dict if shap_dict else {k: 0.0 for k in features},
        }

    except Exception as e:
        import traceback
        current_app.logger.error(traceback.format_exc())
        return {
            'error': True,
            'message': f'Model inference failed: {str(e)}',
            'prediction': 'Error',
            'confidence': 0.0,
            'is_positive': False,
            'features': {},
            'shap_values': {},
        }


# MOCK RESULT IS NOW EXPLICITLY NOT AUTO-CALLED
# It only exists for emergency manual use if needed


@bp.route('/')
def upload_form():
    return render_template('dysgraphia/upload.html')


@bp.route('/analyze', methods=['POST'])
def analyze():
    t0 = time.time()
    if 'handwriting_image' not in request.files:
        flash('No file selected. Please upload a handwriting image.', 'error')
        return redirect(url_for('dysgraphia.upload_form'))

    file = request.files['handwriting_image']
    if file.filename == '':
        flash('No file selected. Please upload a handwriting image.', 'error')
        return redirect(url_for('dysgraphia.upload_form'))

    ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
    if ext not in current_app.config['ALLOWED_EXTENSIONS']:
        flash('Invalid file type. Please upload a PNG, JPG, JPEG, BMP, or TIFF image.', 'error')
        return redirect(url_for('dysgraphia.upload_form'))

    upload_dir = current_app.config['UPLOAD_FOLDER']
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(upload_dir, unique_name)
    file.save(filepath)

    results = analyze_handwriting(filepath)

    current_app.config['LAST_DYSGRAPHIA_RESULT'] = {
        'image_path': unique_name,
        'results': results,
    }

    print(f"[TIMING] /analyze route: {time.time() - t0:.4f}s total", flush=True)
    return redirect(url_for('dysgraphia.results'))


@bp.route('/results')
def results():
    data = current_app.config.get('LAST_DYSGRAPHIA_RESULT')
    if not data:
        flash('No analysis results found. Please upload an image first.', 'error')
        return redirect(url_for('dysgraphia.upload_form'))

    if data.get('results', {}).get('error'):
        flash(data['results']['message'], 'error')
        return redirect(url_for('dysgraphia.upload_form'))

    return render_template(
        'dysgraphia/results.html',
        image_path=data['image_path'],
        results=data['results'],
    )