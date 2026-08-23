import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    # Models live in app/models/
    MODEL_FOLDER = os.path.join(BASE_DIR, 'app', 'models')
    
    # Uploads must live inside static so url_for('static') can serve them
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app', 'static', 'uploads')
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
    
    # Ensure folders exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)