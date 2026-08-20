from flask import Flask
from .config import Config


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    from .routes.main import bp as main_bp
    from .routes.dysgraphia import bp as dysgraphia_bp
    from .routes.dyslexia import bp as dyslexia_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(dysgraphia_bp, url_prefix='/dysgraphia')
    app.register_blueprint(dyslexia_bp, url_prefix='/dyslexia')

    return app
