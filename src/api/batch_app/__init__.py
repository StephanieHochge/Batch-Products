"""
This package contains the main logic of the batch processing and database access.
"""

from flask import Flask
from src.api.batch_app.config import Config
from src.api.batch_app.db_models import db
from src.api.batch_app.routes import register_routes
import logging
from src.utils import load_app_config


def create_app(logger: logging.Logger, is_test: bool = False) -> Flask:
    """
    Create a flask application.

    :param logger: A logger instance to handle logging.
    :param is_test: Boolean indicating whether to load test configuration.
    :return: Flask application instance.
    """
    # initialize the Flask application
    app = Flask(__name__)

    # load configuration from the Config object
    app.config.from_object(Config)

    # load additional configuration from the YAML config file
    app_config = load_app_config(is_test=is_test)
    app.config["App_config"] = app_config

    # set the logger in the Flask app config
    app.config["Logger"] = logger

    # set up the SQLAlchemy database URI for testing / production
    db_uri = app_config["db_path"]
    if db_uri.startswith("postgresql://"):
        app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
    else:
        app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{str(db_uri)}"

    logger.info(f"Using DB URI: {db_uri}")

    # initialize the database with the Flask app context
    db.init_app(app)
    with app.app_context():
        db.create_all()  # create database tables if they do not exist

    # store the model path in the config
    app.config["ModelPath"] = app_config["model_path"]
    app.config["Model"] = None

    # create a logger for the current module
    logger.info("App created and logging initialized.")

    # log loaded configuration
    logger.info(f"Loaded app configuration: {app_config}")

    # register routes with the application
    register_routes(app)

    return app
