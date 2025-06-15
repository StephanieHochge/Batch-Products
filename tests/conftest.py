import logging

import pytest
from pathlib import Path
from flask import Flask
from flask.testing import FlaskClient
from typing import Generator, Dict, Any
from src.api.batch_app import create_app
from src.api.batch_app.db_models import db
from src.utils import setup_logging, load_app_config


@pytest.fixture(scope="session", autouse=True)
def logger():
    """
    Configure and provide the root logger for test executions. Ensure that the logger is configured once per test
    session and remains available through the session.

    :return: The configured root logger.
    """
    setup_logging()
    return logging.getLogger()


@pytest.fixture(scope="session", autouse=True)
def config() -> Dict[str, Any]:
    """
    Load and provide application configuration for test sessions.

    :return: The application configuration loaded with test-specific settings.
    """
    return load_app_config(is_test=True)


@pytest.fixture(scope="session")
def app(logger) -> Generator[Flask, None, None]:
    """
    Fixture to create and configure a Flask application for testing.

    This fixture sets up the application and database once per test session.
    It ensures that the database is created before tests run and cleaned up after the session ends.

    :param logger: The logger fixture.
    :yield: Flask app instance configured for testing.
    """
    app = create_app(logger=logger, is_test=True)

    # set up the database schema for the session
    with app.app_context():
        db.create_all()

    yield app

    # teardown database after the session
    with app.app_context():
        db.drop_all()


@pytest.fixture(scope="session")
def client(app: Flask) -> FlaskClient:
    """
    Fixture to provide a Flask test client used to simulate HTTP requests to the application without running
    the server.

    :param app: the Flask application instance.
    :return: Flask test client.
    """
    return app.test_client()


@pytest.fixture(scope="function", autouse=True)
def provide_app_context(app: Flask) -> Generator[None, None, None]:
    """
    Fixture to ensure each test runs within an application context.
    Important for accessing app-specific resources like the database.

    :param app: the Flask application instance.
    :yield: Application context for the duration of the test.
    """
    with app.app_context():
        yield


@pytest.fixture(scope="function")
def setup_test_data(config: Dict[str, Any]) -> Generator[None, None, None]:
    """
    Fixture to prepare and clean up test data directories.

    :param config: The application configuration for test settings.
    :yield: The setup and teardown actions are automatically handled.
    """

    incoming_folder = Path(config["directories"]["target_directory"])
    processed_folder = Path(config["directories"]["processed_directory"])

    # make sure that folders exist
    incoming_folder.mkdir(parents=True, exist_ok=True)
    processed_folder.mkdir(parents=True, exist_ok=True)

    # clean up any existing files in these directories before each test
    extensions = ["*.png", "*.jpg", "*.jpeg"]
    for folder in [incoming_folder, processed_folder]:
        for ext in extensions:
            for file in folder.glob(ext):
                file.unlink()

    yield

    # clean up any test-generated files after each test
    for folder in [incoming_folder, processed_folder]:
        for ext in extensions:
            for file in folder.glob(ext):
                file.unlink()


@pytest.fixture(scope="function")
def ingestion_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fixture to provide sample ingestion data for testing.

    :param config: The application configuration for test settings.
    :return: a dictionary containing source and target directory paths along with min and max image counts.
    """
    return {
        "source_directory": config["directories"]["source_directory"],
        "target_directory": config["directories"]["target_directory"],
        "min_images_per_run": config["batch"]["min_images_per_run"],
        "max_images_per_run": config["batch"]["max_images_per_run"],
    }
