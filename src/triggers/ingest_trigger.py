import logging
from pathlib import Path
from src.triggers.trigger_utils import send_request
from flask import Response
from typing import Dict

from src.utils import load_app_config, setup_logging


def trigger_ingestion(logger: logging.Logger,
                      source_dir: str | Path,
                      target_dir: str | Path,
                      min_images: int = 5,
                      max_images: int = 100,
                      base_url: str = "http://localhost:5000") -> Response | None:
    """
    Send a request to trigger the ingestion of images from a source directory to a target directory.

    :param logger: A logger instance to handle logging.
    :param source_dir: Path to the source directory containing images to ingest.
    :param target_dir: Path to the target directory where images should be ingested.
    :param min_images: Minimum number of images to ingest in one run.
    :param max_images: Maximum number of images to ingest in one run.
    :param base_url: The base URL of the ingestion endpoint, default is 'http://localhost:5000/ingest_images'.
    :return: The JSON response from the ingestion endpoint.
    """
    url = f"{base_url}/ingest_images"

    ingestion_data = {
        "source_directory": str(source_dir),
        "target_directory": str(target_dir),
        "min_images_per_run": min_images,
        "max_images_per_run": max_images
    }

    # send the request and retrieve the response in JSON format
    response_json = send_request(url, ingestion_data, logger)
    return response_json


def trigger_ingestion_config(config: Dict, is_test: bool = False) -> Response | None:
    """
    Configure and trigger the ingestion process for images based on the provided configuration.

    :param config: A dictionary containing configuration settings. It should include:
        - directories: A dictionary with 'source_directory' and 'target_directory' keys.
        - batch: A dictionary with 'min_images_per_run' and 'max_images_per_run' keys.
        - flask: A dictionary with host and port keys for the Flask server.
    :param is_test: A boolean variable indicating whether the function is used during tests.
    :return: The JSON response from the ingestion endpoint, or None if an error occurs.
    """
    # retrieve relevant settings from the config
    source_directory = config["directories"]["source_directory"]
    target_directory = config["directories"]["target_directory"]
    minimum_images = config["batch"]["min_images_per_run"]
    maximum_images = config["batch"]["max_images_per_run"]
    flask_host = config["flask"]["host"]
    flask_port = config["flask"]["port"]

    # construct the base URL dynamically from the host and port
    base_url_config = f"http://{flask_host}:{flask_port}"

    # set up logging
    setup_logging()
    if not is_test:
        trigger_logger = logging.getLogger("trigger_logger")
    else:
        trigger_logger = logging.getLogger()

    # trigger the ingestion
    return trigger_ingestion(trigger_logger, source_directory, target_directory,
                             minimum_images, maximum_images, base_url_config)


if __name__ == "__main__":
    # load the configuration file
    app_config = load_app_config()
    trigger_ingestion_config(app_config)
