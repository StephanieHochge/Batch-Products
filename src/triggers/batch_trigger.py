from src.triggers.trigger_utils import send_request
from flask import Response
import logging
from typing import Dict

from src.utils import load_app_config, setup_logging


def trigger_batch_process(image_dir: str,
                          logger: logging.Logger,
                          base_url: str = "http://localhost:5000") -> Response | None:
    """
    Send a request to trigger the categorization of a batch of images in a specified directory.

    :param image_dir: Path to the directory containing images to be processed.
    :param logger: A logger instance to handle logging.
    :param base_url: The base URL of the batch processing endpoint, default is 'http://localhost:5000/batch_process'.
    :return: The JSON response from the batch process endpoint.
    """
    url = f"{base_url}/batch_process"

    data = {
        "image_directory": image_dir
    }

    # send the request and retrieve the response in JSON format
    response_json = send_request(url, data, logger)
    return response_json


def trigger_batch_process_config(config: Dict, is_test: bool = False) -> Response | None:
    """
    Configure and trigger the batch processing of images based on the provided configuration.

    :param config: A dictionary containing configuration settings. It should include
        - directories: A dictionary with a target_directory key for the image directory.
        - flask: A dictionary with host and port keys for the Flask server.
    :param is_test: A boolean variable indicating whether the function is used during testing.
    :return: The JSON response from the batch process endpoint, or None if an error occurs.
    """
    # retrieve relevant settings from the config
    image_directory = config["directories"]["target_directory"]
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

    # trigger the batch process and return the response
    result = trigger_batch_process(image_directory, trigger_logger, base_url_config)

    return result


if __name__ == "__main__":
    # load the configuration file
    app_config = load_app_config()

    trigger_batch_process_config(app_config)
