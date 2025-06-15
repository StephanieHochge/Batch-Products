import yaml
import os
import logging
import logging.config
from typing import Dict, List
from pathlib import Path

# get the base directory
base_dir = Path(__file__).resolve().parent.parent


def collect_image_paths(directory: str | Path, extensions: List[str] = None) -> List[Path]:
    """
    Collect all image paths with the given extensions from a directory.

    :param directory: Path to the directory to search for images.
    :param extensions: List of file extensions to include (default: ["*.png", "*.jpg", "*.jpeg"]).
    :return: List of image paths matching the given extensions.
    """
    # use default extensions if none were provided
    if extensions is None:
        extensions = ["*.png", "*.jpg", "*.jpeg"]
    dir_path = Path(directory)

    if not dir_path.is_dir():
        raise ValueError(f"The path {dir_path} is not a valid folder.")

    image_paths = []

    # look for the specified extensions in the provided directory
    for ext in extensions:
        image_paths.extend(dir_path.glob(ext))
    return image_paths


def is_running_in_docker() -> bool:
    """
    Check if the application is running inside a Docker container.

    :return: True if the application is running inside a Docker container. False otherwise.
    """
    return os.getenv("DOCKER_CONTAINER") == "true"


def load_app_config(is_test: bool = False) -> Dict:
    """
    Load the YAML app configuration file and return its content as a dictionary.

    :param is_test: Boolean indicating whether to load test configuration.
    :return: A dictionary containing the data from the YAML file.
    """
    in_docker = is_running_in_docker()

    # load app config depending on whether the app is running in a docker container
    if in_docker:
        # choose the config file based on is_test
        config_filename = "test_app_config_docker.yaml" if is_test else "app_config_docker.yaml"
    else:
        # choose the config file based on is_test
        config_filename = "test_app_config.yaml" if is_test else "app_config.yaml"

    # define the path
    config_path = base_dir / f"configs/{config_filename}"
    # open the YAML configuration file in read mode
    with open(config_path, "r") as file:
        # parse the YAML file and return the content as a dictionary
        return yaml.safe_load(file)


def setup_logging(log_path: Path = None):
    """
    Set up logging configuration from a YAML file.

    :param log_path: Optional path for the log file to override the default in YAML.
    :return:
    """
    is_docker = is_running_in_docker()
    # load logging config depending on whether the app is running in a docker container
    logging_config = "logging_config_docker" if is_docker else "logging_config"
    config_path = base_dir / f"configs/{logging_config}.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # ensure log directory and files exist
    for handler in config["handlers"].values():
        if "filename" in handler:
            log_file = Path(handler["filename"])
            log_file.parent.mkdir(parents=True, exist_ok=True)  # create directory if it does not exist
            log_file.touch(exist_ok=True)  # create file if it does not exist

    # override the default file path in the YAML if log_path is provided
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        for handler in config["handlers"].values():
            if "filename" in handler:
                handler["filename"] = str(log_path)

    logging.config.dictConfig(config)
