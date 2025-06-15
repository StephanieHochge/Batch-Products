from src.utils import load_app_config, setup_logging
from src.triggers.trigger_utils import get_db_contents

from typing import Dict
import pandas as pd
import logging


def get_db_contents_config(config: Dict, is_test: bool = False) -> pd.DataFrame | None:
    """
    Returns a DataFrame containing the contents of the database based on the provided configuration.

    :param config: A dictionary containing configuration settings. It should include
        - flask: A dictionary with host and port keys for the Flask server.
    :param is_test: A boolean variable indicating whether the function is used during testing.
    :return: A DataFrame containing the contents of the database based on the provided configuration.
    """
    # retrieve relevant settings from the config
    flask_host = config["flask"]["host"]
    flask_port = config["flask"]["port"]

    # construct the base URL dynamically from the host and port
    base_url = f"http://{flask_host}:{flask_port}"

    # set up logging
    setup_logging()
    if not is_test:
        trigger_logger = logging.getLogger("trigger_logger")
    else:
        trigger_logger = logging.getLogger()

    # log information
    trigger_logger.info(f"Triggering get_db_contents_config with base_url: {base_url}")

    # get the DataFrame containing the database contents based on the constructed base_url
    return get_db_contents(base_url)


if __name__ == "__main__":
    # show all columns and all rows
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    # suppress scientific notation
    pd.options.display.float_format = "{:,.3f}".format

    # load the configuration file
    app_config = load_app_config()

    # get the database contents as a DataFrame
    df = get_db_contents_config(app_config)

    # print the DataFrame
    if isinstance(df, pd.DataFrame):
        print(df)




