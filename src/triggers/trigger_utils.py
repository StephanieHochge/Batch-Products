import requests
import logging
import pandas as pd
from typing import Any, Dict, Optional, Union

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def send_request(url: str, data: Dict[str, Union[str, int]],
                 logger: logging.Logger) -> Optional[Dict[str, Any]]:
    """
    Send a POST request with JSON data to the specified URL and handle the response.

    :param url: The URL to which the request is sent.
    :param data: A dictionary containing the JSON data to send with the request.
    :param logger: A logger instance to handle logging.
    :return: The JSON response as a dictionary if the request is successful; None otherwise.
    """
    try:
        # post and log the request
        logger.info(f"Requesting to {url} with data: {data}...")
        response = requests.post(url, json=data)
        response.raise_for_status()  # return an HTTP error object if an error has occurred

        # log and return responses based on the status code of the response
        if response.status_code == 200:
            logger.info(f"Request to {url} successful.")
            return response.json()
        elif response.status_code == 204:
            logger.warning("No images available for processing found.")
        else:
            logger.error(f"Error: {response.status_code}, {response.text}")

    # handle possible errors
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error: {e}")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {e}")


def get_db_contents(base_url: str, n_rows: Optional[int] = None) -> pd.DataFrame | None:
    """
    Fetch database contents from a given base URL in JSON format and converts it into a pandas DataFrame.

    :param base_url: The base URL to fetch contents from.
    :param n_rows: Optional number of rows to return from the end of the DataFrame.
    :return: DataFrame containing the database contents, or None if the request fails.
    """
    url = f"{base_url}/get_db_as_json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df.tail(n_rows) if n_rows else df
    except requests.RequestException as e:
        print(f"Error during data access: {e}")
        return None
