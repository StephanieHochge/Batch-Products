from src.triggers.ingest_trigger import trigger_ingestion, trigger_ingestion_config
from src.triggers.batch_trigger import trigger_batch_process, trigger_batch_process_config
from src.triggers.print_db_contents import get_db_contents_config
from src.triggers.trigger_utils import get_db_contents
import requests
import pytest
import logging
import pandas as pd

from typing import Dict, Any, Type


@pytest.fixture
def base_url(config: [Dict[str, Any]]) -> str:
    """
    Fixture to construct the base URL using the provided configuration.

    :param config: Configuration dictionary containing host and port.
    :return: The complete base URL as a string, e.g., 'http://localhost:5000'.
    """
    return f"http://{config["flask"]["host"]}:{config["flask"]["port"]}"


def mock_post_request(requests_mock, url: str, response: Any = None,
                      exception: Type[ConnectionError] | Exception = None) -> None:
    """
    Mock a POST request to the given url.

    :param requests_mock: The mock object for HTTP requests.
    :param url: The URL to mock.
    :param response: JSON response to return.
    :param exception: An optional exception to simulate a network error.
    """
    if exception:
        # simulate a network error by raising the given exception
        requests_mock.post(url, exc=exception)
    else:
        # simulate a successful POST request and return the provided response
        requests_mock.post(url, json=response)


def mock_get_request(requests_mock, url: str, response: Any = None,
                     exception: Type[ConnectionError] | Exception = None,
                     status_code: int = 200) -> None:
    """
    Mock a GET request to the given URL.

    :param requests_mock: The mock object for HTTP requests.
    :param url: The URL to mock.
    :param response: JSON response to return.
    :param exception: An optional exception to simulate a network error.
    :param status_code: HTTP status code to return in the response (default: 200).
    """
    if exception:
        # simulate a network error by raising the given exception
        requests_mock.get(url, exc=exception)
    else:
        # simulate a successful or failed GET request
        requests_mock.get(url, json=response, status_code=status_code)


class TestTriggerIngestion:
    """
    Test class for testing ingestion trigger functions.
    """

    def test_trigger_ingestion_success(self,
                                       requests_mock,
                                       ingestion_data: Dict[str, Any],
                                       logger: logging.Logger,
                                       base_url: str) -> None:
        """
        Test the successful execution of the ingestion trigger.

        :param requests_mock: The mock object for HTTP requests.
        :param ingestion_data: Sample data for ingestion including source and target directories.
        :param logger: Logger object used for logging during the process.
        :param base_url: The base URL of the API.
        """
        url = f"{base_url}/ingest_images"
        mock_response = {"status": "success", "ingested_images": 10}

        # mock the POST request and return a successful response
        mock_post_request(requests_mock, url, response=mock_response)

        # call the trigger function
        response = trigger_ingestion(
            logger,
            ingestion_data["source_directory"],
            ingestion_data["target_directory"],
            ingestion_data["min_images_per_run"],
            ingestion_data["max_images_per_run"],
            base_url
        )

        # assertions to verify the behavior
        assert response == mock_response
        assert requests_mock.called
        assert requests_mock.call_count == 1
        assert requests_mock.last_request.json() == ingestion_data

    def test_trigger_ingestion_connection_error(self,
                                                requests_mock,
                                                ingestion_data: Dict[str, Any],
                                                logger: logging.Logger,
                                                base_url: str) -> None:
        """
        Test the ingestion trigger when a connection error occurs (e.g., a server is down).

        :param requests_mock: The mock object for HTTP requests.
        :param ingestion_data: Sample data for ingestion including source and target directories.
        :param logger: Logger object used for logging during the process.
        :param base_url: The base URL of the API.
        """
        url = f"{base_url}/ingest_images"

        # mock the POST request to raise a connection error
        mock_post_request(requests_mock, url, exception=requests.exceptions.ConnectionError)

        # call the trigger function and expect it to return None due to the error
        response = trigger_ingestion(
            logger,
            ingestion_data["source_directory"],
            ingestion_data["target_directory"],
            ingestion_data["min_images_per_run"],
            ingestion_data["max_images_per_run"],
            base_url
        )

        # assertions to verify the behavior
        assert response is None
        assert requests_mock.called

    def test_trigger_ingestion_config_success(self,
                                              requests_mock,
                                              config: Dict[str, Any],
                                              base_url: str) -> None:
        """
        Test the ingestion trigger using configuration values.

        :param requests_mock: The mock object for HTTP requests.
        :param config: The configuration dictionary to use for testing.
        :param base_url: The base URL of the API.
        """
        url = f"{base_url}/ingest_images"
        mock_response = {"status": "success", "ingested_images": 10}

        # mock the POST request to ingest images
        mock_post_request(requests_mock, url, response=mock_response)
        response = trigger_ingestion_config(config, is_test=True)

        # ensure the mocked response is returned correctly
        assert response == mock_response
        assert requests_mock.called
        assert requests_mock.call_count == 1
        assert requests_mock.last_request.json() == {
            "source_directory": config["directories"]["source_directory"],
            "target_directory": config["directories"]["target_directory"],
            "min_images_per_run": config["batch"]["min_images_per_run"],
            "max_images_per_run": config["batch"]["max_images_per_run"]
        }

    def test_trigger_ingestion_config_connection_error(self,
                                                       requests_mock,
                                                       config: Dict[str, Any],
                                                       base_url: str) -> None:
        """
        Test the ingestion trigger using configuration values when a connection error occurs.

        :param requests_mock: The mock object for HTTP requests
        :param config: The configuration dictionary to use for testing.
        :param base_url: The base URL of the API.
        """
        url = f"{base_url}/ingest_images"

        # mock the POST request to raise a connection error
        mock_post_request(requests_mock, url, exception=requests.exceptions.ConnectionError)

        response = trigger_ingestion_config(config, is_test=True)

        # assertions to verify the behavior
        assert response is None
        assert requests_mock.called


class TestTriggerBatchProcess:
    """
    Test class for testing batch process trigger functions.
    """
    def test_trigger_batch_process_success(self,
                                           requests_mock,
                                           logger: logging.Logger,
                                           config: Dict[str, Any],
                                           base_url: str) -> None:
        """
        Test the successful execution of the batch process trigger.

        :param requests_mock: The mock object for HTTP requests.
        :param logger: Logger object used for logging during the process.
        :param config: The configuration dictionary to use for testing.
        :param base_url: The base URL of the API.
        """
        url = f"{base_url}/batch_process"
        image_data = {"image_directory": config["directories"]["target_directory"]}
        mock_response = {"status": "success", "processed_files": 5}

        # mock the POST request and return a successful response
        mock_post_request(requests_mock, url, response=mock_response)

        response = trigger_batch_process(image_data["image_directory"], logger, base_url=base_url)

        # assertions to verify the behavior
        assert response == mock_response
        assert requests_mock.called
        assert requests_mock.call_count == 1
        assert requests_mock.last_request.json() == image_data

    def test_trigger_batch_process_connection_error(self,
                                                    requests_mock,
                                                    logger: logging.Logger,
                                                    config: Dict[str, Any],
                                                    base_url: str) -> None:
        """
        Test the batch process trigger when a connection error occurs.

        :param requests_mock: The mock object for HTTP requests.
        :param logger: Logger object used for logging during the process.
        :param config: The configuration dictionary to use for testing.
        :param base_url: The base URL of the API.
        """
        url = f"{base_url}/batch_process"

        # mock the POST request to raise a connection error
        mock_post_request(requests_mock, url, exception=requests.exceptions.ConnectionError)

        response = trigger_batch_process(config["directories"]["target_directory"], logger, base_url)

        # assertions to verify the behavior
        assert response is None
        assert requests_mock.called

    def test_trigger_batch_process_config_success(self,
                                                  requests_mock,
                                                  config: Dict[str, Any],
                                                  base_url: str) -> None:
        """
        Test the batch process trigger using configuration values.

        :param requests_mock: The mock object for HTTP requests.
        :param config: The configuration dictionary to use for testing.
        :param base_url: The base URL of the API.
        """
        url = f"{base_url}/batch_process"
        image_data = {"image_directory": config["directories"]["target_directory"]}
        mock_response = {"status": "success", "processed_files": 5}

        # mock the POST request and return a successful response
        mock_post_request(requests_mock, url, response=mock_response)

        response = trigger_batch_process_config(config, is_test=True)

        # assertions to verify behavior
        assert response == mock_response
        assert requests_mock.called
        assert requests_mock.call_count == 1
        assert requests_mock.last_request.json() == image_data

    def test_trigger_batch_process_config_connection_error(self,
                                                           requests_mock,
                                                           config: Dict[str, Any],
                                                           base_url: str) -> None:
        """
        Test the batch process trigger using configuration values when a connection error occurs.

        :param requests_mock: The mock object for HTTP requests.
        :param config: The configuration dictionary to use for testing.
        :param base_url: The base URL of the API.
        """
        url = f"{base_url}/batch_process"

        # mock the POST request to raise a connection error
        mock_post_request(requests_mock, url, exception=requests.exceptions.ConnectionError)

        response = trigger_batch_process_config(config, is_test=True)

        # assertions to verify the behavior
        assert response is None
        assert requests_mock.called


class TestTriggerPrintDBContents:
    """
    Test class for testing the fetching of db contents.
    """

    def test_get_db_contents_success(self, requests_mock, base_url: str) -> None:
        """
        Test the successful fetching of db contents.

        :param requests_mock: The mock object for HTTP requests.
        :param base_url: The base URL of the API.
        """
        url = f"{base_url}/get_db_as_json"
        mock_response = [{"id": 1, "name": "Bag.png"}, {"id": 2, "name": "Shirt.jpg"}]

        # mock the successful get request
        mock_get_request(requests_mock, url, response=mock_response)

        result = get_db_contents(base_url)

        # assertions to verify behavior
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["id", "name"]

    def test_get_db_contents_failure(self, requests_mock, base_url: str) -> None:
        """
        Test failed retrieval of database contents.

        :param requests_mock: The mock object for HTTP requests.
        :param base_url: The base URL of the API.
        """
        url = f"{base_url}/get_db_as_json"

        # mock get request with a 500 status code
        mock_get_request(requests_mock, url, status_code=500)

        result = get_db_contents(base_url)

        # assertions to verify behavior
        assert result is None
        assert requests_mock.called

    def test_get_db_contents_config_success(self, requests_mock,
                                            config: Dict[str, Any],
                                            base_url: str) -> None:
        """
        Test the get db contents trigger using configuration values.

        :param requests_mock: The mock object for HTTP requests.
        :param config: The configuration dictionary to use for testing.
        :param base_url: The base URL of the API.
        """
        url = f"{base_url}/get_db_as_json"
        mock_response = [{"id": 1, "name": "Bag.png"}, {"id": 2, "name": "Shirt.jpg"}]

        # mock the get request and return a successful response
        mock_get_request(requests_mock, url, response=mock_response)

        result = get_db_contents_config(config=config, is_test=True)

        # assertions to verify behavior
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result.columns) == ["id", "name"]

    def test_get_db_contents_config_failure(self, requests_mock,
                                            config: Dict[str, Any],
                                            base_url: str) -> None:
        """
        Test the get db contents trigger using configuration values when a connection error occurs.

        :param requests_mock: The mock object for HTTP requests.
        :param config: The configuration dictionary to use for testing.
        :param base_url: The base URL of the API.
        """
        url = f"{base_url}/get_db_as_json"

        # mock the POST request to raise a connection error
        mock_get_request(requests_mock, url, exception=requests.exceptions.ConnectionError)

        result = get_db_contents_config(config=config, is_test=True)

        # assertions to verify the behavior
        assert result is None
        assert requests_mock.called
