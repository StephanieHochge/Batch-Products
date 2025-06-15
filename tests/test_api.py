import pytest
from typing import Dict
from flask.testing import FlaskClient


def test_index_page(client: FlaskClient) -> None:
    """
    Test the root (/) endpoint to ensure it serves the index page correctly.

    :param client: Flask test client for sending requests.
    """
    response = client.get("/")

    # assert that the request was successful
    assert response.status_code == 200

    # assert that the response contains expected HTML content
    assert b"Welcome" in response.data
    assert b"Show Database" in response.data


def test_ingest_images(client: FlaskClient,
                       ingestion_data: Dict[str, str], setup_test_data) -> None:
    """
    Test the /ingest_images endpoint to ensure it correctly ingests a range of images.

    :param client: Flask test client for sending requests.
    :param ingestion_data: Fixture providing sample ingestion data.
    :param setup_test_data: Fixture to prepare test data setup.
    """
    url_ingestion = "/ingest_images"
    response = client.post(url_ingestion, json=ingestion_data)

    # assert that the request was successful
    assert response.status_code == 200

    # assert that the number of ingested images is within the specified range
    result = response.get_json()
    assert result["ingested_images"] >= ingestion_data["min_images_per_run"]
    assert result["ingested_images"] <= ingestion_data["max_images_per_run"]


def test_ingest_images_incomplete_data(client: FlaskClient, setup_test_data) -> None:
    """
    Test the /ingest_images endpoint with incomplete data to ensure default configuration is used.

    :param client: Flask test client for sending requests.
    :param setup_test_data: Fixture to prepare test data setup.
    """
    incomplete_data = {
        "min_images_per_run": 5,
        "max_images_per_run": 10
    }

    url_ingestion = "/ingest_images"
    response = client.post(url_ingestion, json=incomplete_data)

    # assert that the request was successful
    assert response.status_code == 200

    # assert that the number of ingested images is within the specified range
    result = response.get_json()
    assert result["ingested_images"] >= incomplete_data["min_images_per_run"]
    assert result["ingested_images"] <= incomplete_data["max_images_per_run"]


def test_batch_process(client: FlaskClient, ingestion_data: Dict[str, str],
                       setup_test_data) -> None:
    """
    Test the /batch_process endpoint to ensure it processed images correctly after ingestion.

    :param client: Flask test client for sending requests.
    :param ingestion_data: Fixture providing sample ingestion data.
    :param setup_test_data: Fixture to prepare test data setup.
    """
    url_ingestion = "/ingest_images"
    url_batch_process = "/batch_process"

    # ingest the images
    client.post(url_ingestion, json=ingestion_data)

    # trigger batch processing on the ingested images
    response = client.post(url_batch_process, json={"image_directory": ingestion_data["target_directory"]})

    # assert that the batch processing request was successful
    assert response.status_code == 200

    # assert that some files were processed
    result = response.get_json()
    assert result["processed_files"] > 0


def test_batch_process_incomplete_data(client: FlaskClient, setup_test_data) -> None:
    """
    Test the /batch_process endpoint without a specified image directory to ensure default configuration is used.

    :param client: Flask test client for sending requests.
    :param setup_test_data: Fixture providing sample ingestion data.
    """
    url_ingestion = "/ingest_images"
    url_batch_process = "/batch_process"

    # ingest the images
    client.post(url_ingestion, json={})

    # trigger batch processing on the ingested images
    response = client.post(url_batch_process, json={})

    # assert that the batch processing request was successful
    assert response.status_code == 200

    # assert that some files were processed
    result = response.get_json()
    assert result["processed_files"] > 0


def test_view_db(client: FlaskClient) -> None:
    """
    Test the /view_db endpoint to ensure it correctly renders the database content page.

    :param client: Flask test client for sending requests.
    """
    # render an HTML response to view the database
    url = "/view_db"
    response = client.get(url)

    # assert that the request was successful
    assert response.status_code == 200

    # assert that the response contains expected HTML content
    assert b"<table" in response.data


def test_view_db_error(client: FlaskClient, mocker) -> None:
    """
    Test the /view_db endpoint to ensure it handles errors correctly.

    :param client: Flask test client for sending requests.
    :param mocker: Mocking utility to simulate database errors.
    """
    mocker.patch("src.api.batch_app.routes.fetch_db_contents", return_value={"error": "Database error"})
    url = "/view_db"
    response = client.get(url)

    # assert that an error response is returned
    assert response.status_code == 500
    assert b"Database error" in response.data


def test_get_db_as_json(client: FlaskClient) -> None:
    """
    Test the /get_db_as_json endpoint to ensure it retrieves data from the database.

    :param client: Flask test client for sending requests.
    """
    url = "/get_db_as_json"
    response = client.get(url)

    # assert that the request was successful
    assert response.status_code == 200

    # assert that the database returns some data
    data = response.get_json()
    assert len(data) > 0


def test_get_db_as_json_error(client: FlaskClient, mocker) -> None:
    """
    Test the /get_db_as_json endpoint to ensure it handles errors correctly.

    :param client: Flask test client for sending requests.
    :param mocker: Mocking utility to simulate database errors.
    """
    mocker.patch("src.api.batch_app.routes.fetch_db_contents", return_value={"error": "Database error"})
    url = "/get_db_as_json"
    response = client.get(url)

    # assert that an error response is returned
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data
    assert data["error"] == "Database error"
