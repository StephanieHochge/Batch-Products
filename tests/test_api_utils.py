import logging

import pytest
from src.api.batch_app.utils import (truncate_image_name, save_ingestion_to_db,
                                     save_categorization_to_db, get_all_data_as_dataframe,
                                     fetch_db_contents)
from src.api.batch_app.db_models import ImageClassifications, db

from flask import Flask
import datetime
import pandas as pd
from typing import Generator


@pytest.fixture(scope="function")
def init_database(app: Flask) -> Generator[db, None, None]:
    """
    Fixture to initialize and tear down the database for each test.

    :param app: Flask app instance.
    :yield: SQLAlchemy database instance.
    """
    from src.api.batch_app.db_models import db
    with app.app_context():
        db.create_all()
    yield db
    with app.app_context():
        db.session.remove()
        db.drop_all()


def test_truncate_image_name() -> None:
    """
    Test the truncate image function to ensure it truncates or preserves image names correctly.
    """
    # normal length, should remain unchanged
    image_name = "image_name.png"
    truncated_name = truncate_image_name(image_name)
    assert truncated_name == image_name

    # long name, should be truncated to at most 255 characters
    long_name = "a" * 300 + ".png"
    truncated_name = truncate_image_name(long_name)
    assert len(truncated_name) <= 255
    assert truncated_name.endswith(".png")

    # name with counter, which should not be truncated
    name_with_counter = "image_12345678901234567890123456789012345678901234567890123456789012345678901234567890123456789_1.png"
    truncated_name = truncate_image_name(name_with_counter)
    assert truncated_name.endswith("_1.png")
    assert len(truncated_name) <= 255


def test_save_ingestion_to_db(init_database: db) -> None:
    """
    Test the save_ingestion_to_db function to ensure it saves image ingestion records correctly.

    :param init_database: Initialized test database fixture.
    """
    image_name = "test_image.png"

    # save the image name once
    result = save_ingestion_to_db(image_name, timestamp_ingestion=datetime.datetime.now())
    assert result == "test_image.png"

    # save the same image name again, should add a counter
    result = save_ingestion_to_db(image_name, timestamp_ingestion=datetime.datetime.now())
    assert result.startswith("test_image_1.png")

    # save again, counter should increment
    result = save_ingestion_to_db(image_name, timestamp_ingestion=datetime.datetime.now())
    assert result.startswith("test_image_2.png")


def test_save_categorization_to_db(init_database: db) -> None:
    """
    Test the save_categorization_to_db function to ensure it updates image categorizations correctly.

    :param init_database: Initialized test database fixture.
    """
    image_name = "test_image_for_categorization.png"
    save_ingestion_to_db(image_name, timestamp_ingestion=datetime.datetime.now())

    # define predicted class and probabilities
    predicted_class = "T-shirt/top"
    probabilities = {
        "prob_tshirt_top": 0.95,
        "prob_trouser": 0.02,
        "prob_pullover": 0.01,
        "prob_dress": 0.01,
        "prob_coat": 0.01,
        "prob_sandal": 0.0,
        "prob_shirt": 0.0,
        "prob_sneaker": 0.0,
        "prob_bag": 0.0,
        "prob_ankle_boot": 0.0
    }

    # save categorization
    result = save_categorization_to_db(image_name, predicted_class, probabilities)
    assert result == predicted_class

    # verify saved data in the database
    image = ImageClassifications.query.filter_by(image_name=image_name).first()
    assert image is not None
    assert image.predicted_class == predicted_class
    assert image.prob_tshirt_top == probabilities["prob_tshirt_top"]

    # save categorization for image not previously saved to the database
    image_name_not_db = "img_not_in_db.jpg"
    result = save_categorization_to_db(image_name_not_db, predicted_class, probabilities)
    assert result is None


def test_get_all_data_as_dataframe(init_database: db) -> None:
    """
    Test the get_all_data_as_dataframe function to ensure it retrieves all data as a DataFrame.

    :param init_database: Initialized test database fixture
    """
    image_name = "test_image_for_dataframe.png"
    save_ingestion_to_db(image_name, timestamp_ingestion=datetime.datetime.now())

    # define predicted class and probabilities
    predicted_class = "Pullover"
    probabilities = {
        "prob_tshirt_top": 0.1,
        "prob_trouser": 0.1,
        "prob_pullover": 0.6,
        "prob_dress": 0.05,
        "prob_coat": 0.05,
        "prob_sandal": 0.05,
        "prob_shirt": 0.05,
        "prob_sneaker": 0.0,
        "prob_bag": 0.0,
        "prob_ankle_boot": 0.0
    }
    save_categorization_to_db(image_name, predicted_class, probabilities)

    # retrieve all data as a DataFrame and check that the image was correctly saved
    df = get_all_data_as_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert image_name in df["image_name"].values
    assert predicted_class in df["predicted_class"].values


def test_fetch_db_contents(logger: logging.Logger, mocker) -> None:
    """
    Test the fetch_db_contents function to ensure it fetches content correctly.

    :param logger: Logger fixture.
    """
    # mock successful database retrieval
    mocker.patch("src.api.batch_app.utils.get_all_data_as_dataframe", return_value=
                 pd.DataFrame({"id": [1, 2], "name": ["Item1", "Item2"],
                               "timestamp_ingestion": [None, "2024-02-01"],
                               "timestamp_prediction": [None, "2024-02-02"]}))

    result = fetch_db_contents(logger)
    assert "error" not in result
    assert len(result["data"]) == 2
    assert "columns" in result

    # mock empty database
    mocker.patch("src.api.batch_app.utils.get_all_data_as_dataframe",
                 return_value=pd.DataFrame())
    result = fetch_db_contents(logger)
    assert result["data"] == []
    assert result["columns"] == ["No data available"]

    # mock database error
    mocker.patch("src.api.batch_app.utils.get_all_data_as_dataframe",
                 side_effect=Exception("Database failure"))
    result = fetch_db_contents(logger)
    assert "error" in result
    assert result["error"] == "Database failure"
