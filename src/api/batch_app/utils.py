import datetime
import logging

from src.api.batch_app.db_models import db, ImageClassifications
import pandas as pd
from sqlalchemy.exc import IntegrityError
import re
from typing import Dict, Any


def truncate_image_name(image_name: str, max_length: int = 255) -> str:
    """
    Trim the image name to a specified maximum length, considering any counter that may be appended to the base name.

    :param image_name: The original image name.
    :param max_length: Maximum allowed length for the image name (default: 255).
    :return: The truncated image name if it exceeds the max_length; otherwise, the original name.
    """
    # check whether the original image name exceeds the maximum allowed length
    if len(image_name) > max_length:
        # split the image name into the base name and the extension
        base_name, extension = image_name.rsplit(".", 1)

        # look for a counter (e.g., "_1") at the end of the base name
        match = re.search(r"_(\d+)$", base_name)
        if match:
            # counter found, calculate available space for the base name
            counter_length = len(match.group(0))  # length of counter (e.g., "_1")
            max_base_name_length = max_length - len(extension) - counter_length - 1  # -1 for the point
        else:
            # no counter found, calculate available space for the whole image name
            max_base_name_length = max_length - len(extension) - 1  # -1 for the point

        # truncate the base name and reconstruct the full image name
        base_name = base_name[:max_base_name_length]
        image_name = f"{base_name}.{extension}"

    return image_name


def save_ingestion_to_db(image_name: str,
                         timestamp_ingestion: datetime = None) -> str | None:
    """
    Save the image ingestion record to the database, ensuring that the image name is unique.

    :param image_name: The name of the image name to be saved.
    :param timestamp_ingestion: The timestamp of ingestion (default: now).
    :return: The unique image name saved in the database or None if an error occurred.
    """
    # use the current time if no timestamp is provided
    if timestamp_ingestion is None:
        timestamp_ingestion = datetime.datetime.now()

    # truncate the image name if it exceeds the maximum length
    image_name = truncate_image_name(image_name)

    # check if the image name already exists in the database
    existing_image = ImageClassifications.query.filter_by(image_name=image_name).first()

    # if the image name already exists
    if existing_image:
        # split the existing image name into the base name and the extension
        base_name, extension = image_name.rsplit(".", 1)

        # initialize a counter to create a unique name
        counter = 1

        # create a new image name by appending the counter to the base name
        new_image_name = f"{base_name}_{counter}.{extension}"

        # loop to find a unique image name by incrementing the counter
        while ImageClassifications.query.filter_by(image_name=new_image_name).first():
            counter += 1
            # generate a new image name with the updated counter
            new_image_name = f"{base_name}_{counter}.{extension}"

        # truncate the new image name if necessary (considering the counter)
        new_image_name = truncate_image_name(new_image_name)

    else:
        new_image_name = image_name

    # create a new image record
    new_image = ImageClassifications(
        image_name=new_image_name,
        timestamp_ingestion=timestamp_ingestion
    )

    # save the new image record to the database
    try:
        db.session.add(new_image)
        db.session.commit()
        print(f"Image saved with name: {new_image_name}")
        return new_image_name
    except IntegrityError:
        db.session.rollback()
        print("An error occurred while saving the image.")
        return None


def save_categorization_to_db(image_name: str, predicted_class: str,
                              probabilities: Dict[str, float],
                              logger: logging.Logger = None) -> str | None:
    """
    Update the categorization information for an image in the database.

    :param image_name: The name of the image to update.
    :param predicted_class: The predicted class of the image.
    :param probabilities: A dictionary containing probabilities for various classes.
    :param logger: A logger instance to handle logging.
    :return: The predicted class that was saved, or None if an error occurred.
    """
    # retrieve the image record from the database
    image = ImageClassifications.query.filter_by(image_name=image_name).first()

    # log the image name if no record was found in the database.
    if image is None:
        if logger is not None:
            logger.info(f"Image with name '{image_name}' not found in the database.")
        else:
            print(f"[Info]: Image with name '{image_name}' not found in the database.")
        return

    # update the categorization information
    image.predicted_class = predicted_class
    image.timestamp_prediction = datetime.datetime.now()
    image.prob_tshirt_top = probabilities.get("prob_tshirt_top", None)
    image.prob_trouser = probabilities.get("prob_trouser", None)
    image.prob_pullover = probabilities.get("prob_pullover", None)
    image.prob_dress = probabilities.get("prob_dress", None)
    image.prob_coat = probabilities.get("prob_coat", None)
    image.prob_sandal = probabilities.get("prob_sandal", None)
    image.prob_shirt = probabilities.get("prob_shirt", None)
    image.prob_sneaker = probabilities.get("prob_sneaker", None)
    image.prob_bag = probabilities.get("prob_bag", None)
    image.prob_ankle_boot = probabilities.get("prob_ankle_boot", None)

    # commit the changes to the database
    try:
        db.session.commit()
        print(f"Categorization saved for image: {image_name}")
        return predicted_class
    except Exception as e:
        db.session.rollback()
        print(f"An error occurred while saving the categorization: {e}")
        return None


def get_all_data_as_dataframe() -> pd.DataFrame:
    """
    Retrieve all records from the ImageClassifications table and return them as a pandas DataFrame.

    :return: A DataFrame containing all image classification records.
    """
    # retrieve all records from the ImageClassifications table
    images = ImageClassifications.query.all()

    # convert the records to a list of dictionaries
    data = [
        {
            "img_id": image.img_id,
            "image_name": image.image_name,
            "timestamp_ingestion": image.timestamp_ingestion,
            "timestamp_prediction": image.timestamp_prediction,
            "predicted_class": image.predicted_class,
            "prob_tshirt_top": image.prob_tshirt_top,
            "prob_trouser": image.prob_trouser,
            "prob_pullover": image.prob_pullover,
            "prob_dress": image.prob_dress,
            "prob_coat": image.prob_coat,
            "prob_sandal": image.prob_sandal,
            "prob_shirt": image.prob_shirt,
            "prob_sneaker": image.prob_sneaker,
            "prob_bag": image.prob_bag,
            "prob_ankle_boot": image.prob_ankle_boot
        }
        for image in images
    ]

    # create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)

    return df


def fetch_db_contents(logger: logging.Logger) -> Dict[str, Any]:
    """
    Fetches all data from the database and returns it as a dictionary.

    :param logger: A logger instance to handle logging.
    :return: A dictionary containing all image classification records, including column names and data orws.
    If an error occurs, returns an error message.
    """
    try:
        # retrieve all data as a pandas DataFrame
        df = get_all_data_as_dataframe()
        if df.empty:
            logger.warning("Database is empty.")
            return {"data": [], "columns": ["No data available"]}

        # ensure timestamps are converted to strings and handle missing values
        df["timestamp_ingestion"] = df["timestamp_ingestion"].fillna("").astype(str)
        df["timestamp_prediction"] = df["timestamp_prediction"].fillna("").astype(str)

        logger.info(f"Fetched {len(df)} records from the database.")

        return {"data": df.to_dict(orient="records"), "columns": df.columns.tolist()}
    except Exception as e:
        logger.error(f"An error occurred while fetching the database: {e}", exc_info=True)
        return {"error": str(e), "data": [], "columns": []}
