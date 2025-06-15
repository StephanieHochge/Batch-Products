import logging
from threading import Lock
import os
import shutil
import random
from typing import Tuple

from flask import Flask, Blueprint, jsonify, request, current_app, Response, render_template
from pathlib import Path

from werkzeug.exceptions import RequestEntityTooLarge

from src.api.batch_app.utils import save_ingestion_to_db, save_categorization_to_db, fetch_db_contents
from src.model.predict import batch_predict, load_onnx_model
from src.utils import collect_image_paths

# create a Blueprint for the main routes
main = Blueprint("main", __name__, template_folder="src/api/batch_app/templates")

# lock-object for thread synchronization when accessing shared resources
model_lock = Lock()


def get_logger():
    """
    Get the logger from the flask app configuration.

    :return: The configured logger if present, otherwise a standard logger named after the current module's name.
    """
    return current_app.config.get("Logger", logging.getLogger(__name__))


@main.route("/")
def index():
    """
    Serves the index page for the application. This endpoint returns a welcome page that provides a link
    to the database view.

    :return: Rendered HTML template for the index page.
    """
    return render_template("index.html")


@main.route("/batch_process", methods=["POST"])
def batch_process() -> Tuple[Response, int]:
    """
    Endpoint for batch processing images using the trained model.

    Expects a JSON request containing 'image_directory' (str) for batch processing.
    :return: a JSON response indicating the result of the batch processing.
    """
    logger = get_logger()
    logger.info("batch_process called.")
    try:
        # load and access the model using a lock for thread safety
        with model_lock:
            if current_app.config["Model"] is None:
                logger.info("Model not loaded in app config. Loading model...")
                model_path = current_app.config["ModelPath"]
                model = load_onnx_model(model_path)
                current_app.config["Model"] = model
            else:
                model = current_app.config.get("Model")

        # load the current_config
        current_config = current_app.config.get("App_config")

        # check whether an image directory was provided and whether it exists
        image_directory = request.json.get("image_directory", current_config["directories"]["target_directory"])
        if not image_directory or not os.path.exists(image_directory):
            logger.error(f"Invalid image directory {image_directory} provided.")
            return jsonify({"error": "Invalid image directory"}), 400

        # create the processed directory if it does not already exist
        processed_dir = Path(current_config["directories"]["processed_directory"])
        processed_dir.mkdir(exist_ok=True)

        # collect all images with supported extensions
        image_paths = collect_image_paths(image_directory)

        # check whether images exist
        if not image_paths:
            logger.warning(f"No images found in {image_directory}.")
            return jsonify({"message": "No images found."}), 204

        # image processing
        with model_lock:
            # make a batch prediction for the images
            predictions = batch_predict([str(img) for img in image_paths], model)

        # move each processed image to the processed directory and save the categorization in the database
        success_count = 0
        for image_path, (predicted_class, probabilities) in zip(image_paths, predictions):
            new_path = processed_dir / image_path.name
            try:
                shutil.move(str(image_path), str(new_path))
                save_categorization_to_db(image_path.name, predicted_class, probabilities, logger)
                success_count += 1
            except Exception as e:
                logger.error(f"Error while moving file {image_path} and saving its categorization: {e}")

        logger.info(f"Batch processing complete: {success_count} images processed and moved into {processed_dir}.")
        return jsonify({"message": "Batch processing complete", "processed_files": success_count}), 200

    except Exception as e:
        logger.error(f"Error during batch processing: {e}")
        return jsonify({"error": str(e)}), 500


@main.route("/view_db", methods=["GET"])
def view_db() -> Tuple[Response | str, int]:
    """
    Endpoint to view the contents of the database as an HTML response.

    :return: renders an HTML page displaying the database content or an error message.
    """
    logger = get_logger()
    logger.info("view_db called.")

    # get database contents
    db_contents = fetch_db_contents(logger)

    # render the error template if an error occurred during database retrieval
    if "error" in db_contents:
        return render_template("error.html", error=db_contents["error"]), 500

    # render the view_db template if no error occurred
    return render_template("view_db.html",
                           columns=db_contents["columns"],
                           data=db_contents["data"]), 200


@main.route("/get_db_as_json", methods=["GET"])
def get_db_as_json() -> Tuple[Response | str, int]:
    """
    Endpoint to retrieve the database contents as a JSON response.

    :return: A JSON response containing the database contents or an error message.
    """
    logger = get_logger()
    logger.info("get_db_as_json called.")

    # get database contents
    db_contents = fetch_db_contents(logger)

    # return an error if an error occurred during database retrieval
    if "error" in db_contents:
        return jsonify({"error": db_contents["error"]}), 500

    # return a JSON response if no error occurred
    return jsonify(db_contents["data"]), 200


@main.route("/ingest_images", methods=["POST"])
def ingest_images() -> Tuple[Response, int]:
    """
    Endpoint for ingesting images into the system.

    Expects a JSON request with source and target directories, an optional image range.
    :return: a JSON response indicating the result of the image ingestion.
    """
    logger = get_logger()
    logger.info("ingest_images called.")
    try:
        # load the current config
        current_config = current_app.config.get("App_config")

        # extract JSON-data from the request
        data = request.json
        source_directory = data.get("source_directory", current_config["directories"]["source_directory"])
        target_directory = data.get("target_directory", current_config["directories"]["target_directory"])
        min_images_per_run = data.get("min_images_per_run", current_config["batch"]["min_images_per_run"])
        # limit to a maximum of 5000
        max_images_per_run = min(data.get("max_images_per_run", current_config["batch"]["max_images_per_run"]), 5000)

        source_directory = Path(source_directory)
        target_directory = Path(target_directory)

        # check whether the source directory exists
        if not source_directory.exists() or not source_directory.is_dir():
            logger.error("Invalid source directory provided.")
            return jsonify({"error": "Invalid source directory"}), 400

        # collect all images with supported extensions
        source_images = collect_image_paths(source_directory)

        # check whether there are any images in the source_directory
        if not source_images:
            logger.warning(f"No images found in {source_directory}.")
            return jsonify({"message": f"No images found in {source_directory}"}), 204

        # randomly determine the number of images to ingest
        num_images = random.randint(min_images_per_run, max_images_per_run)

        # create the target directory if it does not exist
        target_directory.mkdir(parents=True, exist_ok=True)

        success_count = 0
        for _ in range(num_images):
            image = random.choice(source_images)
            # save metadata of the ingested image in the database (creates a unique name, if the name already exists)
            new_image_name = save_ingestion_to_db(image.name)

            # once the image was saved, it is copied to the target directory for later batch processing
            if new_image_name:
                new_target_path = target_directory / new_image_name
                shutil.copy(image, new_target_path)
                success_count += 1
            else:
                logger.warning(f"Failed to save image {image.name} to database.")

        logger.info(f"Image ingestion completed successfully: {success_count} images ingested into {target_directory}.")
        return jsonify({"message": "Image ingestion completed", "ingested_images": success_count}), 200

    except Exception as e:
        logger.error(f"Error during image ingestion: {e}")
        return jsonify({"error": str(e)}), 500


def register_routes(app: Flask) -> None:
    """
    Register the blueprint routes with the Flask application.

    :param app: The Flask application instance.
    """
    app.register_blueprint(main)


@main.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    """
    Handles the RequestEntityTooLarge exception caused by exceeding the MAX_CONTENT_LENGTH limit.

    :param e: The exception object.
    :return: A JSON response indicating the result of the requested exception.
    """
    logger = get_logger()
    logger.error("File upload exceeded size limit.")
    return jsonify({"error": f"File upload exceeded size limit."}), 413
