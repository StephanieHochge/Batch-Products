import pytest

from PIL import Image
from typing import Dict, Any
import onnxruntime as ort

import src.model.predict as pr
from src.utils import collect_image_paths


@pytest.fixture(scope="module")
def model(config: Dict[str, Any]) -> ort.InferenceSession:
    """
    Fixture to load and provide the model for testing.

    :param config: The application configuration for test settings.
    :return: Loaded model object.
    """
    model_path = config["model_path"]
    return pr.load_onnx_model(model_path)


@pytest.fixture(scope="module")
def test_prediction_data(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Fixture to provide paths to test data for prediction tests.

    :param config: The application configuration for test settings.
    :return: A dictionary containing the directory of test images and the path to a single test image.
    """
    test_image_dir = config["directories"]["source_directory"]
    return {
        "test_image_dir": test_image_dir,
        "test_img": f"{test_image_dir}/0006_Bag.png"
    }


def test_preprocess_image(test_prediction_data: Dict[str, str]) -> None:
    """
    Test the preprocessing of an image for prediction. Ensure that the output tensor shape is consistent
    with the expected dimension.

    :param test_prediction_data: Dictionary containing paths to test data.
    """
    image = Image.open(test_prediction_data["test_img"])
    processed_image = pr.preprocess_image(image)
    assert processed_image.shape == (1, 1, 224, 224)


def test_batch_predict(test_prediction_data: Dict[str, str], model: ort.InferenceSession) -> None:
    """
    Test the batch prediction functionality to ensure all images are processed correctly.

    :param test_prediction_data: Dictionary containing paths to test images.
    :param model: The model object to use for prediction.
    """
    # ensure that predictions are made for all input images
    image_paths = collect_image_paths(test_prediction_data["test_image_dir"])[:10]
    predictions = pr.batch_predict(image_paths, model)
    assert len(predictions) == 10

    # validate that the first prediction exists
    first_pred = predictions[0]
    assert first_pred[0] is not None
