import onnxruntime as ort
import numpy as np

from typing import List, Dict, Tuple
from pathlib import Path
from PIL import Image


def get_class_names() -> List[str]:
    """
    Return the list of class names corresponding to the model's output classes.

    :return: List of class names.
    """
    return [
        "T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
    ]


def load_onnx_model(model_path: str | Path) -> ort.InferenceSession:
    """
    Load the ONNX model from the specified path.

    :param model_path: Path to the model file.
    :return: loaded ONNX model.
    """
    return ort.InferenceSession(str(model_path))


def preprocess_image(image: Image) -> np.ndarray:
    """
    Preprocess an image for input into the model. Apply resizing, grayscale conversion, normalization and tensor
    conversion.

    :param image: PIL Image object to be processed.
    :return: Preprocessed image as a NumPy array (1, 1, 224, 224).
    """
    # resize to 224 x 224
    image = image.resize((224, 224))

    # convert to grayscale (if necessary)
    if image.mode != "L":
        image = image.convert("L")

    # convert to NumPy array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5

    # adjust dimensions: (1, 224, 224) -> (1, 1, 224, 224)
    img_array = np.expand_dims(img_array, axis=(0, 1))

    return img_array


def batch_predict(image_paths: List[str | Path], model: ort.InferenceSession) -> List[Tuple[str, Dict[str, float]]]:
    """
    Predict classes for a batch of images using the given model.

    :param image_paths: List of paths to images to be processed.
    :param model: Pre-loaded ONNX model for prediction.
    :return: List of tuples containing the predicted class and a dictionary of class probabilities.
    """
    class_names = get_class_names()

    # preprocess images
    image_inputs = [preprocess_image(Image.open(image_path)) for image_path in image_paths]

    # generate batch
    batch = np.vstack(image_inputs).astype(np.float32)

    # get model input name
    input_names = model.get_inputs()[0].name

    # run inference
    outputs = model.run(None, {input_names: batch})

    # apply softmax to get probabilities
    probabilities = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]), axis=1, keepdims=True)

    # extract predictions
    results = []
    for i, class_probs in enumerate(probabilities):
        class_probs = [float(np.round(prob, 3)) for prob in class_probs]
        prob_dict = {
            "prob_tshirt_top": class_probs[0],
            "prob_trouser": class_probs[1],
            "prob_pullover": class_probs[2],
            "prob_dress": class_probs[3],
            "prob_coat": class_probs[4],
            "prob_sandal": class_probs[5],
            "prob_shirt": class_probs[6],
            "prob_sneaker": class_probs[7],
            "prob_bag": class_probs[8],
            "prob_ankle_boot": class_probs[9]
        }
        predicted_class = class_names[np.argmax(class_probs)]
        results.append((predicted_class, prob_dict))

    return results
