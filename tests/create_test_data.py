from pathlib import Path
from PIL import Image
import time
import random
from typing import List

from sklearn.model_selection import train_test_split
from torchvision import datasets
from src.utils import load_app_config, collect_image_paths


def save_images_with_labels(data: List[Image.Image],
                            labels: List[int],
                            output_folder: str,
                            class_names: List[str] = None,
                            seed: int = 42) -> None:
    """
    Save images with their corresponding labels to the specified output folder.

    :param data: list of Image objects representing the images.
    :param labels: list of integer labels corresponding to the images.
    :param output_folder: path to the folder where images will be saved.
    :param class_names: optional list of class names corresponding to the labels.
    :param seed: random seed for reproducibility.
    """
    # check if output folder exists
    start_time = time.time()
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # define class names if none were provided
    if class_names is None:
        class_names = [
            "T-Shirt-Top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
        ]

    # set random seed for reproducibility
    random.seed(seed)

    # save each image with an increasing index and the corresponding label
    saved_images_count = 0
    for idx, (image, label) in enumerate(zip(data, labels)):
        # only save images that are of type Image.Image
        if isinstance(image, Image.Image):
            # get class name
            label = class_names[label]

            # create image name
            filename = f"{idx + 1:04d}_{label}"  # format: 0001_label

            # randomly select a file extension (png, jpg, or jpeg)
            file_extension = random.choice([".png", ".jpg", ".jpeg"])

            # final image filename with the random extension
            filename += file_extension

            # save image
            image_pil = image
            image_pil.save(output_path / filename)

            saved_images_count += 1

    # check if images were saved
    if saved_images_count == 0:
        raise ValueError(f"No images were saved to {output_folder}. Please check the input data.")

    print(f"Saved {saved_images_count} images to {output_folder}. Duration: {time.time() - start_time:.2f} seconds")


def create_test_data(output_folder=None) -> None:
    """
    Create test data by saving a subset of FashionMNIST dataset images and labels to the specified output folder.
    This function checks if the output folder already exists and skips data creation if it does.

    :param output_folder: path to the folder where images will be saved.
    """
    # get the output folder from the configuration
    config = load_app_config(is_test=True)
    if output_folder is None:
        output_folder = config["directories"]["source_directory"]

    # check if the folder already contains images
    if Path(output_folder).exists():
        img_paths = collect_image_paths(output_folder)
        if len(img_paths) > 0:
            # if it contains images, skip data creation
            print(f"Folder {output_folder} already exists and contains images. Skipping data creation.")
            return

    print(f"Creating test data in {output_folder}...")

    start_time = time.time()
    # load the FashionMNIST dataset
    fashion_mnist_path = config["train_data"]["path"]
    full_train_dataset = datasets.FashionMNIST(root=fashion_mnist_path, train=True, download=True)
    data = [full_train_dataset[i][0] for i in range(len(full_train_dataset))]
    labels = [full_train_dataset[i][1] for i in range(len(full_train_dataset))]
    print(f"loading data and labels took: {time.time() - start_time:.2f} seconds")

    # load the data that was not used for training
    train_data, later_data, train_labels, later_labels = train_test_split(
        data, labels, test_size=0.05, stratify=labels, random_state=42
    )

    print(f"Size of the later data: {len(later_data)}")
    # save these images with their corresponding labels to the output folder
    save_images_with_labels(later_data, later_labels, output_folder)


if __name__ == "__main__":
    create_test_data()
