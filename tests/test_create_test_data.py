import pytest
from unittest.mock import patch, MagicMock
from PIL import Image
from tests.create_test_data import save_images_with_labels, create_test_data
from pathlib import Path
from typing import List


@pytest.fixture(scope="function")
def output_folder(config) -> Path:
    """
    Fixture to create and clean up a fixed output folder for tests.

    :yield: The output folder Path.
    """
    # define the output path for the output folder
    source_dir = Path(config["directories"]["source_directory"])
    parent_dir = source_dir.parent
    output_path = parent_dir / "test_images"

    # ensure the folder is clean before the test
    if output_path.exists():
        for file in output_path.glob("*"):
            file.unlink()  # delete files in the folder
        output_path.rmdir()  # remove the folder

    output_path.mkdir(parents=True, exist_ok=True)  # create a fresh folder
    yield output_path  # provide the folder to the test

    # cleanup after the test
    if output_path.exists():
        for file in output_path.glob("*"):
            file.unlink()
        output_path.rmdir()


@pytest.fixture
def mock_images() -> List[Image.Image]:
    """
    Fixture to create mock images for testing.

    :return: A list of mock images.
    """
    # create 10 red images
    return [Image.new("RGB", (100, 100), color="red") for _ in range(10)]


def test_save_images_with_labels_success(mock_images: List[Image.Image], output_folder: Path) -> None:
    """
    Test that 'save_images_with_labels' successfully saves images with corresponding labels.

    :param mock_images: Mock images to save.
    :param output_folder: Output folder to save the images.
    """
    # mock labels and class names
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    class_names = ["Class 0", "Class 1", "Class 2", "Class 3", "Class 4",
                   "Class 5", "Class 6", "Class 7", "Class 8", "Class 9"]

    save_images_with_labels(mock_images, labels, str(output_folder), class_names)

    # check if images are saved in the output folder
    saved_images = list(output_folder.glob("*"))
    assert len(saved_images) == len(mock_images)
    assert all(image.suffix in [".png", ".jpg", ".jpeg"] for image in saved_images)


def test_save_images_with_labels_no_images(output_folder: Path) -> None:
    """
    Test that 'save_images_with_labels' raises a ValueError when no images are provided.

    :param output_folder: Output folder for the test.
    """
    with pytest.raises(ValueError, match="No images were saved to"):
        save_images_with_labels([], [], str(output_folder))


@patch('src.utils.load_app_config')
@patch('torchvision.datasets.FashionMNIST')
def test_create_test_data(mock_fashion_mnist: MagicMock,
                          mock_load_app_config: MagicMock,
                          output_folder: Path) -> None:
    """
    Test the 'create_test_data' function to ensure it saves images correctly.

    :param mock_fashion_mnist: Mock for the FashionMNIST dataset.
    :param mock_load_app_config: Mock for the app configuration loader.
    :param output_folder: The output folder for saving images.
    """
    # ensure that the output folder does not exist before starting
    if output_folder.exists():
        for file in output_folder.glob("*"):
            file.unlink()
        output_folder.rmdir()

    # mock the application configuration
    mock_load_app_config.return_value = {
        "directories": {
            "source_directory": str(output_folder)  # use the output folder for testing
        },
        "train_data": {
            "path": str(output_folder)  # use the same folder as the data path
        }
    }

    # mock the FashionMNIST dataset
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 500  # dataset has 500 images
    mock_dataset.__getitem__.side_effect = lambda index: (Image.new("RGB", (100, 100)), index % 10)
    mock_fashion_mnist.return_value = mock_dataset

    # call the function to create test data
    create_test_data(output_folder)

    # check if images were saved in the mocked output folder
    saved_images = list(output_folder.glob("*"))
    assert len(saved_images) > 0
