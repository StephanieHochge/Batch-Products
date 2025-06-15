from torch.utils.data import Dataset
from typing import List, Any, Tuple


class CustomFashionMNISTDataset(Dataset):
    """
    A custom dataset class for the Fashion MNIST dataset.
    This class inherits from PyTorch's Dataset and provides a way to load data and labels
    for the Fashion MNIST dataset.

    Attributes:
        - data: A list or array-like structure containing the images.
        - labels: A list of integers representing the labels for each image.
    """

    def __init__(self, data: List[Any], labels: List[int]) -> None:
        """
        Initialize the CustomFashionMNISTDataset with data and labels.

        :param data: The images to be stored in the dataset.
        :param labels: The corresponding labels for the images.
        """
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        :return: the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """
        Retrieve the image and label at the specified index.

        :param idx: The index of the sample to retrieve.
        :return: A tuple containing the image and its corresponding label.
        """
        return self.data[idx], self.labels[idx]
