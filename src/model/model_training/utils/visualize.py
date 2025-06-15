import matplotlib.pyplot as plt
import numpy as np
import torch

from .custom_dataset import CustomFashionMNISTDataset


def visualize_sample(dataset: CustomFashionMNISTDataset) -> None:
    """
    Visualize a sample of nine images from the given dataset.

    :param dataset: An instance of the CustomFashionMNISTDataset class.
    """
    # prepare the plotting grid
    plt.figure(figsize=(8, 8))
    for i in range(9):
        # randomly select an index
        idx = np.random.randint(0, len(dataset))

        # retrieve the image and label
        image, label = dataset[idx]

        # convert image tensor to plottable format
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()

        # handle channel permutation if necessary
        if image.ndim == 3 and image.shape[0] == 1:  # if shape is (1, H, W)
            image = image.squeeze(0)  # remove the channel dimension
        elif image.ndim == 3 and image.shape[0] == 3:  # if shape is (3, H, W)
            image = image.transpose(1, 2, 0)  # convert to (H, W, C)

        # plot the image and label it with the corresponding class
        plt.subplot(3, 3, i + 1)
        plt.imshow(image, cmap="gray" if image.ndim == 2 else None)
        class_names = [
            "T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
        ]
        plt.title(f"Label: {class_names[label]}")
        plt.axis("off")

    # Show the full plot
    plt.tight_layout()
    plt.show()
