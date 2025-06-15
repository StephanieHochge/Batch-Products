import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from .model import get_model
from .early_stopping import EarlyStopping
from .custom_dataset import CustomFashionMNISTDataset


def set_global_seed(seed: int) -> None:
    """
    Set the seed for generating random numbers to ensure reproducibility.

    :param seed: The seed value to be used for random number generation.
    :return:
    """
    # set seed for Python's random module
    random.seed(seed)

    # set seed for NumPy
    np.random.seed(seed)

    # set seed for PyTorch
    torch.manual_seed(seed)

    # if CUDA is available, set the seed for all CUDA devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(device: torch.device,
                    model: torch.nn.Module,
                    loader: DataLoader,
                    criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer) -> tuple[float, float, float, float, float]:
    """
    Train the model for one epoch using the provided data loader.

    :param device: The device to perform computation on (e.g., "cpu" or "cuda").
    :param model: The model to be trained.
    :param loader: A DataLoader object that provides batches of images and labels.
    :param criterion: The loss function to compute the loss.
    :param optimizer: The optimizer used for updating model parameters.
    :return: A tuple containing:
        - the average loss over the epoch.
        - the accuracy of the model on the training data.
        - the F1 score of the model on the training data.
        - the precision of the model on the training data.
        - the recall of the model on the training data.
    """
    # set the model to training mode
    model.train()

    running_loss = 0.0  # to accumulate the total loss
    all_preds, all_labels = [], []  # to store predictions and labels for metric calculations

    # iterate over batches of images and labels
    for images, labels in loader:
        # move images and labels to the specified device
        images, labels = images.to(device), labels.to(device)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # accumulate loss for the epoch
        running_loss += loss.item() * images.size(0)

        # get predicted class
        _, predicted = torch.max(outputs, 1)

        # store predictions and labels for metric calculations
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)

    # return average loss and metrics
    return running_loss / len(loader.dataset), accuracy, f1, precision, recall


def evaluate_metrics(device: torch.device,
                     model: torch.nn.Module,
                     loader: DataLoader,
                     criterion: torch.nn.Module) -> Tuple[float, float, float, float, float]:
    """
    Evaluate the model on the provided data loader and calculate metrics.

    :param device: The device to perform computation on (e.g., "cpu" or "cuda").
    :param model: The model to be trained.
    :param loader: A DataLoader object that provides batches of images and labels.
    :param criterion: The loss function to compute the loss.
    :return: A tuple containing:
        - the average loss over the validation set.
        - the accuracy of the model on the validation data.
        - the F1 score of the model on the validation data.
        - the precision of the model on the validation data.
        - the recall of the model on the validation data.
    """
    # set the model to evaluation mode
    model.eval()

    running_loss = 0.0  # to accumulate the total loss
    all_preds, all_labels = [], []  # to store predictions and true labels for metric calculations

    # disable gradient calculations for evaluation
    with torch.no_grad():
        # iterate over batches of images and labels
        for images, labels in loader:
            # move images and labels to the specified device
            images, labels = images.to(device), labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # accumulate loss for the validation set
            running_loss += loss.item() * images.size(0)

            # get predicted class
            _, predicted = torch.max(outputs, 1)

            # store predictions and labels for metric calculations
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted")

    # return average loss and metrics
    return running_loss / len(loader.dataset), accuracy, f1, precision, recall


def train_cross_validation(config: Dict[str, Any],
                           device: torch.device,
                           train_labels: List[int],
                           train_data: List[Any]) -> pd.DataFrame:
    """
    Perform 5-fold cross validation on the training data.

    :param config: A dictionary containing hyperparameters for training: batch_size, epochs, dropout,
    learning_rate, freeze.
    :param device: The device to perform computation on (e.g., "cpu" or "cuda").
    :param train_labels: A list of labels corresponding to the training data.
    :param train_data: A list or array-like structure containing the training images.
    :return: A DataFrame containing mean metrics (accuracy, F1 score, precision, recall).
    """
    print("-----------------------------------")
    print(f"Training using config: {config}")

    # transform the training data into the custom dataset format
    train_data = CustomFashionMNISTDataset(train_data, train_labels)

    # perform 5-fold cross validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    targets = np.array(train_labels)

    # get hyperparameters from config
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    dropout_rate = config["dropout"]
    lr = config["learning_rate"]
    freeze = config["freeze"]

    # initialize fold metrics
    fold_metrics = {"accuracy": [], "f1_score": [], "precision": [], "recall": []}

    # iterate through each fold
    for n, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(train_data)), targets)):
        print(f"--------------Fold {n + 1}----------------")

        # generate subsets and data loaders for the current split
        train_subset = Subset(train_data, train_idx)
        val_subset = Subset(train_data, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # initialize the model
        model = get_model(device=device, num_classes=10, dropout_rate=dropout_rate, freeze=freeze)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # iterate through each epoch
        for epoch in tqdm(range(num_epochs)):
            # perform model training and calculate metrics for the training data
            train_loss, train_acc, train_f1, train_precision, train_recall = train_one_epoch(device, model,
                                                                                             train_loader,
                                                                                             criterion,
                                                                                             optimizer)
            # perform model evaluation by calculating metrics for the validation data
            val_loss, val_acc, val_f1, val_precision, val_recall = evaluate_metrics(device=device, model=model,
                                                                                    loader=val_loader,
                                                                                    criterion=criterion)

            print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f},"
                  f"Train Precision={train_precision:.4f}, Train Recall={train_recall:.4f}")
            print(f"           Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}, "
                  f"Val Precision={val_precision:.4f}, Val Recall={val_recall:.4f}")

        print("Trained the model for one fold.")

        # calculate and print metrics for the fold
        loss, accuracy, f1, precision, recall = evaluate_metrics(device, model, val_loader, criterion)
        fold_metrics["accuracy"].append(accuracy)
        fold_metrics["f1_score"].append(f1)
        fold_metrics["precision"].append(precision)
        fold_metrics["recall"].append(recall)
        print(f"Accuracy: {accuracy}, F1 Score: {f1}, Precision: {precision}, Recall: {recall}")

    # calculate mean metrics
    mean_metrics = {metric: np.mean(values) for metric, values in fold_metrics.items()}
    df_metrics = pd.DataFrame(list(mean_metrics.items()), columns=["Metric", "Mean Value"])
    return df_metrics


def train_tuning(config: Dict[str, Any], device: torch.device,
                 train_data: CustomFashionMNISTDataset,
                 val_data: CustomFashionMNISTDataset,
                 phase: str = "val") -> Dict[str, float] | nn.Module:
    """
    Train and tune a model based on the provided configuration and data.
    This function initializes the model, optimizer, and loss criterion based on the configuration. It uses early
    stopping to halt training if the validation loss does not improve, and prints out training and validation
    metrics for each epoch.

    :param config: A dictionary containing hyperparameters for training: batch_size, epochs, dropout,
    learning_rate, freeze.
    :param device: The device to perform computation on (e.g., "cpu" or "cuda").
    :param train_data: The training dataset.
    :param val_data: The validation or testing dataset.
    :param phase: The phase of training, either "val" for validation or another phase (e.g. "test") for different
    purposes. Default is "val".
    :return: a dictionary containing validation metrics if phase is "val", else the trained model.
    """
    print("-----------------------------------")
    print(f"Using config {config} in phase {phase}")

    # extract hyperparameters
    batch_size = config["batch_size"]
    num_epochs = config["epochs"]
    dropout_rate = config["dropout"]
    lr = config["learning_rate"]
    freeze = config["freeze"]

    # prepare data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # initialize model, loss criterion, optimizer, and early stopping
    model = get_model(device, num_classes=10, dropout_rate=dropout_rate, freeze=freeze)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=5)

    # initialize lists to store loss values
    train_losses = []
    val_losses = []

    # training loop
    for epoch in tqdm(range(num_epochs)):
        # train for one epoch
        train_loss, train_acc, train_f1, train_precision, train_recall = train_one_epoch(device, model, train_loader,
                                                                                         criterion,
                                                                                         optimizer)
        # evaluate on validation data
        val_loss, val_acc, val_f1, val_precision, val_recall = evaluate_metrics(device=device, model=model,
                                                                                loader=val_loader,
                                                                                criterion=criterion)

        # store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # print training and validation metrics
        print(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Train F1={train_f1:.4f},"
              f"Train Precision={train_precision:.4f}, Train Recall={train_recall:.4f}")
        print(f"           {phase} Loss={val_loss:.4f}, {phase} Acc={val_acc:.4f}, {phase} F1={val_f1:.4f}, "
              f"{phase} Precision={val_precision:.4f}, {phase} Recall={val_recall:.4f}")

        # check for early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # return metrics if in validation phase, otherwise return the model
    if phase == "val":
        return {"accuracy": val_acc, "f1_score": val_f1, "precision": val_precision, "recall": val_recall}
    else:
        # plot loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.show()

        return model



