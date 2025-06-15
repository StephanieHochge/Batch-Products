class EarlyStopping:
    """
    A class to implement early stopping for training models.
    This class stops training when the validation loss does not improve after a certain patience.

    Attributes:
        - patience: The number of epochs with no improvement after which training will be stopped.
        - delta: The minimum change in validation loss to qualify as an improvement.
        - counter: The number of consecutive times validation lass has not improved.
        - best_score: The best recorded validation loss.
        - early_stop: Flag indicating whether early stopping should be triggered.
    """
    def __init__(self, patience: int = 5, delta: float = 0.0) -> None:
        """
        Initialize the EarlyStopping object.

        :param patience: The number of epochs to wait for improvement before stopping. Default is 5.
        :param delta: Minimum change in validation loss to qualify as an improvement. Default is 0.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        """
        Check if the validation loss has improved and update the early stopping parameters.

        :param val_loss: The current epoch's validation loss.
        """
        if self.best_score is None:
            # initialize best_score with the first validation loss
            self.best_score = val_loss

        elif val_loss > self.best_score + self.delta:
            # increment counter if no improvement
            self.counter += 1
            if self.counter >= self.patience:
                # trigger early stopping if patience is exceeded
                self.early_stop = True

        else:
            # update best_score and reset counter if there is an improvement
            self.best_score = val_loss
            self.counter = 0
