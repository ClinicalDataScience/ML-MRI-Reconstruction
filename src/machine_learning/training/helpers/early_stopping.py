"""Class for early stopping."""


class EarlyStopping:
    """
    Early stopping.
    """

    def __init__(self, patience=30):
        """Initialize EarlyStopping."""
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        """Stop training if validation loss has not been improved (within a defined min_delta) for a defined patience."""
        if self.best_loss == None:
            min_delta = 0
        else:
            min_delta = 0.0001 * self.best_loss

        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < min_delta:
            self.counter += 1
            print(
                f'Early stopping counter: {self.counter}.'
            )
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'Early stopping (counter={self.counter}; patience={self.patience}).')
