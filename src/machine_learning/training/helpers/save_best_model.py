"""Class for saving the best model while training."""
import os
from typing import Union

import torch


class SaveBestModel:
    """Save the best model during training."""

    def __init__(self, best_val_loss=float('inf')) -> None:
        """Initialize SaveBestModel."""
        self.best_val_loss = best_val_loss

    def __call__(
        self,
        current_val_loss: float,
        epoch: int,
        model,
        optimizer,
        path_to_save: Union[str, os.PathLike],
        num_spokes: int,
    ) -> None:
        """Save the best model while training."""
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                os.path.join(
                    path_to_save,
                    'ML_' + str(num_spokes) + '_spokes' + '_model_best.pth',
                ),
            )
            print(f'Saved best model at epoch {epoch+1}.')
