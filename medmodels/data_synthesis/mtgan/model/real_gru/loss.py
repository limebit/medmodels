"""Loss function for RealGRU model.

Prediction Loss for RealGRU training to predict codes in next windows."""

import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.masks import find_sequence_mask


class PredictionLoss(nn.Module):
    """Computing masked binary cross entropy Loss for RealGRU Training to predict codes
    in next windows.

    This will be used to train the RealGRU model to predict the next windows of the
    patients."""

    loss_function: nn.BCELoss

    def __init__(self, device: torch.device) -> None:
        """Constructor of the PredictionLoss.

        Args:
            device (torch.device): The device to use for the loss function.
        """
        super().__init__()
        self.to(device)
        self.loss_function = nn.BCELoss(reduction="none")

    def forward(
        self,
        input_data: torch.Tensor,
        predictions: torch.Tensor,
        number_of_windows_per_patient: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward method with masked input: compute binary cross entropy.

        Args:
            input_data (torch.Tensor): Output of the GRU.
            predictions (torch.Tensor): boolean matrix of shape (batch size, maximum number of
                windows, total number of concepts) showing next windows predictions.
            number_of_windows_per_patient (torch.Tensor): number of windows per patient.

        Returns:
            torch.Tensor: Loss of the GRU model.
        """
        maximum_number_of_windows = input_data.shape[1]
        mask = find_sequence_mask(
            number_of_windows_per_patient, maximum_number_of_windows
        ).unsqueeze(dim=-1)
        loss = self.loss_function(input_data, predictions)
        loss = loss * mask
        return loss.sum(dim=-1).sum(dim=-1).mean()
