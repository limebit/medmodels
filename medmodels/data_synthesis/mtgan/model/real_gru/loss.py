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
        real_prediction_data: torch.Tensor,
        synthetic_prediction_data: torch.Tensor,
        number_of_windows_per_patient: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward method with masked input: compute binary cross entropy.

        The predictions of the real data are the real data itself shifted by one
        window. This way we can compute the loss of the RealGRU model to predict the
        next window of the patients. For each patient, the loss is computed only for
        the windows that are part of the sequence. The loss is computed for each
        concept and then summed over all concepts and windows.

        Args:
            real_prediction_data (torch.Tensor): boolean matrix of shape (batch size,
                maximum number of windows - 1, total number of concepts) showing the
                training data corresponding to the input data shifted by one window.
            synthetic_prediction_data (torch.Tensor): Predictions of the RealGRU, of
                shape (batch size, maximum number of windows - 1, total number of
                concepts).
            number_of_windows_per_patient (torch.Tensor): number of windows per patient.

        Returns:
            torch.Tensor: Prediction loss of the GRU model.
        """
        maximum_number_of_windows = synthetic_prediction_data.shape[1]
        mask = find_sequence_mask(
            number_of_windows_per_patient, maximum_number_of_windows
        ).unsqueeze(dim=-1)
        loss = self.loss_function(real_prediction_data, synthetic_prediction_data)
        loss = loss * mask
        return loss.sum(dim=-1).sum(dim=-1).mean()
