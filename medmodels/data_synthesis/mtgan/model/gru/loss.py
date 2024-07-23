"""Loss function for GRU model.

Prediction Loss for GRU Training to Predict codes in next admissions."""

import torch
from torch import nn

from medmodels.data_synthesis.mtgan.model.masks import sequence_mask


class PredictionLoss(nn.Module):
    """Computing masked binary cross entropy Loss for GRU Training to Predict codes in next admissions."""

    def __init__(self) -> None:
        """Constructor of the PredictionLoss."""
        super().__init__()
        self.loss_fn = nn.BCELoss(reduction="none")

    def forward(
        self,
        input_: torch.Tensor,
        predictions: torch.Tensor,
        number_admissions: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward method with masked input: compute binary cross entropy.

        Args:
            input_ (torch.Tensor): Output of the GRU.
            predictions (torch.Tensor): boolean matrix of shape (num_patients, max_number_admissions, number_codes)
                showing next admissions predictions.

        Returns:
            torch.Tensor: Loss of the GRU model.
        """
        max_number_admissions = input_.shape[1]
        mask = sequence_mask(number_admissions, max_number_admissions).unsqueeze(dim=-1)
        loss: torch.Tensor = self.loss_fn(input_, predictions)
        loss = loss * mask
        return loss.sum(dim=-1).sum(dim=-1).mean()
