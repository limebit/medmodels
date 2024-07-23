"""Create masks for the real and synthetic data."""

import torch


def sequence_mask(
    number_admissions: torch.Tensor, max_number_admissions: int
) -> torch.Tensor:
    """Creates a mask for the number of admissions of the sequences.

    The mask is structured as a  boolean tensor with shape
    (batch_size, max_number_admissions, 1). The mask is True for positions
    that are within the length of the sequence and False for positions that
    are outside the length of the sequence.

    Example:
        number_admissions = torch.tensor([1,3,2,2])
        max_number_admissions = 3
        sequence_mask(number_admissions, max_number_admissions)
        tensor([[[ True], # 1 > 0
            [False], # 1 !> 1
            [False]], # 1 !> 2

            [[ True], # 3 > 0
            [ True], # 3 > 1
            [ True]], # 3 > 2

            [[ True], # 2 > 0
            [ True], # 2 > 1
            [False]], # 2 !> 2

            [[ True], # 2 > 0
            [ True], # 2 > 1
            [False]]]) # 2 !> 2

    Args:
        number_admissions (torch.Tensor): Number of admissions per patient.
        max_number_admissions (int): Maximum number of admissions for a patient.

    Returns:
        torch.Tensor: Mask for the results with 0s for positions outside of the length of the
            sequence and 1s for positions within the length of the sequence.
    """
    number_admissions_reshaped = number_admissions.view((-1, 1))
    result = (
        torch.arange(max_number_admissions).view((1, -1)).to(number_admissions.device)
    )
    return result < number_admissions_reshaped


def masked_softmax(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Computes safe softmax for masked points.

    Args:
        x (torch.Tensor): Input tensor.
        mask (torch.Tensor): Mask tensor.

    Returns:
        torch.Tensor: Softmax of the input tensor.
    """
    mask = (~mask).to(x.dtype)
    mask[mask == 1] = float("-inf")
    x = x + mask
    return torch.softmax(x, dim=-1)
