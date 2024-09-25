import torch


def find_sequence_mask(
    number_of_windows_per_patient: torch.Tensor, maximum_number_of_windows: int
) -> torch.Tensor:
    """Creates a mask for the synthetic data matrix depending on the number of windows of the sequences.

    The mask is structured as a  boolean tensor with shape (batch size, maximum number
    of windows, 1). The mask is True for positions that are within the length of the
    sequence and False for positions that are outside the length of the sequence.

    Example:
        number_windows = torch.tensor([1,3,2,2])
        maximum_number_of_windows = 3
        sequence_mask(number_windows, maximum_number_of_windows)
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
        number_of_windows_per_patient (torch.Tensor): Number of windows per patient.
        maximum_number_of_windows (int): Maximum number of windows for a patient.

    Returns:
        torch.Tensor: Mask for the results with 0s for positions outside of the length of the
            sequence and 1s for positions within the length of the sequence.
    """
    number_windows_reshaped = number_of_windows_per_patient.view((-1, 1))
    result = (
        torch.arange(maximum_number_of_windows)
        .view((1, -1))
        .to(number_of_windows_per_patient.device)
    )
    return result < number_windows_reshaped
