import torch


def find_sequence_mask(
    number_of_windows_per_patient: torch.Tensor, maximum_number_of_windows: int
) -> torch.Tensor:
    """Creates a mask for the synthetic data matrix depending on the number of windows of the sequences.

    The mask is structured as a  boolean tensor with shape (batch size, maximum number
    of windows, 1). The mask is True for positions that are within the length of the
    sequence and False for positions that are outside the length of the sequence.

    Consider `number_of_windows_per_patient` to be `torch.tensor([1,3,2,2])` and
    `maximum_number_of_windows` to be 3. The function would output the following
    tensor (4, 3) in shape

    tensor([[ True, False, False],  # Patient 1, 1 window
            [ True,  True,  True],  # Patient 2, 3 windows
            [ True,  True, False],  # Patient 3, 2 windows
            [ True,  True, False]]  # Patient 4, 2 windows
    )

    Args:
        number_of_windows_per_patient (torch.Tensor): Number of windows per patient.
        maximum_number_of_windows (int): Maximum number of windows for a patient.

    Returns:
        torch.Tensor: Mask for the results with 0s for positions outside of the length of the
            sequence and 1s for positions within the length of the sequence.
    """
    result = (
        torch.arange(maximum_number_of_windows)
        .view((1, -1))
        .to(number_of_windows_per_patient.device)
    )
    return result < number_of_windows_per_patient.view((-1, 1))


def apply_masked_softmax(data: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply softmax to a tensor while masking out specified elements.

    This function computes the softmax of the input tensor 'data', but first sets the
    elements specified by the 'mask' to negative infinity (-inf). This ensures that
    after applying softmax, the masked positions will have a value of zero, since
    softmax(-inf) is 0.

    This approach is useful for ignoring or masking out certain values when computing
    softmax, such as in attention mechanisms where specific positions should not
    contribute to the attention scores.

    Args:
        data (torch.Tensor): Input tensor.
        mask (torch.Tensor): Mask tensor.

    Returns:
        torch.Tensor: Softmax output.
    """
    mask = (~mask).to(data.dtype)
    mask[mask == 1] = float("-inf")
    data = data + mask
    return torch.softmax(data, dim=-1)
