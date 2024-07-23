from typing import Union

import sparse
import torch

from medmodels import MedRecord
from medmodels.data_synthesis.synthesizer import Synthesizer


class SynthesizerModel(torch.nn.Module):
    def __init__(
        self,
        synthesizer: Synthesizer,
    ) -> None:
        super().__init__()
        self.input_dimension = -1  # Needs to be set by the child class
        self.number_samples = -1  # Needs to be set by the child class

        self.device = synthesizer.device
        self.to(self.device)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Forward pass for the synthesizer.

        Args:
            noise (torch.Tensor): Noise input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError(
            "Forward method must be implemented in the child class."
        )

    def generate_synthetic_data(self) -> MedRecord:
        """Generates synthetic data in the form of a MedRecord object.

        Returns:
            MedRecord: Synthetic MedRecord object.

        Raises:
            ValueError: If the NUM_SAMPLES hyperparameter is missing, invalid or non-positive.
        """

        noise = torch.randn(self.number_samples, self.input_dimension).to(self.device)
        synthetic_data = self(noise)
        return self.postprocess(synthetic_data)

    def postprocess(self, synthetic_data: Union[MedRecord, sparse.COO]) -> MedRecord:
        """Postprocesses the synthetic data using the provided postprocessor.

        Args:
            synthetic_medrecord (MedRecord): Synthetic MedRecord object.

        Returns:
            MedRecord: Postprocessed MedRecord object.

        Raises:
            NotImplementedError: If the postprocessor is not callable or does not have a 'postprocess' method.
        """
        if callable(self.postprocessor) and hasattr(self.postprocessor, "postprocess"):
            return self.postprocessor.postprocess(synthetic_data)

        raise NotImplementedError(
            "Postprocessing method not implemented for the synthesizer."
        )
