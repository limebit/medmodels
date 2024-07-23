"""Base class for synthesizers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import torch
from torch import nn

from medmodels.data_synthesis.builder import SynthesizerBuilder
from medmodels.medrecord.medrecord import MedRecord

if TYPE_CHECKING:
    from medmodels.data_synthesis.synthesizer_model import SynthesizerModel


class Synthesizer(nn.Module):
    """Synthesizer is an abstract class that serves as a blueprint for synthesizers.

    It ensures the correct instantiation, training, and persistence of
    models.
    """

    def __init__(
        self,
        preprocessor: nn.Module,
        postprocessor: nn.Module,
    ) -> None:
        """Initializes the synthesizer with the required components.

        Args:
            preprocessor (nn.Module): Preprocessor module.
            postprocessor (nn.Module): Postprocessor module.
        """
        super(Synthesizer, self).__init__()
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @classmethod
    def builder(cls) -> SynthesizerBuilder:
        """Creates a MTGANBuilder instance for the MTGAN class."""

        return SynthesizerBuilder()

    def preprocess(self, medrecord: MedRecord) -> MedRecord:
        return self.preprocessor.preprocess(medrecord)

    def fit(
        self, medrecord: MedRecord, save_directory: Optional[Path]
    ) -> SynthesizerModel:
        """Trains the synthesizer on the given data.

        Note that this method should be implemented in the child class.

        Args:
            medrecord (MedRecord): MedRecord object.
            save_directory (Optional[Path]): Directory to save the model to.

        Returns:
            SynthesizerModel: Trained synthesizer.

        Raises:
            NotImplementedError: If the training method is not implemented.
        """
        raise NotImplementedError(
            "Training method not implemented for the synthesizer."
        )

    def fit_from(self, **kwargs: Dict[str, Path]) -> SynthesizerModel:
        """Trains the synthesizer on the data at the given path.

        Note that this method should be implemented in the child class.

        Args:
            **kwargs (Dict[str, Path]): Dictionary containing the paths to the data.

        Returns:
            Synthesizer: Trained synthesizer.

        Raises:
            NotImplementedError: If the training method is not implemented.
        """
        raise NotImplementedError(
            "Training method not implemented for the synthesizer."
        )

    def save_model(self, path: Path) -> None:
        """Saves the model to a specified path.

        Args:
            path (Path): Path to save the model parameters to.
        """
        try:
            torch.save(self.state_dict(), path)
        except Exception as e:
            msg = f"Error saving model to '{path}': {e}"
            raise Exception(msg)

    def load_model(self, path: Path) -> SynthesizerModel:
        """Loads the model from a specified path.

        Args:
            path (Path): Path to load the model parameters from.

        Returns:
            SynthesizerModel: The loaded model.

        Raises:
            FileNotFoundError: If the model file is not found.
            Exception: If there is an error loading the model.
        """
        if not path.exists():
            msg = f"Model file '{path}' not found."
            raise FileNotFoundError(msg)

        try:
            self.load_state_dict(torch.load(path, map_location=self.device))
        except Exception as e:
            msg = f"Error loading model from '{path}': {e}"
            raise Exception(msg)

        return SynthesizerModel(self)
