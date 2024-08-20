"""Abstract class for synthesizers."""

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

    It ensures the correct instantiation, training, and persistence of models.
    """
    preprocessor: nn.Module
    postprocessor: nn.Module
    device: torch.device

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
        """Creates a SynthesizerBuilder instance.

        Returns:
            SynthesizerBuilder: An instance of SynthesizerBuilder.
        """
        return SynthesizerBuilder()

    def preprocess(self, medrecord: MedRecord) -> MedRecord:
        """Preprocesses a MedRecord instance to be used by the synthesizer.

        Args:
            medrecord (MedRecord): A MedRecord instance.

        Returns:
            MedRecord: A preprocessed MedRecord instance.

        Raises:
            NotImplementedError: If the preprocess method is not implemented in the
                preprocessor.
        """
        if not callable(self.preprocessor) or not hasattr(
            self.preprocessor, "preprocess"
        ):
            raise NotImplementedError("Preprocessor must have a preprocess method.")

        return self.preprocessor.preprocess(medrecord)

    def fit(
        self, medrecord: MedRecord, save_directory: Optional[Path]
    ) -> SynthesizerModel:
        """Trains the synthesizer on the given data and saves the model if a directory is provided.

        This method is meant to fit the synthesizer on the given data on scratch (no
        pre-trained model is used).

        It is an abstract method that should be implemented in the child class.

        Args:
            medrecord (MedRecord): MedRecord object.
            save_directory (Optional[Path]): Directory to save the model to.

        Returns:
            SynthesizerModel: Trained synthesizer.

        Raises:
            NotImplementedError: If the fit method is not implemented in the child
                class.
        """
        raise NotImplementedError("Fit method must be implemented in the child class.")

    def fit_from(
        self,
        medrecord: MedRecord,
        save_directory: Optional[Path],
        **kwargs: Dict[str, Path],
    ) -> SynthesizerModel:
        """Trains the synthesizer on the given data, using a pre-trained model that is provided.

        This method is meant to fit the synthesizer on the given data using a
        pre-trained model that is provided. This model will be saved to the given
        directory if it is provided.

        It is an abstract method that should be implemented in the child class.

        Args:
            medrecord (MedRecord): MedRecord object.
            save_directory (Optional[Path]): Directory to save the model to.
            **kwargs (Dict[str, Path]): Dictionary containing the paths to the
                saved model/checkpoints of the model.

        Returns:
            Synthesizer: Trained synthesizer.

        Raises:
            NotImplementedError: If the fitting method is not implemented in the child
                class.
        """
        raise NotImplementedError(
            "Fit from method must be implemented in the child class."
        )

    def save_model(self, path: Path) -> None:
        """Saves the model to a specified path.

        This method is meant to save the model to the given path. It is an abstract
        method that should be implemented in the child class.

        Args:
            path (Path): Path to save the model parameters to.

        Raises:
            NotImplementedError: If the save model method is not implemented in the
                child class.
        """
        raise NotImplementedError(
            "Save model method must be implemented in the child class."
        )

    def load_model(self, path: Path) -> SynthesizerModel:
        """Loads the model from a specified path.

        This method is meant to load the model from the given path. It is an abstract
        method that should be implemented in the child class.

        Args:
            path (Path): Path to load the model parameters from.

        Returns:
            SynthesizerModel: Loaded synthesizer model.

        Raises:
            NotImplementedError: If the load model method is not implemented in the
                child class.
        """
        raise NotImplementedError(
            "Load model method must be implemented in the child class."
        )
