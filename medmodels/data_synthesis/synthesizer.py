"""Abstract class for synthesizers."""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Type

import torch
from torch import nn

from medmodels.medrecord.medrecord import MedRecord
from medmodels.data_synthesis.synthesizer_model import SynthesizerModel

if TYPE_CHECKING:
    from medmodels.data_synthesis.builder import SynthesizerBuilder


class Synthesizer(nn.Module, metaclass=ABCMeta):
    """Synthesizer is an abstract class that serves as a blueprint for synthesizers.

    It ensures the correct instantiation, training, and persistence of models.
    """

    _preprocessor: Type[nn.Module]
    _postprocessor: Type[nn.Module]
    device: torch.device

    def __init__(
        self,
        preprocessor: Type[nn.Module],
        postprocessor: Type[nn.Module],
    ) -> None:
        """Initializes the synthesizer with the required components.

        Args:
            preprocessor (Type[nn.Module]): Type of preprocessor module.
            postprocessor (Type[nn.Module]): Type of postprocessor module.

        Raises:
            NotImplementedError: If the preprocessor or postprocessor does not have the
                required methods (preprocess and postprocess, respectively).
        """
        super(Synthesizer, self).__init__()
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor

        if not callable(getattr(self._preprocessor, "preprocess", None)):
            raise NotImplementedError("Preprocessor must have a preprocess method.")

        if not callable(getattr(self._postprocessor, "postprocess", None)):
            raise NotImplementedError("Postprocessor must have a postprocess method.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    @classmethod
    def builder(cls) -> SynthesizerBuilder:
        """Creates a SynthesizerBuilder instance.

        Returns:
            SynthesizerBuilder: An instance of SynthesizerBuilder.
        """
        from medmodels.data_synthesis.builder import SynthesizerBuilder
        return SynthesizerBuilder()

    @abstractmethod
    def fit(
        self, medrecord: MedRecord, checkpoint_directory: Optional[Path],
    ) -> SynthesizerModel:
        """Trains the synthesizer on the given data and saves the model if a directory is provided.

        This method is meant to fit the synthesizer on the given data on scratch (no
        pre-trained model is used).

        It is an abstract method that should be implemented in the child class.

        Args:
            medrecord (MedRecord): MedRecord object.
            checkpoint_directory (Optional[Path]): Directory to save the model to.

        Returns:
            SynthesizerModel: Trained synthesizer.
        """
        pass

    @abstractmethod
    def fit_from(
        self,
        medrecord: MedRecord,
        checkpoint_directory: Optional[Path],
        **kwargs: Dict[str, Path],
    ) -> SynthesizerModel:
        """Trains the synthesizer on the given data, using a pre-trained model that is provided.

        This method is meant to fit the synthesizer on the given data using a
        pre-trained model that is provided. This model will be saved to the given
        checkpoint directory if it is provided. It will take the checkpoint paths
        where the model is saved as keyword arguments to start fitting from them.

        It is an abstract method that should be implemented in the child class.

        Args:
            medrecord (MedRecord): MedRecord object.
            save_directory (Optional[Path]): Directory to save the model to.
            **kwargs (Dict[str, Path]): Dictionary containing the paths to the
                saved model/checkpoints of the model.

        Returns:
            Synthesizer: Trained synthesizer.
        """
        pass

    @abstractmethod
    def load_model(self, medrecord: MedRecord, path: Path) -> SynthesizerModel:
        """Loads the model from a specified path.

        This method is meant to load the model from the given path. It is an abstract
        method that should be implemented in the child class.

        Args:
            medrecord (MedRecord): MedRecord object.
            path (Path): Path to load the model parameters from.

        Returns:
            SynthesizerModel: Loaded synthesizer model.
        """
        pass
