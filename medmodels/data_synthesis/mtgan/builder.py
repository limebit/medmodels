from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from typing_extensions import Unpack

import medmodels.data_synthesis.mtgan.mtgan as mtgan
from medmodels.data_synthesis.mtgan.modules.postprocessor import (
    AttributeType,
    PostprocessingHyperparametersOptional,
)
from medmodels.data_synthesis.mtgan.modules.preprocessor import (
    PreprocessingHyperparameters,
)
from medmodels.data_synthesis.mtgan.train.gan_trainer import TrainingHyperparameters
from medmodels.medrecord.types import MedRecordAttribute


class MTGANBuilder:
    seed: int

    training_hyperparameters: TrainingHyperparameters
    preprocessing_hyperparameters: PreprocessingHyperparameters
    postprocessing_hyperparameters: PostprocessingHyperparametersOptional

    training_hyperparameters_json: TrainingHyperparameters
    preprocessing_hyperparameters_json: PreprocessingHyperparameters
    postprocessing_hyperparameters_json: PostprocessingHyperparametersOptional

    def with_seed(self, seed: int) -> MTGANBuilder:
        """Sets the seed for the MTGAN model.

        Args:
            seed (int): The seed to set.

        Returns:
            MTGANBuilder: The current instance of the MTGANBuilder.
        """
        self.seed = seed

        return self

    def with_preprocessor_hyperparameters(
        self, **kwargs: Unpack[PreprocessingHyperparameters]
    ) -> MTGANBuilder:
        """Sets the preprocessor hyperparameters for the MTGAN model.

        Args:
            **kwargs: The hyperparameters for the preprocessor. The hyperparameters
                should be passed as keyword arguments and they can be any of the
                following:
                - minimum_occurrences_concept(int) - The minimum number of occurrences required
                    of a concept in the dataset.
                - time_interval_days(int) - The time interval in days for the time window.
                - number_sampled_patients(int) - The number of patients to sample from the dataset.
                - minimum_codes_per_window(int) - The minimum number of codes required in a time window.

        Returns:
            MTGANBuilder: The current instance of the MTGANBuilder.
        """
        self.preprocessing_hyperparameters = kwargs

        return self

    def with_training_hyperparameters(
        self, **kwargs: Unpack[TrainingHyperparameters]
    ) -> MTGANBuilder:
        """Sets the training hyperparameters for the MTGAN model.

        Args:
            **kwargs: The hyperparameters for the training. The hyperparameters
                should be passed as keyword arguments and they can be any of the
                following:
                - batch_size(int) - The batch size for training.
                - learning_rate(float) - The learning rate for training.
                - number_epochs(int) - The number of epochs for training.
                - number_previous_admissions(int) - The number of previous admissions to consider.
                - top_k_codes(int) - The number of top codes to consider.

        Returns:
            MTGANBuilder: The current instance of the MTGANBuilder.
        """
        self.training_hyperparameters = kwargs

        return self

    def with_postprocessor_hyperparameters(
        self, **kwargs: Unpack[PostprocessingHyperparametersOptional]
    ) -> MTGANBuilder:
        """Sets the postprocessor hyperparameters for the MTGAN model.

        Args:
            **kwargs: The hyperparameters for the postprocessor. The hyperparameters
                should be passed as keyword arguments and they can be any of the
                following:
                - training_epochs(int) - The number of training epochs.
                - hidden_dim(int) - The hidden dimension of the model.
                - number_layers(int) - The number of layers in the model.
                - learning_rate(float) - The learning rate for training.
                - batch_size(int) - The batch size for training.
                - number_previous_admissions(int) - The number of previous admissions to consider.
                - top_k_codes(int) - The number of top codes to consider.

        Returns:
            MTGANBuilder: The current instance of the MTGANBuilder.
        """
        self.postprocessing_hyperparameters = kwargs

        return self

    def with_postprocessor_attributes(
        self, attributes_types: Dict[MedRecordAttribute, AttributeType]
    ) -> MTGANBuilder:
        """Sets the postprocessor attributes for the MTGAN model.

        Args:
            attributes_types (Dict[MedRecordAttribute, AttributeType]): The attributes
                postprocessing for the model.

        Returns:
            MTGANBuilder: The current instance of the MTGANBuilder.
        """
        self.attributes_types = attributes_types

        return self

    def load_hyperparameters_from(self, path: Path) -> MTGANBuilder:
        """Loads the hyperparameters from a JSON file.

        Args:
            path (Path): The path to the JSON file containing the hyperparameters.

        Returns:
            MTGANBuilder: The current instance of the MTGANBuilder.

        Raises:
            FileNotFoundError: If the path does not exist or if the path is not a file.
            ValueError: If the file does not contain any of the following keys:
                'training', 'preprocessing', 'postprocessing'.
        """

        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")

        if not path.is_file():
            raise FileNotFoundError(f"Path {path} is not a file.")

        with open(path, "r") as file:
            hyperparameters = json.load(file)
            if "training" in hyperparameters:
                self.training_hyperparameters_json = hyperparameters["training"]
                for key in self.training_hyperparameters_json.keys():
                    if key not in TrainingHyperparameters.__annotations__.keys():
                        raise ValueError(
                            f"Invalid training hyperparameter key: {key} in JSON. The key should be one of the following: {list(TrainingHyperparameters.__annotations__.keys())}"
                        )
            if "preprocessing" in hyperparameters:
                self.preprocessing_hyperparameters_json = hyperparameters[
                    "preprocessing"
                ]
                for key in self.preprocessing_hyperparameters_json.keys():
                    if key not in PreprocessingHyperparameters.__annotations__.keys():
                        raise ValueError(
                            f"Invalid preprocessing hyperparameter key: {key} in JSON. The key should be one of the following: {list(PreprocessingHyperparameters.__annotations__.keys())}"
                        )
            if "postprocessing" in hyperparameters:
                self.postprocessing_hyperparameters_json = hyperparameters[
                    "postprocessing"
                ]
                for key in self.postprocessing_hyperparameters_json.keys():
                    if (
                        key
                        not in PostprocessingHyperparametersOptional.__annotations__.keys()
                    ):
                        raise ValueError(
                            f"Invalid postprocessing hyperparameter key: {key} in JSON. The key should be one of the following: {list(PostprocessingHyperparametersOptional.__annotations__.keys())}"
                        )
            if (
                "training" not in hyperparameters
                and "preprocessing" not in hyperparameters
                and "postprocessing" not in hyperparameters
            ):
                raise ValueError(
                    "Invalid hyperparameters file format. The file should contain at least one of the following keys: 'training', 'preprocessing', 'postprocessing'."
                )

        return self

    def build(self) -> mtgan.MTGAN:
        """Builds the treatment effect with all the provided configurations.

        Returns:
            MTGAN: The MTGAN model with the provided configurations.
        """
        mtgan_model = mtgan.MTGAN.__new__(mtgan.MTGAN)
        mtgan.MTGAN._set_configuration(mtgan_model, **vars(self))
        return mtgan_model
