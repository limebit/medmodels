"""Class for the MTGAN model

MTGAN is a generative adversarial network (GAN) that generates synthetic
electronic health records (EHRs) with the same statistical properties as the
real data. The model is trained on real EHRs and generates synthetic EHRs.

It has 4 main methods:
- fit: trains the MTGAN model.
- fit_from: fits the MTGAN model from a saved RealGRU model (and optionally a saved model).
- load_model: loads a MTGAN model from a pre-saved one.
- generate_synthetic_data: generates synthetic data with the MTGAN model.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional

from medmodels.data_synthesis.mtgan.builder import MTGANBuilder
from medmodels.data_synthesis.mtgan.mtgan_model import MTGANModel
from medmodels.medrecord.types import MedRecordAttribute

if TYPE_CHECKING:
    from medmodels.data_synthesis.mtgan.mtgan import MTGAN

import numpy as np
import pandas as pd
import torch

from medmodels.data_synthesis.mtgan.model.critic.critic import Critic
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator
from medmodels.data_synthesis.mtgan.model.gru.real_gru import RealGRU
from medmodels.data_synthesis.mtgan.model.loaders import (
    MTGANDataLoader,
    MTGANDataset,
    MTGANDatasetPrediction,
)
from medmodels.data_synthesis.mtgan.modules.postprocessor import (
    AttributeType,
    MTGANPostprocessor,
    PostprocessingHyperparameters,
    PostprocessingHyperparametersOptional,
)
from medmodels.data_synthesis.mtgan.modules.preprocessor import (
    MTGANPreprocessor,
    PreprocessingHyperparameters,
    PreprocessingHyperparametersTotal,
)
from medmodels.data_synthesis.mtgan.train.gan_trainer import (
    GANTrainer,
    TrainingHyperparameters,
    TrainingHyperparametersTotal,
)
from medmodels.data_synthesis.mtgan.train.real_gru_trainer import (
    RealGRUTrainer,
)
from medmodels.data_synthesis.synthesizer import Synthesizer
from medmodels.medrecord.medrecord import MedRecord


class GeneratorInputs(NamedTuple):
    number_codes: int
    max_number_admissions: int
    generator_hidden_dim: int
    generator_attention_dim: int
    device: torch.device


class MTGAN(Synthesizer):
    """Class for the MTGAN model.

    MTGAN is a generative adversarial network (GAN) that generates synthetic
    electronic health records (EHRs) with the same statistical properties as the
    real data. The model is trained on real EHRs and generates synthetic EHRs.

    It has 4 main methods:
    - fit: trains the MTGAN model.
    - fit_from: fits the MTGAN model from a saved RealGRU model (and optionally a saved
        model).
    - load_model: loads a MTGAN model from a pre-saved one.
    - generate_synthetic_data: generates synthetic data with the MTGAN model.
    """

    preprocessor: MTGANPreprocessor
    postprocessor: MTGANPostprocessor
    generator: Optional[Generator]

    _preprocessing_hyperparameters: PreprocessingHyperparametersTotal
    _postprocessing_hyperparameters: PostprocessingHyperparameters
    _training_hyperparameters: TrainingHyperparametersTotal

    def __init__(
        self,
    ) -> None:
        """Constructor of the MTGAN class."""
        # TODO: Set logger
        MTGAN._set_configuration(self)

    def _initialize_parent_class(
        self, preprocessor: MTGANPreprocessor, postprocessor: MTGANPostprocessor
    ) -> None:
        """Initializes the parent class.

        Args:
            preprocessor (MTGANPreprocessor): Preprocessor for the MTGAN model.
            postprocessor (MTGANPostprocessor): Postprocessor for the MTGAN model.
        """
        super().__init__(preprocessor, postprocessor)

    def _initalize_seed(self, seed: int) -> None:
        """Initializes the seed for the random number generation.

        Args:
            seed (int): Seed for random number generation.
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    @classmethod
    def builder(cls) -> MTGANBuilder:
        """Creates a MTGANBuilder instance for the MTGAN class."""

        return MTGANBuilder()

    @staticmethod
    def _set_configuration(
        mtgan: MTGAN,
        *,
        preprocessor: MTGANPreprocessor = MTGANPreprocessor(),
        postprocessor: MTGANPostprocessor = MTGANPostprocessor(),
        preprocessing_hyperparameters: PreprocessingHyperparameters = {},
        training_hyperparameters: TrainingHyperparameters = {},
        postprocessing_hyperparameters: PostprocessingHyperparametersOptional = {},
        preprocessing_hyperparameters_json: PreprocessingHyperparameters = {},
        training_hyperparameters_json: TrainingHyperparameters = {},
        postprocessing_hyperparameters_json: PostprocessingHyperparametersOptional = {},
        attributes_types: Dict[MedRecordAttribute, AttributeType] = {},
        seed: int = 0,
    ) -> None:
        """Sets the configuration for the MTGAN model.

        Args:
            mtgan (MTGAN): The MTGAN model.
            preprocessor (MTGANPreprocessor): Preprocessor for the MTGAN model.
            postprocessor (MTGANPostprocessor): Postprocessor for the MTGAN model.
            preprocessing_hyperparameters (PreprocessingHyperparameters): Preprocessing hyperparameters.
            training_hyperparameters (TrainingHyperparameters): Training hyperparameters.
            postprocessing_hyperparameters (PostprocessingHyperparameters): Postprocessing hyperparameters.
            preprocessing_hyperparameters_json (PreprocessingHyperparameters): Preprocessing hyperparameters from a JSON file.
            training_hyperparameters_json (TrainingHyperparameters): Training hyperparameters from a JSON file.
            postprocessing_hyperparameters_json (PostprocessingHyperparameters): Postprocessing hyperparameters from a JSON file.
            attributes_types (Dict[MedRecordAttribute, AttributeType]): Attributes types for the postprocessor.
            seed (int): Seed for random number generation.
        """
        mtgan._initialize_parent_class(preprocessor, postprocessor)
        mtgan._initalize_seed(seed)

        mtgan._preprocessing_hyperparameters = PreprocessingHyperparametersTotal(
            minimum_occurrences_concept=1,
            time_interval_days=1,
            minimum_codes_per_window=1,
            number_sampled_patients=0,
        )
        mtgan._preprocessing_hyperparameters.update(preprocessing_hyperparameters_json)
        mtgan._preprocessing_hyperparameters.update(preprocessing_hyperparameters)

        mtgan._training_hyperparameters = TrainingHyperparametersTotal(
            batch_size=8,
            real_gru_training_epochs=200,
            real_gru_lr=1e-3,
            gan_training_epochs=300,
            critic_hidden_dim=64,
            generator_hidden_dim=256,
            generator_attention_dim=64,
            critic_iterations=1,
            generator_iterations=1,
            critic_lr=1e-5,
            generator_lr=1e-4,
            beta0=0.5,
            beta1=0.9,
            decay_step=100000,
            decay_rate=0.1,
            lambda_gradient=10.0,
            lambda_sparseness=1.0,
            test_freq=100,
        )
        mtgan._training_hyperparameters.update(training_hyperparameters_json)
        mtgan._training_hyperparameters.update(training_hyperparameters)

        mtgan._postprocessing_hyperparameters = PostprocessingHyperparameters(
            number_patients_generated=1000,
            training_epochs=100,
            hidden_dim=64,
            number_layers=2,
            batch_size=32,
            number_previous_admissions=5,
            learning_rate=1e-3,
            top_k_codes=500,
        )
        mtgan._postprocessing_hyperparameters.update(
            postprocessing_hyperparameters_json
        )
        mtgan._postprocessing_hyperparameters.update(postprocessing_hyperparameters)

        mtgan.preprocessor.hyperparameters.update(mtgan._preprocessing_hyperparameters)
        mtgan.postprocessor.hyperparameters.update(
            mtgan._postprocessing_hyperparameters
        )
        mtgan.postprocessor.attributes_types.update(attributes_types)

    def _setup_real_gru(
        self, medrecord: MedRecord, saved_gru_path: Optional[Path] = None
    ) -> RealGRU:
        """Set up the RealGRU model.

        Args:
            medrecord (MedRecord): The medical record data.
            saved_gru_path (Optional[Path]): Path to the saved RealGRU model. Defaults
                to None.

        Returns:
            RealGRU: The RealGRU model.

        Raises:
            FileNotFoundError: If the saved RealGRU model is not found and the path is
                provided.
        """
        number_codes = len(medrecord.nodes_in_group(self.preprocessor.concepts_group))
        real_gru = RealGRU(
            number_codes=number_codes,
            real_gru_hidden_dimension=self._training_hyperparameters.get(
                "generator_hidden_dim"
            ),
        ).to(self.device)

        if saved_gru_path and os.path.exists(saved_gru_path):
            real_gru.load_state_dict(
                torch.load(saved_gru_path, map_location=self.device)
            )
        elif saved_gru_path and not os.path.exists(saved_gru_path):
            raise FileNotFoundError(f"RealGRU model file {saved_gru_path} not found.")
        else:
            real_gru = self._train_real_gru(real_gru, medrecord=medrecord)

        real_gru.eval()
        return real_gru

    def _train_real_gru(self, real_gru: RealGRU, medrecord: MedRecord) -> RealGRU:
        """Train the RealGRU model.

        Args:
            real_gru (RealGRU): The RealGRU model before training.

        Returns:
            RealGRU: The trained RealGRU model.
        """
        real_gru_trainloader = MTGANDataLoader(
            MTGANDatasetPrediction(self.preprocessor, medrecord=medrecord),
            batch_size=self._training_hyperparameters.get("batch_size"),
            shuffle=True,
        )
        real_gru_trainer = RealGRUTrainer(
            real_gru=real_gru,
            train_loader=real_gru_trainloader,
            hyperparameters=self._training_hyperparameters,
        )
        logging.info("Training RealGRU...")
        real_gru_trainer.train()

        return real_gru

    def fit(
        self,
        medrecord: MedRecord,
        save_directory: Optional[Path] = None,
    ) -> MTGANModel:
        """Trains MTGAN on the given data.

        We will train the generator and the critic needed for a GAN architecture.
        The return is a trained MTGAN model.

        Args:
            medrecord (MedRecord): The medical record data.
            save_directory (Optional[Path]): Path to save the trained model. Defaults
                to None.

        Returns:
            MTGANModel: The trained MTGAN model.
        """
        if save_directory is not None and not os.path.exists(save_directory):
            raise FileNotFoundError(
                f"Saving directory {save_directory} does not exist."
            )

        preprocessed_medrecord = self.preprocess(medrecord)
        max_number_admissions = np.array(
            list(
                medrecord.node[
                    preprocessed_medrecord.nodes_in_group(
                        self.preprocessor.patients_group
                    ),
                    self.preprocessor.number_admissions_attribute,
                ].values()
            )
        ).max()
        number_codes = len(
            preprocessed_medrecord.nodes_in_group(self.preprocessor.concepts_group)
        )
        generator_inputs = GeneratorInputs(
            number_codes=number_codes,
            max_number_admissions=max_number_admissions,
            generator_hidden_dim=self._training_hyperparameters["generator_hidden_dim"],
            generator_attention_dim=self._training_hyperparameters.get(
                "generator_attention_dim"
            ),
            device=self.device,
        )

        # Pretrain RealGRU and save it
        real_gru = self._setup_real_gru(medrecord=preprocessed_medrecord)
        if save_directory is not None:
            time_gru = pd.Timestamp.now().strftime("%d-%m-%y_%H-%M")
            saved_gru_path = Path(
                os.path.join(save_directory, f"{time_gru}_saved_gru.pth")
            )
            torch.save(real_gru.state_dict(), saved_gru_path)
            logging.info("RealGRU saved at %s", saved_gru_path)

        # Create Critic and Generator
        generator = Generator(*generator_inputs).to(self.device)
        critic = Critic(
            number_codes=number_codes,
            critic_hidden_dimension=self._training_hyperparameters.get(
                "critic_hidden_dim"
            ),
            generator_hidden_dimension=self._training_hyperparameters.get(
                "generator_hidden_dim"
            ),
        ).to(self.device)

        trainer = GANTrainer(
            generator=generator,
            critic=critic,
            real_gru=real_gru,
            dataset=MTGANDataset(self.preprocessor, medrecord=preprocessed_medrecord),
            hyperparameters=self._training_hyperparameters,
            device=self.device,
        )

        # Train Generator and Critic
        logging.info("Training Generator and Critic...")
        generator, critic = trainer.train()
        saved_model = {
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
        }
        if save_directory is not None:
            time_training = pd.Timestamp.now().strftime("%d-%m-%y_%H-%M")
            new_saved_model = os.path.join(
                save_directory, f"{time_training}_saved_model.pth"
            )
            torch.save(saved_model, new_saved_model)

        return MTGANModel(
            medrecord=preprocessed_medrecord, mtgan=self, generator=generator
        )

    def fit_from(
        self,
        medrecord: MedRecord,
        saved_gru_path: Path,
        saved_model_path: Optional[Path] = None,
        save_directory: Optional[Path] = None,
    ) -> MTGANModel:
        """Fits the MTGAN model from a saved RealGRU model (and optionally a saved Generator model).

        Args:
            medrecord (MedRecord): The medical record data.
            saved_gru_path (Path): Path to the saved RealGRU model.
            saved_model_path (Optional[Path]): Path to the saved model. Defaults to
                None.
            save_directory (Optional[Path]): Path to save the trained model. Defaults
                to None. If None, the model will be saved in the same directory as the
                saved RealGRU model.

        Returns:
            MTGANModel: The trained MTGAN model.

        Raises:
            FileNotFoundError: If the saving directory does not exist.
        """
        if save_directory is not None and not os.path.exists(save_directory):
            raise FileNotFoundError(
                f"Saving directory {save_directory} does not exist."
            )
        elif save_directory is None:
            save_directory = saved_gru_path.parent
            logging.info(
                "Saving directory not provided. Saving model at %s", save_directory
            )

        preprocessed_medrecord = self.preprocess(medrecord)

        max_number_admissions = np.array(
            list(
                preprocessed_medrecord.node[
                    preprocessed_medrecord.nodes_in_group(
                        self.preprocessor.patients_group
                    ),
                    self.preprocessor.number_admissions_attribute,
                ].values()
            )
        ).max()
        number_codes = len(
            preprocessed_medrecord.nodes_in_group(self.preprocessor.concepts_group)
        )

        generator_inputs = GeneratorInputs(
            number_codes=number_codes,
            max_number_admissions=max_number_admissions,
            generator_hidden_dim=self._training_hyperparameters["generator_hidden_dim"],
            generator_attention_dim=self._training_hyperparameters.get(
                "generator_attention_dim"
            ),
            device=self.device,
        )
        real_gru = self._setup_real_gru(preprocessed_medrecord, saved_gru_path)

        # Create Critic and Generator
        generator = Generator(*generator_inputs).to(self.device)
        critic = Critic(
            number_codes=number_codes,
            critic_hidden_dimension=self._training_hyperparameters.get(
                "critic_hidden_dim"
            ),
            generator_hidden_dimension=self._training_hyperparameters.get(
                "generator_hidden_dim"
            ),
        ).to(self.device)

        if saved_model_path and os.path.exists(saved_model_path):
            logging.info("Loaded Generator from %s", saved_model_path)
            saved_model = torch.load(saved_model_path, map_location=self.device)
            generator.load_state_dict(saved_model["generator_state_dict"])
            critic.load_state_dict(saved_model["critic_state_dict"])

        trainer = GANTrainer(
            generator=generator,
            critic=critic,
            real_gru=real_gru,
            dataset=MTGANDataset(self.preprocessor, medrecord=preprocessed_medrecord),
            hyperparameters=self._training_hyperparameters,
            device=self.device,
        )

        # Train Generator and Critic
        logging.info("Training Generator and Critic...")
        generator, critic = trainer.train()
        saved_model = {
            "generator_state_dict": generator.state_dict(),
            "critic_state_dict": critic.state_dict(),
        }
        time = pd.Timestamp.now().strftime("%d-%m-%y_%H:%M")
        new_saved_model = os.path.join(save_directory, f"{time}_saved_model.pth")
        torch.save(saved_model, new_saved_model)

        return MTGANModel(
            medrecord=preprocessed_medrecord, mtgan=self, generator=generator
        )

    def load_model(
        self,
        medrecord: MedRecord,
        saved_model_path: Path,
    ) -> MTGANModel:
        """Loads a MTGAN model from a trained one (only Generator).

        Args:
            medrecord (MedRecord): The medical record data.
            saved_model_path (Path): Path to the saved model.

        Returns:
            MTGANModel: The loaded MTGAN model.
        """
        preprocessed_medrecord = self.preprocess(medrecord)

        max_number_admissions = np.array(
            list(
                preprocessed_medrecord.node[
                    preprocessed_medrecord.nodes_in_group(
                        self.preprocessor.patients_group
                    ),
                    self.preprocessor.number_admissions_attribute,
                ].values()
            )
        ).max()
        number_codes = len(
            preprocessed_medrecord.nodes_in_group(self.preprocessor.concepts_group)
        )
        generator_inputs = GeneratorInputs(
            number_codes=number_codes,
            max_number_admissions=max_number_admissions,
            generator_hidden_dim=self._training_hyperparameters["generator_hidden_dim"],
            generator_attention_dim=self._training_hyperparameters.get(
                "generator_attention_dim"
            ),
            device=self.device,
        )

        saved_model = torch.load(saved_model_path, map_location=self.device)
        generator = Generator(*generator_inputs).to(self.device)
        generator.load_state_dict(saved_model["generator_state_dict"])

        return MTGANModel(
            medrecord=preprocessed_medrecord, mtgan=self, generator=generator
        )
