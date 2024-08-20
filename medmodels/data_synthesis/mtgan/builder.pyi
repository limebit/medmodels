from __future__ import annotations

from pathlib import Path
from typing import Dict

from typing_extensions import Unpack

from medmodels.data_synthesis.mtgan.modules.postprocessor import (
    AttributeType,
    PostprocessingHyperparameters,
)
from medmodels.data_synthesis.mtgan.modules.preprocessor import (
    PreprocessingHyperparameters,
)
from medmodels.data_synthesis.mtgan.mtgan import MTGAN
from medmodels.data_synthesis.mtgan.train.gan_trainer import TrainingHyperparameters
from medmodels.medrecord.types import MedRecordAttribute

class MTGANBuilder:
    seed: int

    training_hyperparameters: TrainingHyperparameters
    preprocessing_hyperparameters: PreprocessingHyperparameters
    postprocessing_hyperparameters: PostprocessingHyperparameters

    training_hyperparameters_json: TrainingHyperparameters
    preprocessing_hyperparameters_json: PreprocessingHyperparameters
    postprocessing_hyperparameters_json: PostprocessingHyperparameters

    def with_seed(self, seed: int) -> MTGANBuilder: ...
    def with_preprocessor_hyperparameters(
        self, **kwargs: Unpack[PreprocessingHyperparameters]
    ) -> MTGANBuilder: ...
    def with_training_hyperparameters(
        self, **kwargs: Unpack[TrainingHyperparameters]
    ) -> MTGANBuilder: ...
    def with_postprocessor_hyperparameters(
        self, **kwargs: Unpack[PostprocessingHyperparameters]
    ) -> MTGANBuilder: ...
    def with_postprocessor_attributes(
        self, attributes_types: Dict[MedRecordAttribute, AttributeType]
    ) -> MTGANBuilder: ...
    def load_hyperparameters_from(self, path: Path) -> MTGANBuilder: ...
    def build(self) -> MTGAN: ...
