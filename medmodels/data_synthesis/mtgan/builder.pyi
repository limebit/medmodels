from pathlib import Path
from typing import Dict

from typing_extensions import Unpack

from medmodels.data_synthesis.mtgan.model.gan import TrainingHyperparametersOptional
from medmodels.data_synthesis.mtgan.modules.postprocessor import (
    AttributeType,
    PostprocessingHyperparametersOptional,
)
from medmodels.data_synthesis.mtgan.modules.preprocessor import (
    PreprocessingHyperparametersOptional,
)
from medmodels.data_synthesis.mtgan.mtgan import MTGAN
from medmodels.data_synthesis.mtgan.mtgan_model import MTGANModel
from medmodels.medrecord.types import Group, MedRecordAttribute


class MTGANBuilder:
    seed: int

    patients_group: Group
    concepts_group: Group
    time_attribute: MedRecordAttribute

    training_hyperparameters: TrainingHyperparametersOptional
    preprocessing_hyperparameters: PreprocessingHyperparametersOptional
    postprocessing_hyperparameters: PostprocessingHyperparametersOptional

    def with_seed(self, seed: int) -> MTGANBuilder: ...
    def with_patients_group(self, patients_group: Group) -> MTGANBuilder: ...
    def with_concepts_group(self, concepts_group: Group) -> MTGANBuilder: ...
    def with_time_attribute(
        self, time_attribute: MedRecordAttribute
    ) -> MTGANBuilder: ...
    def with_preprocessor_hyperparameters(
        self, **kwargs: Unpack[PreprocessingHyperparametersOptional]
    ) -> MTGANBuilder: ...
    def with_training_hyperparameters(
        self, **kwargs: Unpack[TrainingHyperparametersOptional]
    ) -> MTGANBuilder: ...
    def with_postprocessor_hyperparameters(
        self, **kwargs: Unpack[PostprocessingHyperparametersOptional]
    ) -> MTGANBuilder: ...
    def with_postprocessor_attributes(
        self, attributes_types: Dict[MedRecordAttribute, AttributeType]
    ) -> MTGANBuilder: ...
    def load_hyperparameters_from(self, path: Path) -> MTGANBuilder: ...
    def load_model(self, path: Path) -> MTGANModel: ...
    def build(self) -> MTGAN: ...
