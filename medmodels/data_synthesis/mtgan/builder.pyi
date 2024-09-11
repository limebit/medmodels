from pathlib import Path
from typing import Dict

from typing_extensions import Unpack

from medmodels.data_synthesis.mtgan.model.gan import TrainingHyperparametersOptional
from medmodels.data_synthesis.mtgan.modules.postprocessor import (
    PostprocessingHyperparametersOptional,
)
from medmodels.data_synthesis.mtgan.modules.preprocessor import (
    PreprocessingHyperparametersOptional,
)
from medmodels.data_synthesis.mtgan.mtgan import MTGAN
from medmodels.data_synthesis.mtgan.mtgan_model import MTGANModel
from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.schema import AttributeType
from medmodels.medrecord.types import Group, MedRecordAttribute

class MTGANBuilder:
    seed: int

    training_hyperparameters: TrainingHyperparametersOptional
    preprocessing_hyperparameters: PreprocessingHyperparametersOptional
    postprocessing_hyperparameters: PostprocessingHyperparametersOptional

    def with_seed(self, seed: int) -> MTGANBuilder: ...
    def with_preprocessor_hyperparameters(
        self, **kwargs: Unpack[PreprocessingHyperparametersOptional]
    ) -> MTGANBuilder: ...
    def with_training_hyperparameters(
        self, **kwargs: Unpack[TrainingHyperparametersOptional]
    ) -> MTGANBuilder: ...
    def with_postprocessor_hyperparameters(
        self, **kwargs: Unpack[PostprocessingHyperparametersOptional]
    ) -> MTGANBuilder: ...
    def load_hyperparameters_from(self, path: Path) -> MTGANBuilder: ...
    def load_model(
        self,
        medrecord: MedRecord,
        model_path: Path,
        attribute_types_patients: Dict[MedRecordAttribute, AttributeType] = {},
        attribute_types_concepts: Dict[MedRecordAttribute, AttributeType] = {},
        patients_group: Group = "patients",
        concepts_group: Group = "concepts",
        time_attribute: MedRecordAttribute = "time",
    ) -> MTGANModel: ...
    def build(self) -> MTGAN: ...
