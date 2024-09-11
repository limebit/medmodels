from pathlib import Path
from typing import Dict, Optional, Type

from torch import nn

from medmodels.data_synthesis.mtgan.builder import MTGANBuilder
from medmodels.data_synthesis.mtgan.model.gan import (
    TrainingHyperparameters,
    TrainingHyperparametersOptional,
)
from medmodels.data_synthesis.mtgan.modules.postprocessor import (
    PostprocessingHyperparameters,
    PostprocessingHyperparametersOptional,
)
from medmodels.data_synthesis.mtgan.modules.preprocessor import (
    PreprocessingHyperparameters,
    PreprocessingHyperparametersOptional,
)
from medmodels.data_synthesis.mtgan.mtgan_model import MTGANModel
from medmodels.data_synthesis.synthesizer import Synthesizer
from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.schema import AttributeType
from medmodels.medrecord.types import Group, MedRecordAttribute

class MTGAN(Synthesizer):
    seed: int

    _preprocessing_hyperparameters: PreprocessingHyperparameters
    _postprocessing_hyperparameters: PostprocessingHyperparameters
    _training_hyperparameters: TrainingHyperparameters

    def __init__(
        self,
        preprocessor: Type[nn.Module],
        postprocessor: Type[nn.Module],
    ) -> None: ...
    @classmethod
    def builder(cls) -> MTGANBuilder: ...
    @staticmethod
    def _set_configuration(
        mtgan: MTGAN,
        *,
        preprocessor: Type[nn.Module],
        postprocessor: Type[nn.Module],
        preprocessing_hyperparameters: PreprocessingHyperparametersOptional = {},
        training_hyperparameters: TrainingHyperparametersOptional = {},
        postprocessing_hyperparameters: PostprocessingHyperparametersOptional = {},
        seed: int = 0,
    ) -> None: ...
    def fit(
        self,
        medrecord: MedRecord,
        attribute_types_patients: Dict[MedRecordAttribute, AttributeType] = {},
        attribute_types_concepts: Dict[MedRecordAttribute, AttributeType] = {},
        patients_group: Group = "patients",
        concepts_group: Group = "concepts",
        time_attribute: MedRecordAttribute = "time",
        checkpoint_directory: Optional[Path] = None,
    ) -> MTGANModel: ...
    def fit_from(
        self,
        medrecord: MedRecord,
        gru_path: Path,
        model_path: Optional[Path] = None,
        attribute_types_patients: Dict[MedRecordAttribute, AttributeType] = {},
        attribute_types_concepts: Dict[MedRecordAttribute, AttributeType] = {},
        patients_group: Group = "patients",
        concepts_group: Group = "concepts",
        time_attribute: MedRecordAttribute = "time",
        checkpoint_directory: Optional[Path] = None,
    ) -> MTGANModel: ...
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
