from pathlib import Path
from typing import Dict, Optional, Type

from torch import nn

from medmodels.data_synthesis.mtgan.builder import MTGANBuilder
from medmodels.data_synthesis.mtgan.model.gan import (
    TrainingHyperparameters,
    TrainingHyperparametersOptional,
)
from medmodels.data_synthesis.mtgan.modules.postprocessor import (
    AttributeType,
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
from medmodels.medrecord.types import Group, MedRecordAttribute

class MTGAN(Synthesizer):
    seed: int

    _patients_group: Group
    _concepts_group: Group
    _time_attribute: MedRecordAttribute
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
        patients_group: Group = "patients",
        concepts_group: Group = "concepts",
        time_attribute: MedRecordAttribute = "time",
        preprocessing_hyperparameters: PreprocessingHyperparametersOptional = {},
        training_hyperparameters: TrainingHyperparametersOptional = {},
        postprocessing_hyperparameters: PostprocessingHyperparametersOptional = {},
        attributes_types: Dict[MedRecordAttribute, AttributeType] = {},
        seed: int = 0,
    ) -> None: ...
    def fit(
        self,
        medrecord: MedRecord,
        checkpoint_directory: Optional[Path] = None,
    ) -> MTGANModel: ...
    def fit_from(
        self,
        medrecord: MedRecord,
        checkpoint_gru_path: Path,
        checkpoint_model_path: Optional[Path] = None,
        checkpoint_directory: Optional[Path] = None,
    ) -> MTGANModel: ...
    def load_model(
        self,
        medrecord: MedRecord,
        checkpoint_model_path: Path,
    ) -> MTGANModel: ...
