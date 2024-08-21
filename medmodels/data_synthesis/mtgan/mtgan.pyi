from pathlib import Path
from typing import Dict, Optional

from medmodels.data_synthesis.mtgan.builder import MTGANBuilder
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator
from medmodels.data_synthesis.mtgan.model.real_gru.real_gru import RealGRU
from medmodels.data_synthesis.mtgan.modules.postprocessor import (
    AttributeType,
    MTGANPostprocessor,
    PostprocessingHyperparameters,
    PostprocessingHyperparametersTotal,
)
from medmodels.data_synthesis.mtgan.modules.preprocessor import (
    MTGANPreprocessor,
    PreprocessingHyperparameters,
    PreprocessingHyperparametersTotal,
)
from medmodels.data_synthesis.mtgan.mtgan_model import MTGANModel
from medmodels.data_synthesis.mtgan.train.gan_trainer import (
    TrainingHyperparameters,
    TrainingHyperparametersTotal,
)
from medmodels.data_synthesis.synthesizer import Synthesizer
from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import MedRecordAttribute

class MTGAN(Synthesizer):
    generator: Optional[Generator]

    _preprocessing_hyperparameters: PreprocessingHyperparametersTotal
    _postprocessing_hyperparameters: PostprocessingHyperparametersTotal
    _training_hyperparameters: TrainingHyperparametersTotal

    def __init__(
        self,
    ) -> None: ...
    def _initialize_parent_class(
        self, preprocessor: MTGANPreprocessor, postprocessor: MTGANPostprocessor
    ) -> None: ...
    def _initalize_seed(self, seed: int) -> None: ...
    @classmethod
    def builder(cls) -> MTGANBuilder: ...
    @staticmethod
    def _set_configuration(
        mtgan: MTGAN,
        *,
        preprocessor: MTGANPreprocessor,
        postprocessor: MTGANPostprocessor,
        preprocessing_hyperparameters: PreprocessingHyperparameters = {},
        training_hyperparameters: TrainingHyperparameters = {},
        postprocessing_hyperparameters: PostprocessingHyperparameters = {},
        preprocessing_hyperparameters_json: PreprocessingHyperparameters = {},
        training_hyperparameters_json: TrainingHyperparameters = {},
        postprocessing_hyperparameters_json: PostprocessingHyperparameters = {},
        attributes_types: Dict[MedRecordAttribute, AttributeType] = {},
        seed: int = 0,
    ) -> None: ...
    def _setup_real_gru(
        self, medrecord: MedRecord, saved_gru_path: Optional[Path] = None
    ) -> RealGRU: ...
    def _train_real_gru(self, real_gru: RealGRU, medrecord: MedRecord) -> RealGRU: ...
    def fit(
        self,
        medrecord: MedRecord,
        save_directory: Optional[Path] = None,
    ) -> MTGANModel: ...
    def fit_from(
        self,
        medrecord: MedRecord,
        saved_gru_path: Path,
        saved_model_path: Optional[Path] = None,
        save_directory: Optional[Path] = None,
    ) -> MTGANModel: ...
    def load_model(
        self,
        medrecord: MedRecord,
        saved_model_path: Path,
    ) -> MTGANModel: ...
