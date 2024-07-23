from __future__ import annotations

import numpy as np
import sparse
import torch
from typing_extensions import TYPE_CHECKING

from medmodels import MedRecord
from medmodels.data_synthesis.mtgan.model.generator.generator import Generator
from medmodels.data_synthesis.mtgan.modules.preprocessor import MTGANPreprocessor
from medmodels.data_synthesis.synthesizer_model import SynthesizerModel

if TYPE_CHECKING:
    from medmodels.data_synthesis.mtgan.mtgan import MTGAN


class MTGANModel(SynthesizerModel):
    def __init__(
        self, medrecord: MedRecord, mtgan: MTGAN, generator: Generator
    ) -> None:
        """Initializes the MTGANModel with the required components.

        Args:
            medrecord (MedRecord): MedRecord object.
            mtgan (MTGAN): MTGAN object.
            generator (Generator): Generator object.
        """
        super(MTGANModel, self).__init__(synthesizer=mtgan)
        self.medrecord = medrecord
        self.preprocessor = mtgan.preprocessor
        self.postprocessor = mtgan.postprocessor
        self.generator = generator

        self.postprocessor.real_medrecord = medrecord
        self.postprocessor.preprocessor = self.preprocessor

        self.input_dimension = mtgan._training_hyperparameters["generator_hidden_dim"]
        self.number_samples = mtgan.postprocessor.hyperparameters[
            "number_patients_generated"
        ]
        self.batch_size = mtgan._training_hyperparameters.get("batch_size")

    def _find_admissions_distribution(
        self, medrecord: MedRecord, preprocessor: MTGANPreprocessor
    ) -> torch.Tensor:
        """Finds the distribution of the number of admissions per patient in the dataset.

        Args:
            medrecord (MedRecord): The medical record data.
            preprocessor (MTGANPreprocessor): Preprocessor object.

        Returns:
            torch.Tensor: The distribution of the number of admissions per patient.
        """
        number_admissions_per_patient = np.array(
            list(
                medrecord.node[
                    sorted(medrecord.nodes_in_group(preprocessor.patients_group)),
                    preprocessor.number_admissions_attribute,
                ].values()
            )
        )
        num_admissions_distribution = np.bincount(number_admissions_per_patient) / (
            len(number_admissions_per_patient)
        )
        return torch.from_numpy(num_admissions_distribution).to(self.device)

    def forward(
        self,
        noise: torch.Tensor,
    ) -> sparse.COO:
        """Synthesizes data with MTGAN.

        Args:
            noise (torch.Tensor): Noise to be used for generating synthetic data.

        Returns:
            sparse.COO: Synthetic data. It has the shape (number_samples,
                max_number_windows, number_codes) and is a boolean matrix that
                indicates the presence of a code in a window.
        """
        number_admissions_distribution = self._find_admissions_distribution(
            self.medrecord, self.preprocessor
        )
        return self.generator.generate_(
            number_patients=noise.shape[0],
            windows_distribution=number_admissions_distribution,
            batch_size=self.batch_size,
            noise=noise,
        )
