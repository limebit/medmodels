from typing import List, Tuple, Union

import numpy as np
import torch

from medmodels.data_synthesis.mtgan.model.loaders import MTGANDataset
from medmodels.medrecord.querying import node


class CodeSampleIter:
    """Class for iterating over the patients who have a specific code."""

    def __init__(self, code: int, patients_indices: List[int], shuffle: bool = True):
        """Creates an iterable object that iterates over the patients included in
        samples who have the code.

        Example:
        >>> code_sample_iter = CodeSampleIter(1, [1, 2, 3])
        >>> next(code_sample_iter)
        1, 2, or 3 (shuffled)

        Args:
            code (int): Code ID.
            patients_indices (List[int]): List of indices of patients who have the code.
            shuffle (bool, optional): Whether to shuffle the list of patients. Defaults to True.
        """
        self.code = code
        self.patients_indices = patients_indices

        self.current_index = 0
        self.length = len(patients_indices)
        if shuffle:
            np.random.shuffle(self.patients_indices)

    def __next__(self) -> int:
        """Returns the next patient ID in the list.

        If the end of the list is reached, the list is shuffled and the first element
        is returned.

        Returns:
            int: Patient index.
        """
        sample = self.patients_indices[self.current_index]
        self.current_index += 1
        if self.current_index == self.length:
            self.current_index = 0
        return sample


class MTGANDataSampler:
    """Class for sampling batches of data from the EHR data."""

    def __init__(
        self,
        dataset: MTGANDataset,
        device: Union[torch.device, torch.cuda.device],
    ) -> None:
        """Creates a data sampler.

        Args:
            dataset (MTGANDataset): Dataset.
            device (Union[torch.device, torch.cuda.device]): Device.
        """
        self.dataset = dataset
        self.device = device

        self.size = len(dataset)
        self.code_samples = self._get_code_sample_map(dataset)

    def _get_code_sample_map(self, dataset: MTGANDataset) -> List[CodeSampleIter]:
        """Creates a map of codes to the respective patients who have the code.

        Example:
        >>> code_sample_map = self._get_code_sample_map()
        >>> code_sample_map[1]
        CodeSampleIter(1, [1, 2, 3])

        Args:
            dataset (MTGANDataset): Dataset.
        """
        medrecord = dataset.medrecord
        patients = sorted(medrecord.nodes_in_group(dataset.patients_group))
        code_samples = [CodeSampleIter(i, []) for i in range(dataset.number_codes)]

        for concept in sorted(medrecord.nodes_in_group(dataset.concepts_group)):
            index = dataset.concept_to_index_dict[concept]
            patient_ids = medrecord.select_nodes(
                node().has_neighbor_with(node().index() == concept, directed=False)
                & node().in_group(dataset.patients_group)
            )
            patient_indices = np.where(np.isin(patients, patient_ids))[0]
            code_samples[index] = CodeSampleIter(index, list(patient_indices))
        return code_samples

    def sample(self, target_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples a batch of data from the EHR data.

        Args:
            target_codes (torch.Tensor): Target codes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Real data and real number of admissions.
        """
        lines = np.array(
            [next(self.code_samples[code]) for code in target_codes], dtype=int
        )
        return self.dataset[lines]
