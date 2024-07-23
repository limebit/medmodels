from typing import Any, Dict, List, Literal, Tuple, Union

import numpy as np
import sparse
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

from medmodels.data_synthesis.mtgan.modules.preprocessor import MTGANPreprocessor
from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.querying import edge, node
from medmodels.medrecord.types import MedRecordAttribute, MedRecordValue


class MTGANDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        preprocessing_object: MTGANPreprocessor,
        medrecord: MedRecord,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Creates a dataset for the MTGAN model, so that it can be used in the DataLoader.

        Args:
            preprocessing_object (MTGANPreprocessor): The preprocessing object.
            medrecord (MedRecord): The MedRecord object.
            device (torch.device, optional): The device to run the model. Defaults to torch.device("cpu").

        Raises:
            ValueError: If the patient index is out of bounds.
        """
        self.medrecord = medrecord
        self.device = device

        self.patients_group = preprocessing_object.patients_group
        self.concepts_group = preprocessing_object.concepts_group

        self.time_window_attribute = preprocessing_object.time_window_attribute
        self.absolute_time_window_attribute = (
            preprocessing_object.absolute_time_window_attribute
        )
        self.concept_index_attribute = preprocessing_object.concept_index_attribute
        self.concept_edge_attribute = preprocessing_object.concept_edge_attribute
        self.number_admissions_attribute = (
            preprocessing_object.number_admissions_attribute
        )

        self.concept_to_index_dict: Dict[MedRecordValue, int] = {
            concept: index
            for index, concept in preprocessing_object.index_to_concept_dict.items()
        }
        self.number_codes = len(self.medrecord.nodes_in_group(self.concepts_group))
        self.max_number_admissions = np.array(
            list(
                self.medrecord.node[
                    self.medrecord.nodes_in_group(self.patients_group),
                    self.number_admissions_attribute,
                ].values()
            )
        ).max()

    def __len__(self) -> int:
        return len(self.medrecord.nodes_in_group(self.patients_group))

    def _convert_idx(
        self, idx: Union[List[int], int, torch.Tensor, NDArray[np.int16], np.int64]
    ) -> List[int]:
        """
        Ensures that the index is in a list of ints, suitable for indexing arrays.

        Args:
            idx (Union[List[int], int, torch.Tensor, NDArray[np.int16], np.int64]):
                Input index.

        Returns:
            List[int]: A list of ints.
        """
        # Normalize idx to always be a list of int for easier processing
        if isinstance(idx, int):
            return [idx]
        elif isinstance(idx, np.integer):
            return [int(idx)]
        elif isinstance(idx, torch.Tensor):
            return idx.tolist()
        elif isinstance(idx, np.ndarray):
            if idx.ndim == 0:  # it's a scalar array
                return [int(idx.item())]  # Convert to Python int
            return [int(item) for item in idx]
        else:
            return idx

    def __getitem__(
        self, idx: Union[List[int], int, torch.Tensor, NDArray[np.int16], np.int64]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the data for the given patient index.

        Args:
            idx (Union[List[int], int, torch.Tensor, NDArray[np.int16]]): The index of the patient.

        Raises:
            ValueError: If the patient index is out of bounds.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The data for the given patient indices
                and the number of admissions.
        """
        idx = self._convert_idx(idx)

        coords = []
        patient_indices = [
            sorted(self.medrecord.nodes_in_group(self.patients_group))[i] for i in idx
        ]
        if max(idx) + 1 > len(self.medrecord.nodes_in_group(self.patients_group)):
            raise ValueError("Patient index out of bounds")

        for array_index, patient_index in enumerate(idx):
            coords.append(self._transform_into_coords(patient_index, array_index))

        matrix = sparse.COO(
            np.vstack(coords).T,
            True,
            shape=(
                len(idx),
                self.max_number_admissions,
                self.number_codes,
            ),
        ).todense()

        number_admissions = [
            self.medrecord.node[patient, self.number_admissions_attribute]
            for patient in patient_indices
        ]

        return (
            torch.tensor(matrix, dtype=torch.float32, device=self.device),
            torch.tensor(number_admissions, dtype=torch.int16, device=self.device),
        )

    def _transform_into_coords(
        self, patient_idx: int, array_idx: int
    ) -> NDArray[np.int16]:
        """Transform the data for the given patient index into coordinates space.

        Example:
        Patient 0 has the following data:
            In time window 0, the patient has concept 0.
            In time window 0, the patient has concept 1.
            In time window 1, the patient has concept 2.
        >>> self._transform_into_coords(0, 0)
        array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 2]])
        - First column: Array index.
        - Second column: Time window.
        - Third column: Concept index.

        If Array index is 25:
        >>> self._transform_into_coords(0, 25)
        array([[25, 0, 0],
               [25, 0, 1],
               [25, 1, 2]])

        Args:
            patient_idx (int): The index of the patient within the MedRecord.
            array_idx (int): The index of the patient within the sparse matrix.

        Returns:
            NDArray[np.int16]: The data for the given patient index in coordinates space.
        """
        edges = self.medrecord.edge[
            self.medrecord.edges_connecting(
                sorted(self.medrecord.nodes_in_group(self.patients_group))[patient_idx],
                sorted(self.medrecord.nodes_in_group(self.concepts_group)),
                directed=False,
            ),
            [self.time_window_attribute, self.concept_edge_attribute],
        ]

        attributes = np.array(
            [
                [
                    attribute[self.time_window_attribute],
                    self.concept_to_index_dict[attribute[self.concept_edge_attribute]],
                ]
                for attribute in edges.values()
            ]
        )
        attributes = np.unique(attributes, axis=0)
        patient_indices = np.full((attributes.shape[0], 1), array_idx)

        return np.hstack((patient_indices, attributes))

    def get_attributes(
        self,
        idx: Union[List[int], int, torch.Tensor, NDArray[np.int16]],
        attribute: MedRecordAttribute,
    ) -> Tuple[NDArray[np.bool], NDArray[Any]]:
        """Get the data for the given patient index and attribute.

        Args:
            idx (Union[List[int], int, torch.Tensor, NDArray[np.int16]]): The index of the patient.
            attribute (MedRecordAttribute): The attribute to get.

        Raises:
            ValueError: If the patient index is out of bounds.

        Returns:
            Tuple[NDArray[np.bool], NDArray[Any]]: The data for the given patient indices and attribute.
        """
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        elif isinstance(idx, int):
            idx = [idx]

        coords = []
        patient_indices = [
            sorted(self.medrecord.nodes_in_group(self.patients_group))[i] for i in idx
        ]
        if max(idx) + 1 > len(self.medrecord.nodes_in_group(self.patients_group)):
            raise ValueError("Patient index out of bounds")

        for array_index, patient_index in enumerate(idx):
            coords.append(self._transform_into_coords(patient_index, array_index))

        matrix = (
            sparse.COO(
                np.vstack(coords).T,
                True,
                shape=(
                    len(idx),
                    self.max_number_admissions,
                    self.number_codes,
                ),
            )
            .todense()
            .astype(bool)
        )

        attributes = np.array(
            [self.medrecord.node[patient, attribute] for patient in patient_indices]
        )

        return (matrix, attributes)


class MTGANDatasetPrediction(MTGANDataset):
    def __init__(
        self,
        preprocessor: MTGANPreprocessor,
        medrecord: MedRecord,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Generates the dataset for the prediction task in the GRU."""
        super().__init__(preprocessor, medrecord, device)

    def __getitem__(
        self, idx: Union[List[int], int, torch.Tensor, NDArray[np.int16], np.int64]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the data for the given patient index for the prediction task in the GRU.

        It gives the data of the patient without the last time window for the input and
        the data of the patient without the first time window for the output (predicting
        what the next time window will be).

        Args:
            idx (Union[List[int], int, torch.Tensor, NDArray[np.int16], np.int64]):
                The index of the patient.

        Raises:
            ValueError: If the patient index is out of bounds.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The data for the given patient indices,
                the data for the given patient indices for prediction, and the number of admissions.
        """
        idx = self._convert_idx(idx)

        predict_x_coords = []
        predict_y_coords = []
        patient_indices = [
            sorted(self.medrecord.nodes_in_group(self.patients_group))[i] for i in idx
        ]
        if max(idx) + 1 > len(self.medrecord.nodes_in_group(self.patients_group)):
            raise ValueError("Patient index out of bounds")

        for array_index, patient_index in enumerate(idx):
            predict_x_coords.append(
                self._transform_into_coords(
                    patient_index, array_index, remove_window="last"
                )
            )
            predict_y_coords.append(
                self._transform_into_coords(
                    patient_index, array_index, remove_window="first"
                )
            )

        predict_x = sparse.COO(
            np.vstack(predict_x_coords).T,
            True,
            shape=(
                len(idx),
                self.max_number_admissions - 1,
                self.number_codes,
            ),
        ).todense()

        predict_y = sparse.COO(
            np.vstack(predict_y_coords).T,
            True,
            shape=(
                len(idx),
                self.max_number_admissions - 1,
                self.number_codes,
            ),
        ).todense()

        number_admissions = (
            np.array(
                [
                    self.medrecord.node[patient, self.number_admissions_attribute]
                    for patient in patient_indices
                ]
            )
            - 1
        )

        return (
            torch.tensor(predict_x, dtype=torch.float32, device=self.device),
            torch.tensor(predict_y, dtype=torch.float32, device=self.device),
            torch.tensor(number_admissions, dtype=torch.int16, device=self.device),
        )

    def _transform_into_coords(
        self, patient_idx: int, array_idx: int, remove_window: Literal["first", "last"]
    ) -> NDArray[np.int16]:
        """Transform the data for the given patient index into coordinates space.

        Takes into account the removal of the first or last time window. It also takes the
        array index to keep track of where in the sparse matrix the data should be placed.

        Example:
        Patient 0 has the following data:
            In time window 0, the patient has concept 0.
            In time window 0, the patient has concept 1.
            In time window 1, the patient has concept 2.
        >>> self._transform_into_coords(0, 0, remove_window="first")
        array([[0, 0, 2]])
        - First column: Array index.
        - Second column: Time window.
        - Third column: Concept index.
        The first time window is removed.

        If Array index is 25 and the last time window is removed:
        >>> self._transform_into_coords(0, 25, remove_window="last")
        array([[25, 0, 0],
               [25, 0, 1]])

        Args:
            patient_idx (int): The index of the patient within the MedRecord.
            array_idx (int): The index of the patient within the sparse matrix.
            remove_window (Literal["first", "last"]): Whether to remove the first or last time
                window.

        Returns:
            NDArray[np.int16]: The data for the given patient index in coordinates space.
        """
        edges = self.medrecord.edge[
            self.medrecord.edges_connecting(
                sorted(self.medrecord.nodes_in_group(self.patients_group))[patient_idx],
                sorted(self.medrecord.nodes_in_group(self.concepts_group)),
                directed=False,
            ),
            [self.time_window_attribute, self.concept_edge_attribute],
        ]

        attributes = np.array(
            [
                [
                    attribute[self.time_window_attribute],
                    self.concept_to_index_dict[attribute[self.concept_edge_attribute]],
                ]
                for attribute in edges.values()
            ]
        )
        attributes = np.unique(attributes, axis=0)

        if remove_window == "first":
            # Remove the first time window
            attributes = attributes[attributes[:, 0] != 0]
            attributes[:, 0] -= 1
        elif remove_window == "last":
            # Remove the last time window
            max_time_window = attributes[:, 0].max()
            attributes = attributes[attributes[:, 0] != max_time_window]

        patient_indices = np.full((attributes.shape[0], 1), array_idx)
        return np.hstack((patient_indices, attributes))


class MTGANDatasetTime(MTGANDataset):
    def __init__(
        self,
        preprocessor: MTGANPreprocessor,
        medrecord: MedRecord,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Generates the dataset for predicting the absolute time windows in the postprocessing task.

        Args:
            preprocessor (MTGANPreprocessor): The preprocessor object.
            medrecord (MedRecord): The MedRecord object.
            device (torch.device, optional): The device to run the model. Defaults to torch.device("cpu").
        """
        super().__init__(preprocessor, medrecord, device)

    def __getitem__(
        self, idx: Union[List[int], int, torch.Tensor, NDArray[np.int16], np.int64]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the data for the given patient index for the prediction task in the GRU.

        It returns the data of the patient with the absolute time windows and the
        number of windows

        Args:
            idx (Union[List[int], int, torch.Tensor, NDArray[np.int16], np.int64]):
                The index of the patient.

        Raises:
            ValueError: If the patient index is out of bounds.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The data for the given
                patient indices, the number of admissions, and the absolute time
                windows.
        """
        idx = self._convert_idx(idx)

        coords = []
        absolute_time_windows = []
        patient_indices = [
            sorted(self.medrecord.nodes_in_group(self.patients_group))[i] for i in idx
        ]
        if max(idx) + 1 > len(self.medrecord.nodes_in_group(self.patients_group)):
            raise ValueError("Patient index out of bounds")

        for array_index, patient_index in enumerate(idx):
            coords_index, time_windows_index = self._transform_into_coords_with_time(
                patient_index, array_index
            )
            coords.append(coords_index)
            absolute_time_windows.append(time_windows_index)

        absolute_time_windows = np.concat(absolute_time_windows)
        matrix = sparse.COO(
            np.vstack(coords).T,
            True,
            shape=(
                len(idx),
                self.max_number_admissions,
                self.number_codes,
            ),
        ).todense()

        number_admissions = [
            self.medrecord.node[patient, self.number_admissions_attribute]
            for patient in patient_indices
        ]

        return (
            torch.tensor(matrix, dtype=torch.float32, device=self.device),
            torch.tensor(number_admissions, dtype=torch.int16, device=self.device),
            torch.tensor(absolute_time_windows, dtype=torch.int16, device=self.device),
        )

    def _transform_into_coords_with_time(
        self, patient_idx: int, array_idx: int
    ) -> Tuple[NDArray[np.int16], NDArray[np.int16]]:
        """Transforms the data for the given patient index into coordinates space and returns the amount of absolute time windows between time windows.

        Example:
        Patient 0 has the following data:
            In time window 0, the patient has concept 0, absolute time window 0.
            In time window 0, the patient has concept 1, absolute time window 0.
            In time window 1, the patient has concept 2, absolute time window 10.
        >>> self._transform_into_coords(0, 0)
        (array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 2]]), array([0, 10]))
        - First column: Array index.
        - Second column: Time window.
        - Third column: Concept index.
        - Second array: Absolute time windows.

        If Array index is 25:
        >>> self._transform_into_coords(0, 25)
        array([[25, 0, 0],
               [25, 0, 1],
               [25, 1, 2]]), array([0, 10])

        Args:
            patient_idx (int): The index of the patient within the MedRecord.
            array_idx (int): The index of the patient within the sparse matrix.

        Returns:
            Tuple[NDArray[np.int16], NDArray[np.int16]]: The data for the given patient
                index in coordinates space and the absolute time windows.
        """
        edges = self.medrecord.edge[
            self.medrecord.edges_connecting(
                sorted(self.medrecord.nodes_in_group(self.patients_group))[patient_idx],
                sorted(self.medrecord.nodes_in_group(self.concepts_group)),
                directed=False,
            ),
            [
                self.time_window_attribute,
                self.concept_edge_attribute,
                self.absolute_time_window_attribute,
            ],
        ]

        attributes = np.array(
            [
                [
                    attribute[self.time_window_attribute],
                    self.concept_to_index_dict[attribute[self.concept_edge_attribute]],
                    attribute[self.absolute_time_window_attribute],
                ]
                for attribute in edges.values()
            ]
        )
        # Extract the last column from attrtibutes
        absolute_time_window = attributes[:, 2]
        attributes = np.unique(attributes[:, :2], axis=0)
        patient_indices = np.full((attributes.shape[0], 1), array_idx)

        return np.hstack((patient_indices, attributes)), np.sort(
            np.unique(absolute_time_window)
        ).astype(np.int16)


class MTGANDatasetAttribute(MTGANDataset):
    def __init__(
        self,
        preprocessor: MTGANPreprocessor,
        medrecord: MedRecord,
        attribute: MedRecordAttribute,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Generates the dataset for predicting attributes in the postprocessing.

        Args:
            preprocessor (MTGANPreprocessor): The preprocessor object.
            medrecord (MedRecord): The MedRecord object.
            attribute (MedRecordAttribute): The attribute to predict.
            device (torch.device, optional): The device to run the model. Defaults to torch.device("cpu").
        """
        super().__init__(preprocessor, medrecord, device)
        self.attribute = attribute

        edges_with_attribute = medrecord.select_edges(
            edge().has_attribute(attribute)
            & edge().connected_with(node().in_group(self.concepts_group))
        )
        self.concepts_with_attribute = list(
            set(
                medrecord.edge[
                    edges_with_attribute, self.concept_edge_attribute
                ].values()
            )
        )
        self.concepts_with_attribute = sorted(
            self.concepts_with_attribute, key=lambda x: self.concept_to_index_dict[x]
        )

    def __getitem__(
        self, idx: Union[List[int], int, torch.Tensor, NDArray[np.int16], np.int64]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the data for the given patient index for the prediction task in the GRU.

        It returns the data of the patient with the absolute time windows and the
        number of windows

        Args:
            idx (Union[List[int], int, torch.Tensor, NDArray[np.int16], np.int64]):
                The index of the patient.

        Raises:
            ValueError: If the patient index is out of bounds.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The data for the given
                patient indices, the number of admissions, and the sparse matrix with
                the attributes.
        """
        idx = self._convert_idx(idx)

        coords = []
        total_attributes = []
        total_coords_attributes = []
        patient_indices = [
            sorted(self.medrecord.nodes_in_group(self.patients_group))[i] for i in idx
        ]
        if max(idx) + 1 > len(self.medrecord.nodes_in_group(self.patients_group)):
            raise ValueError("Patient index out of bounds")

        for array_index, patient_index in enumerate(idx):
            coords_index = self._transform_into_coords(patient_index, array_index)
            coords_attributes_index, attributes_index = (
                self._transform_into_coords_with_attribute(
                    patient_index,
                    array_index,
                )
            )

            coords.append(coords_index)
            total_coords_attributes.append(coords_attributes_index)
            total_attributes.append(attributes_index)

        total_attributes = np.concatenate(total_attributes)
        matrix = sparse.COO(
            np.vstack(coords).T,
            True,
            shape=(
                len(idx),
                self.max_number_admissions,
                self.number_codes,
            ),
        ).todense()

        matrix_attributes = sparse.COO(
            np.vstack(total_coords_attributes).T,
            total_attributes,
            shape=(
                len(idx),
                self.max_number_admissions,
                self.number_codes,
            ),
            fill_value=np.nan,
        ).todense()

        number_admissions = [
            self.medrecord.node[patient, self.number_admissions_attribute]
            for patient in patient_indices
        ]

        return (
            torch.tensor(matrix, dtype=torch.float32, device=self.device),
            torch.tensor(number_admissions, dtype=torch.int16, device=self.device),
            torch.tensor(matrix_attributes, device=self.device),
        )

    def _transform_into_coords_with_attribute(
        self,
        patient_idx: int,
        array_idx: int,
    ) -> Tuple[NDArray[np.int16], NDArray[Any]]:
        """Transforms the data for the given patient index into coordinates space and returns the attribute for the concepts that have it.

        It gives back only the coordinates of the concepts that have the attribute and the attribute itself.

        Example:
        Concepts in self.concepts_with_attribute: [0, 1]
        Patient 0 has the following data:
            In time window 0, the patient has concept 0, absolute time window 0. Attribute: 20
            In time window 0, the patient has concept 1, absolute time window 0. Attribute: 25
            In time window 1, the patient has concept 2, absolute time window 10. Attribute: 30
        >>> self._transform_into_coords(0, 0)
        (array([[0, 0, 0],
               [0, 0, 1]]), array([20, 25, 30]))
        - First column: Array index.
        - Second column: Time window.
        - Third column: Concept index.
        - Second array: Absolute time windows.

        If Array index is 25:
        >>> self._transform_into_coords(0, 25)
        array([[25, 0, 0],
               [25, 0, 1],
               [25, 1, 2]]), array([20, 25, 30])

        Args:
            patient_idx (int): The index of the patient within the MedRecord.
            array_idx (int): The index of the patient within the sparse matrix.

        Returns:
            Tuple[NDArray[np.int16], NDArray[np.int16]]: The data for the given patient
                index in coordinates space and the absolute time windows.
        """
        edges = self.medrecord.edge[
            self.medrecord.edges_connecting(
                sorted(self.medrecord.nodes_in_group(self.patients_group))[patient_idx],
                self.concepts_with_attribute,  # pyright: ignore
                directed=False,
            ),
            [self.time_window_attribute, self.concept_edge_attribute, self.attribute],
        ]

        concept_to_position = {
            concept: index for index, concept in enumerate(self.concepts_with_attribute)
        }
        attributes = np.array(
            [
                [
                    attribute[self.time_window_attribute],
                    concept_to_position[attribute[self.concept_edge_attribute]],
                    attribute[self.attribute],
                ]
                for attribute in edges.values()
            ]
        )
        # Extract the last column from attrtibutes
        concept_attributes = attributes[:, 2]
        attributes = np.unique(attributes[:, :2], axis=0)
        patient_indices = np.full((attributes.shape[0], 1), array_idx)

        return np.hstack((patient_indices, attributes)), concept_attributes


class MTGANDataLoader(DataLoader[torch.Tensor]):
    """MTGAN DataLoader for the MTGAN model."""

    def __init__(
        self, dataset: MTGANDataset, batch_size: int = 32, shuffle: bool = True
    ) -> None:
        """Creates an iterable object that iterates over the data.

        Args:
            dataset (MTGANDataset): The dataset.
            batch_size (int, optional): The batch size. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.size = len(dataset)  # num of patients
        self.patient_indices = np.arange(self.size)  # patient IDs
        self.n_batches = np.ceil(self.size / batch_size).astype(int)

        self.counter = 0
        if shuffle:
            np.random.shuffle(self.patient_indices)

    def _get_item(self, index: int) -> torch.Tensor:
        """Returns the data of the batch, starting from the index * batch_size.

        Args:
            index (int): The index of the batch.

        Raises:
            ValueError: If the batch size is None.

        Returns:
            torch.Tensor: The data of the batch.
        """
        if self.batch_size is None:
            raise ValueError("Batch size is None")
        start = index * self.batch_size
        end = start + self.batch_size
        slice_of_indices = self.patient_indices[start:end]
        return self.dataset[slice_of_indices]

    def __next__(self) -> torch.Tensor:
        if self.counter >= self.n_batches:
            self.counter = 0
            raise StopIteration
        data = self._get_item(self.counter)
        self.counter += 1
        return data

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        return self
