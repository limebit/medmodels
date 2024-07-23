from datetime import datetime, timedelta
from typing import Any, Dict, List, Literal, TypedDict, Union

import numpy as np
import polars as pl
import sparse
import torch
import xgboost as xgb
from sklearn.model_selection import train_test_split
from typing_extensions import TypeAlias

from medmodels.data_synthesis.mtgan.model.loaders import (
    MTGANDataLoader,
    MTGANDataset,
    MTGANDatasetAttribute,
    MTGANDatasetTime,
)
from medmodels.data_synthesis.mtgan.modules.classifiers import (
    AttributeClassifier,
    AttributeRegression,
    TimeLSTM,
)
from medmodels.data_synthesis.mtgan.modules.preprocessor import MTGANPreprocessor
from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import Group, MedRecordAttribute

AttributeType: TypeAlias = Literal["categorical", "regression", "temporal"]


class PostprocessingHyperparameters(TypedDict, total=True):
    number_patients_generated: int
    training_epochs: int
    hidden_dim: int
    learning_rate: float
    number_layers: int
    batch_size: int
    number_previous_admissions: int
    top_k_codes: int


class PostprocessingHyperparametersOptional(TypedDict, total=False):
    number_patients_generated: int
    training_epochs: int
    hidden_dim: int
    learning_rate: float
    number_layers: int
    batch_size: int
    number_previous_admissions: int
    top_k_codes: int


class MTGANPostprocessor(torch.nn.Module):
    """Postprocessing class for the MTGAN model."""

    attributes_types: Dict[MedRecordAttribute, AttributeType]
    attributes_concepts_types: Dict[MedRecordAttribute, AttributeType]
    hyperparameters: PostprocessingHyperparameters

    def __init__(
        self,
    ) -> None:
        """Constructor for the Postprocessing class."""
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TODO: change the attributes_postprocessing so it is inferred from the schema.
        self.attributes_types: Dict[MedRecordAttribute, AttributeType] = {
            "first_admission": "temporal"
        }
        self.attributes_concepts_types: Dict[MedRecordAttribute, AttributeType] = {}

        self.hyperparameters: PostprocessingHyperparameters = {
            "number_patients_generated": 1000,
            "training_epochs": 100,
            "hidden_dim": 64,
            "number_layers": 2,
            "batch_size": 32,
            "learning_rate": 1e-3,
            "number_previous_admissions": 5,
            "top_k_codes": 500,
        }
        self.preprocessor = MTGANPreprocessor()
        self.real_medrecord = MedRecord()
        self.to(self.device)

    def _load_processed_data(
        self,
        real_medrecord: MedRecord,
        preprocessor: MTGANPreprocessor,
    ) -> None:
        """Load the processed data from the preprocessing class.

        Args:
            real_medrecord (MedRecord): Real medical record.
            preprocessor (MTGANPreprocessor): Preprocessor class.
        """
        self.real_medrecord = real_medrecord
        self.preprocessor = preprocessor

    def postprocess(self, synthetic_data: sparse.COO) -> MedRecord:
        """Postprocess the synthetic data.

        Args:
            synthetic_data (sparse.COO): Synthetic data in sparse format.

        Returns:
            MedRecord: Postprocessed synthetic medical record.
        """
        # TODO: check for a better way to find the attributes.
        # attributes_patients = list(
        #     self.real_medrecord.node[
        #         self.real_medrecord.nodes_in_group(self.preprocessor.patients_group)[0]
        #     ].keys()
        # )
        synthetic_attributes_patients = {}

        for attribute, attribute_type in self.attributes_types.items():
            if attribute_type == "categorical":
                synthetic_attributes_patients.update(
                    self._patients_categorical_postprocessing(
                        synthetic_data,
                        group=self.preprocessor.patients_group,
                        attribute=attribute,
                    )
                )
            elif attribute_type == "regression":
                synthetic_attributes_patients.update(
                    self._patients_regression_postprocessing(
                        synthetic_data,
                        group=self.preprocessor.patients_group,
                        attribute=attribute,
                    )
                )
            elif attribute_type == "temporal":
                synthetic_attributes_patients.update(
                    self._patients_temporal_postprocessing(
                        synthetic_data,
                        group=self.preprocessor.patients_group,
                        attribute=attribute,
                    )
                )

        # TODO: some way of saving evaluation data on training of postprocessing.
        synthetic_attributes_concepts = {}
        synthetic_attributes_concepts.update(
            self.find_time(
                synthetic_data,
                synthetic_attributes_patients[
                    self.preprocessor.first_admission_attribute
                ],
            )
        )

        for attribute, attribute_type in self.attributes_concepts_types.items():
            if attribute_type == "regression":
                synthetic_attributes_concepts.update(
                    self._concepts_regression_postprocessing(
                        synthetic_data,
                        group=self.preprocessor.concepts_group,
                        attribute=attribute,
                    )
                )
            elif attribute_type == "categorical":
                synthetic_attributes_concepts.update(
                    self._concepts_categorical_postprocessing(
                        synthetic_data,
                        group=self.preprocessor.concepts_group,
                        attribute=attribute,
                    )
                )

        # attributes_concepts = list(
        #     real_medrecord.node[
        #         real_medrecord.nodes_in_group(self.preprocessor.concepts_group)[0]
        #     ].keys()
        # )

        return self.convert_to_medrecord(
            synthetic_data, synthetic_attributes_patients, synthetic_attributes_concepts
        )

    def _patients_categorical_postprocessing(
        self,
        synthetic_data: sparse.COO,
        group: Group,
        attribute: MedRecordAttribute,
    ) -> Dict[MedRecordAttribute, List[Any]]:
        """Postprocess a categorical attribute for the patients group.

        Args:
            synthetic_data (sparse.COO): Synthetic data in sparse format.
            group (Group): Group to postprocess.
            attribute (MedRecordAttribute): Attribute to postprocess.

        Returns:
            Dict[MedRecordAttribute, List[Any]]: Postprocessed attribute.

        Raises:
            ValueError: If the attribute has only one category, the number of samples
                is too small or the train and test data types do not match the real
                data type.
        """
        number_real_patients = len(self.real_medrecord.nodes_in_group(group))
        real_data, real_attributes = MTGANDataset(
            self.preprocessor, medrecord=self.real_medrecord, device=self.device
        ).get_attributes(idx=np.arange(number_real_patients), attribute=attribute)
        # Encode the attributes as integers, create a dictionary
        real_attributes_dictionary = {
            attribute: i for i, attribute in enumerate(np.unique(real_attributes))
        }
        real_attributes = np.array(
            [real_attributes_dictionary[attribute] for attribute in real_attributes]
        )
        if len(np.unique(real_attributes)) == 1:
            raise ValueError(f"{attribute} has only one category.")

        try:
            train_data, test_data, train_attributes, test_attributes = train_test_split(
                real_data,
                real_attributes,
                test_size=0.2,
                random_state=42,
                stratify=real_attributes,
            )
        except ValueError:
            try:
                train_data, test_data, train_attributes, test_attributes = (
                    train_test_split(
                        real_data,
                        real_attributes,
                        test_size=0.5,
                        random_state=42,
                        stratify=real_attributes,
                    )
                )
            except ValueError:
                raise ValueError(
                    f"The number of samples for {attribute} is too small to stratify."
                )
        if not (
            isinstance(train_data, type(real_data))
            and isinstance(test_data, type(real_data))
        ):
            raise ValueError(
                f"Train and test data types do not match the real data type for {attribute}."
            )
        train_set = xgb.DMatrix(
            train_data.sum(axis=1), label=train_attributes, enable_categorical=True
        )
        test_set = xgb.DMatrix(
            test_data.sum(axis=1), label=test_attributes, enable_categorical=True
        )
        model = xgb.train(
            {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "num_class": len(np.unique(train_attributes)),
                "verbosity": 0,
            },
            train_set,
            self.hyperparameters["training_epochs"],
            evals=[(test_set, "eval")],
            verbose_eval=False,
        )

        synthetic_attributes = np.argmax(
            model.predict(xgb.DMatrix(synthetic_data.todense().sum(axis=1))), axis=1
        )
        inverse_real_attributes_dictionary = {
            v: k for k, v in real_attributes_dictionary.items()
        }

        synthetic_attributes = np.array(
            [
                inverse_real_attributes_dictionary[attribute]
                for attribute in synthetic_attributes
            ]
        )

        return {attribute: synthetic_attributes.tolist()}

    def _patients_regression_postprocessing(
        self,
        synthetic_data: sparse.COO,
        group: Group,
        attribute: MedRecordAttribute,
    ) -> Dict[MedRecordAttribute, List[Union[float, int]]]:
        """Postprocess a regression attribute for patients.

        Args:
            synthetic_data (sparse.COO): Synthetic data in sparse format.
            group (Group): Group to postprocess.
            attribute (MedRecordAttribute): Attribute to postprocess.

        Returns:
            Dict[MedRecordAttribute, List[Union[float, int]]]: Postprocessed attribute.

        Raises:
            ValueError: If the train and test data types do not match the real data
                type or that is the case for the attributes.
        """
        number_real_patients = len(self.real_medrecord.nodes_in_group(group))
        real_data, real_attributes = MTGANDataset(
            self.preprocessor, medrecord=self.real_medrecord, device=self.device
        ).get_attributes(idx=np.arange(number_real_patients), attribute=attribute)
        train_data, test_data, train_attributes, test_attributes = train_test_split(
            real_data, real_attributes, test_size=0.2, random_state=42
        )
        if not (
            isinstance(train_data, type(real_data))
            and isinstance(test_data, type(real_data))
        ):
            raise ValueError(
                f"Train and test data types do not match the real data type for {attribute}."
            )
        if not (
            isinstance(train_attributes, type(real_attributes))
            and isinstance(test_attributes, type(real_attributes))
        ):
            raise ValueError(
                f"Train and test attributes types do not match the real data type for {attribute}."
            )
        train_set = xgb.DMatrix(
            train_data.sum(axis=1), label=train_attributes, enable_categorical=True
        )
        test_set = xgb.DMatrix(
            test_data.sum(axis=1), label=test_attributes, enable_categorical=True
        )
        model = xgb.train(
            {
                "objective": "reg:tweedie",
                "eval_metric": "rmse",
                "verbosity": 0,
            },
            train_set,
            self.hyperparameters["training_epochs"],
            evals=[(test_set, "eval")],
            verbose_eval=False,
        )

        # Synthetic data
        synthetic_attributes = model.predict(
            xgb.DMatrix(synthetic_data.todense().sum(axis=1))
        )
        # check if the training data was integer or float to round the synthetic data
        if np.issubdtype(train_attributes.dtype, np.integer):
            synthetic_attributes = np.round(synthetic_attributes).astype(int)
        else:
            synthetic_attributes = synthetic_attributes.astype(float)

        return {attribute: synthetic_attributes.tolist()}

    def _patients_temporal_postprocessing(
        self,
        synthetic_data: sparse.COO,
        group: Group,
        attribute: MedRecordAttribute,
    ) -> Dict[MedRecordAttribute, List[datetime]]:
        """Postprocess a temporal attribute for patients.

        Args:
            synthetic_data (sparse.COO): Synthetic data in sparse format.
            group (Group): Group to postprocess.
            attribute (MedRecordAttribute): Attribute to postprocess.

        Returns:
            Dict[MedRecordAttribute, List[datetime]]: Postprocessed attribute.

        Raises:
            ValueError: If the train and test data types do not match the real data
                type.
        """
        number_real_patients = len(self.real_medrecord.nodes_in_group(group))
        time_interval_days = self.preprocessor.hyperparameters["time_interval_days"]
        real_data, real_attributes = MTGANDataset(
            self.preprocessor, medrecord=self.real_medrecord, device=self.device
        ).get_attributes(idx=np.arange(number_real_patients), attribute=attribute)

        # Convert to datetime and normalize
        min_date = np.min(real_attributes)
        real_attributes = (
            np.array([delta.days for delta in (real_attributes - min_date)])
            // time_interval_days
        )

        train_data, test_data, train_attributes, test_attributes = train_test_split(
            real_data, real_attributes, test_size=0.2, random_state=42
        )
        if not (
            isinstance(train_data, type(real_data))
            and isinstance(test_data, type(real_data))
        ):
            raise ValueError(
                f"Train and test data types do not match the real data type for {attribute}."
            )
        train_set = xgb.DMatrix(
            train_data.sum(axis=1), label=train_attributes, enable_categorical=True
        )
        test_set = xgb.DMatrix(
            test_data.sum(axis=1), label=test_attributes, enable_categorical=True
        )
        model = xgb.train(
            {
                "objective": "reg:tweedie",
                "eval_metric": "rmse",
                "verbosity": 0,
            },
            train_set,
            self.hyperparameters["training_epochs"],
            evals=[(test_set, "eval")],
            verbose_eval=False,
        )

        # Synthetic data
        synthetic_attributes = model.predict(
            xgb.DMatrix(synthetic_data.todense().sum(axis=1))
        ).tolist()
        synthetic_attributes = [
            min_date + timedelta(days=int(round(x * time_interval_days)))
            for x in synthetic_attributes
        ]

        return {attribute: synthetic_attributes}

    def find_time(
        self, synthetic_data: sparse.COO, first_admissions: List[datetime]
    ) -> Dict[MedRecordAttribute, List[datetime]]:
        """Find time to the edges of the synthetic MedRecord.

        Args:
            synthetic_data (sparse.COO): Synthetic data in sparse format.
            first_admissions (List[datetime]): List of first admissions.

        Returns:
            Dict[MedRecordAttribute, List[datetime]]: Postprocessed attribute.
        """
        time_dataset = MTGANDatasetTime(
            self.preprocessor, medrecord=self.real_medrecord, device=self.device
        )
        train_loader = MTGANDataLoader(
            dataset=time_dataset,
            batch_size=self.hyperparameters["batch_size"],
            shuffle=False,
        )

        model = TimeLSTM(
            input_size=time_dataset[0][0].size(2),
            hyperparameters=self.hyperparameters,
            device=self.device,
        )
        model.train_model(train_loader)
        absolute_time_windows = model.predict(synthetic_data)

        # Convert to datetime with respect to the first admission
        time_windows_days = (
            absolute_time_windows
            * self.preprocessor.hyperparameters["time_interval_days"]
        ).round().astype(int) * timedelta(days=1)
        first_admissions_array = np.array(first_admissions).reshape(-1, 1)
        times = (first_admissions_array + time_windows_days).tolist()

        return {self.preprocessor.time_attribute: times}

    def _concepts_regression_postprocessing(
        self,
        synthetic_data: sparse.COO,
        group: Group,
        attribute: MedRecordAttribute,
    ) -> Dict[MedRecordAttribute, List[Any]]:
        """Postprocess a regression attribute for the concepts group.

        Args:
            synthetic_data (sparse.COO): Synthetic data in sparse format.
            group (Group): Group to postprocess.
            attribute (MedRecordAttribute): Attribute to postprocess.

        Returns:
            Dict[MedRecordAttribute, List[Any]]: Postprocessed attribute.

        Raises:
            ValueError: If the attribute has only one category, the number of samples
                is too small or the train and test data types do not match the real
                data type.
        """
        attribute_dataset = MTGANDatasetAttribute(
            self.preprocessor,
            medrecord=self.real_medrecord,
            attribute=attribute,
            device=self.device,
        )
        train_loader = MTGANDataLoader(
            dataset=attribute_dataset,
            batch_size=self.hyperparameters["batch_size"],
            shuffle=False,
        )

        model = AttributeClassifier(
            model=AttributeRegression(
                input_size=attribute_dataset[0][0].size(2),
                output_size=len(attribute_dataset.concepts_with_attribute),
                hyperparameters=self.hyperparameters,
                device=self.device,
            ),
        )

        model.train_model(train_loader)
        attributes = model.predict(
            synthetic_data, attribute_dataset.concepts_with_attribute
        )
        # TODO finish the padding on tha attributes
        return {attribute: attributes}

    def convert_to_medrecord(
        self,
        synthetic_data: sparse.COO,
        synthetic_attributes_patients: Dict[MedRecordAttribute, List[Any]],
        synthetic_attributes_concepts: Dict[MedRecordAttribute, List[Any]],
    ) -> MedRecord:
        """Converts synthetic data sparse object to a MedRecord object.

        Args:
            synthetic_data (sparse.COO): Synthetic data in sparse format.
            synthetic_attributes_patients (Dict[MedRecordAttribute, List[Any]]):
                Synthetic attributes for patients.
            synthetic_attributes_concepts (Dict[MedRecordAttribute, List[Any]]):
                Synthetic attributes for concepts.

        Returns:
            MedRecord: MedRecord object with synthetic data.
        """
        synthetic_attributes_patients.pop(self.preprocessor.first_admission_attribute)
        patient_nodes = pl.DataFrame(
            {
                "index": np.arange(synthetic_data.shape[0]),
                **synthetic_attributes_patients,
            }
        )

        concepts_indices = np.arange(synthetic_data.shape[2])
        concepts_translated = [
            self.preprocessor.index_to_concept_dict[concept]
            for concept in concepts_indices
        ]
        concept_nodes = pl.DataFrame(
            {
                "index": concepts_translated,
            }
        )

        concepts_edge = [
            self.preprocessor.index_to_concept_dict[coord]
            for coord in synthetic_data.coords[2]
        ]
        times = np.array(
            synthetic_attributes_concepts[self.preprocessor.time_attribute]
        )
        edges = pl.DataFrame(
            {
                "source": concepts_edge,
                "target": synthetic_data.coords[0],
                self.preprocessor.time_attribute: times[
                    synthetic_data.coords[0], synthetic_data.coords[1]
                ].tolist(),
            }
        )

        synthetic_medrecord = MedRecord.from_polars(
            nodes=[(patient_nodes, "index"), (concept_nodes, "index")],
            edges=[(edges, "source", "target")],
        )
        synthetic_medrecord.add_group(
            self.preprocessor.patients_group, nodes=patient_nodes["index"].to_list()
        )
        synthetic_medrecord.add_group(
            self.preprocessor.concepts_group, nodes=concept_nodes["index"].to_list()
        )

        return synthetic_medrecord
