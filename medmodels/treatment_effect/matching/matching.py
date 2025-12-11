"""Module containing the matching abstract class.

Matching is the process of selecting control subjects that are similar to treated
subjects. The class provides the base for the matching algorithms, such as propensity
score matching and nearest neighbor matching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeAlias,
    TypedDict,
    Union,
)

import polars as pl

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.schema import AttributesSchema, AttributeType
from medmodels.medrecord.types import (
    Attributes,
    EdgeIndex,
    NodeIndex,
)

if TYPE_CHECKING:
    from medmodels.medrecord.medrecord import MedRecord
    from medmodels.medrecord.querying import NodeIndicesOperand, NodeOperand
    from medmodels.medrecord.types import Group, MedRecordAttribute

MatchingMethod: TypeAlias = Literal["propensity", "nearest_neighbors"]


class AttributeInfo(TypedDict):
    """A dictionary containing info about nodes/edges and their attributes."""

    count: int
    attribute: Dict[
        MedRecordAttribute,
        Union[TemporalAttributeInfo, NumericAttributeInfo, StringAttributeInfo],
    ]


class TemporalAttributeInfo(TypedDict):
    """Dictionary for a temporal attribute and its metrics."""

    type: Literal["Temporal"]
    min: datetime
    max: datetime


class NumericAttributeInfo(TypedDict):
    """Dictionary for a numeric attribute and its metrics."""

    type: Literal["Continuous"]
    min: Union[int, float]
    max: Union[int, float]
    mean: Union[int, float]


class StringAttributeInfo(TypedDict):
    """Dictionary for a string attribute and its values."""

    type: Literal["Categorical"]
    values: str


AttributeDictionary: TypeAlias = Union[
    Dict[EdgeIndex, Attributes], Dict[NodeIndex, Attributes]
]


def _extract_numeric_attribute_info(
    attribute_values: List[Union[int, float]],
) -> NumericAttributeInfo:
    """Extracts info about attributes with numeric format.

    Args:
        attribute_values (List[Union[int, float]]): List containing attribute values.

    Returns:
        NumericAttributeInfo: Dictionary containg attribute metrics.
    """
    min_value = min(attribute_values)
    max_value = max(attribute_values)
    mean_value = sum(attribute_values) / len(attribute_values)

    # assertion to ensure correct typing
    # never fails, because the series never contains None values and is always numeric
    assert isinstance(min_value, (int, float))
    assert isinstance(max_value, (int, float))
    assert isinstance(mean_value, (int, float))

    return {
        "type": "Continuous",
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
    }


def _extract_temporal_attribute_info(
    attribute_values: List[datetime],
) -> TemporalAttributeInfo:
    """Extracts info about attributes with temporal format.

    Args:
        attribute_values (List[datetime]): List containing temporal attribute values.

    Returns:
        TemporalAttributeInfo: Dictionary containg attribute metrics.
    """
    return {
        "type": "Temporal",
        "min": min(attribute_values),
        "max": max(attribute_values),
    }


def _extract_string_attribute_info(
    attribute_values: List[str],
    short_string_prefix: Literal["Values", "Categories"] = "Values",
    long_string_suffix: str = "unique values",
    max_number_values: int = 5,
    max_line_length: int = 100,
) -> StringAttributeInfo:
    """Extracts info about attributes with string format.

    Args:
        attribute_values (List[str]): List containing attribute values.
        short_string_prefix (Literal["Values", "Categories"], optional): Prefix for
            information string in case of listing all the values. Defaults to "Values".
        long_string_suffix (str, optional): Suffix for attribute information in case of
            too many values to list. Here only the count will be displayed. Defaults to
            "unique values".
        max_number_values (int, optional): Maximum values that should be listed in the
            information string. Defaults to 5.
        max_line_length (int, optional): Maximum line length for the information string.
            Defaults to 100.

    Returns:
        StringAttributeInfo: Dictionary containg attribute metrics.
    """
    values = sorted(set(attribute_values))

    values_string = f"{short_string_prefix}: {', '.join(list(values))}"

    if (len(values) > max_number_values) | (len(values_string) > max_line_length):
        values_string = f"{len(values)} {long_string_suffix}"

    return {
        "type": "Categorical",
        "values": values_string,
    }


def extract_attribute_summary(
    attribute_dictionary: AttributeDictionary,
    schema: Optional[AttributesSchema] = None,
) -> Dict[
    MedRecordAttribute,
    Union[TemporalAttributeInfo, NumericAttributeInfo, StringAttributeInfo],
]:
    """Extracts a summary from a node or edge attribute dictionary.

    Args:
        attribute_dictionary (AttributeDictionary): Edges or Nodes and their attributes
            and values.
        schema (Optional[AttributesSchema], optional): Attribute Schema for the group
            nodes or edges. Defaults to None.
        decimal (int): Decimal points to round the numeric values to. Defaults to 2.

    Returns:
        Dict[MedRecordAttribute, Union[TemporalAttributeInfo, NumericAttributeInfo,
            StringAttributeInfo]: Summary of node or edge attributes.
    """
    data = {}

    for dictionary in attribute_dictionary.values():
        for key, value in dictionary.items():
            data.setdefault(key, []).append(value)

    data_dict = {}

    for attribute in sorted(data):
        attribute_values = [value for value in data[attribute] if value is not None]

        if len(attribute_values) == 0:
            attribute_info = {"type": "-", "values": "-"}

        # check if the attribute has as an attribute type defined in the schema
        elif schema and attribute in schema and schema[attribute][1]:
            if schema[attribute][1] == AttributeType.Continuous:
                attribute_info = _extract_numeric_attribute_info(attribute_values)
            elif schema[attribute][1] == AttributeType.Temporal:
                attribute_info = _extract_temporal_attribute_info(attribute_values)
            else:
                attribute_info = _extract_string_attribute_info(
                    attribute_values=attribute_values,
                    long_string_suffix="categories",
                    short_string_prefix="Categories",
                )

        # Without Schema
        else:
            if all(isinstance(value, (int, float)) for value in attribute_values):
                attribute_info = _extract_numeric_attribute_info(attribute_values)
            elif all(isinstance(value, datetime) for value in attribute_values):
                attribute_info = _extract_temporal_attribute_info(attribute_values)
            else:
                attribute_info = _extract_string_attribute_info(
                    attribute_values=[str(value) for value in attribute_values]
                )

        data_dict[attribute] = attribute_info

    return data_dict


class Matching(ABC):
    """The Abstract Class for matching."""

    number_of_neighbors: int

    def __init__(self, number_of_neighbors: int) -> None:
        """Initializes the matching class.

        Args:
            number_of_neighbors (int): Number of nearest neighbors to find for each
                treated patient.
        """
        self.number_of_neighbors = number_of_neighbors

    def _preprocess_data(
        self,
        *,
        medrecord: MedRecord,
        control_set: Set[NodeIndex],
        treated_set: Set[NodeIndex],
        patients_group: Group,
        essential_covariates: Optional[List[MedRecordAttribute]] = None,
        one_hot_covariates: Optional[List[MedRecordAttribute]] = None,
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """Prepared the data for the matching algorithms.

        Args:
            medrecord (MedRecord):  MedRecord object containing the data.
            control_set (Set[NodeIndex]): Set of treated subjects.
            treated_set (Set[NodeIndex]): Set of control subjects.
            patients_group (Group): The group of patients.
            essential_covariates (Optional[List[MedRecordAttribute]], optional):
                Covariates that are essential for matching. Defaults to None, meaning
                all the attributes of the patients are used.
            one_hot_covariates (Optional[List[MedRecordAttribute]], optional):
                Covariates that are one-hot encoded for matching. Defaults to None,
                meaning all the categorical attributes of the patients are used.

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: Treated and control groups with their
                preprocessed covariates

        Raises:
            AssertionError: If the one-hot covariates are not in the essential
                covariates.
        """
        if essential_covariates is None:
            # If no essential covariates provided, use all attributes of patients group
            nodes_attributes = medrecord.node[medrecord.nodes_in_group(patients_group)]
            essential_covariates = list(
                {key for attributes in nodes_attributes.values() for key in attributes}
            )

        control_set = self._check_nodes(
            medrecord=medrecord,
            treated_set=treated_set,
            control_set=control_set,
            essential_covariates=essential_covariates,
        )

        if "id" not in essential_covariates:
            essential_covariates.append("id")

        # Dataframe with the essential covariates
        data = pl.DataFrame(
            data=[
                {"id": k, **v}
                for k, v in medrecord.node[list(control_set | treated_set)].items()
            ]
        )
        original_columns = data.columns

        # If no one-hot covariates provided, use all categorical attributes of patients
        if one_hot_covariates is None:
            attributes = extract_attribute_summary(
                medrecord.node[medrecord.nodes_in_group(patients_group)]
            )
            one_hot_covariates = [
                covariate
                for covariate, values in attributes.items()
                if "Categorical" in values["type"]
            ]

            one_hot_covariates = [
                covariate
                for covariate in one_hot_covariates
                if covariate in essential_covariates
            ]

        # If there are one-hot covariates, check if all are in the essential covariates
        if (
            not all(
                covariate in essential_covariates for covariate in one_hot_covariates
            )
            and one_hot_covariates
        ):
            msg = "One-hot covariates must be in the essential covariates"
            raise AssertionError(msg)

        # One-hot encode the categorical variables
        data = data.to_dummies(
            columns=[str(covariate) for covariate in one_hot_covariates],
            drop_first=True,
        )
        new_columns = [col for col in data.columns if col not in original_columns]

        # Add to essential covariates the new columns created by one-hot encoding and
        # delete the ones that were one-hot encoded
        essential_covariates.extend(new_columns)
        [essential_covariates.remove(col) for col in one_hot_covariates]
        data = data.select(essential_covariates)

        # Select the sets of treated and control subjects
        data_treated = data.filter(pl.col("id").is_in(treated_set))
        data_control = data.filter(pl.col("id").is_in(control_set))

        return data_treated, data_control

    def _check_nodes(
        self,
        medrecord: MedRecord,
        treated_set: Set[NodeIndex],
        control_set: Set[NodeIndex],
        essential_covariates: List[MedRecordAttribute],
    ) -> Set[NodeIndex]:
        """Check if the treated and control sets are disjoint.

        Args:
            medrecord (MedRecord): MedRecord object containing the data.
            treated_set (Set[NodeIndex]): Set of treated subjects.
            control_set (Set[NodeIndex]): Set of control subjects.
            essential_covariates (List[MedRecordAttribute]): Covariates that are
                essential for matching.

        Returns:
            Set[NodeIndex]: The control set.

        Raises:
            ValueError: If not enough control subjects to match the treated subjects.
            ValueError: If some treated nodes do not have all the essential covariates.
        """

        def query_essential_covariates(
            node: NodeOperand, patients_set: Set[NodeIndex]
        ) -> NodeIndicesOperand:
            """Query the nodes that have all the essential covariates.

            Returns:
                NodeIndicesOperand: The node indices of the queried node.
            """
            node.has_attribute(essential_covariates)

            node.index().is_in(list(patients_set))

            return node.index()

        control_set = set(
            medrecord.query_nodes(
                lambda node: query_essential_covariates(node, control_set)
            )
        )

        if len(control_set) < self.number_of_neighbors * len(treated_set):
            msg = (
                f"Not enough control subjects to match the treated subjects. "
                f"Number of controls: {len(control_set)}, "
                f"Number of treated subjects: {len(treated_set)}, "
                f"Number of neighbors required per treated subject: {self.number_of_neighbors}, "
                f"Total controls needed: {self.number_of_neighbors * len(treated_set)}."
            )
            raise ValueError(msg)

        if len(treated_set) != len(
            medrecord.query_nodes(
                lambda node: query_essential_covariates(node, treated_set)
            )
        ):
            msg = "Some treated nodes do not have all the essential covariates"
            raise ValueError(msg)

        return control_set

    @abstractmethod
    def match_controls(
        self,
        *,
        medrecord: MedRecord,
        control_set: Set[NodeIndex],
        treated_set: Set[NodeIndex],
        essential_covariates: Optional[Sequence[MedRecordAttribute]] = None,
        one_hot_covariates: Optional[Sequence[MedRecordAttribute]] = None,
    ) -> Set[NodeIndex]:
        """Matches the controls based on the matching algorithm.

        Args:
            medrecord (MedRecord): MedRecord object containing the data.
            control_set (Set[NodeIndex]): Set of control subjects.
            treated_set (Set[NodeIndex]): Set of treated subjects.
            essential_covariates (Optional[Sequence[MedRecordAttribute]], optional):
                Covariates that are essential for matching. Defaults to None.
            one_hot_covariates (Optional[Sequence[MedRecordAttribute]], optional):
                Covariates that are one-hot encoded for matching. Defaults to None.

        Returns:
            Set[NodeIndex]: Node Ids of the matched controls.
        """
        ...
