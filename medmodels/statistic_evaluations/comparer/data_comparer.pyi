from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import polars as pl

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.querying import NodeOperation
from medmodels.medrecord.types import (
    Group,
    GroupInputList,
    MedRecordAttribute,
    MedRecordAttributeInputList,
)
from medmodels.statistic_evaluations.comparer.builder import ComparerBuilder

class DataComparer:
    patients_group: Group
    time_attribute: MedRecordAttribute
    concept_groups: GroupInputList
    top_k_concepts: int
    alpha: float
    subcohorts_groups: Optional[GroupInputList]
    subcohorts_queries: Optional[List[NodeOperation]]
    control_cohort: Optional[Union[Group, NodeOperation]]

    @classmethod
    def builder(cls) -> ComparerBuilder: ...
    @staticmethod
    def _set_configuration(
        data_comparer: DataComparer,
        *,
        patient_group: Group = "patients",
        time_attribute: MedRecordAttribute = "time",
        concept_groups: GroupInputList = ["concepts"],
        top_k_concepts: int = 10,
        alpha: float = 0.05,
        subcohorts_groups: Optional[GroupInputList],
        subcohorts_queries: Optional[List[NodeOperation]],
        control_cohorts: Optional[Union[Group, NodeOperation]],
    ) -> None: ...
    def compare_attributes(
        self,
        medrecords: List[MedRecord],
        attributes: MedRecordAttributeInputList,
    ) -> Dict[MedRecordAttribute, pl.DataFrame]: ...
    def calculate_absolute_relative_difference(
        self,
        medrecords: Union[MedRecord, List[MedRecord]],
        attributes: MedRecordAttributeInputList,
        control_data: Optional[MedRecord],
    ) -> Tuple[float, pl.DataFrame]: ...
    def test_difference_attributes(
        self,
        medrecords: Union[MedRecord, List[MedRecord]],
        attributes: MedRecordAttributeInputList,
    ) -> Dict[MedRecordAttribute, Dict[str, Union[str, bool, float]]]: ...
    def get_top_k_concepts(
        self,
        medrecord: MedRecord,
    ) -> Dict[Union[Group, str], List[Any]]: ...
    def test_difference_top_k_concepts(
        self, medrecords: Union[List[MedRecord], MedRecord]
    ) -> Dict[Group, Dict[str, Union[str, bool, float]]]: ...
    def calculate_distance_concepts(
        self, medrecord: Union[List[MedRecord], MedRecord]
    ) -> Dict[str, float]: ...
    def full_report(
        self, medrecord: Union[MedRecord, List[MedRecord]]
    ) -> Dict[Union[MedRecordAttribute, Group], Dict[str, Any]]: ...
