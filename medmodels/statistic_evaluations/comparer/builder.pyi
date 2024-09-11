from __future__ import annotations
from typing import List, Optional, Union

from medmodels.medrecord.types import Group, GroupInputList, MedRecordAttribute

from medmodels.statistic_evaluations.comparer.data_comparer import DataComparer
from medmodels.medrecord.querying import NodeOperation

class ComparerBuilder:
    patients_group: Optional[Group]
    time_attribute: Optional[MedRecordAttribute]
    concept_group: Optional[Union[GroupInputList, Group]]
    subpopulations_groups: Optional[GroupInputList]
    subpopulations_queries: Optional[List[NodeOperation]]
    control_population: Optional[Union[Group, NodeOperation]]
    top_k_concepts: Optional[int]
    alpha: Optional[float]

    def with_patients_group(self, group: Group) -> ComparerBuilder: ...
    def with_time_attributes(
        self, attribute: MedRecordAttribute
    ) -> ComparerBuilder: ...
    def with_concept_group(
        self, group: Union[Group, GroupInputList]
    ) -> ComparerBuilder: ...
    def with_subpopulations(
        self,
        groups: GroupInputList,
        queries: List[NodeOperation],
        control_population: Union[Group, NodeOperation],
    ) -> ComparerBuilder: ...
    def with_top_k(self, k: int) -> ComparerBuilder: ...
    def with_significance_level(self, alpha: float) -> ComparerBuilder: ...
    def build(self) -> DataComparer: ...
