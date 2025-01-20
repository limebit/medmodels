from typing import Dict, List, Optional, Tuple, Union

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.querying import NodeQuery
from medmodels.medrecord.schema import AttributeType
from medmodels.medrecord.types import (
    AttributeSummary,
    Group,
    GroupInputList,
    MedRecordAttribute,
    NodeIndex,
)

class CohortEvaluator:
    medrecord: MedRecord
    name: str
    cohort_group: Group
    time_attribute: MedRecordAttribute
    attributes: Optional[Dict[str, MedRecordAttribute]]
    concepts_groups: Optional[GroupInputList]
    attribute_summary: Dict[Group, AttributeSummary]
    attribute_types: Dict[MedRecordAttribute, AttributeType]

    def __init__(
        self,
        medrecord: MedRecord,
        name: str,
        cohort_group: Union[Group, NodeQuery] = "patients",
        time_attribute: MedRecordAttribute = "time",
        attributes: Optional[Dict[str, MedRecordAttribute]] = None,
        concepts_groups: Optional[GroupInputList] = None,
    ) -> None: ...
    def get_concept_counts(
        self,
    ) -> List[Tuple[NodeIndex, int]]: ...
    def get_top_k_concepts(
        self,
        top_k: int,
    ) -> List[NodeIndex]: ...
    def get_attribute_summary(
        self,
    ) -> Dict[Group, AttributeSummary]: ...
