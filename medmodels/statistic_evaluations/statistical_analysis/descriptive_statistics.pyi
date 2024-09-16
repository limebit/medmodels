from typing import Dict

import polars as pl

from medmodels.medrecord import MedRecord
from medmodels.medrecord.querying import NodeOperation
from medmodels.medrecord.schema import AttributeType

def determine_attribute_type(attribute_values: pl.Series) -> AttributeType: ...
def get_continuos_attribute_statistics(
    medrecord: MedRecord, attribute_query: NodeOperation
) -> Dict[str, float]: ...
def get_temporal_attribute_statistics(
    medrecord: MedRecord, attribute_query: NodeOperation
) -> Dict[str, str]: ...
def get_categorical_attribute_statistics(
    medrecord: MedRecord, attribute_query: NodeOperation
) -> Dict[str, str]: ...
