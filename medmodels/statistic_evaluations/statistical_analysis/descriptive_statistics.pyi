import polars as pl

from medmodels.medrecord import MedRecord
from medmodels.medrecord.querying import NodeOperation
from medmodels.medrecord.schema import AttributeType
from medmodels.medrecord.types import (
    NumericAttributeInfo,
    StringAttributeInfo,
    TemporalAttributeInfo,
)

def determine_attribute_type(attribute_values: pl.Series) -> AttributeType: ...
def get_continuos_attribute_statistics(
    medrecord: MedRecord, attribute_query: NodeOperation
) -> NumericAttributeInfo: ...
def get_temporal_attribute_statistics(
    medrecord: MedRecord, attribute_query: NodeOperation
) -> TemporalAttributeInfo: ...
def get_categorical_attribute_statistics(
    medrecord: MedRecord, attribute_query: NodeOperation
) -> StringAttributeInfo: ...
