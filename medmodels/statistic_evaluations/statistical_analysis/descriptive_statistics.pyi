import polars as pl

from medmodels.medrecord import MedRecord
from medmodels.medrecord.querying import NodeQuery
from medmodels.medrecord.schema import AttributeType
from medmodels.medrecord.types import (
    NumericAttributeInfo,
    StringAttributeInfo,
    TemporalAttributeInfo,
)

def determine_attribute_type(attribute_values: pl.Series) -> AttributeType: ...
def get_continuous_attribute_statistics(
    medrecord: MedRecord, attribute_query: NodeQuery
) -> NumericAttributeInfo: ...
def get_temporal_attribute_statistics(
    medrecord: MedRecord, attribute_query: NodeQuery
) -> TemporalAttributeInfo: ...
def get_categorical_attribute_statistics(
    medrecord: MedRecord, attribute_query: NodeQuery
) -> StringAttributeInfo: ...
