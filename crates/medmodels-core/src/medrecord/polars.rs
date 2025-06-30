use crate::{
    errors::MedRecordError,
    medrecord::{Attributes, MedRecordAttribute, MedRecordValue, NodeIndex},
};
use chrono::DateTime;
use polars::{datatypes::AnyValue, frame::DataFrame};

// TODO: Add tests for Duration
impl<'a> TryFrom<AnyValue<'a>> for MedRecordValue {
    type Error = MedRecordError;

    fn try_from(value: AnyValue<'a>) -> Result<Self, Self::Error> {
        match value {
            AnyValue::String(value) => Ok(MedRecordValue::String(value.into())),
            AnyValue::StringOwned(value) => Ok(MedRecordValue::String((*value).into())),
            AnyValue::Int8(value) => Ok(MedRecordValue::Int(value.into())),
            AnyValue::Int16(value) => Ok(MedRecordValue::Int(value.into())),
            AnyValue::Int32(value) => Ok(MedRecordValue::Int(value.into())),
            AnyValue::Int64(value) => Ok(MedRecordValue::Int(value)),
            AnyValue::UInt8(value) => Ok(MedRecordValue::Int(value.into())),
            AnyValue::UInt16(value) => Ok(MedRecordValue::Int(value.into())),
            AnyValue::UInt32(value) => Ok(MedRecordValue::Int(value.into())),
            AnyValue::Float32(value) => Ok(MedRecordValue::Float(value.into())),
            AnyValue::Float64(value) => Ok(MedRecordValue::Float(value)),
            AnyValue::Boolean(value) => Ok(MedRecordValue::Bool(value)),
            AnyValue::Datetime(value, unit, _) => {
                // TODO: handle timezone
                Ok(match unit {
                    polars::prelude::TimeUnit::Nanoseconds => {
                        MedRecordValue::DateTime(DateTime::from_timestamp_nanos(value).naive_utc())
                    }
                    polars::prelude::TimeUnit::Microseconds => MedRecordValue::DateTime(
                        DateTime::from_timestamp_micros(value)
                            .ok_or(MedRecordError::ConversionError(format!(
                                "Cannot convert {value}ms into MedRecordValue"
                            )))?
                            .naive_utc(),
                    ),
                    polars::prelude::TimeUnit::Milliseconds => MedRecordValue::DateTime(
                        DateTime::from_timestamp_millis(value)
                            .ok_or(MedRecordError::ConversionError(format!(
                                "Cannot convert {value}ms into MedRecordValue"
                            )))?
                            .naive_utc(),
                    ),
                })
            }
            AnyValue::Duration(value, unit) => Ok(match unit {
                polars::prelude::TimeUnit::Nanoseconds => MedRecordValue::Duration(
                    std::time::Duration::from_nanos(value.try_into().map_err(|_| {
                        MedRecordError::ConversionError(format!(
                            "Cannot convert {value} into MedRecordValue"
                        ))
                    })?),
                ),
                polars::prelude::TimeUnit::Microseconds => MedRecordValue::Duration(
                    std::time::Duration::from_micros(value.try_into().map_err(|_| {
                        MedRecordError::ConversionError(format!(
                            "Cannot convert {value} into MedRecordValue"
                        ))
                    })?),
                ),
                polars::prelude::TimeUnit::Milliseconds => MedRecordValue::Duration(
                    std::time::Duration::from_millis(value.try_into().map_err(|_| {
                        MedRecordError::ConversionError(format!(
                            "Cannot convert {value} into MedRecordValue"
                        ))
                    })?),
                ),
            }),
            AnyValue::Null => Ok(MedRecordValue::Null),
            _ => Err(MedRecordError::ConversionError(format!(
                "Cannot convert {value} into MedRecordValue"
            ))),
        }
    }
}

impl<'a> TryFrom<AnyValue<'a>> for MedRecordAttribute {
    type Error = MedRecordError;

    fn try_from(value: AnyValue<'a>) -> Result<Self, Self::Error> {
        match value {
            AnyValue::String(value) => Ok(MedRecordAttribute::String(value.into())),
            AnyValue::StringOwned(value) => Ok(MedRecordAttribute::String((*value).into())),
            AnyValue::Int8(value) => Ok(MedRecordAttribute::Int(value.into())),
            AnyValue::Int16(value) => Ok(MedRecordAttribute::Int(value.into())),
            AnyValue::Int32(value) => Ok(MedRecordAttribute::Int(value.into())),
            AnyValue::Int64(value) => Ok(MedRecordAttribute::Int(value)),
            AnyValue::UInt8(value) => Ok(MedRecordAttribute::Int(value.into())),
            AnyValue::UInt16(value) => Ok(MedRecordAttribute::Int(value.into())),
            AnyValue::UInt32(value) => Ok(MedRecordAttribute::Int(value.into())),
            _ => Err(MedRecordError::ConversionError(format!(
                "Cannot convert {value} into MedRecordAttribute"
            ))),
        }
    }
}

pub(crate) fn dataframe_to_nodes(
    mut nodes: DataFrame,
    index_column_name: &str,
) -> Result<Vec<(NodeIndex, Attributes)>, MedRecordError> {
    if nodes.max_n_chunks() > 1 {
        nodes.rechunk_mut();
    }

    let attribute_column_names: Vec<_> = nodes
        .get_column_names()
        .into_iter()
        .filter(|name| *name != index_column_name)
        .collect();

    let index = nodes
        .column(index_column_name)
        .map_err(|_| {
            MedRecordError::ConversionError(format!(
                "Cannot find column with name {index_column_name} in dataframe"
            ))
        })?
        .as_materialized_series()
        .iter();

    let mut columns: Vec<_> = nodes
        .columns(&attribute_column_names)
        .expect("Attribute columns must exist")
        .iter()
        .map(|s| s.as_materialized_series().iter())
        .zip(attribute_column_names)
        .collect();

    index
        .map(|index_value| {
            Ok((
                index_value.try_into()?,
                columns
                    .iter_mut()
                    .map(|(column, column_name)| {
                        Ok((
                            (***column_name).into(),
                            column.next().expect("msg").try_into()?,
                        ))
                    })
                    .collect::<Result<_, MedRecordError>>()?,
            ))
        })
        .collect()
}

pub(crate) fn dataframe_to_edges(
    mut edges: DataFrame,
    source_index_column_name: &str,
    target_index_column_name: &str,
) -> Result<Vec<(NodeIndex, NodeIndex, Attributes)>, MedRecordError> {
    if edges.max_n_chunks() > 1 {
        edges.rechunk_mut();
    }

    let attribute_column_names: Vec<_> = edges
        .get_column_names()
        .into_iter()
        .filter(|name| *name != source_index_column_name && *name != target_index_column_name)
        .collect();

    let source_index = edges
        .column(source_index_column_name)
        .map_err(|_| {
            MedRecordError::ConversionError(format!(
                "Cannot find column with name {source_index_column_name} in dataframe"
            ))
        })?
        .as_materialized_series()
        .iter();
    let target_index = edges
        .column(target_index_column_name)
        .map_err(|_| {
            MedRecordError::ConversionError(format!(
                "Cannot find column with name {target_index_column_name} in dataframe"
            ))
        })?
        .as_materialized_series()
        .iter();

    let mut columns: Vec<_> = edges
        .columns(&attribute_column_names)
        .expect("Attribute columns must exist")
        .iter()
        .map(|s| s.as_materialized_series().iter())
        .zip(attribute_column_names)
        .collect();

    source_index
        .zip(target_index)
        .map(|(source_index_value, target_index_value)| {
            Ok((
                source_index_value.try_into()?,
                target_index_value.try_into()?,
                columns
                    .iter_mut()
                    .map(|(column, column_name)| {
                        Ok((
                            (***column_name).into(),
                            column
                                .next()
                                .expect("Should have as many iterations as rows")
                                .try_into()?,
                        ))
                    })
                    .collect::<Result<_, MedRecordError>>()?,
            ))
        })
        .collect()
}

#[cfg(test)]
mod test {
    use super::{dataframe_to_edges, dataframe_to_nodes, MedRecordValue};
    use crate::errors::MedRecordError;
    use chrono::NaiveDateTime;
    use polars::prelude::*;
    use std::collections::HashMap;

    #[test]
    fn test_try_from_anyvalue_string() {
        let any_value = AnyValue::String("value");

        let value = MedRecordValue::try_from(any_value).unwrap();

        assert_eq!(MedRecordValue::String("value".to_string()), value);
    }

    #[test]
    fn test_from_anyvalue_int8() {
        let any_value = AnyValue::Int8(0);

        let value = MedRecordValue::try_from(any_value).unwrap();

        assert_eq!(MedRecordValue::Int(0), value);
    }

    #[test]
    fn test_from_anyvalue_int16() {
        let any_value = AnyValue::Int16(0);

        let value = MedRecordValue::try_from(any_value).unwrap();

        assert_eq!(MedRecordValue::Int(0), value);
    }

    #[test]
    fn test_from_anyvalue_int32() {
        let any_value = AnyValue::Int32(0);

        let value = MedRecordValue::try_from(any_value).unwrap();

        assert_eq!(MedRecordValue::Int(0), value);
    }

    #[test]
    fn test_from_anyvalue_int64() {
        let any_value = AnyValue::Int64(0);

        let value = MedRecordValue::try_from(any_value).unwrap();

        assert_eq!(MedRecordValue::Int(0), value);
    }

    #[test]
    fn test_from_anyvalue_float32() {
        let any_value = AnyValue::Float32(0.0);

        let value = MedRecordValue::try_from(any_value).unwrap();

        assert_eq!(MedRecordValue::Float(0.0), value);
    }

    #[test]
    fn test_from_anyvalue_float64() {
        let any_value = AnyValue::Float64(0.0);

        let value = MedRecordValue::try_from(any_value).unwrap();

        assert_eq!(MedRecordValue::Float(0.0), value);
    }

    #[test]
    fn test_from_anyvalue_bool() {
        let any_value = AnyValue::Boolean(false);

        let value = MedRecordValue::try_from(any_value).unwrap();

        assert_eq!(MedRecordValue::Bool(false), value);
    }

    #[test]
    fn test_from_anyvalue_datetime() {
        let any_value = AnyValue::Datetime(0, polars::prelude::TimeUnit::Microseconds, None);

        let value = MedRecordValue::try_from(any_value).unwrap();

        assert_eq!(
            MedRecordValue::DateTime(
                NaiveDateTime::parse_from_str("1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
            ),
            value
        );

        let any_value = AnyValue::Datetime(0, polars::prelude::TimeUnit::Milliseconds, None);

        let value = MedRecordValue::try_from(any_value).unwrap();

        assert_eq!(
            MedRecordValue::DateTime(
                NaiveDateTime::parse_from_str("1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
            ),
            value
        );

        let any_value = AnyValue::Datetime(0, polars::prelude::TimeUnit::Nanoseconds, None);

        let value = MedRecordValue::try_from(any_value).unwrap();

        assert_eq!(
            MedRecordValue::DateTime(
                NaiveDateTime::parse_from_str("1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").unwrap()
            ),
            value
        );
    }

    #[test]
    fn test_from_anyvalue_null() {
        let any_value = AnyValue::Null;

        let value = MedRecordValue::try_from(any_value).unwrap();

        assert_eq!(MedRecordValue::Null, value);
    }

    #[test]
    fn test_dataframe_to_nodes() {
        let s0 = Series::new("index".into(), &["0", "1"]);
        let s1 = Series::new("attribute".into(), &[1, 2]);
        let nodes_dataframe = DataFrame::new(vec![s0.into(), s1.into()]).unwrap();

        let nodes = dataframe_to_nodes(nodes_dataframe, "index").unwrap();

        assert_eq!(
            vec![
                ("0".into(), HashMap::from([("attribute".into(), 1.into())])),
                ("1".into(), HashMap::from([("attribute".into(), 2.into())]))
            ],
            nodes
        );
    }

    #[test]
    fn test_invalid_dataframe_to_nodes() {
        let s0 = Series::new("index".into(), &["0", "1"]);
        let s1 = Series::new("attribute".into(), &[1, 2]);
        let nodes_dataframe = DataFrame::new(vec![s0.into(), s1.into()]).unwrap();

        // Providing the wrong index column name should fail
        assert!(dataframe_to_nodes(nodes_dataframe, "wrong_column")
            .is_err_and(|e| matches!(e, MedRecordError::ConversionError(_))));
    }

    #[test]
    fn test_dataframe_to_edges() {
        let s0 = Series::new("source".into(), &["0", "1"]);
        let s1 = Series::new("target".into(), &["1", "0"]);
        let s2 = Series::new("attribute".into(), &[1, 2]);
        let edges_dataframe = DataFrame::new(vec![s0.into(), s1.into(), s2.into()]).unwrap();

        let edges = dataframe_to_edges(edges_dataframe, "source", "target").unwrap();

        assert_eq!(
            vec![
                (
                    "0".into(),
                    "1".into(),
                    HashMap::from([("attribute".into(), 1.into())])
                ),
                (
                    "1".into(),
                    "0".into(),
                    HashMap::from([("attribute".into(), 2.into())])
                )
            ],
            edges
        );
    }

    #[test]
    fn test_invalid_dataframe_to_edges() {
        let s0 = Series::new("source".into(), &["0", "1"]);
        let s1 = Series::new("target".into(), &["1", "0"]);
        let s2 = Series::new("attribute".into(), &[1, 2]);
        let edges_dataframe = DataFrame::new(vec![s0.into(), s1.into(), s2.into()]).unwrap();

        // Providing the wrong source index column name should fail
        assert!(
            dataframe_to_edges(edges_dataframe.clone(), "wrong_column", "target")
                .is_err_and(|e| matches!(e, MedRecordError::ConversionError(_)))
        );

        // Providing the wrong target index column name should fail
        assert!(
            dataframe_to_edges(edges_dataframe, "source", "wrong_column")
                .is_err_and(|e| matches!(e, MedRecordError::ConversionError(_)))
        );
    }
}
