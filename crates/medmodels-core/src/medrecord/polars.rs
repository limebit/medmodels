use crate::{
    errors::MedRecordError,
    medrecord::{Attributes, MedRecordAttribute, MedRecordValue, NodeIndex},
};
use polars::{datatypes::AnyValue, frame::DataFrame};

impl<'a> TryFrom<AnyValue<'a>> for MedRecordValue {
    type Error = MedRecordError;

    fn try_from(value: AnyValue<'a>) -> Result<Self, Self::Error> {
        match value {
            AnyValue::String(value) => Ok(MedRecordValue::String(value.into())),
            AnyValue::StringOwned(value) => Ok(MedRecordValue::String(value.into())),
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
            AnyValue::Null => Ok(MedRecordValue::Null),
            _ => Err(MedRecordError::ConversionError(format!(
                "Cannot convert {} into MedRecordValue",
                value
            ))),
        }
    }
}

impl<'a> TryFrom<AnyValue<'a>> for MedRecordAttribute {
    type Error = MedRecordError;

    fn try_from(value: AnyValue<'a>) -> Result<Self, Self::Error> {
        match value {
            AnyValue::String(value) => Ok(MedRecordAttribute::String(value.into())),
            AnyValue::StringOwned(value) => Ok(MedRecordAttribute::String(value.into())),
            AnyValue::Int8(value) => Ok(MedRecordAttribute::Int(value.into())),
            AnyValue::Int16(value) => Ok(MedRecordAttribute::Int(value.into())),
            AnyValue::Int32(value) => Ok(MedRecordAttribute::Int(value.into())),
            AnyValue::Int64(value) => Ok(MedRecordAttribute::Int(value)),
            AnyValue::UInt8(value) => Ok(MedRecordAttribute::Int(value.into())),
            AnyValue::UInt16(value) => Ok(MedRecordAttribute::Int(value.into())),
            AnyValue::UInt32(value) => Ok(MedRecordAttribute::Int(value.into())),
            _ => Err(MedRecordError::ConversionError(format!(
                "Cannot convert {} into MedRecordAttribute",
                value
            ))),
        }
    }
}

pub(crate) fn dataframe_to_nodes(
    nodes: DataFrame,
    index_column_name: &str,
) -> Result<Vec<(NodeIndex, Attributes)>, MedRecordError> {
    let attribute_column_names = nodes
        .get_column_names()
        .into_iter()
        .filter(|name| *name != index_column_name)
        .collect::<Vec<_>>();

    let index = nodes
        .column(index_column_name)
        .map_err(|_| {
            MedRecordError::ConversionError(format!(
                "Cannot find column with name {} in dataframe",
                index_column_name
            ))
        })?
        .iter();

    let mut columns = nodes
        .columns(&attribute_column_names)
        .expect("Attribute columns must exist")
        .iter()
        .map(|s| s.iter())
        .zip(attribute_column_names)
        .collect::<Vec<_>>();

    index
        .map(|index_value| {
            Ok((
                index_value.try_into()?,
                columns
                    .iter_mut()
                    .map(|(column, column_name)| {
                        Ok((
                            (*column_name).into(),
                            column.next().expect("msg").try_into()?,
                        ))
                    })
                    .collect::<Result<_, MedRecordError>>()?,
            ))
        })
        .collect()
}

pub(crate) fn dataframe_to_edges(
    edges: DataFrame,
    from_index_column_name: &str,
    to_index_column_name: &str,
) -> Result<Vec<(NodeIndex, NodeIndex, Attributes)>, MedRecordError> {
    let attribute_column_names = edges
        .get_column_names()
        .into_iter()
        .filter(|name| *name != from_index_column_name && *name != to_index_column_name)
        .collect::<Vec<_>>();

    let from_index = edges
        .column(from_index_column_name)
        .map_err(|_| {
            MedRecordError::ConversionError(format!(
                "Cannot find column with name {} in dataframe",
                from_index_column_name
            ))
        })?
        .iter();
    let to_index = edges
        .column(to_index_column_name)
        .map_err(|_| {
            MedRecordError::ConversionError(format!(
                "Cannot find column with name {} in dataframe",
                to_index_column_name
            ))
        })?
        .iter();

    let mut columns = edges
        .columns(&attribute_column_names)
        .expect("Attribute columns must exist")
        .iter()
        .map(|s| s.iter())
        .zip(attribute_column_names)
        .collect::<Vec<_>>();

    from_index
        .zip(to_index)
        .map(|(from_index_value, to_index_value)| {
            Ok((
                from_index_value.try_into()?,
                to_index_value.try_into()?,
                columns
                    .iter_mut()
                    .map(|(column, column_name)| {
                        Ok((
                            (*column_name).into(),
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
    fn test_from_anyvalue_null() {
        let any_value = AnyValue::Null;

        let value = MedRecordValue::try_from(any_value).unwrap();

        assert_eq!(MedRecordValue::Null, value);
    }

    #[test]
    fn test_dataframe_to_nodes() {
        let s0 = Series::new("index", &["0", "1"]);
        let s1 = Series::new("attribute", &[1, 2]);
        let nodes_dataframe = DataFrame::new(vec![s0, s1]).unwrap();

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
        let s0 = Series::new("index", &["0", "1"]);
        let s1 = Series::new("attribute", &[1, 2]);
        let nodes_dataframe = DataFrame::new(vec![s0, s1]).unwrap();

        // Providing the wrong index column name should fail
        assert!(dataframe_to_nodes(nodes_dataframe, "wrong_column")
            .is_err_and(|e| matches!(e, MedRecordError::ConversionError(_))));
    }

    #[test]
    fn test_dataframe_to_edges() {
        let s0 = Series::new("from", &["0", "1"]);
        let s1 = Series::new("to", &["1", "0"]);
        let s2 = Series::new("attribute", &[1, 2]);
        let edges_dataframe = DataFrame::new(vec![s0, s1, s2]).unwrap();

        let edges = dataframe_to_edges(edges_dataframe, "from", "to").unwrap();

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
        let s0 = Series::new("from", &["0", "1"]);
        let s1 = Series::new("to", &["1", "0"]);
        let s2 = Series::new("attribute", &[1, 2]);
        let edges_dataframe = DataFrame::new(vec![s0, s1, s2]).unwrap();

        // Providing the wrong from index column name should fail
        assert!(
            dataframe_to_edges(edges_dataframe.clone(), "wrong_column", "to")
                .is_err_and(|e| matches!(e, MedRecordError::ConversionError(_)))
        );

        // Providing the wrong to index column name should fail
        assert!(dataframe_to_edges(edges_dataframe, "from", "wrong_column")
            .is_err_and(|e| matches!(e, MedRecordError::ConversionError(_))));
    }
}
