use crate::{
    errors::MedRecordError,
    medrecord::{Dictionary, MedRecordValue},
};
use polars::{datatypes::AnyValue, frame::DataFrame};

impl<'a> TryFrom<AnyValue<'a>> for MedRecordValue {
    type Error = MedRecordError;

    fn try_from(value: AnyValue<'a>) -> Result<Self, Self::Error> {
        match value {
            AnyValue::String(value) => Ok(MedRecordValue::String(value.to_string())),
            AnyValue::Int8(value) => Ok(MedRecordValue::Int(value.into())),
            AnyValue::Int16(value) => Ok(MedRecordValue::Int(value.into())),
            AnyValue::Int32(value) => Ok(MedRecordValue::Int(value.into())),
            AnyValue::Int64(value) => Ok(MedRecordValue::Int(value)),
            AnyValue::Float32(value) => Ok(MedRecordValue::Float(value.into())),
            AnyValue::Float64(value) => Ok(MedRecordValue::Float(value)),
            AnyValue::Boolean(value) => Ok(MedRecordValue::Bool(value)),
            _ => Err(MedRecordError::ConversionError(format!(
                "Could not convert {} into MedRecordValue",
                value
            ))),
        }
    }
}

pub(crate) fn dataframe_to_nodes(
    nodes: DataFrame,
    index_column_name: &str,
) -> Result<Vec<(String, Dictionary)>, MedRecordError> {
    let index_column = nodes
        .column(index_column_name)
        .map_err(|_| {
            MedRecordError::ConversionError(format!(
                "Could not find column with name {} in dataframe",
                index_column_name
            ))
        })?
        .str()
        .map_err(|_| {
            MedRecordError::ConversionError("Could not convert index column to utf8".to_string())
        })?;

    let attribute_column_names = nodes
        .get_column_names()
        .into_iter()
        .filter(|name| *name != index_column_name)
        .collect::<Vec<_>>();

    let attribute_columns = nodes
        .columns(attribute_column_names.clone())
        .expect("Attribute columns need to exist");

    index_column
        .into_iter()
        .enumerate()
        .map(|(row_index, nodex_index)| {
            let id = nodex_index.ok_or(MedRecordError::ConversionError(
                "Failed to read id in index column".to_string(),
            ))?;

            Ok((
                id.to_string(),
                attribute_column_names
                    .iter()
                    .zip(attribute_columns.clone())
                    .map(|(column_name, column)| {
                        Ok((
                            column_name.to_string(),
                            column
                                .get(row_index)
                                .expect("Entry needs to exist")
                                .try_into()?,
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
) -> Result<Vec<(String, String, Dictionary)>, MedRecordError> {
    let from_index_column = edges
        .column(from_index_column_name)
        .map_err(|_| {
            MedRecordError::ConversionError(format!(
                "Could not find column with name {} in dataframe",
                from_index_column_name
            ))
        })?
        .str()
        .map_err(|_| {
            MedRecordError::ConversionError("Could not convert index column to string".to_string())
        })?;
    let to_index_column = edges
        .column(to_index_column_name)
        .map_err(|_| {
            MedRecordError::ConversionError(format!(
                "Could not find column with name {} in dataframe",
                to_index_column_name
            ))
        })?
        .str()
        .map_err(|_| {
            MedRecordError::ConversionError("Could not convert index column to string".to_string())
        })?;

    let attribute_column_names = edges
        .get_column_names()
        .into_iter()
        .filter(|name| *name != from_index_column_name && *name != to_index_column_name)
        .collect::<Vec<_>>();

    let attribute_columns = edges
        .columns(attribute_column_names.clone())
        .expect("Attribute columns need to exist");

    from_index_column
        .into_iter()
        .zip(to_index_column)
        .enumerate()
        .map(|(row_index, (from_nodex_index, to_node_index))| {
            let from_id = from_nodex_index.ok_or(MedRecordError::ConversionError(
                "Failed to read id in from index column".to_string(),
            ))?;
            let to_id = to_node_index.ok_or(MedRecordError::ConversionError(
                "Failed to read id in to index column".to_string(),
            ))?;

            Ok((
                from_id.to_string(),
                to_id.to_string(),
                attribute_column_names
                    .iter()
                    .zip(attribute_columns.clone())
                    .map(|(column_name, column)| {
                        Ok((
                            column_name.to_string(),
                            column
                                .get(row_index)
                                .expect("Entry needs to exist")
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
    use std::collections::HashMap;

    use crate::{errors::MedRecordError, polars::dataframe_to_edges};

    use super::{dataframe_to_nodes, MedRecordValue};
    use polars::prelude::*;

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
    fn test_dataframe_to_nodes() {
        let s0 = Series::new("index", &["0", "1"]);
        let s1 = Series::new("attribute", &[1, 2]);
        let nodes_dataframe = DataFrame::new(vec![s0, s1]).unwrap();

        let nodes = dataframe_to_nodes(nodes_dataframe, "index").unwrap();

        assert_eq!(
            vec![
                (
                    "0".to_string(),
                    HashMap::from([("attribute".to_string(), 1.into())])
                ),
                (
                    "1".to_string(),
                    HashMap::from([("attribute".to_string(), 2.into())])
                )
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

        let s0 = Series::new("index", &[1, 2]);
        let s1 = Series::new("attribute", &[1, 2]);
        let nodes_dataframe = DataFrame::new(vec![s0, s1]).unwrap();

        // The index column should be a string column, otherwise fail
        assert!(dataframe_to_nodes(nodes_dataframe, "index")
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
                    "0".to_string(),
                    "1".to_string(),
                    HashMap::from([("attribute".to_string(), 1.into())])
                ),
                (
                    "1".to_string(),
                    "0".to_string(),
                    HashMap::from([("attribute".to_string(), 2.into())])
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

        let s0 = Series::new("from", &[0, 1]);
        let s1 = Series::new("to", &["1", "0"]);
        let s2 = Series::new("attribute", &[1, 2]);
        let edges_dataframe = DataFrame::new(vec![s0, s1, s2]).unwrap();

        // The from index column should be a string column, otherwise fail
        assert!(dataframe_to_edges(edges_dataframe, "from", "to")
            .is_err_and(|e| matches!(e, MedRecordError::ConversionError(_))));

        let s0 = Series::new("from", &["0", "1"]);
        let s1 = Series::new("to", &[1, 0]);
        let s2 = Series::new("attribute", &[1, 2]);
        let edges_dataframe = DataFrame::new(vec![s0, s1, s2]).unwrap();

        // The to index column should be a string column, otherwise fail
        assert!(dataframe_to_edges(edges_dataframe, "from", "to")
            .is_err_and(|e| matches!(e, MedRecordError::ConversionError(_))));
    }
}
