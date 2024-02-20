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
