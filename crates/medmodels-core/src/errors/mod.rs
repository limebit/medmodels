mod graph;
mod medrecord;

pub use graph::GraphError;
pub use medrecord::MedRecordError;

impl From<GraphError> for MedRecordError {
    fn from(value: GraphError) -> Self {
        match value {
            GraphError::IndexError(value) => Self::IndexError(value),
            GraphError::AssertionError(value) => Self::AssertionError(value),
            GraphError::SchemaError(value) => Self::SchemaError(value),
        }
    }
}

pub type MedRecordResult<T> = Result<T, MedRecordError>;

#[cfg(test)]
mod test {
    use super::{GraphError, MedRecordError};

    #[test]
    fn test_from() {
        assert_eq!(
            MedRecordError::IndexError("value".to_string()),
            MedRecordError::from(GraphError::IndexError("value".to_string()))
        );
        assert_eq!(
            MedRecordError::AssertionError("value".to_string()),
            MedRecordError::from(GraphError::AssertionError("value".to_string()))
        );
        assert_eq!(
            MedRecordError::SchemaError("value".to_string()),
            MedRecordError::from(GraphError::SchemaError("value".to_string()))
        );
    }
}
