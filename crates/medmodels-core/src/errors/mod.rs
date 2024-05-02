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
