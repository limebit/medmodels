use std::{
    error::Error,
    fmt::{Display, Formatter, Result},
};

#[derive(Debug)]
pub enum GraphError {
    IndexError(String),
    AssertionError(String),
    SchemaError(String),
}

impl Error for GraphError {
    fn description(&self) -> &str {
        match self {
            GraphError::IndexError(message) => message,
            GraphError::AssertionError(message) => message,
            GraphError::SchemaError(message) => message,
        }
    }
}

impl Display for GraphError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            GraphError::IndexError(message) => write!(f, "IndexError: {message}"),
            GraphError::AssertionError(message) => write!(f, "AssertionError: {message}"),
            GraphError::SchemaError(message) => write!(f, "SchemaError: {message}"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::GraphError;

    #[test]
    fn test_display() {
        assert_eq!(
            "IndexError: value",
            GraphError::IndexError("value".to_string()).to_string()
        );
        assert_eq!(
            "AssertionError: value",
            GraphError::AssertionError("value".to_string()).to_string()
        );
        assert_eq!(
            "SchemaError: value",
            GraphError::SchemaError("value".to_string()).to_string()
        );
    }
}
