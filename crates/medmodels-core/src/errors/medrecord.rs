use std::{
    error::Error,
    fmt::{Display, Formatter, Result},
};

#[derive(Debug, PartialEq)]
pub enum MedRecordError {
    IndexError(String),
    KeyError(String),
    ConversionError(String),
    AssertionError(String),
    SchemaError(String),
    QueryError(String),
}

impl Error for MedRecordError {
    fn description(&self) -> &str {
        match self {
            MedRecordError::IndexError(message) => message,
            MedRecordError::KeyError(message) => message,
            MedRecordError::ConversionError(message) => message,
            MedRecordError::AssertionError(message) => message,
            MedRecordError::SchemaError(message) => message,
            MedRecordError::QueryError(message) => message,
        }
    }
}

impl Display for MedRecordError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::IndexError(message) => write!(f, "IndexError: {message}"),
            Self::KeyError(message) => write!(f, "KeyError: {message}"),
            Self::ConversionError(message) => write!(f, "ConversionError: {message}"),
            Self::AssertionError(message) => write!(f, "AssertionError: {message}"),
            Self::SchemaError(message) => write!(f, "SchemaError: {message}"),
            Self::QueryError(message) => write!(f, "QueryError: {message}"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::MedRecordError;

    #[test]
    fn test_display() {
        assert_eq!(
            "IndexError: value",
            MedRecordError::IndexError("value".to_string()).to_string()
        );
        assert_eq!(
            "KeyError: value",
            MedRecordError::KeyError("value".to_string()).to_string()
        );
        assert_eq!(
            "ConversionError: value",
            MedRecordError::ConversionError("value".to_string()).to_string()
        );
        assert_eq!(
            "AssertionError: value",
            MedRecordError::AssertionError("value".to_string()).to_string()
        );
        assert_eq!(
            "SchemaError: value",
            MedRecordError::SchemaError("value".to_string()).to_string()
        );
    }
}
