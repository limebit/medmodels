use std::{
    error::Error,
    fmt::{Display, Formatter, Result},
};

#[derive(Debug)]
pub enum MedRecordError {
    IndexError(String),
    KeyError(String),
    ConversionError(String),
    AssertionError(String),
}

impl Error for MedRecordError {
    fn description(&self) -> &str {
        match self {
            MedRecordError::IndexError(message) => message,
            MedRecordError::KeyError(message) => message,
            MedRecordError::ConversionError(message) => message,
            MedRecordError::AssertionError(message) => message,
        }
    }
}

impl Display for MedRecordError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            Self::IndexError(message) => write!(f, "IndexError: {}", message),
            Self::KeyError(message) => write!(f, "KeyError: {}", message),
            Self::ConversionError(message) => write!(f, "ConversionError: {}", message),
            Self::AssertionError(message) => write!(f, "AssertionError: {}", message),
        }
    }
}
