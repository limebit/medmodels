use std::{
    error::Error,
    fmt::{Display, Formatter, Result},
};

#[derive(Debug)]
pub enum MedRecordError {
    IndexError(String),
    ConversionError(String),
    AssertionError(String),
}

impl Error for MedRecordError {
    fn description(&self) -> &str {
        match self {
            MedRecordError::IndexError(message) => message,
            MedRecordError::ConversionError(message) => message,
            MedRecordError::AssertionError(message) => message,
        }
    }
}

impl Display for MedRecordError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            MedRecordError::IndexError(message) => write!(f, "{}", message),
            MedRecordError::ConversionError(message) => write!(f, "{}", message),
            MedRecordError::AssertionError(message) => write!(f, "{}", message),
        }
    }
}
