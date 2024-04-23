use std::{
    error::Error,
    fmt::{Display, Formatter, Result},
};

#[derive(Debug)]
pub enum GraphError {
    IndexError(String),
}

impl Error for GraphError {
    fn description(&self) -> &str {
        match self {
            GraphError::IndexError(message) => message,
        }
    }
}

impl Display for GraphError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            GraphError::IndexError(message) => write!(f, "IndexError: {}", message),
        }
    }
}
