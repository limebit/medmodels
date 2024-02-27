use medmodels_utils::implement_from_for_wrapper;
use std::{fmt::Display, hash::Hash};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MedRecordAttribute {
    String(String),
    Int(i64),
}

impl Hash for MedRecordAttribute {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Int(value) => value.hash(state),
            Self::String(value) => value.hash(state),
        }
    }
}

impl From<&str> for MedRecordAttribute {
    fn from(value: &str) -> Self {
        value.to_string().into()
    }
}

impl Display for MedRecordAttribute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(value) => write!(f, "{}", value),
            Self::Int(value) => write!(f, "{}", value),
        }
    }
}

implement_from_for_wrapper!(MedRecordAttribute, String, String);
implement_from_for_wrapper!(MedRecordAttribute, i64, Int);
