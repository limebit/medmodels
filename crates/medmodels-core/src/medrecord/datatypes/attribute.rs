use super::{Contains, EndsWith, MedRecordValue, StartsWith};
use crate::errors::MedRecordError;
use medmodels_utils::implement_from_for_wrapper;
use std::{cmp::Ordering, fmt::Display, hash::Hash};

#[derive(Debug, Clone)]
pub enum MedRecordAttribute {
    String(String),
    Int(i64),
}

impl StartsWith for MedRecordAttribute {
    fn starts_with(&self, other: &Self) -> bool {
        match (self, other) {
            (MedRecordAttribute::String(value), MedRecordAttribute::String(other)) => {
                value.starts_with(other)
            }
            (MedRecordAttribute::String(value), MedRecordAttribute::Int(other)) => {
                value.starts_with(&other.to_string())
            }
            (MedRecordAttribute::Int(value), MedRecordAttribute::String(other)) => {
                value.to_string().starts_with(other)
            }
            (MedRecordAttribute::Int(value), MedRecordAttribute::Int(other)) => {
                value.to_string().starts_with(&other.to_string())
            }
        }
    }
}

impl EndsWith for MedRecordAttribute {
    fn ends_with(&self, other: &Self) -> bool {
        match (self, other) {
            (MedRecordAttribute::String(value), MedRecordAttribute::String(other)) => {
                value.ends_with(other)
            }
            (MedRecordAttribute::String(value), MedRecordAttribute::Int(other)) => {
                value.ends_with(&other.to_string())
            }
            (MedRecordAttribute::Int(value), MedRecordAttribute::String(other)) => {
                value.to_string().ends_with(other)
            }
            (MedRecordAttribute::Int(value), MedRecordAttribute::Int(other)) => {
                value.to_string().ends_with(&other.to_string())
            }
        }
    }
}

impl Contains for MedRecordAttribute {
    fn contains(&self, other: &Self) -> bool {
        match (self, other) {
            (MedRecordAttribute::String(value), MedRecordAttribute::String(other)) => {
                value.contains(other)
            }
            (MedRecordAttribute::String(value), MedRecordAttribute::Int(other)) => {
                value.contains(&other.to_string())
            }
            (MedRecordAttribute::Int(value), MedRecordAttribute::String(other)) => {
                value.to_string().contains(other)
            }
            (MedRecordAttribute::Int(value), MedRecordAttribute::Int(other)) => {
                value.to_string().contains(&other.to_string())
            }
        }
    }
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

impl TryFrom<MedRecordValue> for MedRecordAttribute {
    type Error = MedRecordError;

    fn try_from(value: MedRecordValue) -> Result<Self, Self::Error> {
        match value {
            MedRecordValue::String(value) => Ok(MedRecordAttribute::String(value)),
            MedRecordValue::Int(value) => Ok(MedRecordAttribute::Int(value)),
            _ => Err(MedRecordError::ConversionError(format!(
                "Cannot convert {} into MedRecordAttribute",
                value
            ))),
        }
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

impl PartialEq for MedRecordAttribute {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MedRecordAttribute::String(value), MedRecordAttribute::String(other)) => {
                value == other
            }
            (MedRecordAttribute::Int(value), MedRecordAttribute::Int(other)) => value == other,
            _ => false,
        }
    }
}

impl Eq for MedRecordAttribute {}

impl PartialOrd for MedRecordAttribute {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (MedRecordAttribute::String(value), MedRecordAttribute::String(other)) => {
                Some(value.cmp(other))
            }
            (MedRecordAttribute::Int(value), MedRecordAttribute::Int(other)) => {
                Some(value.cmp(other))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::MedRecordAttribute;

    #[test]
    fn test_from_string() {
        let attribute = MedRecordAttribute::from("value".to_string());

        assert_eq!(MedRecordAttribute::String("value".to_string()), attribute);
    }

    #[test]
    fn test_from_i64() {
        let attribute = MedRecordAttribute::from(0_i64);

        assert_eq!(MedRecordAttribute::Int(0), attribute);
    }

    #[test]
    fn test_from_str() {
        let attribute = MedRecordAttribute::from("value");

        assert_eq!(MedRecordAttribute::String("value".to_string()), attribute)
    }

    #[test]
    fn test_partial_eq() {
        assert!(MedRecordAttribute::from(0_i64) == MedRecordAttribute::from(0_i64));
        assert!(MedRecordAttribute::from(1_i64) != MedRecordAttribute::from(0_i64));

        assert!(
            MedRecordAttribute::from("attribute".to_string())
                == MedRecordAttribute::from("attribute".to_string())
        );
        assert!(
            MedRecordAttribute::from("attribute2".to_string())
                != MedRecordAttribute::from("attribute".to_string())
        );
    }

    #[test]
    fn test_display() {
        assert_eq!(
            "value",
            MedRecordAttribute::from("value".to_string()).to_string()
        );

        assert_eq!("0", MedRecordAttribute::from(0_i64).to_string());
    }
}
