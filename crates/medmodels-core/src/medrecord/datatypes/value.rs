use super::{
    Ceil, Contains, EndsWith, Floor, Lowercase, Round, Slice, StartsWith, Trim, TrimEnd, TrimStart,
    Uppercase,
};
use crate::errors::MedRecordError;
use medmodels_utils::implement_from_for_wrapper;
use std::{
    cmp::Ordering,
    fmt::Display,
    ops::{Add, Div, Mul, Range, Sub},
};

#[derive(Debug, Clone)]
pub enum MedRecordValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl Add for MedRecordValue {
    type Output = Result<Self, MedRecordError>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MedRecordValue::String(value), MedRecordValue::String(rhs)) => {
                Ok(MedRecordValue::String(value + rhs.as_str()))
            }
            (MedRecordValue::String(value), MedRecordValue::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {} to {}", rhs, value)),
            ),
            (MedRecordValue::String(value), MedRecordValue::Float(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {} to {}", rhs, value)),
            ),
            (MedRecordValue::String(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {} to {}", rhs, value)),
            ),
            (MedRecordValue::Int(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {} to {}", rhs, value)),
            ),
            (MedRecordValue::Int(value), MedRecordValue::Int(rhs)) => {
                Ok(MedRecordValue::Int(value + rhs))
            }
            (MedRecordValue::Int(value), MedRecordValue::Float(rhs)) => {
                Ok(MedRecordValue::Float(value as f64 + rhs))
            }
            (MedRecordValue::Int(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {} to {}", rhs, value)),
            ),
            (MedRecordValue::Float(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {} to {}", rhs, value)),
            ),
            (MedRecordValue::Float(value), MedRecordValue::Int(rhs)) => {
                Ok(MedRecordValue::Float(value + rhs as f64))
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(rhs)) => {
                Ok(MedRecordValue::Float(value + rhs))
            }
            (MedRecordValue::Float(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {} to {}", rhs, value)),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {} to {}", rhs, value)),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {} to {}", rhs, value)),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Float(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {} to {}", rhs, value)),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {} to {}", rhs, value)),
            ),
        }
    }
}

impl Sub for MedRecordValue {
    type Output = Result<Self, MedRecordError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MedRecordValue::String(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {} from {}", rhs, value)),
            ),
            (MedRecordValue::String(value), MedRecordValue::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {} from {}", rhs, value)),
            ),
            (MedRecordValue::String(value), MedRecordValue::Float(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {} from {}", rhs, value)),
            ),
            (MedRecordValue::String(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {} from {}", rhs, value)),
            ),
            (MedRecordValue::Int(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {} from {}", rhs, value)),
            ),
            (MedRecordValue::Int(value), MedRecordValue::Int(rhs)) => {
                Ok(MedRecordValue::Int(value - rhs))
            }
            (MedRecordValue::Int(value), MedRecordValue::Float(rhs)) => {
                Ok(MedRecordValue::Float(value as f64 - rhs))
            }
            (MedRecordValue::Int(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {} from {}", rhs, value)),
            ),
            (MedRecordValue::Float(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {} from {}", rhs, value)),
            ),
            (MedRecordValue::Float(value), MedRecordValue::Int(rhs)) => {
                Ok(MedRecordValue::Float(value - rhs as f64))
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(rhs)) => {
                Ok(MedRecordValue::Float(value - rhs))
            }
            (MedRecordValue::Float(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {} from {}", rhs, value)),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {} from {}", rhs, value)),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {} from {}", rhs, value)),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Float(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {} from {}", rhs, value)),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {} from {}", rhs, value)),
            ),
        }
    }
}

impl Mul for MedRecordValue {
    type Output = Result<Self, MedRecordError>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Int(other)) => {
                let mut result = String::new();

                for _ in 0..other {
                    result.push_str(&value)
                }

                Ok(MedRecordValue::String(result))
            }
            (MedRecordValue::String(value), MedRecordValue::Float(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Int(value), MedRecordValue::String(other)) => {
                let mut result = String::new();

                for _ in 0..value {
                    result.push_str(&other)
                }

                Ok(MedRecordValue::String(result))
            }
            (MedRecordValue::Int(value), MedRecordValue::Int(other)) => {
                Ok(MedRecordValue::Int(value * other))
            }
            (MedRecordValue::Int(value), MedRecordValue::Float(other)) => {
                Ok(MedRecordValue::Float(value as f64 * other))
            }
            (MedRecordValue::Int(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::Int(other)) => {
                Ok(MedRecordValue::Float(value * other as f64))
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => {
                Ok(MedRecordValue::Float(value * other))
            }
            (MedRecordValue::Float(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Int(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Float(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
        }
    }
}

impl Div for MedRecordValue {
    type Output = Result<Self, MedRecordError>;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Int(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Float(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Int(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Int(value), MedRecordValue::Int(other)) => {
                Ok(MedRecordValue::Float(value as f64 / other as f64))
            }
            (MedRecordValue::Int(value), MedRecordValue::Float(other)) => {
                Ok(MedRecordValue::Float(value as f64 * other))
            }
            (MedRecordValue::Int(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::Int(other)) => {
                Ok(MedRecordValue::Float(value / other as f64))
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => {
                Ok(MedRecordValue::Float(value / other))
            }
            (MedRecordValue::Float(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Int(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Float(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value, other
                )))
            }
        }
    }
}

impl StartsWith for MedRecordValue {
    fn starts_with(&self, other: &Self) -> bool {
        match (self, other) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => {
                value.starts_with(other)
            }
            (MedRecordValue::String(value), MedRecordValue::Int(other)) => {
                value.starts_with(&other.to_string())
            }
            (MedRecordValue::String(value), MedRecordValue::Float(other)) => {
                value.starts_with(&other.to_string())
            }
            (MedRecordValue::Int(value), MedRecordValue::String(other)) => {
                value.to_string().starts_with(other)
            }
            (MedRecordValue::Int(value), MedRecordValue::Int(other)) => {
                value.to_string().starts_with(&other.to_string())
            }
            (MedRecordValue::Float(value), MedRecordValue::String(other)) => {
                value.to_string().starts_with(other)
            }
            (MedRecordValue::Float(value), MedRecordValue::Int(other)) => {
                value.to_string().starts_with(&other.to_string())
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => {
                value.to_string().starts_with(&other.to_string())
            }
            _ => false,
        }
    }
}

impl EndsWith for MedRecordValue {
    fn ends_with(&self, other: &Self) -> bool {
        match (self, other) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => {
                value.ends_with(other)
            }
            (MedRecordValue::String(value), MedRecordValue::Int(other)) => {
                value.ends_with(&other.to_string())
            }
            (MedRecordValue::String(value), MedRecordValue::Float(other)) => {
                value.ends_with(&other.to_string())
            }
            (MedRecordValue::Int(value), MedRecordValue::String(other)) => {
                value.to_string().ends_with(other)
            }
            (MedRecordValue::Int(value), MedRecordValue::Int(other)) => {
                value.to_string().ends_with(&other.to_string())
            }
            (MedRecordValue::Float(value), MedRecordValue::String(other)) => {
                value.to_string().ends_with(other)
            }
            (MedRecordValue::Float(value), MedRecordValue::Int(other)) => {
                value.to_string().ends_with(&other.to_string())
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => {
                value.to_string().ends_with(&other.to_string())
            }
            _ => false,
        }
    }
}

impl Contains for MedRecordValue {
    fn contains(&self, other: &Self) -> bool {
        match (self, other) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => value.contains(other),
            (MedRecordValue::String(value), MedRecordValue::Int(other)) => {
                value.contains(&other.to_string())
            }
            (MedRecordValue::String(value), MedRecordValue::Float(other)) => {
                value.contains(&other.to_string())
            }
            (MedRecordValue::Int(value), MedRecordValue::String(other)) => {
                value.to_string().contains(other)
            }
            (MedRecordValue::Int(value), MedRecordValue::Int(other)) => {
                value.to_string().contains(&other.to_string())
            }
            (MedRecordValue::Float(value), MedRecordValue::String(other)) => {
                value.to_string().contains(other)
            }
            (MedRecordValue::Float(value), MedRecordValue::Int(other)) => {
                value.to_string().contains(&other.to_string())
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => {
                value.to_string().contains(&other.to_string())
            }
            _ => false,
        }
    }
}

impl Slice for MedRecordValue {
    fn slice(self, range: Range<usize>) -> Self {
        match self {
            MedRecordValue::String(value) => value[range].into(),
            MedRecordValue::Int(value) => value.to_string()[range].into(),
            MedRecordValue::Float(value) => value.to_string()[range].into(),
            MedRecordValue::Bool(value) => value.to_string()[range].into(),
        }
    }
}

impl Round for MedRecordValue {
    fn round(self) -> Self {
        match self {
            MedRecordValue::Float(value) => MedRecordValue::Float(value.round()),
            _ => self,
        }
    }
}

impl Ceil for MedRecordValue {
    fn ceil(self) -> Self {
        match self {
            MedRecordValue::Float(value) => MedRecordValue::Float(value.ceil()),
            _ => self,
        }
    }
}

impl Floor for MedRecordValue {
    fn floor(self) -> Self {
        match self {
            MedRecordValue::Float(value) => MedRecordValue::Float(value.floor()),
            _ => self,
        }
    }
}

impl Trim for MedRecordValue {
    fn trim(self) -> Self {
        match self {
            MedRecordValue::String(value) => MedRecordValue::String(value.trim().to_string()),
            _ => self,
        }
    }
}

impl TrimStart for MedRecordValue {
    fn trim_start(self) -> Self {
        match self {
            MedRecordValue::String(value) => MedRecordValue::String(value.trim_start().to_string()),
            _ => self,
        }
    }
}

impl TrimEnd for MedRecordValue {
    fn trim_end(self) -> Self {
        match self {
            MedRecordValue::String(value) => MedRecordValue::String(value.trim_end().to_string()),
            _ => self,
        }
    }
}

impl Lowercase for MedRecordValue {
    fn lowercase(self) -> Self {
        match self {
            MedRecordValue::String(value) => MedRecordValue::String(value.to_lowercase()),
            _ => self,
        }
    }
}

impl Uppercase for MedRecordValue {
    fn uppercase(self) -> Self {
        match self {
            MedRecordValue::String(value) => MedRecordValue::String(value.to_uppercase()),
            _ => self,
        }
    }
}

impl From<&str> for MedRecordValue {
    fn from(value: &str) -> Self {
        value.to_string().into()
    }
}

implement_from_for_wrapper!(MedRecordValue, String, String);
implement_from_for_wrapper!(MedRecordValue, i64, Int);
implement_from_for_wrapper!(MedRecordValue, f64, Float);
implement_from_for_wrapper!(MedRecordValue, bool, Bool);

impl PartialEq for MedRecordValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => value == other,
            (MedRecordValue::Int(value), MedRecordValue::Int(other)) => value == other,
            (MedRecordValue::Int(value), MedRecordValue::Float(other)) => &(*value as f64) == other,
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => value == other,
            (MedRecordValue::Float(value), MedRecordValue::Int(other)) => value == &(*other as f64),
            (MedRecordValue::Bool(value), MedRecordValue::Bool(other)) => value == other,
            _ => false,
        }
    }
}

impl PartialOrd for MedRecordValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => {
                Some(value.cmp(other))
            }
            (MedRecordValue::Int(value), MedRecordValue::Int(other)) => Some(value.cmp(other)),
            (MedRecordValue::Int(value), MedRecordValue::Float(other)) => {
                other.partial_cmp(&(*value as f64))
            }
            (MedRecordValue::Float(value), MedRecordValue::Int(other)) => {
                value.partial_cmp(&(*other as f64))
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => {
                value.partial_cmp(other)
            }
            (MedRecordValue::Bool(value), MedRecordValue::Bool(other)) => Some(value.cmp(other)),
            _ => None,
        }
    }
}

impl Display for MedRecordValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(value) => write!(f, "{}", value),
            Self::Int(value) => write!(f, "{}", value),
            Self::Float(value) => write!(f, "{}", value),
            Self::Bool(value) => write!(f, "{}", value),
        }
    }
}

#[cfg(test)]
mod test {
    use super::MedRecordValue;

    #[test]
    fn test_from_string() {
        let value = MedRecordValue::from("value".to_string());

        assert_eq!(MedRecordValue::String("value".to_string()), value);
    }

    #[test]
    fn test_from_i64() {
        let value = MedRecordValue::from(0_i64);

        assert_eq!(MedRecordValue::Int(0), value);
    }

    #[test]
    fn test_from_f64() {
        let value = MedRecordValue::from(0_f64);

        assert_eq!(MedRecordValue::Float(0.0), value);
    }

    #[test]
    fn test_from_bool() {
        let value = MedRecordValue::from(false);

        assert_eq!(MedRecordValue::Bool(false), value);
    }

    #[test]
    fn test_from_str() {
        let value = MedRecordValue::from("value");

        assert_eq!(MedRecordValue::String("value".to_string()), value)
    }

    #[test]
    fn test_partial_eq() {
        assert!(MedRecordValue::from(0_i64) == MedRecordValue::from(0_i64));
        assert!(MedRecordValue::from(1_i64) != MedRecordValue::from(0_i64));

        assert!(MedRecordValue::from(0_f64) == MedRecordValue::from(0_f64));
        assert!(MedRecordValue::from(1_f64) != MedRecordValue::from(0_f64));

        assert!(
            MedRecordValue::from("value".to_string()) == MedRecordValue::from("value".to_string())
        );
        assert!(
            MedRecordValue::from("value2".to_string()) != MedRecordValue::from("value".to_string())
        );

        assert!(MedRecordValue::from(false) == MedRecordValue::from(false));
        assert!(MedRecordValue::from(true) != MedRecordValue::from(false));
    }

    #[test]
    fn test_partial_ord() {
        // assert!(MedRecordValue::from(0_f64) < MedRecordValue::from(1_f64));
        // assert!(MedRecordValue::from(1_i64)  MedRecordValue::from(0_i64));
        let test = "0.1".to_string().parse::<f64>();
        println!("{:?}", test)
    }

    #[test]
    fn test_display() {
        assert_eq!(
            "value",
            MedRecordValue::from("value".to_string()).to_string()
        );

        assert_eq!("0", MedRecordValue::from(0_i64).to_string());

        assert_eq!("0.5", MedRecordValue::from(0.5).to_string());

        assert_eq!("false", MedRecordValue::from(false).to_string());
    }
}
