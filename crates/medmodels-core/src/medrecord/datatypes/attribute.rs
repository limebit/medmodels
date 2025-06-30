use super::{
    Abs, Contains, EndsWith, Lowercase, MedRecordValue, Mod, Pow, Slice, StartsWith, Trim, TrimEnd,
    TrimStart, Uppercase,
};
use crate::errors::{MedRecordError, MedRecordResult};
use medmodels_utils::implement_from_for_wrapper;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    fmt::Display,
    hash::Hash,
    ops::{Add, Mul, Sub},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MedRecordAttribute {
    Int(i64),
    String(String),
}

impl Hash for MedRecordAttribute {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::String(value) => value.hash(state),
            Self::Int(value) => value.hash(state),
        }
    }
}

impl From<&str> for MedRecordAttribute {
    fn from(value: &str) -> Self {
        value.to_string().into()
    }
}

implement_from_for_wrapper!(MedRecordAttribute, String, String);
implement_from_for_wrapper!(MedRecordAttribute, i64, Int);

impl TryFrom<MedRecordValue> for MedRecordAttribute {
    type Error = MedRecordError;

    fn try_from(value: MedRecordValue) -> Result<Self, Self::Error> {
        match value {
            MedRecordValue::String(value) => Ok(MedRecordAttribute::String(value)),
            MedRecordValue::Int(value) => Ok(MedRecordAttribute::Int(value)),
            _ => Err(MedRecordError::ConversionError(format!(
                "Cannot convert {value} into MedRecordAttribute"
            ))),
        }
    }
}

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

impl Display for MedRecordAttribute {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(value) => write!(f, "{value}"),
            Self::Int(value) => write!(f, "{value}"),
        }
    }
}

// TODO: Add tests
impl Add for MedRecordAttribute {
    type Output = MedRecordResult<Self>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MedRecordAttribute::String(value), MedRecordAttribute::String(rhs)) => {
                Ok(MedRecordAttribute::String(value + rhs.as_str()))
            }
            (MedRecordAttribute::String(value), MedRecordAttribute::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordAttribute::Int(value), MedRecordAttribute::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordAttribute::Int(value), MedRecordAttribute::Int(rhs)) => {
                Ok(MedRecordAttribute::Int(value + rhs))
            }
        }
    }
}

// TODO: Add tests
impl Sub for MedRecordAttribute {
    type Output = MedRecordResult<Self>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MedRecordAttribute::String(value), MedRecordAttribute::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordAttribute::String(value), MedRecordAttribute::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordAttribute::Int(value), MedRecordAttribute::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordAttribute::Int(value), MedRecordAttribute::Int(rhs)) => {
                Ok(MedRecordAttribute::Int(value - rhs))
            }
        }
    }
}

// TODO: Add tests
impl Mul for MedRecordAttribute {
    type Output = MedRecordResult<Self>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MedRecordAttribute::String(value), MedRecordAttribute::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiply {value} by {rhs}")),
            ),
            (MedRecordAttribute::String(value), MedRecordAttribute::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiply {value} by {rhs}")),
            ),
            (MedRecordAttribute::Int(value), MedRecordAttribute::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiply {value} by {rhs}")),
            ),
            (MedRecordAttribute::Int(value), MedRecordAttribute::Int(rhs)) => {
                Ok(MedRecordAttribute::Int(value * rhs))
            }
        }
    }
}

// TODO: Add tests
impl Pow for MedRecordAttribute {
    fn pow(self, rhs: Self) -> MedRecordResult<Self> {
        match (self, rhs) {
            (MedRecordAttribute::String(value), MedRecordAttribute::String(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {rhs}"
                )))
            }
            (MedRecordAttribute::String(value), MedRecordAttribute::Int(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {rhs}"
                )))
            }
            (MedRecordAttribute::Int(value), MedRecordAttribute::String(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {rhs}"
                )))
            }
            (MedRecordAttribute::Int(value), MedRecordAttribute::Int(rhs)) => {
                Ok(MedRecordAttribute::Int(value.pow(rhs as u32)))
            }
        }
    }
}

// TODO: Add tests
impl Mod for MedRecordAttribute {
    fn r#mod(self, rhs: Self) -> MedRecordResult<Self> {
        match (self, rhs) {
            (MedRecordAttribute::String(value), MedRecordAttribute::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} by {rhs}")),
            ),
            (MedRecordAttribute::String(value), MedRecordAttribute::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} by {rhs}")),
            ),
            (MedRecordAttribute::Int(value), MedRecordAttribute::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} by {rhs}")),
            ),
            (MedRecordAttribute::Int(value), MedRecordAttribute::Int(rhs)) => {
                Ok(MedRecordAttribute::Int(value % rhs))
            }
        }
    }
}

// TODO: Add tests
impl Abs for MedRecordAttribute {
    fn abs(self) -> Self {
        match self {
            MedRecordAttribute::Int(value) => MedRecordAttribute::Int(value.abs()),
            _ => self,
        }
    }
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

// TODO: Add tests
impl Slice for MedRecordAttribute {
    fn slice(self, range: std::ops::Range<usize>) -> Self {
        match self {
            MedRecordAttribute::String(value) => value[range].into(),
            MedRecordAttribute::Int(value) => value.to_string()[range].into(),
        }
    }
}

// TODO: Add tests
impl Trim for MedRecordAttribute {
    fn trim(self) -> Self {
        match self {
            MedRecordAttribute::String(value) => {
                MedRecordAttribute::String(value.trim().to_string())
            }
            _ => self,
        }
    }
}

// TODO: Add tests
impl TrimStart for MedRecordAttribute {
    fn trim_start(self) -> Self {
        match self {
            MedRecordAttribute::String(value) => {
                MedRecordAttribute::String(value.trim_start().to_string())
            }
            _ => self,
        }
    }
}

// TODO: Add tests
impl TrimEnd for MedRecordAttribute {
    fn trim_end(self) -> Self {
        match self {
            MedRecordAttribute::String(value) => {
                MedRecordAttribute::String(value.trim_end().to_string())
            }
            _ => self,
        }
    }
}

// TODO: Add tests
impl Lowercase for MedRecordAttribute {
    fn lowercase(self) -> Self {
        match self {
            MedRecordAttribute::String(value) => MedRecordAttribute::String(value.to_lowercase()),
            _ => self,
        }
    }
}

// TODO: Add tests
impl Uppercase for MedRecordAttribute {
    fn uppercase(self) -> Self {
        match self {
            MedRecordAttribute::String(value) => MedRecordAttribute::String(value.to_uppercase()),
            _ => self,
        }
    }
}

#[cfg(test)]
mod test {
    use super::MedRecordAttribute;
    use crate::{
        errors::MedRecordError,
        medrecord::{
            datatypes::{Contains, EndsWith, StartsWith},
            MedRecordValue,
        },
    };

    #[test]
    fn test_from_str() {
        let attribute = MedRecordAttribute::from("value");

        assert_eq!(MedRecordAttribute::String("value".to_string()), attribute)
    }

    #[test]
    fn test_from_string() {
        let attribute = MedRecordAttribute::from("value".to_string());

        assert_eq!(MedRecordAttribute::String("value".to_string()), attribute);
    }

    #[test]
    fn test_from_int() {
        let attribute = MedRecordAttribute::from(0);

        assert_eq!(MedRecordAttribute::Int(0), attribute);
    }

    #[test]
    fn test_try_from_medrecord_value() {
        let attribute = MedRecordAttribute::try_from(MedRecordValue::from("value")).unwrap();

        assert_eq!(MedRecordAttribute::String("value".to_string()), attribute);

        let attribute = MedRecordAttribute::try_from(MedRecordValue::from(0)).unwrap();

        assert_eq!(MedRecordAttribute::Int(0), attribute);

        assert!(MedRecordAttribute::try_from(MedRecordValue::from(true))
            .is_err_and(|e| matches!(e, MedRecordError::ConversionError(_))));

        assert!(MedRecordAttribute::try_from(MedRecordValue::from(0.0))
            .is_err_and(|e| matches!(e, MedRecordError::ConversionError(_))));
    }

    #[test]
    fn test_display() {
        assert_eq!(
            "value",
            MedRecordAttribute::from("value".to_string()).to_string()
        );

        assert_eq!("0", MedRecordAttribute::from(0).to_string());
    }

    #[test]
    fn test_partial_eq() {
        assert!(
            MedRecordAttribute::String("attribute".to_string())
                == MedRecordAttribute::String("attribute".to_string())
        );
        assert!(
            MedRecordAttribute::String("attribute2".to_string())
                != MedRecordAttribute::String("attribute".to_string())
        );

        assert!(MedRecordAttribute::Int(0) == MedRecordAttribute::Int(0));
        assert!(MedRecordAttribute::Int(1) != MedRecordAttribute::Int(0));
    }

    #[test]
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn test_partial_ord() {
        assert!(
            MedRecordAttribute::String("b".to_string())
                > MedRecordAttribute::String("a".to_string())
        );
        assert!(
            MedRecordAttribute::String("b".to_string())
                >= MedRecordAttribute::String("a".to_string())
        );
        assert!(
            MedRecordAttribute::String("a".to_string())
                < MedRecordAttribute::String("b".to_string())
        );
        assert!(
            MedRecordAttribute::String("a".to_string())
                <= MedRecordAttribute::String("b".to_string())
        );
        assert!(
            MedRecordAttribute::String("a".to_string())
                >= MedRecordAttribute::String("a".to_string())
        );
        assert!(
            MedRecordAttribute::String("a".to_string())
                <= MedRecordAttribute::String("a".to_string())
        );

        assert!(MedRecordAttribute::Int(1) > MedRecordAttribute::Int(0));
        assert!(MedRecordAttribute::Int(1) >= MedRecordAttribute::Int(0));
        assert!(MedRecordAttribute::Int(0) < MedRecordAttribute::Int(1));
        assert!(MedRecordAttribute::Int(0) <= MedRecordAttribute::Int(1));
        assert!(MedRecordAttribute::Int(0) >= MedRecordAttribute::Int(0));
        assert!(MedRecordAttribute::Int(0) <= MedRecordAttribute::Int(0));

        assert!(!(MedRecordAttribute::String("a".to_string()) > MedRecordAttribute::Int(1)));
        assert!(!(MedRecordAttribute::String("a".to_string()) >= MedRecordAttribute::Int(1)));
        assert!(!(MedRecordAttribute::String("a".to_string()) < MedRecordAttribute::Int(1)));
        assert!(!(MedRecordAttribute::String("a".to_string()) <= MedRecordAttribute::Int(1)));
        assert!(!(MedRecordAttribute::String("a".to_string()) >= MedRecordAttribute::Int(1)));
        assert!(!(MedRecordAttribute::String("a".to_string()) <= MedRecordAttribute::Int(1)));

        assert!(!(MedRecordAttribute::Int(1) > MedRecordAttribute::String("a".to_string())));
        assert!(!(MedRecordAttribute::Int(1) >= MedRecordAttribute::String("a".to_string())));
        assert!(!(MedRecordAttribute::Int(1) < MedRecordAttribute::String("a".to_string())));
        assert!(!(MedRecordAttribute::Int(1) <= MedRecordAttribute::String("a".to_string())));
        assert!(!(MedRecordAttribute::Int(1) >= MedRecordAttribute::String("a".to_string())));
        assert!(!(MedRecordAttribute::Int(1) <= MedRecordAttribute::String("a".to_string())));
    }

    #[test]
    fn test_starts_with() {
        assert!(MedRecordAttribute::String("value".to_string())
            .starts_with(&MedRecordAttribute::String("val".to_string())));
        assert!(!MedRecordAttribute::String("value".to_string())
            .starts_with(&MedRecordAttribute::String("not_val".to_string())));
        assert!(
            MedRecordAttribute::String("10".to_string()).starts_with(&MedRecordAttribute::Int(1))
        );
        assert!(
            !MedRecordAttribute::String("10".to_string()).starts_with(&MedRecordAttribute::Int(0))
        );

        assert!(
            MedRecordAttribute::Int(10).starts_with(&MedRecordAttribute::String("1".to_string()))
        );
        assert!(
            !MedRecordAttribute::Int(10).starts_with(&MedRecordAttribute::String("0".to_string()))
        );
        assert!(MedRecordAttribute::Int(10).starts_with(&MedRecordAttribute::Int(1)));
        assert!(!MedRecordAttribute::Int(10).starts_with(&MedRecordAttribute::Int(0)));
    }

    #[test]
    fn test_ends_with() {
        assert!(MedRecordAttribute::String("value".to_string())
            .ends_with(&MedRecordAttribute::String("ue".to_string())));
        assert!(!MedRecordAttribute::String("value".to_string())
            .ends_with(&MedRecordAttribute::String("not_ue".to_string())));
        assert!(MedRecordAttribute::String("10".to_string()).ends_with(&MedRecordAttribute::Int(0)));
        assert!(
            !MedRecordAttribute::String("10".to_string()).ends_with(&MedRecordAttribute::Int(1))
        );

        assert!(MedRecordAttribute::Int(10).ends_with(&MedRecordAttribute::String("0".to_string())));
        assert!(
            !MedRecordAttribute::Int(10).ends_with(&MedRecordAttribute::String("1".to_string()))
        );
        assert!(MedRecordAttribute::Int(10).ends_with(&MedRecordAttribute::Int(0)));
        assert!(!MedRecordAttribute::Int(10).ends_with(&MedRecordAttribute::Int(1)));
    }

    #[test]
    fn test_contains() {
        assert!(MedRecordAttribute::String("value".to_string())
            .contains(&MedRecordAttribute::String("al".to_string())));
        assert!(!MedRecordAttribute::String("value".to_string())
            .contains(&MedRecordAttribute::String("not_al".to_string())));
        assert!(MedRecordAttribute::String("101".to_string()).contains(&MedRecordAttribute::Int(0)));
        assert!(
            !MedRecordAttribute::String("101".to_string()).contains(&MedRecordAttribute::Int(2))
        );

        assert!(MedRecordAttribute::Int(101).contains(&MedRecordAttribute::String("0".to_string())));
        assert!(
            !MedRecordAttribute::Int(101).contains(&MedRecordAttribute::String("2".to_string()))
        );
        assert!(MedRecordAttribute::Int(101).contains(&MedRecordAttribute::Int(0)));
        assert!(!MedRecordAttribute::Int(101).contains(&MedRecordAttribute::Int(2)));
    }
}
