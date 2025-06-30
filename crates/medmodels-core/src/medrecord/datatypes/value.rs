use super::{
    Abs, Ceil, Contains, EndsWith, Floor, Lowercase, Mod, Pow, Round, Slice, Sqrt, StartsWith,
    Trim, TrimEnd, TrimStart, Uppercase,
};
use crate::errors::MedRecordError;
use chrono::{DateTime, NaiveDateTime};
use medmodels_utils::implement_from_for_wrapper;
use serde::{Deserialize, Serialize};
use std::{
    cmp::Ordering,
    fmt::Display,
    ops::{Add, Div, Mul, Range, Sub},
    time::Duration,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MedRecordValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    DateTime(NaiveDateTime),
    Duration(Duration),
    Null,
}

impl Default for MedRecordValue {
    fn default() -> Self {
        Self::Null
    }
}

impl From<&str> for MedRecordValue {
    fn from(value: &str) -> Self {
        value.to_string().into()
    }
}

// TODO: Add tests for Duration
implement_from_for_wrapper!(MedRecordValue, String, String);
implement_from_for_wrapper!(MedRecordValue, i64, Int);
implement_from_for_wrapper!(MedRecordValue, f64, Float);
implement_from_for_wrapper!(MedRecordValue, bool, Bool);
implement_from_for_wrapper!(MedRecordValue, NaiveDateTime, DateTime);
implement_from_for_wrapper!(MedRecordValue, Duration, Duration);

impl<T> From<Option<T>> for MedRecordValue
where
    T: Into<MedRecordValue>,
{
    fn from(value: Option<T>) -> Self {
        match value {
            Some(value) => value.into(),
            None => Self::Null,
        }
    }
}

// TODO: Add tests for Duration
impl PartialEq for MedRecordValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => value == other,
            (MedRecordValue::Int(value), MedRecordValue::Int(other)) => value == other,
            (MedRecordValue::Int(value), MedRecordValue::Float(other)) => &(*value as f64) == other,
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => value == other,
            (MedRecordValue::Float(value), MedRecordValue::Int(other)) => value == &(*other as f64),
            (MedRecordValue::Bool(value), MedRecordValue::Bool(other)) => value == other,
            (MedRecordValue::DateTime(value), MedRecordValue::DateTime(other)) => value == other,
            (MedRecordValue::Duration(value), MedRecordValue::Duration(other)) => value == other,
            (MedRecordValue::Null, MedRecordValue::Null) => true,
            _ => false,
        }
    }
}

// TODO: Add tests for Duration
impl PartialOrd for MedRecordValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => {
                Some(value.cmp(other))
            }
            (MedRecordValue::Int(value), MedRecordValue::Int(other)) => Some(value.cmp(other)),
            (MedRecordValue::Int(value), MedRecordValue::Float(other)) => {
                (*value as f64).partial_cmp(other)
            }
            (MedRecordValue::Float(value), MedRecordValue::Int(other)) => {
                value.partial_cmp(&(*other as f64))
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => {
                value.partial_cmp(other)
            }
            (MedRecordValue::Bool(value), MedRecordValue::Bool(other)) => Some(value.cmp(other)),
            (MedRecordValue::DateTime(value), MedRecordValue::DateTime(other)) => {
                Some(value.cmp(other))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Duration(other)) => {
                Some(value.cmp(other))
            }
            _ => None,
        }
    }
}

// TODO: Add tests for Duration
impl Display for MedRecordValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(value) => write!(f, "{value}"),
            Self::Int(value) => write!(f, "{value}"),
            Self::Float(value) => write!(f, "{value}"),
            Self::Bool(value) => write!(f, "{value}"),
            Self::DateTime(value) => write!(f, "{value}"),
            Self::Duration(value) => write!(f, "{}", value.as_secs()),
            Self::Null => write!(f, "Null"),
        }
    }
}

// TODO: Add tests for Duration
impl Add for MedRecordValue {
    type Output = Result<Self, MedRecordError>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MedRecordValue::String(value), MedRecordValue::String(rhs)) => {
                Ok(MedRecordValue::String(value + rhs.as_str()))
            }
            (MedRecordValue::String(value), MedRecordValue::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Float(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::DateTime(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Duration(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot add {} to {}",
                    rhs.as_secs(),
                    value
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot add None to {value}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::Int(rhs)) => {
                Ok(MedRecordValue::Int(value + rhs))
            }
            (MedRecordValue::Int(value), MedRecordValue::Float(rhs)) => {
                Ok(MedRecordValue::Float(value as f64 + rhs))
            }
            (MedRecordValue::Int(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::DateTime(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::Duration(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot add {} to {}",
                    rhs.as_secs(),
                    value
                )))
            }
            (MedRecordValue::Int(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot add None to {value}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::Int(rhs)) => {
                Ok(MedRecordValue::Float(value + rhs as f64))
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(rhs)) => {
                Ok(MedRecordValue::Float(value + rhs))
            }
            (MedRecordValue::Float(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::DateTime(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::Duration(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot add {} to {}",
                    rhs.as_secs(),
                    value
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot add None to {value}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Float(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::DateTime(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Duration(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot add {} to {}",
                    rhs.as_secs(),
                    value
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot add None to {value}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Float(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to {value}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::DateTime(rhs)) => {
                Ok(DateTime::from_timestamp(
                    value.and_utc().timestamp() + rhs.and_utc().timestamp(),
                    0,
                )
                .ok_or(MedRecordError::AssertionError(
                    "Invalid timestamp".to_string(),
                ))?
                .naive_utc()
                .into())
            }
            (MedRecordValue::DateTime(value), MedRecordValue::Duration(rhs)) => {
                Ok(value.add(rhs).into())
            }
            (MedRecordValue::DateTime(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot add None to {value}")),
            ),
            (MedRecordValue::Duration(value), MedRecordValue::String(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot add {} to {}",
                    rhs,
                    value.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Int(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot add {} to {}",
                    rhs,
                    value.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Float(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot add {} to {}",
                    rhs,
                    value.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Bool(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot add {} to {}",
                    rhs,
                    value.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::DateTime(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot add {} to {}",
                    rhs,
                    value.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Duration(rhs)) => {
                Ok((value + rhs).into())
            }
            (MedRecordValue::Duration(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot add None to {}", value.as_secs())),
            ),
            (MedRecordValue::Null, MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to None")),
            ),
            (MedRecordValue::Null, MedRecordValue::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to None")),
            ),
            (MedRecordValue::Null, MedRecordValue::Float(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to None")),
            ),
            (MedRecordValue::Null, MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to None")),
            ),
            (MedRecordValue::Null, MedRecordValue::DateTime(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {rhs} to None")),
            ),
            (MedRecordValue::Null, MedRecordValue::Duration(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot add {} to None", rhs.as_secs())),
            ),
            (MedRecordValue::Null, MedRecordValue::Null) => Err(MedRecordError::AssertionError(
                "Cannot add None to None".to_string(),
            )),
        }
    }
}

// TODO: Add tests for Duration
impl Sub for MedRecordValue {
    type Output = Result<Self, MedRecordError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MedRecordValue::String(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Float(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::DateTime(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Duration(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot subtract {} from {}",
                    rhs.as_secs(),
                    value
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract None from {value}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::Int(rhs)) => {
                Ok(MedRecordValue::Int(value - rhs))
            }
            (MedRecordValue::Int(value), MedRecordValue::Float(rhs)) => {
                Ok(MedRecordValue::Float(value as f64 - rhs))
            }
            (MedRecordValue::Int(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::DateTime(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::Duration(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot subtract {} from {}",
                    rhs.as_secs(),
                    value
                )))
            }
            (MedRecordValue::Int(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract None from {value}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::Int(rhs)) => {
                Ok(MedRecordValue::Float(value - rhs as f64))
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(rhs)) => {
                Ok(MedRecordValue::Float(value - rhs))
            }
            (MedRecordValue::Float(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::DateTime(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::Duration(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot subtract {} from {}",
                    rhs.as_secs(),
                    value
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract None from {value}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Float(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::DateTime(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Duration(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot subtract {} from {}",
                    rhs.as_secs(),
                    value
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract None from {value}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Float(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from {value}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::DateTime(rhs)) => {
                let duration = (value - rhs)
                    .to_std()
                    .map_err(|_| MedRecordError::AssertionError("Invalid timestamp".to_string()))?;

                Ok(duration.into())
            }
            (MedRecordValue::DateTime(value), MedRecordValue::Duration(rhs)) => {
                Ok((value - rhs).into())
            }
            (MedRecordValue::DateTime(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract None from {value}")),
            ),
            (MedRecordValue::Duration(value), MedRecordValue::String(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot subtract {} from {}",
                    rhs,
                    value.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Int(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot subtract {} from {}",
                    rhs,
                    value.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Float(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot subtract {} from {}",
                    rhs,
                    value.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Bool(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot subtract {} from {}",
                    rhs,
                    value.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::DateTime(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot subtract {} from {}",
                    rhs,
                    value.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Duration(rhs)) => {
                Ok((value + rhs).into())
            }
            (MedRecordValue::Duration(value), MedRecordValue::Null) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot subtract None from {}",
                    value.as_secs()
                )))
            }
            (MedRecordValue::Null, MedRecordValue::String(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from None")),
            ),
            (MedRecordValue::Null, MedRecordValue::Int(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from None")),
            ),
            (MedRecordValue::Null, MedRecordValue::Float(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from None")),
            ),
            (MedRecordValue::Null, MedRecordValue::Bool(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from None")),
            ),
            (MedRecordValue::Null, MedRecordValue::DateTime(rhs)) => Err(
                MedRecordError::AssertionError(format!("Cannot subtract {rhs} from None")),
            ),
            (MedRecordValue::Null, MedRecordValue::Duration(rhs)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot subtract {} from None",
                    rhs.as_secs()
                )))
            }
            (MedRecordValue::Null, MedRecordValue::Null) => Err(MedRecordError::AssertionError(
                "Cannot subtract None from None".to_string(),
            )),
        }
    }
}

// TODO: Add tests for Duration
impl Mul for MedRecordValue {
    type Output = Result<Self, MedRecordError>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Int(other)) => {
                let mut result = String::new();

                for _ in 0..other {
                    result.push_str(&value)
                }

                Ok(MedRecordValue::String(result))
            }
            (MedRecordValue::String(value), MedRecordValue::Float(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with None")),
            ),
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
            (MedRecordValue::Int(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::Duration(other)) => {
                Ok((other * (value as u32)).into())
            }
            (MedRecordValue::Int(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with None")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::Int(other)) => {
                Ok(MedRecordValue::Float(value * other as f64))
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => {
                Ok(MedRecordValue::Float(value * other))
            }
            (MedRecordValue::Float(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with None")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Int(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Float(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with None")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Int(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Float(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::DateTime(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty {value} with None")),
            ),
            (MedRecordValue::Duration(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Int(other)) => {
                Ok((value * (other as u32)).into())
            }
            (MedRecordValue::Duration(value), MedRecordValue::Float(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::DateTime(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with {}",
                    value.as_secs(),
                    other.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Null) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty {} with None",
                    value.as_secs()
                )))
            }
            (MedRecordValue::Null, MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty None with {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::Int(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty None with {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::Float(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty None with {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty None with {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot multiplty None with {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot multiplty None with {}",
                    other.as_secs()
                )))
            }
            (MedRecordValue::Null, MedRecordValue::Null) => Err(MedRecordError::AssertionError(
                "Cannot multiplty None with None".to_string(),
            )),
        }
    }
}

// TODO: Add tests for Duration
impl Div for MedRecordValue {
    type Output = Result<Self, MedRecordError>;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Int(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Float(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot divide {} by {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by None")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::Int(other)) => {
                Ok(MedRecordValue::Float(value as f64 / other as f64))
            }
            (MedRecordValue::Int(value), MedRecordValue::Float(other)) => {
                Ok(MedRecordValue::Float(value as f64 / other))
            }
            (MedRecordValue::Int(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot divide {} by {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::Int(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by None")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::Int(other)) => {
                Ok(MedRecordValue::Float(value / other as f64))
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => {
                Ok(MedRecordValue::Float(value / other))
            }
            (MedRecordValue::Float(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot divide {} by {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by None")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Int(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Float(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot divide {} by {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by None")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Int(other)) => {
                Ok(DateTime::from_timestamp(
                    (value.and_utc().timestamp() as f64 / other as f64).floor() as i64,
                    0,
                )
                .ok_or(MedRecordError::AssertionError(
                    "Invalid timestamp".to_string(),
                ))?
                .naive_utc()
                .into())
            }
            (MedRecordValue::DateTime(value), MedRecordValue::Float(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot divide {} by {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::DateTime(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot divide {value} by None")),
            ),
            (MedRecordValue::Duration(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot divide {} by {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Int(other)) => {
                Ok((value / (other as u32)).into())
            }
            (MedRecordValue::Duration(value), MedRecordValue::Float(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot divide {} by {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot divide {} by {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::DateTime(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot divide {} by {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot divide {} by {}",
                    value.as_secs(),
                    other.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Null) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot divide {} by None",
                    value.as_secs()
                )))
            }
            (MedRecordValue::Null, MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide None by {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::Int(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide None by {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::Float(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide None by {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide None by {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot divide None by {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot divide None by {}",
                    other.as_secs()
                )))
            }
            (MedRecordValue::Null, MedRecordValue::Null) => Err(MedRecordError::AssertionError(
                "Cannot divide None by None".to_string(),
            )),
        }
    }
}

// TODO: Add tests for Duration
impl Pow for MedRecordValue {
    fn pow(self, exp: Self) -> Result<Self, MedRecordError> {
        match (self, exp) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Int(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Float(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::DateTime(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {} to the power of {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Null) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of None"
                )))
            }
            (MedRecordValue::Int(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::Int(value), MedRecordValue::Int(exp)) => {
                Ok(MedRecordValue::Int(value.pow(exp as u32)))
            }
            (MedRecordValue::Int(value), MedRecordValue::Float(exp)) => {
                Ok(MedRecordValue::Float((value as f64).powf(exp)))
            }
            (MedRecordValue::Int(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::Int(value), MedRecordValue::DateTime(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::Int(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {} to the power of {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::Int(value), MedRecordValue::Null) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of None"
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::Int(exp)) => {
                Ok(MedRecordValue::Float(value.powi(exp as i32)))
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(exp)) => {
                Ok(MedRecordValue::Float(value.powf(exp)))
            }
            (MedRecordValue::Float(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::DateTime(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {} to the power of {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::Null) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of None"
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Int(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Float(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::DateTime(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {} to the power of {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Null) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of None"
                )))
            }
            (MedRecordValue::DateTime(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::DateTime(value), MedRecordValue::Int(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::DateTime(value), MedRecordValue::Float(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::DateTime(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::DateTime(value), MedRecordValue::DateTime(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of {other}"
                )))
            }
            (MedRecordValue::DateTime(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {} to the power of {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::DateTime(value), MedRecordValue::Null) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {value} to the power of Null"
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {} to the power of {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Int(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {} to the power of {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Float(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {} to the power of {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {} to the power of {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::DateTime(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {} to the power of {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {} to the power of {}",
                    value.as_secs(),
                    other.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Null) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise {} to the power of Null",
                    value.as_secs()
                )))
            }
            (MedRecordValue::Null, MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise None to the power of {other}"
                )))
            }
            (MedRecordValue::Null, MedRecordValue::Int(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise None to the power of {other}"
                )))
            }
            (MedRecordValue::Null, MedRecordValue::Float(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise None to the power of {other}"
                )))
            }
            (MedRecordValue::Null, MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise None to the power of {other}"
                )))
            }
            (MedRecordValue::Null, MedRecordValue::DateTime(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise None to the power of {other}"
                )))
            }
            (MedRecordValue::Null, MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot raise None to the power of {}",
                    other.as_secs()
                )))
            }
            (MedRecordValue::Null, MedRecordValue::Null) => Err(MedRecordError::AssertionError(
                "Cannot raise None to the power of None".to_string(),
            )),
        }
    }
}

// TODO: Add tests for Duration
impl Mod for MedRecordValue {
    fn r#mod(self, other: Self) -> Result<Self, MedRecordError> {
        match (self, other) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Int(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Float(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::String(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot mod {} with {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::String(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with None")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::Int(other)) => {
                Ok(MedRecordValue::Int(value % other))
            }
            (MedRecordValue::Int(value), MedRecordValue::Float(other)) => {
                Ok(MedRecordValue::Float(value as f64 % other))
            }
            (MedRecordValue::Int(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::Int(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot mod {} with {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::Int(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with None")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::Int(other)) => {
                Ok(MedRecordValue::Float(value % other as f64))
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => {
                Ok(MedRecordValue::Float(value % other))
            }
            (MedRecordValue::Float(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::Float(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot mod {} with {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::Float(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with None")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Int(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Float(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::Bool(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot mod {} with {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::Bool(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with None")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Int(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Float(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with {other}")),
            ),
            (MedRecordValue::DateTime(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot mod {} with {}",
                    value,
                    other.as_secs()
                )))
            }
            (MedRecordValue::DateTime(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {value} with None")),
            ),
            (MedRecordValue::Duration(value), MedRecordValue::String(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot mod {} with {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Int(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot mod {} with {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Float(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot mod {} with {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Bool(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot mod {} with {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::DateTime(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot mod {} with {}",
                    value.as_secs(),
                    other
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Duration(other)) => {
                Err(MedRecordError::AssertionError(format!(
                    "Cannot mod {} with {}",
                    value.as_secs(),
                    other.as_secs()
                )))
            }
            (MedRecordValue::Duration(value), MedRecordValue::Null) => Err(
                MedRecordError::AssertionError(format!("Cannot mod {} with None", value.as_secs())),
            ),
            (MedRecordValue::Null, MedRecordValue::String(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod None with {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::Int(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod None with {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::Float(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod None with {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::Bool(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod None with {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::DateTime(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod None with {other}")),
            ),
            (MedRecordValue::Null, MedRecordValue::Duration(other)) => Err(
                MedRecordError::AssertionError(format!("Cannot mod None with {}", other.as_secs())),
            ),
            (MedRecordValue::Null, MedRecordValue::Null) => Err(MedRecordError::AssertionError(
                "Cannot mod None with None".to_string(),
            )),
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

impl Abs for MedRecordValue {
    fn abs(self) -> Self {
        match self {
            MedRecordValue::Int(value) => MedRecordValue::Int(value.abs()),
            MedRecordValue::Float(value) => MedRecordValue::Float(value.abs()),
            _ => self,
        }
    }
}

impl Sqrt for MedRecordValue {
    fn sqrt(self) -> Self {
        match self {
            MedRecordValue::Int(value) => MedRecordValue::Float((value as f64).sqrt()),
            MedRecordValue::Float(value) => MedRecordValue::Float(value.sqrt()),
            _ => self,
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
            (MedRecordValue::Int(value), MedRecordValue::Float(other)) => {
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
            (MedRecordValue::Int(value), MedRecordValue::Float(other)) => {
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
            (MedRecordValue::Int(value), MedRecordValue::Float(other)) => {
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

#[cfg(test)]
mod test {
    use super::{Contains, EndsWith, MedRecordValue, StartsWith};
    use crate::{
        errors::MedRecordError,
        medrecord::datatypes::{
            Abs, Ceil, Floor, Lowercase, Mod, Pow, Round, Slice, Sqrt, Trim, TrimEnd, TrimStart,
            Uppercase,
        },
    };
    use chrono::{NaiveDate, NaiveDateTime, NaiveTime};

    #[test]
    fn test_default() {
        let value = MedRecordValue::default();

        assert_eq!(MedRecordValue::Null, value);
    }

    #[test]
    fn test_from_str() {
        let value = MedRecordValue::from("value");

        assert_eq!(MedRecordValue::String("value".to_string()), value)
    }

    #[test]
    fn test_from_string() {
        let value = MedRecordValue::from("value".to_string());

        assert_eq!(MedRecordValue::String("value".to_string()), value);
    }

    #[test]
    fn test_from_int() {
        let value = MedRecordValue::from(0);

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
    fn test_from_datetime() {
        let value = MedRecordValue::from(NaiveDateTime::MIN);

        assert_eq!(MedRecordValue::DateTime(NaiveDateTime::MIN), value);
    }

    #[test]
    fn test_from_option() {
        let value = MedRecordValue::from(Some("value"));

        assert_eq!(MedRecordValue::String("value".to_string()), value);

        let value = MedRecordValue::from(None::<String>);

        assert_eq!(MedRecordValue::Null, value);
    }

    #[test]
    fn test_partial_eq() {
        assert!(
            MedRecordValue::String("value".to_string())
                == MedRecordValue::String("value".to_string())
        );
        assert!(
            MedRecordValue::String("value2".to_string())
                != MedRecordValue::String("value".to_string())
        );

        assert!(MedRecordValue::Int(0) == MedRecordValue::Int(0));
        assert!(MedRecordValue::Int(1) != MedRecordValue::Int(0));

        assert!(MedRecordValue::Int(0) == MedRecordValue::Float(0_f64));
        assert!(MedRecordValue::Int(1) != MedRecordValue::Float(0_f64));

        assert!(MedRecordValue::Float(0_f64) == MedRecordValue::Float(0_f64));
        assert!(MedRecordValue::Float(1_f64) != MedRecordValue::Float(0_f64));

        assert!(MedRecordValue::Float(0_f64) == MedRecordValue::Int(0));
        assert!(MedRecordValue::Float(1_f64) != MedRecordValue::Int(0));

        assert!(MedRecordValue::Bool(false) == MedRecordValue::Bool(false));
        assert!(MedRecordValue::Bool(true) != MedRecordValue::Bool(false));

        assert!(
            MedRecordValue::DateTime(NaiveDateTime::MIN)
                == MedRecordValue::DateTime(NaiveDateTime::MIN)
        );
        assert!(
            MedRecordValue::DateTime(NaiveDateTime::MAX)
                != MedRecordValue::DateTime(NaiveDateTime::MIN)
        );

        assert!(MedRecordValue::Null == MedRecordValue::Null);

        assert!(MedRecordValue::String("0".to_string()) != MedRecordValue::Int(0));
        assert!(MedRecordValue::String("0".to_string()) != MedRecordValue::Float(0_f64));
        assert!(MedRecordValue::String("false".to_string()) != MedRecordValue::Bool(false));
        assert!(
            MedRecordValue::String("false".to_string())
                != MedRecordValue::DateTime(NaiveDateTime::MIN)
        );
        assert!(MedRecordValue::String("false".to_string()) != MedRecordValue::Null);

        assert!(MedRecordValue::Int(0) != MedRecordValue::String("0".to_string()));
        assert!(MedRecordValue::Int(0) != MedRecordValue::Bool(false));
        assert!(MedRecordValue::Int(0) != MedRecordValue::DateTime(NaiveDateTime::MIN));
        assert!(MedRecordValue::Int(0) != MedRecordValue::Null);

        assert!(MedRecordValue::Float(0_f64) != MedRecordValue::String("0.0".to_string()));
        assert!(MedRecordValue::Float(0_f64) != MedRecordValue::Bool(false));
        assert!(MedRecordValue::Float(0_f64) != MedRecordValue::DateTime(NaiveDateTime::MIN));
        assert!(MedRecordValue::Float(0_f64) != MedRecordValue::Null);

        assert!(MedRecordValue::Bool(false) != MedRecordValue::String("false".to_string()));
        assert!(MedRecordValue::Bool(false) != MedRecordValue::Int(0));
        assert!(MedRecordValue::Bool(false) != MedRecordValue::Float(0_f64));
        assert!(MedRecordValue::Bool(false) != MedRecordValue::DateTime(NaiveDateTime::MIN));
        assert!(MedRecordValue::Bool(false) != MedRecordValue::Null);

        assert!(MedRecordValue::Null != MedRecordValue::String("false".to_string()));
        assert!(MedRecordValue::Null != MedRecordValue::Int(0));
        assert!(MedRecordValue::Null != MedRecordValue::Float(0_f64));
        assert!(MedRecordValue::Null != MedRecordValue::Bool(false));
        assert!(MedRecordValue::Null != MedRecordValue::DateTime(NaiveDateTime::MIN));
    }

    #[test]
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn test_partial_ord() {
        assert!(MedRecordValue::String("b".to_string()) > MedRecordValue::String("a".to_string()));
        assert!(MedRecordValue::String("b".to_string()) >= MedRecordValue::String("a".to_string()));
        assert!(MedRecordValue::String("a".to_string()) < MedRecordValue::String("b".to_string()));
        assert!(MedRecordValue::String("a".to_string()) <= MedRecordValue::String("b".to_string()));
        assert!(MedRecordValue::String("a".to_string()) >= MedRecordValue::String("a".to_string()));
        assert!(MedRecordValue::String("a".to_string()) <= MedRecordValue::String("a".to_string()));

        assert!(MedRecordValue::Int(1) > MedRecordValue::Int(0));
        assert!(MedRecordValue::Int(1) >= MedRecordValue::Int(0));
        assert!(MedRecordValue::Int(0) < MedRecordValue::Int(1));
        assert!(MedRecordValue::Int(0) <= MedRecordValue::Int(1));
        assert!(MedRecordValue::Int(0) >= MedRecordValue::Int(0));
        assert!(MedRecordValue::Int(0) <= MedRecordValue::Int(0));

        assert!(MedRecordValue::Int(1) > MedRecordValue::Float(0_f64));
        assert!(MedRecordValue::Int(1) >= MedRecordValue::Float(0_f64));
        assert!(MedRecordValue::Int(0) < MedRecordValue::Float(1_f64));
        assert!(MedRecordValue::Int(0) <= MedRecordValue::Float(1_f64));
        assert!(MedRecordValue::Int(0) >= MedRecordValue::Float(0_f64));
        assert!(MedRecordValue::Int(0) <= MedRecordValue::Float(0_f64));

        assert!(MedRecordValue::Float(1_f64) > MedRecordValue::Int(0));
        assert!(MedRecordValue::Float(1_f64) >= MedRecordValue::Int(0));
        assert!(MedRecordValue::Float(0_f64) < MedRecordValue::Int(1));
        assert!(MedRecordValue::Float(0_f64) <= MedRecordValue::Int(1));
        assert!(MedRecordValue::Float(0_f64) >= MedRecordValue::Int(0));
        assert!(MedRecordValue::Float(0_f64) <= MedRecordValue::Int(0));

        assert!(MedRecordValue::Bool(true) > MedRecordValue::Bool(false));
        assert!(MedRecordValue::Bool(true) >= MedRecordValue::Bool(false));
        assert!(MedRecordValue::Bool(false) < MedRecordValue::Bool(true));
        assert!(MedRecordValue::Bool(false) <= MedRecordValue::Bool(true));
        assert!(MedRecordValue::Bool(false) >= MedRecordValue::Bool(false));
        assert!(MedRecordValue::Bool(false) <= MedRecordValue::Bool(false));

        assert!(
            MedRecordValue::DateTime(NaiveDateTime::MAX)
                > MedRecordValue::DateTime(NaiveDateTime::MIN)
        );
        assert!(
            MedRecordValue::DateTime(NaiveDateTime::MAX)
                >= MedRecordValue::DateTime(NaiveDateTime::MIN)
        );
        assert!(
            MedRecordValue::DateTime(NaiveDateTime::MIN)
                < MedRecordValue::DateTime(NaiveDateTime::MAX)
        );
        assert!(
            MedRecordValue::DateTime(NaiveDateTime::MIN)
                <= MedRecordValue::DateTime(NaiveDateTime::MAX)
        );
        assert!(
            MedRecordValue::DateTime(NaiveDateTime::MIN)
                >= MedRecordValue::DateTime(NaiveDateTime::MIN)
        );
        assert!(
            MedRecordValue::DateTime(NaiveDateTime::MIN)
                <= MedRecordValue::DateTime(NaiveDateTime::MIN)
        );

        assert!(!(MedRecordValue::String("a".to_string()) > MedRecordValue::Int(1)));
        assert!(!(MedRecordValue::String("a".to_string()) >= MedRecordValue::Int(1)));
        assert!(!(MedRecordValue::String("a".to_string()) < MedRecordValue::Int(1)));
        assert!(!(MedRecordValue::String("a".to_string()) <= MedRecordValue::Int(1)));

        assert!(!(MedRecordValue::String("a".to_string()) > MedRecordValue::Float(1_f64)));
        assert!(!(MedRecordValue::String("a".to_string()) >= MedRecordValue::Float(1_f64)));
        assert!(!(MedRecordValue::String("a".to_string()) < MedRecordValue::Float(1_f64)));
        assert!(!(MedRecordValue::String("a".to_string()) <= MedRecordValue::Float(1_f64)));

        assert!(!(MedRecordValue::String("a".to_string()) > MedRecordValue::Bool(true)));
        assert!(!(MedRecordValue::String("a".to_string()) >= MedRecordValue::Bool(true)));
        assert!(!(MedRecordValue::String("a".to_string()) < MedRecordValue::Bool(true)));
        assert!(!(MedRecordValue::String("a".to_string()) <= MedRecordValue::Bool(true)));

        assert!(
            !(MedRecordValue::String("a".to_string())
                > MedRecordValue::DateTime(NaiveDateTime::MAX))
        );
        assert!(
            !(MedRecordValue::String("a".to_string())
                >= MedRecordValue::DateTime(NaiveDateTime::MAX))
        );
        assert!(
            !(MedRecordValue::String("a".to_string())
                < MedRecordValue::DateTime(NaiveDateTime::MAX))
        );
        assert!(
            !(MedRecordValue::String("a".to_string())
                <= MedRecordValue::DateTime(NaiveDateTime::MAX))
        );

        assert!(!(MedRecordValue::String("a".to_string()) > MedRecordValue::Null));
        assert!(!(MedRecordValue::String("a".to_string()) >= MedRecordValue::Null));
        assert!(!(MedRecordValue::String("a".to_string()) < MedRecordValue::Null));
        assert!(!(MedRecordValue::String("a".to_string()) <= MedRecordValue::Null));

        assert!(!(MedRecordValue::Int(1) > MedRecordValue::String("a".to_string())));
        assert!(!(MedRecordValue::Int(1) >= MedRecordValue::String("a".to_string())));
        assert!(!(MedRecordValue::Int(1) < MedRecordValue::String("a".to_string())));
        assert!(!(MedRecordValue::Int(1) <= MedRecordValue::String("a".to_string())));

        assert!(!(MedRecordValue::Int(1) > MedRecordValue::Bool(true)));
        assert!(!(MedRecordValue::Int(1) >= MedRecordValue::Bool(true)));
        assert!(!(MedRecordValue::Int(1) < MedRecordValue::Bool(true)));
        assert!(!(MedRecordValue::Int(1) <= MedRecordValue::Bool(true)));

        assert!(!(MedRecordValue::Int(1) > MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!(MedRecordValue::Int(1) >= MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!(MedRecordValue::Int(1) < MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!(MedRecordValue::Int(1) <= MedRecordValue::DateTime(NaiveDateTime::MAX)));

        assert!(!(MedRecordValue::Int(1) > MedRecordValue::Null));
        assert!(!(MedRecordValue::Int(1) >= MedRecordValue::Null));
        assert!(!(MedRecordValue::Int(1) < MedRecordValue::Null));
        assert!(!(MedRecordValue::Int(1) <= MedRecordValue::Null));

        assert!(!(MedRecordValue::Float(1_f64) > MedRecordValue::String("a".to_string())));
        assert!(!(MedRecordValue::Float(1_f64) >= MedRecordValue::String("a".to_string())));
        assert!(!(MedRecordValue::Float(1_f64) < MedRecordValue::String("a".to_string())));
        assert!(!(MedRecordValue::Float(1_f64) <= MedRecordValue::String("a".to_string())));

        assert!(!(MedRecordValue::Float(1_f64) > MedRecordValue::Bool(true)));
        assert!(!(MedRecordValue::Float(1_f64) >= MedRecordValue::Bool(true)));
        assert!(!(MedRecordValue::Float(1_f64) < MedRecordValue::Bool(true)));
        assert!(!(MedRecordValue::Float(1_f64) <= MedRecordValue::Bool(true)));

        assert!(!(MedRecordValue::Float(1_f64) > MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!(MedRecordValue::Float(1_f64) >= MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!(MedRecordValue::Float(1_f64) < MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!(MedRecordValue::Float(1_f64) <= MedRecordValue::DateTime(NaiveDateTime::MAX)));

        assert!(!(MedRecordValue::Float(1_f64) > MedRecordValue::Null));
        assert!(!(MedRecordValue::Float(1_f64) >= MedRecordValue::Null));
        assert!(!(MedRecordValue::Float(1_f64) < MedRecordValue::Null));
        assert!(!(MedRecordValue::Float(1_f64) <= MedRecordValue::Null));

        assert!(!(MedRecordValue::Bool(true) > MedRecordValue::String("a".to_string())));
        assert!(!(MedRecordValue::Bool(true) >= MedRecordValue::String("a".to_string())));
        assert!(!(MedRecordValue::Bool(true) < MedRecordValue::String("a".to_string())));
        assert!(!(MedRecordValue::Bool(true) <= MedRecordValue::String("a".to_string())));

        assert!(!(MedRecordValue::Bool(true) > MedRecordValue::Int(1)));
        assert!(!(MedRecordValue::Bool(true) >= MedRecordValue::Int(1)));
        assert!(!(MedRecordValue::Bool(true) < MedRecordValue::Int(1)));
        assert!(!(MedRecordValue::Bool(true) <= MedRecordValue::Int(1)));

        assert!(!(MedRecordValue::Bool(true) > MedRecordValue::Float(1_f64)));
        assert!(!(MedRecordValue::Bool(true) >= MedRecordValue::Float(1_f64)));
        assert!(!(MedRecordValue::Bool(true) < MedRecordValue::Float(1_f64)));
        assert!(!(MedRecordValue::Bool(true) <= MedRecordValue::Float(1_f64)));

        assert!(!(MedRecordValue::Bool(true) > MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!(MedRecordValue::Bool(true) >= MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!(MedRecordValue::Bool(true) < MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!(MedRecordValue::Bool(true) <= MedRecordValue::DateTime(NaiveDateTime::MAX)));

        assert!(!(MedRecordValue::Bool(true) > MedRecordValue::Null));
        assert!(!(MedRecordValue::Bool(true) >= MedRecordValue::Null));
        assert!(!(MedRecordValue::Bool(true) < MedRecordValue::Null));
        assert!(!(MedRecordValue::Bool(true) <= MedRecordValue::Null));

        assert!(
            !(MedRecordValue::DateTime(NaiveDateTime::MAX)
                > MedRecordValue::String("a".to_string()))
        );
        assert!(
            !(MedRecordValue::DateTime(NaiveDateTime::MAX)
                >= MedRecordValue::String("a".to_string()))
        );
        assert!(
            !(MedRecordValue::DateTime(NaiveDateTime::MAX)
                < MedRecordValue::String("a".to_string()))
        );
        assert!(
            !(MedRecordValue::DateTime(NaiveDateTime::MAX)
                <= MedRecordValue::String("a".to_string()))
        );

        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) > MedRecordValue::Int(1)));
        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) >= MedRecordValue::Int(1)));
        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) < MedRecordValue::Int(1)));
        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) <= MedRecordValue::Int(1)));

        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) > MedRecordValue::Float(1_f64)));
        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) >= MedRecordValue::Float(1_f64)));
        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) < MedRecordValue::Float(1_f64)));
        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) <= MedRecordValue::Float(1_f64)));

        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) > MedRecordValue::Bool(true)));
        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) >= MedRecordValue::Bool(true)));
        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) < MedRecordValue::Bool(true)));
        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) <= MedRecordValue::Bool(true)));

        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) > MedRecordValue::Null));
        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) >= MedRecordValue::Null));
        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) < MedRecordValue::Null));
        assert!(!(MedRecordValue::DateTime(NaiveDateTime::MAX) <= MedRecordValue::Null));

        assert!(!(MedRecordValue::Null > MedRecordValue::String("a".to_string())));
        assert!(!(MedRecordValue::Null >= MedRecordValue::String("a".to_string())));
        assert!(!(MedRecordValue::Null < MedRecordValue::String("a".to_string())));
        assert!(!(MedRecordValue::Null <= MedRecordValue::String("a".to_string())));

        assert!(!(MedRecordValue::Null > MedRecordValue::Int(0)));
        assert!(!(MedRecordValue::Null >= MedRecordValue::Int(0)));
        assert!(!(MedRecordValue::Null < MedRecordValue::Int(0)));
        assert!(!(MedRecordValue::Null <= MedRecordValue::Int(0)));

        assert!(!(MedRecordValue::Null > MedRecordValue::Float(0_f64)));
        assert!(!(MedRecordValue::Null >= MedRecordValue::Float(0_f64)));
        assert!(!(MedRecordValue::Null < MedRecordValue::Float(0_f64)));
        assert!(!(MedRecordValue::Null <= MedRecordValue::Float(0_f64)));

        assert!(!(MedRecordValue::Null > MedRecordValue::Bool(false)));
        assert!(!(MedRecordValue::Null >= MedRecordValue::Bool(false)));
        assert!(!(MedRecordValue::Null < MedRecordValue::Bool(false)));
        assert!(!(MedRecordValue::Null <= MedRecordValue::Bool(false)));

        assert!(!(MedRecordValue::Null > MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!(MedRecordValue::Null >= MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!(MedRecordValue::Null < MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!(MedRecordValue::Null <= MedRecordValue::DateTime(NaiveDateTime::MAX)));

        assert!(!(MedRecordValue::Null > MedRecordValue::Null));
        assert!(!(MedRecordValue::Null >= MedRecordValue::Null));
        assert!(!(MedRecordValue::Null < MedRecordValue::Null));
        assert!(!(MedRecordValue::Null <= MedRecordValue::Null));
    }

    #[test]
    fn test_display() {
        assert_eq!(
            "value",
            MedRecordValue::String("value".to_string()).to_string()
        );

        assert_eq!("0", MedRecordValue::Int(0).to_string());

        assert_eq!("0.5", MedRecordValue::Float(0.5).to_string());

        assert_eq!("false", MedRecordValue::Bool(false).to_string());

        assert_eq!(
            "-262143-01-01 00:00:00",
            MedRecordValue::DateTime(NaiveDateTime::MIN).to_string()
        );

        assert_eq!("Null", MedRecordValue::Null.to_string());
    }

    #[test]
    fn test_add() {
        assert_eq!(
            MedRecordValue::String("value".to_string()),
            (MedRecordValue::String("val".to_string()) + MedRecordValue::String("ue".to_string()))
                .unwrap()
        );
        assert!(
            (MedRecordValue::String("value".to_string()) + MedRecordValue::Int(0))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::String("value".to_string()) + MedRecordValue::Float(0_f64))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::String("value".to_string()) + MedRecordValue::Bool(false))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::String("value".to_string())
            + MedRecordValue::DateTime(NaiveDateTime::MIN))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::String("value".to_string()) + MedRecordValue::Null)
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );

        assert!(
            (MedRecordValue::Int(0) + MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert_eq!(
            MedRecordValue::Int(10),
            (MedRecordValue::Int(5) + MedRecordValue::Int(5)).unwrap()
        );
        assert_eq!(
            MedRecordValue::Float(10_f64),
            (MedRecordValue::Int(5) + MedRecordValue::Float(5_f64)).unwrap()
        );
        assert!((MedRecordValue::Int(0) + MedRecordValue::Bool(false))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Int(0) + MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Int(0) + MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!(
            (MedRecordValue::Float(0_f64) + MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert_eq!(
            MedRecordValue::Float(10_f64),
            (MedRecordValue::Float(5_f64) + MedRecordValue::Int(5)).unwrap()
        );
        assert_eq!(
            MedRecordValue::Float(10_f64),
            (MedRecordValue::Float(5_f64) + MedRecordValue::Float(5_f64)).unwrap()
        );
        assert!((MedRecordValue::Float(0_f64) + MedRecordValue::Bool(false))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Float(0_f64) + MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Float(0_f64) + MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!(
            (MedRecordValue::Bool(false) + MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Bool(false) + MedRecordValue::Int(0))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Bool(false) + MedRecordValue::Float(0_f64))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Bool(false) + MedRecordValue::Bool(false))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Bool(false) + MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Bool(false) + MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!((MedRecordValue::DateTime(NaiveDateTime::MIN)
            + MedRecordValue::String("value".to_string()))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) + MedRecordValue::Int(0))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) + MedRecordValue::Float(0_f64))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) + MedRecordValue::Bool(false))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert_eq!(
            MedRecordValue::DateTime(
                NaiveDate::from_ymd_opt(1970, 1, 4)
                    .unwrap()
                    .and_time(NaiveTime::MIN)
            ),
            (MedRecordValue::DateTime(
                NaiveDate::from_ymd_opt(1970, 1, 2)
                    .unwrap()
                    .and_time(NaiveTime::MIN)
            ) + MedRecordValue::DateTime(
                NaiveDate::from_ymd_opt(1970, 1, 3)
                    .unwrap()
                    .and_time(NaiveTime::MIN)
            ))
            .unwrap()
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) + MedRecordValue::Null)
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );

        assert!(
            (MedRecordValue::Null + MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Null + MedRecordValue::Int(0))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Null + MedRecordValue::Float(0_f64))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Null + MedRecordValue::Bool(false))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Null + MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Null + MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_sub() {
        assert!((MedRecordValue::String("value".to_string())
            - MedRecordValue::String("value".to_string()))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::String("value".to_string()) - MedRecordValue::Int(0))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::String("value".to_string()) - MedRecordValue::Float(0_f64))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::String("value".to_string()) - MedRecordValue::Bool(false))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::String("value".to_string())
            - MedRecordValue::DateTime(NaiveDateTime::MIN))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::String("value".to_string()) - MedRecordValue::Null)
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );

        assert!(
            (MedRecordValue::Int(0) - MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert_eq!(
            MedRecordValue::Int(0),
            (MedRecordValue::Int(5) - MedRecordValue::Int(5)).unwrap()
        );
        assert_eq!(
            MedRecordValue::Float(0_f64),
            (MedRecordValue::Int(5) - MedRecordValue::Float(5_f64)).unwrap()
        );
        assert!((MedRecordValue::Int(0) - MedRecordValue::Bool(false))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Int(0) - MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Int(0) - MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!(
            (MedRecordValue::Float(0_f64) - MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert_eq!(
            MedRecordValue::Float(0_f64),
            (MedRecordValue::Float(5_f64) - MedRecordValue::Int(5)).unwrap()
        );
        assert_eq!(
            MedRecordValue::Float(0_f64),
            (MedRecordValue::Float(5_f64) - MedRecordValue::Float(5_f64)).unwrap()
        );
        assert!((MedRecordValue::Float(0_f64) - MedRecordValue::Bool(false))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Float(0_f64) - MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Float(0_f64) - MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!(
            (MedRecordValue::Bool(false) - MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Bool(false) - MedRecordValue::Int(0))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Bool(false) - MedRecordValue::Float(0_f64))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Bool(false) - MedRecordValue::Bool(false))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Bool(false) - MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Bool(false) - MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!((MedRecordValue::DateTime(NaiveDateTime::MIN)
            - MedRecordValue::String("value".to_string()))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) - MedRecordValue::Int(0))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) - MedRecordValue::Float(0_f64))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) - MedRecordValue::Bool(false))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) - MedRecordValue::Null)
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );

        assert!(
            (MedRecordValue::Null - MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Null - MedRecordValue::Int(0))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Null - MedRecordValue::Float(0_f64))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Null - MedRecordValue::Bool(false))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Null - MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Null - MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_mul() {
        assert!((MedRecordValue::String("value".to_string())
            * MedRecordValue::String("value".to_string()))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert_eq!(
            MedRecordValue::String("valuevaluevalue".to_string()),
            (MedRecordValue::String("value".to_string()) * MedRecordValue::Int(3)).unwrap()
        );
        assert!(
            (MedRecordValue::String("value".to_string()) * MedRecordValue::Float(0_f64))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::String("value".to_string()) * MedRecordValue::Bool(false))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::String("value".to_string())
            * MedRecordValue::DateTime(NaiveDateTime::MIN))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::String("value".to_string()) * MedRecordValue::Null)
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );

        assert_eq!(
            MedRecordValue::String("valuevaluevalue".to_string()),
            (MedRecordValue::Int(3) * MedRecordValue::String("value".to_string())).unwrap()
        );
        assert_eq!(
            MedRecordValue::Int(25),
            (MedRecordValue::Int(5) * MedRecordValue::Int(5)).unwrap()
        );
        assert_eq!(
            MedRecordValue::Float(25_f64),
            (MedRecordValue::Int(5) * MedRecordValue::Float(5_f64)).unwrap()
        );
        assert!((MedRecordValue::Int(0) * MedRecordValue::Bool(false))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Int(0) * MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Int(0) * MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!(
            (MedRecordValue::Float(0_f64) * MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert_eq!(
            MedRecordValue::Float(25_f64),
            (MedRecordValue::Float(5_f64) * MedRecordValue::Int(5)).unwrap()
        );
        assert_eq!(
            MedRecordValue::Float(25_f64),
            (MedRecordValue::Float(5_f64) * MedRecordValue::Float(5_f64)).unwrap()
        );
        assert!((MedRecordValue::Float(0_f64) * MedRecordValue::Bool(false))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Float(0_f64) * MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Float(0_f64) * MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!(
            (MedRecordValue::Bool(false) * MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Bool(false) * MedRecordValue::Int(0))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Bool(false) * MedRecordValue::Float(0_f64))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Bool(false) * MedRecordValue::Bool(false))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Bool(false) * MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Bool(false) * MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!((MedRecordValue::DateTime(NaiveDateTime::MIN)
            * MedRecordValue::String("value".to_string()))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) * MedRecordValue::Int(0))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) * MedRecordValue::Float(0_f64))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) * MedRecordValue::Bool(false))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::DateTime(NaiveDateTime::MIN)
            * MedRecordValue::DateTime(NaiveDateTime::MIN))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) * MedRecordValue::Null)
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );

        assert!(
            (MedRecordValue::Null * MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Null * MedRecordValue::Int(0))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Null * MedRecordValue::Float(0_f64))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Null * MedRecordValue::Bool(false))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Null * MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Null * MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_div() {
        assert!((MedRecordValue::String("value".to_string())
            / MedRecordValue::String("value".to_string()))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::String("value".to_string()) / MedRecordValue::Int(1))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::String("value".to_string()) / MedRecordValue::Float(1_f64))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::String("value".to_string()) / MedRecordValue::Bool(true))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::String("value".to_string())
            / MedRecordValue::DateTime(NaiveDateTime::MIN))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::String("value".to_string()) / MedRecordValue::Null)
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );

        assert!(
            (MedRecordValue::Int(0) / MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert_eq!(
            MedRecordValue::Float(1_f64),
            (MedRecordValue::Int(5) / MedRecordValue::Int(5)).unwrap()
        );
        assert_eq!(
            MedRecordValue::Float(1_f64),
            (MedRecordValue::Int(5) / MedRecordValue::Float(5_f64)).unwrap()
        );
        assert!((MedRecordValue::Int(0) / MedRecordValue::Bool(true))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Int(0) / MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Int(0) / MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!(
            (MedRecordValue::Float(0_f64) / MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert_eq!(
            MedRecordValue::Float(1_f64),
            (MedRecordValue::Float(5_f64) / MedRecordValue::Int(5)).unwrap()
        );
        assert_eq!(
            MedRecordValue::Float(1_f64),
            (MedRecordValue::Float(5_f64) / MedRecordValue::Float(5_f64)).unwrap()
        );
        assert!((MedRecordValue::Float(0_f64) / MedRecordValue::Bool(true))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Float(0_f64) / MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Float(0_f64) / MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!(
            (MedRecordValue::Bool(false) / MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Bool(false) / MedRecordValue::Int(1))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Bool(false) / MedRecordValue::Float(1_f64))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Bool(false) / MedRecordValue::Bool(true))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Bool(false) / MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Bool(false) / MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!((MedRecordValue::DateTime(NaiveDateTime::MIN)
            / MedRecordValue::String("value".to_string()))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert_eq!(
            MedRecordValue::DateTime(NaiveDateTime::MIN),
            (MedRecordValue::DateTime(NaiveDateTime::MIN) / MedRecordValue::Int(1)).unwrap()
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) / MedRecordValue::Float(1_f64))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) / MedRecordValue::Bool(true))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::DateTime(NaiveDateTime::MIN)
            / MedRecordValue::DateTime(NaiveDateTime::MIN))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN) / MedRecordValue::Null)
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );

        assert!(
            (MedRecordValue::Null / MedRecordValue::String("value".to_string()))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Null / MedRecordValue::Int(1))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Null / MedRecordValue::Float(1_f64))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Null / MedRecordValue::Bool(true))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Null / MedRecordValue::DateTime(NaiveDateTime::MIN))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Null / MedRecordValue::Null)
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_pow() {
        assert!((MedRecordValue::String("value".to_string())
            .pow(MedRecordValue::String("value".to_string())))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::String("value".to_string()).pow(MedRecordValue::Int(0)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::String("value".to_string()).pow(MedRecordValue::Float(0_f64)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::String("value".to_string()).pow(MedRecordValue::Bool(false)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::String("value".to_string())
            .pow(MedRecordValue::DateTime(NaiveDateTime::MIN)))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::String("value".to_string()).pow(MedRecordValue::Null))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );

        assert!(
            (MedRecordValue::Int(0).pow(MedRecordValue::String("value".to_string())))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert_eq!(
            MedRecordValue::Int(25),
            (MedRecordValue::Int(5).pow(MedRecordValue::Int(2))).unwrap()
        );
        assert_eq!(
            MedRecordValue::Float(25_f64),
            (MedRecordValue::Int(5).pow(MedRecordValue::Float(2_f64))).unwrap()
        );
        assert!((MedRecordValue::Int(0).pow(MedRecordValue::Bool(false)))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Int(0).pow(MedRecordValue::DateTime(NaiveDateTime::MIN)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Int(0).pow(MedRecordValue::Null))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!(
            (MedRecordValue::Float(0_f64).pow(MedRecordValue::String("value".to_string())))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert_eq!(
            MedRecordValue::Float(25_f64),
            (MedRecordValue::Float(5_f64).pow(MedRecordValue::Int(2))).unwrap()
        );
        assert_eq!(
            MedRecordValue::Float(25_f64),
            (MedRecordValue::Float(5_f64).pow(MedRecordValue::Float(2_f64))).unwrap()
        );
        assert!(
            (MedRecordValue::Float(0_f64).pow(MedRecordValue::Bool(false)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::Float(0_f64).pow(MedRecordValue::DateTime(NaiveDateTime::MIN)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Float(0_f64).pow(MedRecordValue::Null))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!(
            (MedRecordValue::Bool(false).pow(MedRecordValue::String("value".to_string())))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Bool(false).pow(MedRecordValue::Int(0)))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Bool(false).pow(MedRecordValue::Float(0_f64)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::Bool(false).pow(MedRecordValue::Bool(false)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::Bool(false).pow(MedRecordValue::DateTime(NaiveDateTime::MIN)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Bool(false).pow(MedRecordValue::Null))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!((MedRecordValue::DateTime(NaiveDateTime::MIN)
            .pow(MedRecordValue::String("value".to_string())))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN).pow(MedRecordValue::Int(0)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN).pow(MedRecordValue::Float(0_f64)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN).pow(MedRecordValue::Bool(false)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::DateTime(NaiveDateTime::MIN)
            .pow(MedRecordValue::DateTime(NaiveDateTime::MIN)))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN).pow(MedRecordValue::Null))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );

        assert!(
            (MedRecordValue::Null.pow(MedRecordValue::String("value".to_string())))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Null.pow(MedRecordValue::Int(0)))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Null.pow(MedRecordValue::Float(0_f64)))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Null.pow(MedRecordValue::Bool(false)))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Null.pow(MedRecordValue::DateTime(NaiveDateTime::MIN)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Null.pow(MedRecordValue::Null))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_mod() {
        assert!((MedRecordValue::String("value".to_string())
            .r#mod(MedRecordValue::String("value".to_string())))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::String("value".to_string()).r#mod(MedRecordValue::Int(0)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::String("value".to_string()).r#mod(MedRecordValue::Float(0_f64)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::String("value".to_string()).r#mod(MedRecordValue::Bool(false)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::String("value".to_string())
            .r#mod(MedRecordValue::DateTime(NaiveDateTime::MIN)))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::String("value".to_string()).r#mod(MedRecordValue::Null))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );

        assert!(
            (MedRecordValue::Int(0).r#mod(MedRecordValue::String("value".to_string())))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert_eq!(
            MedRecordValue::Int(1),
            (MedRecordValue::Int(5).r#mod(MedRecordValue::Int(2))).unwrap()
        );
        assert_eq!(
            MedRecordValue::Float(1_f64),
            (MedRecordValue::Int(5).r#mod(MedRecordValue::Float(2_f64))).unwrap()
        );
        assert!((MedRecordValue::Int(0).r#mod(MedRecordValue::Bool(false)))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Int(0).r#mod(MedRecordValue::DateTime(NaiveDateTime::MIN)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Int(0).r#mod(MedRecordValue::Null))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!(
            (MedRecordValue::Float(0_f64).r#mod(MedRecordValue::String("value".to_string())))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert_eq!(
            MedRecordValue::Float(1_f64),
            (MedRecordValue::Float(5_f64).r#mod(MedRecordValue::Int(2))).unwrap()
        );
        assert_eq!(
            MedRecordValue::Float(1_f64),
            (MedRecordValue::Float(5_f64).r#mod(MedRecordValue::Float(2_f64))).unwrap()
        );
        assert!(
            (MedRecordValue::Float(0_f64).r#mod(MedRecordValue::Bool(false)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::Float(0_f64).r#mod(MedRecordValue::DateTime(NaiveDateTime::MIN)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Float(0_f64).r#mod(MedRecordValue::Null))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!(
            (MedRecordValue::Bool(false).r#mod(MedRecordValue::String("value".to_string())))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Bool(false).r#mod(MedRecordValue::Int(0)))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Bool(false).r#mod(MedRecordValue::Float(0_f64)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::Bool(false).r#mod(MedRecordValue::Bool(false)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::Bool(false).r#mod(MedRecordValue::DateTime(NaiveDateTime::MIN)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Bool(false).r#mod(MedRecordValue::Null))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));

        assert!((MedRecordValue::DateTime(NaiveDateTime::MIN)
            .r#mod(MedRecordValue::String("value".to_string())))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN).r#mod(MedRecordValue::Int(0)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN).r#mod(MedRecordValue::Float(0_f64)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN).r#mod(MedRecordValue::Bool(false)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::DateTime(NaiveDateTime::MIN)
            .r#mod(MedRecordValue::DateTime(NaiveDateTime::MIN)))
        .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::DateTime(NaiveDateTime::MIN).r#mod(MedRecordValue::Null))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );

        assert!(
            (MedRecordValue::Null.r#mod(MedRecordValue::String("value".to_string())))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Null.r#mod(MedRecordValue::Int(0)))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Null.r#mod(MedRecordValue::Float(0_f64)))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!((MedRecordValue::Null.r#mod(MedRecordValue::Bool(false)))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
        assert!(
            (MedRecordValue::Null.r#mod(MedRecordValue::DateTime(NaiveDateTime::MIN)))
                .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_)))
        );
        assert!((MedRecordValue::Null.r#mod(MedRecordValue::Null))
            .is_err_and(|e| matches!(e, MedRecordError::AssertionError(_))));
    }

    #[test]
    fn test_starts_with() {
        assert!(MedRecordValue::String("value".to_string())
            .starts_with(&MedRecordValue::String("val".to_string())));
        assert!(!MedRecordValue::String("value".to_string())
            .starts_with(&MedRecordValue::String("not_val".to_string())));
        assert!(MedRecordValue::String("10".to_string()).starts_with(&MedRecordValue::Int(1)));
        assert!(!MedRecordValue::String("10".to_string()).starts_with(&MedRecordValue::Int(0)));
        assert!(MedRecordValue::String("10".to_string()).starts_with(&MedRecordValue::Float(1_f64)));
        assert!(
            !MedRecordValue::String("10".to_string()).starts_with(&MedRecordValue::Float(0_f64))
        );

        assert!(MedRecordValue::Int(10).starts_with(&MedRecordValue::String("1".to_string())));
        assert!(!MedRecordValue::Int(10).starts_with(&MedRecordValue::String("0".to_string())));
        assert!(MedRecordValue::Int(10).starts_with(&MedRecordValue::Int(1)));
        assert!(!MedRecordValue::Int(10).starts_with(&MedRecordValue::Int(0)));
        assert!(MedRecordValue::Int(10).starts_with(&MedRecordValue::Float(1_f64)));
        assert!(!MedRecordValue::Int(10).starts_with(&MedRecordValue::Float(0_f64)));

        assert!(MedRecordValue::Float(10_f64).starts_with(&MedRecordValue::String("1".to_string())));
        assert!(
            !MedRecordValue::Float(10_f64).starts_with(&MedRecordValue::String("0".to_string()))
        );
        assert!(MedRecordValue::Float(10_f64).starts_with(&MedRecordValue::Int(1)));
        assert!(!MedRecordValue::Float(10_f64).starts_with(&MedRecordValue::Int(0)));
        assert!(MedRecordValue::Float(10_f64).starts_with(&MedRecordValue::Float(1_f64)));
        assert!(!MedRecordValue::Float(10_f64).starts_with(&MedRecordValue::Float(0_f64)));

        assert!(
            !MedRecordValue::String("true".to_string()).starts_with(&MedRecordValue::Bool(true))
        );
        assert!(
            !MedRecordValue::String("-262143-01-01 00:00:00".to_string())
                .starts_with(&MedRecordValue::DateTime(NaiveDateTime::MIN))
        );
        assert!(!MedRecordValue::String("Null".to_string()).starts_with(&MedRecordValue::Null));

        assert!(!MedRecordValue::Int(1).starts_with(&MedRecordValue::Bool(true)));
        assert!(!MedRecordValue::Int(-2).starts_with(&MedRecordValue::DateTime(NaiveDateTime::MIN)));
        assert!(!MedRecordValue::Int(0).starts_with(&MedRecordValue::Null));

        assert!(!MedRecordValue::Float(1_f64).starts_with(&MedRecordValue::Bool(true)));
        assert!(!MedRecordValue::Float(-2_f64)
            .starts_with(&MedRecordValue::DateTime(NaiveDateTime::MIN)));
        assert!(!MedRecordValue::Float(0_f64).starts_with(&MedRecordValue::Null));

        assert!(
            !MedRecordValue::Bool(true).starts_with(&MedRecordValue::String("true".to_string()))
        );
        assert!(!MedRecordValue::Bool(true).starts_with(&MedRecordValue::Int(1)));
        assert!(!MedRecordValue::Bool(true).starts_with(&MedRecordValue::Float(1_f64)));
        assert!(!MedRecordValue::Bool(true).starts_with(&MedRecordValue::Bool(true)));
        assert!(
            !MedRecordValue::Bool(true).starts_with(&MedRecordValue::DateTime(NaiveDateTime::MAX))
        );
        assert!(!MedRecordValue::Bool(false).starts_with(&MedRecordValue::Null));

        assert!(!MedRecordValue::DateTime(NaiveDateTime::MAX).starts_with(
            &MedRecordValue::String("-262143-01-01 00:00:00".to_string())
        ));
        assert!(!MedRecordValue::DateTime(NaiveDateTime::MAX).starts_with(&MedRecordValue::Int(-2)));
        assert!(!MedRecordValue::DateTime(NaiveDateTime::MAX)
            .starts_with(&MedRecordValue::Float(1_f64)));
        assert!(
            !MedRecordValue::DateTime(NaiveDateTime::MAX).starts_with(&MedRecordValue::Bool(false))
        );
        assert!(!MedRecordValue::DateTime(NaiveDateTime::MAX)
            .starts_with(&MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!MedRecordValue::DateTime(NaiveDateTime::MIN).starts_with(&MedRecordValue::Null));

        assert!(!MedRecordValue::Null.starts_with(&MedRecordValue::String("Null".to_string())));
        assert!(!MedRecordValue::Null.starts_with(&MedRecordValue::Int(0)));
        assert!(!MedRecordValue::Null.starts_with(&MedRecordValue::Float(0_f64)));
        assert!(!MedRecordValue::Null.starts_with(&MedRecordValue::Bool(false)));
        assert!(!MedRecordValue::Null.starts_with(&MedRecordValue::DateTime(NaiveDateTime::MIN)));
        assert!(!MedRecordValue::Null.starts_with(&MedRecordValue::Null));
    }

    #[test]
    fn test_ends_with() {
        assert!(MedRecordValue::String("value".to_string())
            .ends_with(&MedRecordValue::String("ue".to_string())));
        assert!(!MedRecordValue::String("value".to_string())
            .ends_with(&MedRecordValue::String("not_ue".to_string())));
        assert!(MedRecordValue::String("10".to_string()).ends_with(&MedRecordValue::Int(0)));
        assert!(!MedRecordValue::String("10".to_string()).ends_with(&MedRecordValue::Int(1)));
        assert!(MedRecordValue::String("10".to_string()).ends_with(&MedRecordValue::Float(0_f64)));
        assert!(!MedRecordValue::String("10".to_string()).ends_with(&MedRecordValue::Float(1_f64)));

        assert!(MedRecordValue::Int(10).ends_with(&MedRecordValue::String("0".to_string())));
        assert!(!MedRecordValue::Int(10).ends_with(&MedRecordValue::String("1".to_string())));
        assert!(MedRecordValue::Int(10).ends_with(&MedRecordValue::Int(0)));
        assert!(!MedRecordValue::Int(10).ends_with(&MedRecordValue::Int(1)));
        assert!(MedRecordValue::Int(10).ends_with(&MedRecordValue::Float(0_f64)));
        assert!(!MedRecordValue::Int(10).ends_with(&MedRecordValue::Float(1_f64)));

        assert!(MedRecordValue::Float(10_f64).ends_with(&MedRecordValue::String("0".to_string())));
        assert!(!MedRecordValue::Float(10_f64).ends_with(&MedRecordValue::String("1".to_string())));
        assert!(MedRecordValue::Float(10_f64).ends_with(&MedRecordValue::Int(0)));
        assert!(!MedRecordValue::Float(10_f64).ends_with(&MedRecordValue::Int(1)));
        assert!(MedRecordValue::Float(10_f64).ends_with(&MedRecordValue::Float(0_f64)));
        assert!(!MedRecordValue::Float(10_f64).ends_with(&MedRecordValue::Float(1_f64)));

        assert!(!MedRecordValue::String("true".to_string()).ends_with(&MedRecordValue::Bool(true)));
        assert!(
            !MedRecordValue::String("-262143-01-01 00:00:00".to_string())
                .ends_with(&MedRecordValue::DateTime(NaiveDateTime::MIN))
        );
        assert!(!MedRecordValue::String("Null".to_string()).ends_with(&MedRecordValue::Null));

        assert!(!MedRecordValue::Int(1).ends_with(&MedRecordValue::Bool(true)));
        assert!(!MedRecordValue::Int(0).ends_with(&MedRecordValue::DateTime(NaiveDateTime::MIN)));
        assert!(!MedRecordValue::Int(0).ends_with(&MedRecordValue::Null));

        assert!(!MedRecordValue::Float(1_f64).ends_with(&MedRecordValue::Bool(true)));
        assert!(
            !MedRecordValue::Float(0_f64).ends_with(&MedRecordValue::DateTime(NaiveDateTime::MIN))
        );
        assert!(!MedRecordValue::Float(0_f64).ends_with(&MedRecordValue::Null));

        assert!(!MedRecordValue::Bool(true).ends_with(&MedRecordValue::String("true".to_string())));
        assert!(!MedRecordValue::Bool(true).ends_with(&MedRecordValue::Int(1)));
        assert!(!MedRecordValue::Bool(true).ends_with(&MedRecordValue::Float(1_f64)));
        assert!(!MedRecordValue::Bool(true).ends_with(&MedRecordValue::Bool(true)));
        assert!(
            !MedRecordValue::Bool(true).ends_with(&MedRecordValue::DateTime(NaiveDateTime::MAX))
        );
        assert!(!MedRecordValue::Bool(false).ends_with(&MedRecordValue::Null));

        assert!(
            !MedRecordValue::DateTime(NaiveDateTime::MIN).ends_with(&MedRecordValue::String(
                "-262143-01-01 00:00:00".to_string()
            ))
        );
        assert!(!MedRecordValue::DateTime(NaiveDateTime::MAX).ends_with(&MedRecordValue::Int(0)));
        assert!(
            !MedRecordValue::DateTime(NaiveDateTime::MAX).ends_with(&MedRecordValue::Float(0_f64))
        );
        assert!(
            !MedRecordValue::DateTime(NaiveDateTime::MAX).ends_with(&MedRecordValue::Bool(false))
        );
        assert!(!MedRecordValue::DateTime(NaiveDateTime::MAX)
            .ends_with(&MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!MedRecordValue::DateTime(NaiveDateTime::MAX).ends_with(&MedRecordValue::Null));

        assert!(!MedRecordValue::Null.ends_with(&MedRecordValue::String("true".to_string())));
        assert!(!MedRecordValue::Null.ends_with(&MedRecordValue::Int(1)));
        assert!(!MedRecordValue::Null.ends_with(&MedRecordValue::Float(1_f64)));
        assert!(!MedRecordValue::Null.ends_with(&MedRecordValue::Bool(false)));
        assert!(!MedRecordValue::Null.ends_with(&MedRecordValue::DateTime(NaiveDateTime::MIN)));
        assert!(!MedRecordValue::Null.ends_with(&MedRecordValue::Null));
    }

    #[test]
    fn test_contains() {
        assert!(MedRecordValue::String("value".to_string())
            .contains(&MedRecordValue::String("al".to_string())));
        assert!(!MedRecordValue::String("value".to_string())
            .contains(&MedRecordValue::String("not_al".to_string())));
        assert!(MedRecordValue::String("10".to_string()).contains(&MedRecordValue::Int(0)));
        assert!(!MedRecordValue::String("10".to_string()).contains(&MedRecordValue::Int(2)));
        assert!(MedRecordValue::String("10".to_string()).contains(&MedRecordValue::Float(0_f64)));
        assert!(!MedRecordValue::String("10".to_string()).contains(&MedRecordValue::Float(2_f64)));

        assert!(MedRecordValue::Int(10).contains(&MedRecordValue::String("0".to_string())));
        assert!(!MedRecordValue::Int(10).contains(&MedRecordValue::String("2".to_string())));
        assert!(MedRecordValue::Int(10).contains(&MedRecordValue::Int(0)));
        assert!(!MedRecordValue::Int(10).contains(&MedRecordValue::Int(2)));
        assert!(MedRecordValue::Int(10).contains(&MedRecordValue::Float(0_f64)));
        assert!(!MedRecordValue::Int(10).contains(&MedRecordValue::Float(2_f64)));

        assert!(MedRecordValue::Float(10_f64).contains(&MedRecordValue::String("0".to_string())));
        assert!(!MedRecordValue::Float(10_f64).contains(&MedRecordValue::String("2".to_string())));
        assert!(MedRecordValue::Float(10_f64).contains(&MedRecordValue::Int(0)));
        assert!(!MedRecordValue::Float(10_f64).contains(&MedRecordValue::Int(2)));
        assert!(MedRecordValue::Float(10_f64).contains(&MedRecordValue::Float(0_f64)));
        assert!(!MedRecordValue::Float(10_f64).contains(&MedRecordValue::Float(2_f64)));

        assert!(!MedRecordValue::String("true".to_string()).contains(&MedRecordValue::Bool(true)));
        assert!(
            !MedRecordValue::String("-262143-01-01 00:00:00".to_string())
                .contains(&MedRecordValue::DateTime(NaiveDateTime::MIN))
        );
        assert!(!MedRecordValue::String("Null".to_string()).contains(&MedRecordValue::Null));

        assert!(!MedRecordValue::Int(1).contains(&MedRecordValue::Bool(true)));
        assert!(!MedRecordValue::Int(0).contains(&MedRecordValue::DateTime(NaiveDateTime::MIN)));
        assert!(!MedRecordValue::Int(0).contains(&MedRecordValue::Null));

        assert!(!MedRecordValue::Float(1_f64).contains(&MedRecordValue::Bool(true)));
        assert!(
            !MedRecordValue::Float(0_f64).contains(&MedRecordValue::DateTime(NaiveDateTime::MIN))
        );
        assert!(!MedRecordValue::Float(0_f64).contains(&MedRecordValue::Null));

        assert!(!MedRecordValue::Bool(true).contains(&MedRecordValue::String("true".to_string())));
        assert!(!MedRecordValue::Bool(true).contains(&MedRecordValue::Int(1)));
        assert!(!MedRecordValue::Bool(true).contains(&MedRecordValue::Float(1_f64)));
        assert!(!MedRecordValue::Bool(true).contains(&MedRecordValue::Bool(true)));
        assert!(!MedRecordValue::Bool(true).contains(&MedRecordValue::DateTime(NaiveDateTime::MAX)));
        assert!(!MedRecordValue::Bool(false).contains(&MedRecordValue::Null));

        assert!(
            !MedRecordValue::DateTime(NaiveDateTime::MIN).contains(&MedRecordValue::String(
                "-262143-01-01 00:00:00".to_string()
            ))
        );
        assert!(!MedRecordValue::DateTime(NaiveDateTime::MIN).contains(&MedRecordValue::Int(0)));
        assert!(
            !MedRecordValue::DateTime(NaiveDateTime::MIN).contains(&MedRecordValue::Float(0_f64))
        );
        assert!(
            !MedRecordValue::DateTime(NaiveDateTime::MIN).contains(&MedRecordValue::Bool(false))
        );
        assert!(!MedRecordValue::DateTime(NaiveDateTime::MIN)
            .contains(&MedRecordValue::DateTime(NaiveDateTime::MIN)));
        assert!(!MedRecordValue::DateTime(NaiveDateTime::MIN).contains(&MedRecordValue::Null));

        assert!(!MedRecordValue::Null.contains(&MedRecordValue::String("true".to_string())));
        assert!(!MedRecordValue::Null.contains(&MedRecordValue::Int(1)));
        assert!(!MedRecordValue::Null.contains(&MedRecordValue::Float(1_f64)));
        assert!(!MedRecordValue::Null.contains(&MedRecordValue::Bool(true)));
        assert!(!MedRecordValue::Null.contains(&MedRecordValue::DateTime(NaiveDateTime::MIN)));
        assert!(!MedRecordValue::Null.contains(&MedRecordValue::Null));
    }

    #[test]
    fn test_slice() {
        assert_eq!(
            MedRecordValue::String("al".to_string()),
            MedRecordValue::String("value".to_string()).slice(1..3)
        );

        assert_eq!(
            MedRecordValue::String("23".to_string()),
            MedRecordValue::Int(1234).slice(1..3)
        );

        assert_eq!(
            MedRecordValue::String("23".to_string()),
            MedRecordValue::Float(1234_f64).slice(1..3)
        );

        assert_eq!(
            MedRecordValue::String("al".to_string()),
            MedRecordValue::Bool(false).slice(1..3)
        );

        assert_eq!(
            MedRecordValue::DateTime(NaiveDateTime::MIN),
            MedRecordValue::DateTime(NaiveDateTime::MIN).slice(1..3)
        );

        assert_eq!(MedRecordValue::Null, MedRecordValue::Null.slice(1..3));
    }

    #[test]
    fn test_round() {
        assert_eq!(
            MedRecordValue::String("value".to_string()),
            MedRecordValue::String("value".to_string()).round()
        );

        assert_eq!(MedRecordValue::Int(1234), MedRecordValue::Int(1234).round());

        assert_eq!(
            MedRecordValue::Float(1234_f64),
            MedRecordValue::Float(1234.3).round()
        );

        assert_eq!(
            MedRecordValue::Bool(false),
            MedRecordValue::Bool(false).round()
        );

        assert_eq!(
            MedRecordValue::DateTime(NaiveDateTime::MIN),
            MedRecordValue::DateTime(NaiveDateTime::MIN).round()
        );

        assert_eq!(MedRecordValue::Null, MedRecordValue::Null.round());
    }

    #[test]
    fn test_ceil() {
        assert_eq!(
            MedRecordValue::String("value".to_string()),
            MedRecordValue::String("value".to_string()).ceil()
        );

        assert_eq!(MedRecordValue::Int(1234), MedRecordValue::Int(1234).ceil());

        assert_eq!(
            MedRecordValue::Float(1235_f64),
            MedRecordValue::Float(1234.3).ceil()
        );

        assert_eq!(
            MedRecordValue::Bool(false),
            MedRecordValue::Bool(false).ceil()
        );

        assert_eq!(
            MedRecordValue::DateTime(NaiveDateTime::MIN),
            MedRecordValue::DateTime(NaiveDateTime::MIN).ceil()
        );

        assert_eq!(MedRecordValue::Null, MedRecordValue::Null.ceil());
    }

    #[test]
    fn test_floor() {
        assert_eq!(
            MedRecordValue::String("value".to_string()),
            MedRecordValue::String("value".to_string()).floor()
        );

        assert_eq!(MedRecordValue::Int(1234), MedRecordValue::Int(1234).floor());

        assert_eq!(
            MedRecordValue::Float(1234_f64),
            MedRecordValue::Float(1234.3).floor()
        );

        assert_eq!(
            MedRecordValue::Bool(false),
            MedRecordValue::Bool(false).floor()
        );

        assert_eq!(
            MedRecordValue::DateTime(NaiveDateTime::MAX),
            MedRecordValue::DateTime(NaiveDateTime::MAX).floor()
        );

        assert_eq!(MedRecordValue::Null, MedRecordValue::Null.floor());
    }

    #[test]
    fn test_abs() {
        assert_eq!(
            MedRecordValue::String("value".to_string()),
            MedRecordValue::String("value".to_string()).abs()
        );

        assert_eq!(MedRecordValue::Int(1234), MedRecordValue::Int(1234).abs());
        assert_eq!(MedRecordValue::Int(1234), MedRecordValue::Int(-1234).abs());

        assert_eq!(
            MedRecordValue::Float(1234_f64),
            MedRecordValue::Float(1234_f64).abs()
        );
        assert_eq!(
            MedRecordValue::Float(1234_f64),
            MedRecordValue::Float(-1234_f64).abs()
        );

        assert_eq!(
            MedRecordValue::Bool(false),
            MedRecordValue::Bool(false).abs()
        );

        assert_eq!(
            MedRecordValue::DateTime(NaiveDateTime::MIN),
            MedRecordValue::DateTime(NaiveDateTime::MIN).abs()
        );

        assert_eq!(MedRecordValue::Null, MedRecordValue::Null.abs());
    }

    #[test]
    fn test_sqrt() {
        assert_eq!(
            MedRecordValue::String("value".to_string()),
            MedRecordValue::String("value".to_string()).sqrt()
        );

        assert_eq!(MedRecordValue::Float(2_f64), MedRecordValue::Int(4).sqrt());

        assert_eq!(
            MedRecordValue::Float(2_f64),
            MedRecordValue::Float(4_f64).sqrt()
        );

        assert_eq!(
            MedRecordValue::Bool(false),
            MedRecordValue::Bool(false).sqrt()
        );

        assert_eq!(
            MedRecordValue::DateTime(NaiveDateTime::MIN),
            MedRecordValue::DateTime(NaiveDateTime::MIN).sqrt()
        );

        assert_eq!(MedRecordValue::Null, MedRecordValue::Null.sqrt());
    }

    #[test]
    fn test_trim() {
        assert_eq!(
            MedRecordValue::String("value".to_string()),
            MedRecordValue::String("  value  ".to_string()).trim()
        );

        assert_eq!(MedRecordValue::Int(1234), MedRecordValue::Int(1234).trim());

        assert_eq!(
            MedRecordValue::Float(1234_f64),
            MedRecordValue::Float(1234_f64).trim()
        );

        assert_eq!(
            MedRecordValue::Bool(false),
            MedRecordValue::Bool(false).trim()
        );

        assert_eq!(
            MedRecordValue::DateTime(NaiveDateTime::MIN),
            MedRecordValue::DateTime(NaiveDateTime::MIN).trim()
        );

        assert_eq!(MedRecordValue::Null, MedRecordValue::Null.trim());
    }

    #[test]
    fn test_trim_start() {
        assert_eq!(
            MedRecordValue::String("value  ".to_string()),
            MedRecordValue::String("  value  ".to_string()).trim_start()
        );

        assert_eq!(
            MedRecordValue::Int(1234),
            MedRecordValue::Int(1234).trim_start()
        );

        assert_eq!(
            MedRecordValue::Float(1234_f64),
            MedRecordValue::Float(1234_f64).trim_start()
        );

        assert_eq!(
            MedRecordValue::Bool(false),
            MedRecordValue::Bool(false).trim_start()
        );

        assert_eq!(
            MedRecordValue::DateTime(NaiveDateTime::MIN),
            MedRecordValue::DateTime(NaiveDateTime::MIN).trim_start()
        );

        assert_eq!(MedRecordValue::Null, MedRecordValue::Null.trim_start());
    }

    #[test]
    fn test_trim_end() {
        assert_eq!(
            MedRecordValue::String("  value".to_string()),
            MedRecordValue::String("  value  ".to_string()).trim_end()
        );

        assert_eq!(
            MedRecordValue::Int(1234),
            MedRecordValue::Int(1234).trim_end()
        );

        assert_eq!(
            MedRecordValue::Float(1234_f64),
            MedRecordValue::Float(1234_f64).trim_end()
        );

        assert_eq!(
            MedRecordValue::Bool(false),
            MedRecordValue::Bool(false).trim_end()
        );

        assert_eq!(
            MedRecordValue::DateTime(NaiveDateTime::MIN),
            MedRecordValue::DateTime(NaiveDateTime::MIN).trim_end()
        );

        assert_eq!(MedRecordValue::Null, MedRecordValue::Null.trim_end());
    }

    #[test]
    fn test_lowercase() {
        assert_eq!(
            MedRecordValue::String("value".to_string()),
            MedRecordValue::String("VaLuE".to_string()).lowercase()
        );

        assert_eq!(
            MedRecordValue::Int(1234),
            MedRecordValue::Int(1234).lowercase()
        );

        assert_eq!(
            MedRecordValue::Float(1234_f64),
            MedRecordValue::Float(1234_f64).lowercase()
        );

        assert_eq!(
            MedRecordValue::Bool(false),
            MedRecordValue::Bool(false).lowercase()
        );

        assert_eq!(
            MedRecordValue::DateTime(NaiveDateTime::MIN),
            MedRecordValue::DateTime(NaiveDateTime::MIN).lowercase()
        );

        assert_eq!(MedRecordValue::Null, MedRecordValue::Null.lowercase());
    }

    #[test]
    fn test_uppercase() {
        assert_eq!(
            MedRecordValue::String("VALUE".to_string()),
            MedRecordValue::String("VaLuE".to_string()).uppercase()
        );

        assert_eq!(
            MedRecordValue::Int(1234),
            MedRecordValue::Int(1234).uppercase()
        );

        assert_eq!(
            MedRecordValue::Float(1234_f64),
            MedRecordValue::Float(1234_f64).uppercase()
        );

        assert_eq!(
            MedRecordValue::Bool(false),
            MedRecordValue::Bool(false).uppercase()
        );

        assert_eq!(
            MedRecordValue::DateTime(NaiveDateTime::MIN),
            MedRecordValue::DateTime(NaiveDateTime::MIN).uppercase()
        );

        assert_eq!(MedRecordValue::Null, MedRecordValue::Null.uppercase());
    }
}
