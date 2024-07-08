mod attribute;
mod value;

pub use self::{attribute::MedRecordAttribute, value::MedRecordValue};
use crate::errors::MedRecordError;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, ops::Range};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DataType {
    String,
    Int,
    Float,
    Bool,
    Null,
    Any,
    Union((Box<DataType>, Box<DataType>)),
    Option(Box<DataType>),
}

impl DataType {
    pub(crate) fn evaluate(&self, other: &Self) -> bool {
        match (self, other) {
            (DataType::Union(_), DataType::Union(_)) => self == other,
            (DataType::Union((first_datatype, second_datatype)), _) => {
                first_datatype.evaluate(other) || second_datatype.evaluate(other)
            }
            (DataType::Option(_), DataType::Option(_)) => self == other,
            (DataType::Option(_), DataType::Null) => true,
            (DataType::Option(datatype), _) => datatype.evaluate(other),
            (DataType::Any, _) => true,
            _ => matches!(
                (self, other),
                (DataType::String, DataType::String)
                    | (DataType::Int, DataType::Int)
                    | (DataType::Float, DataType::Float)
                    | (DataType::Bool, DataType::Bool)
                    | (DataType::Null, DataType::Null)
                    | (DataType::Any, DataType::Any)
            ),
        }
    }
}

impl Default for DataType {
    fn default() -> Self {
        Self::Any
    }
}

impl From<MedRecordValue> for DataType {
    fn from(value: MedRecordValue) -> Self {
        match value {
            MedRecordValue::String(_) => DataType::String,
            MedRecordValue::Int(_) => DataType::Int,
            MedRecordValue::Float(_) => DataType::Float,
            MedRecordValue::Bool(_) => DataType::Bool,
            MedRecordValue::Null => DataType::Null,
        }
    }
}

impl From<&MedRecordValue> for DataType {
    fn from(value: &MedRecordValue) -> Self {
        match value {
            MedRecordValue::String(_) => DataType::String,
            MedRecordValue::Int(_) => DataType::Int,
            MedRecordValue::Float(_) => DataType::Float,
            MedRecordValue::Bool(_) => DataType::Bool,
            MedRecordValue::Null => DataType::Null,
        }
    }
}

impl PartialEq for DataType {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (DataType::Union(first_union), DataType::Union(second_union)) => {
                (first_union.0 == second_union.0 && first_union.1 == second_union.1)
                    || (first_union.1 == second_union.0 && first_union.0 == second_union.1)
            }
            (DataType::Option(first_datatype), DataType::Option(second_datatype)) => {
                first_datatype == second_datatype
            }
            _ => matches!(
                (self, other),
                (DataType::String, DataType::String)
                    | (DataType::Int, DataType::Int)
                    | (DataType::Float, DataType::Float)
                    | (DataType::Bool, DataType::Bool)
                    | (DataType::Null, DataType::Null)
                    | (DataType::Any, DataType::Any)
            ),
        }
    }
}

impl Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::String => write!(f, "String"),
            DataType::Int => write!(f, "Integer"),
            DataType::Float => write!(f, "Float"),
            DataType::Bool => write!(f, "Boolean"),
            DataType::Null => write!(f, "Null"),
            DataType::Any => write!(f, "Any"),
            DataType::Union((first_datatype, second_datatype)) => {
                write!(f, "Union[")?;
                first_datatype.fmt(f)?;
                write!(f, ", ")?;
                second_datatype.fmt(f)?;
                write!(f, "]")
            }
            DataType::Option(data_type) => {
                write!(f, "Option(")?;
                data_type.fmt(f)?;
                write!(f, ")")
            }
        }
    }
}

pub trait Pow: Sized {
    fn pow(self, exp: Self) -> Result<Self, MedRecordError>;
}

pub trait Mod: Sized {
    fn r#mod(self, other: Self) -> Result<Self, MedRecordError>;
}

pub trait StartsWith {
    fn starts_with(&self, other: &Self) -> bool;
}

pub trait EndsWith {
    fn ends_with(&self, other: &Self) -> bool;
}

pub trait Contains {
    fn contains(&self, other: &Self) -> bool;
}

pub trait PartialNeq: PartialEq {
    fn neq(&self, other: &Self) -> bool;
}

pub trait Round {
    fn round(self) -> Self;
}

pub trait Ceil {
    fn ceil(self) -> Self;
}

pub trait Floor {
    fn floor(self) -> Self;
}

pub trait Abs {
    fn abs(self) -> Self;
}

pub trait Sqrt {
    fn sqrt(self) -> Self;
}

pub trait Trim {
    fn trim(self) -> Self;
}

pub trait TrimStart {
    fn trim_start(self) -> Self;
}

pub trait TrimEnd {
    fn trim_end(self) -> Self;
}

pub trait Lowercase {
    fn lowercase(self) -> Self;
}

pub trait Uppercase {
    fn uppercase(self) -> Self;
}

pub trait Slice {
    fn slice(self, range: Range<usize>) -> Self;
}

impl<T> PartialNeq for T
where
    T: PartialOrd,
{
    fn neq(&self, other: &Self) -> bool {
        self != other
    }
}
