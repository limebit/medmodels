mod attribute;
mod value;

pub use self::{attribute::MedRecordAttribute, value::MedRecordValue};
use super::EdgeIndex;
use crate::errors::MedRecordError;
use serde::{Deserialize, Serialize};
use std::{fmt::Display, ops::Range};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DataType {
    String,
    Int,
    Float,
    Bool,
    DateTime,
    Duration,
    Null,
    Any,
    Union((Box<DataType>, Box<DataType>)),
    Option(Box<DataType>),
}

impl Default for DataType {
    fn default() -> Self {
        Self::Any
    }
}

// TODO: Add tests for Duration
impl From<MedRecordValue> for DataType {
    fn from(value: MedRecordValue) -> Self {
        match value {
            MedRecordValue::String(_) => DataType::String,
            MedRecordValue::Int(_) => DataType::Int,
            MedRecordValue::Float(_) => DataType::Float,
            MedRecordValue::Bool(_) => DataType::Bool,
            MedRecordValue::DateTime(_) => DataType::DateTime,
            MedRecordValue::Duration(_) => DataType::Duration,
            MedRecordValue::Null => DataType::Null,
        }
    }
}

// TODO: Add tests for Duration
impl From<&MedRecordValue> for DataType {
    fn from(value: &MedRecordValue) -> Self {
        match value {
            MedRecordValue::String(_) => DataType::String,
            MedRecordValue::Int(_) => DataType::Int,
            MedRecordValue::Float(_) => DataType::Float,
            MedRecordValue::Bool(_) => DataType::Bool,
            MedRecordValue::DateTime(_) => DataType::DateTime,
            MedRecordValue::Duration(_) => DataType::Duration,
            MedRecordValue::Null => DataType::Null,
        }
    }
}

impl From<MedRecordAttribute> for DataType {
    fn from(value: MedRecordAttribute) -> Self {
        match value {
            MedRecordAttribute::String(_) => DataType::String,
            MedRecordAttribute::Int(_) => DataType::Int,
        }
    }
}

impl From<&MedRecordAttribute> for DataType {
    fn from(value: &MedRecordAttribute) -> Self {
        match value {
            MedRecordAttribute::String(_) => DataType::String,
            MedRecordAttribute::Int(_) => DataType::Int,
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
                    | (DataType::DateTime, DataType::DateTime)
                    | (DataType::Null, DataType::Null)
                    | (DataType::Any, DataType::Any)
            ),
        }
    }
}

// TODO: Add tests for Duration
impl Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::String => write!(f, "String"),
            DataType::Int => write!(f, "Int"),
            DataType::Float => write!(f, "Float"),
            DataType::Bool => write!(f, "Bool"),
            DataType::DateTime => write!(f, "DateTime"),
            DataType::Duration => write!(f, "Duration"),
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
                write!(f, "Option[")?;
                data_type.fmt(f)?;
                write!(f, "]")
            }
        }
    }
}

// TODO: Add tests for Duration
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
                    | (DataType::DateTime, DataType::DateTime)
                    | (DataType::Duration, DataType::Duration)
                    | (DataType::Null, DataType::Null)
                    | (DataType::Any, DataType::Any)
            ),
        }
    }
}

pub trait StartsWith {
    fn starts_with(&self, other: &Self) -> bool;
}

// TODO: Add tests
impl StartsWith for EdgeIndex {
    fn starts_with(&self, other: &Self) -> bool {
        self.to_string().starts_with(&other.to_string())
    }
}

pub trait EndsWith {
    fn ends_with(&self, other: &Self) -> bool;
}

// TODO: Add tests
impl EndsWith for EdgeIndex {
    fn ends_with(&self, other: &Self) -> bool {
        self.to_string().ends_with(&other.to_string())
    }
}

pub trait Contains {
    fn contains(&self, other: &Self) -> bool;
}

// TODO: Add tests
impl Contains for EdgeIndex {
    fn contains(&self, other: &Self) -> bool {
        self.to_string().contains(&other.to_string())
    }
}

pub trait Pow: Sized {
    fn pow(self, exp: Self) -> Result<Self, MedRecordError>;
}

pub trait Mod: Sized {
    fn r#mod(self, other: Self) -> Result<Self, MedRecordError>;
}

// TODO: Add tests
impl Mod for EdgeIndex {
    fn r#mod(self, other: Self) -> Result<Self, MedRecordError> {
        Ok(self % other)
    }
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

#[cfg(test)]
mod test {
    use super::{DataType, MedRecordValue};
    use chrono::NaiveDateTime;

    #[test]
    fn test_default() {
        assert_eq!(DataType::Any, DataType::default());
    }

    #[test]
    fn test_from_medrecordvalue() {
        assert_eq!(
            DataType::String,
            DataType::from(MedRecordValue::String("".to_string()))
        );
        assert_eq!(DataType::Int, DataType::from(MedRecordValue::Int(0)));
        assert_eq!(DataType::Float, DataType::from(MedRecordValue::Float(0.0)));
        assert_eq!(DataType::Bool, DataType::from(MedRecordValue::Bool(false)));
        assert_eq!(
            DataType::DateTime,
            DataType::from(MedRecordValue::DateTime(NaiveDateTime::MIN))
        );
        assert_eq!(DataType::Null, DataType::from(MedRecordValue::Null));
    }

    #[test]
    fn test_from_medrecordvalue_reference() {
        assert_eq!(
            DataType::String,
            DataType::from(&MedRecordValue::String("".to_string()))
        );
        assert_eq!(DataType::Int, DataType::from(&MedRecordValue::Int(0)));
        assert_eq!(DataType::Float, DataType::from(&MedRecordValue::Float(0.0)));
        assert_eq!(DataType::Bool, DataType::from(&MedRecordValue::Bool(false)));
        assert_eq!(
            DataType::DateTime,
            DataType::from(&MedRecordValue::DateTime(NaiveDateTime::MIN))
        );
        assert_eq!(DataType::Null, DataType::from(&MedRecordValue::Null));
    }

    #[test]
    fn test_partial_eq() {
        assert!(DataType::String == DataType::String);
        assert!(DataType::Int == DataType::Int);
        assert!(DataType::Float == DataType::Float);
        assert!(DataType::Bool == DataType::Bool);
        assert!(DataType::DateTime == DataType::DateTime);
        assert!(DataType::Null == DataType::Null);
        assert!(DataType::Any == DataType::Any);

        assert!(
            DataType::Union((Box::new(DataType::String), Box::new(DataType::Int)))
                == DataType::Union((Box::new(DataType::String), Box::new(DataType::Int)))
        );
        assert!(
            DataType::Union((Box::new(DataType::String), Box::new(DataType::Int)))
                == DataType::Union((Box::new(DataType::Int), Box::new(DataType::String)))
        );

        assert!(
            DataType::Option(Box::new(DataType::String))
                == DataType::Option(Box::new(DataType::String))
        );

        assert!(DataType::String != DataType::Int);
        assert!(DataType::String != DataType::Float);
        assert!(DataType::String != DataType::Bool);
        assert!(DataType::String != DataType::DateTime);
        assert!(DataType::String != DataType::Null);
        assert!(DataType::String != DataType::Any);

        assert!(DataType::Int != DataType::String);
        assert!(DataType::Int != DataType::Float);
        assert!(DataType::Int != DataType::Bool);
        assert!(DataType::Int != DataType::DateTime);
        assert!(DataType::Int != DataType::Null);
        assert!(DataType::Int != DataType::Any);

        assert!(DataType::Float != DataType::String);
        assert!(DataType::Float != DataType::Int);
        assert!(DataType::Float != DataType::Bool);
        assert!(DataType::Float != DataType::DateTime);
        assert!(DataType::Float != DataType::Null);
        assert!(DataType::Float != DataType::Any);

        assert!(DataType::Bool != DataType::String);
        assert!(DataType::Bool != DataType::Int);
        assert!(DataType::Bool != DataType::Float);
        assert!(DataType::Bool != DataType::DateTime);
        assert!(DataType::Bool != DataType::Null);
        assert!(DataType::Bool != DataType::Any);

        assert!(DataType::DateTime != DataType::String);
        assert!(DataType::DateTime != DataType::Int);
        assert!(DataType::DateTime != DataType::Float);
        assert!(DataType::DateTime != DataType::Bool);
        assert!(DataType::DateTime != DataType::Null);
        assert!(DataType::DateTime != DataType::Any);

        assert!(DataType::Null != DataType::String);
        assert!(DataType::Null != DataType::Int);
        assert!(DataType::Null != DataType::Float);
        assert!(DataType::Null != DataType::Bool);
        assert!(DataType::Null != DataType::DateTime);
        assert!(DataType::Null != DataType::Any);

        assert!(DataType::Any != DataType::String);
        assert!(DataType::Any != DataType::Int);
        assert!(DataType::Any != DataType::Float);
        assert!(DataType::Any != DataType::Bool);
        assert!(DataType::Any != DataType::DateTime);
        assert!(DataType::Any != DataType::Null);

        // If all the basic datatypes have been tested, it should be safe to assume that the
        // Union and Option variants will work as expected.
        assert!(
            DataType::Union((Box::new(DataType::String), Box::new(DataType::Int)))
                != DataType::Union((Box::new(DataType::Int), Box::new(DataType::Float)))
        );
        assert!(
            DataType::Option(Box::new(DataType::String))
                != DataType::Option(Box::new(DataType::Int))
        );
    }

    #[test]
    fn test_display() {
        assert_eq!("String", format!("{}", DataType::String));
        assert_eq!("Int", format!("{}", DataType::Int));
        assert_eq!("Float", format!("{}", DataType::Float));
        assert_eq!("Bool", format!("{}", DataType::Bool));
        assert_eq!("DateTime", format!("{}", DataType::DateTime));
        assert_eq!("Null", format!("{}", DataType::Null));
        assert_eq!("Any", format!("{}", DataType::Any));
        assert_eq!(
            "Union[String, Int]",
            format!(
                "{}",
                DataType::Union((Box::new(DataType::String), Box::new(DataType::Int)))
            )
        );
        assert_eq!(
            "Option[String]",
            format!("{}", DataType::Option(Box::new(DataType::String)))
        );
    }

    #[test]
    fn test_evaluate() {
        assert!(DataType::String.evaluate(&DataType::String));
        assert!(DataType::Int.evaluate(&DataType::Int));
        assert!(DataType::Float.evaluate(&DataType::Float));
        assert!(DataType::Bool.evaluate(&DataType::Bool));
        assert!(DataType::DateTime.evaluate(&DataType::DateTime));
        assert!(DataType::Null.evaluate(&DataType::Null));
        assert!(DataType::Any.evaluate(&DataType::Any));

        assert!(
            DataType::Union((Box::new(DataType::String), Box::new(DataType::Int))).evaluate(
                &DataType::Union((Box::new(DataType::String), Box::new(DataType::Int)))
            )
        );
        assert!(
            DataType::Union((Box::new(DataType::String), Box::new(DataType::Int))).evaluate(
                &DataType::Union((Box::new(DataType::Int), Box::new(DataType::String)))
            )
        );

        assert!(
            DataType::Union((Box::new(DataType::String), Box::new(DataType::Int)))
                .evaluate(&DataType::String)
        );
        assert!(
            DataType::Union((Box::new(DataType::String), Box::new(DataType::Int)))
                .evaluate(&DataType::Int)
        );

        assert!(DataType::Option(Box::new(DataType::String))
            .evaluate(&DataType::Option(Box::new(DataType::String))));
        assert!(DataType::Option(Box::new(DataType::String)).evaluate(&DataType::Null));
        assert!(DataType::Option(Box::new(DataType::String)).evaluate(&DataType::String));

        assert!(DataType::Any.evaluate(&DataType::String));

        assert!(!DataType::String.evaluate(&DataType::Int));
        assert!(!DataType::String.evaluate(&DataType::Float));
        assert!(!DataType::String.evaluate(&DataType::Bool));
        assert!(!DataType::String.evaluate(&DataType::DateTime));
        assert!(!DataType::String.evaluate(&DataType::Null));
        assert!(!DataType::String.evaluate(&DataType::Any));

        assert!(!DataType::Int.evaluate(&DataType::String));
        assert!(!DataType::Int.evaluate(&DataType::Float));
        assert!(!DataType::Int.evaluate(&DataType::Bool));
        assert!(!DataType::Int.evaluate(&DataType::DateTime));
        assert!(!DataType::Int.evaluate(&DataType::Null));
        assert!(!DataType::Int.evaluate(&DataType::Any));

        assert!(!DataType::Float.evaluate(&DataType::String));
        assert!(!DataType::Float.evaluate(&DataType::Int));
        assert!(!DataType::Float.evaluate(&DataType::Bool));
        assert!(!DataType::Float.evaluate(&DataType::DateTime));
        assert!(!DataType::Float.evaluate(&DataType::Null));
        assert!(!DataType::Float.evaluate(&DataType::Any));

        assert!(!DataType::Bool.evaluate(&DataType::String));
        assert!(!DataType::Bool.evaluate(&DataType::Int));
        assert!(!DataType::Bool.evaluate(&DataType::Float));
        assert!(!DataType::Bool.evaluate(&DataType::DateTime));
        assert!(!DataType::Bool.evaluate(&DataType::Null));
        assert!(!DataType::Bool.evaluate(&DataType::Any));

        assert!(!DataType::DateTime.evaluate(&DataType::String));
        assert!(!DataType::DateTime.evaluate(&DataType::Int));
        assert!(!DataType::DateTime.evaluate(&DataType::Float));
        assert!(!DataType::DateTime.evaluate(&DataType::Bool));
        assert!(!DataType::DateTime.evaluate(&DataType::Null));
        assert!(!DataType::DateTime.evaluate(&DataType::Any));

        assert!(!DataType::Null.evaluate(&DataType::String));
        assert!(!DataType::Null.evaluate(&DataType::Int));
        assert!(!DataType::Null.evaluate(&DataType::Float));
        assert!(!DataType::Null.evaluate(&DataType::Bool));
        assert!(!DataType::Null.evaluate(&DataType::DateTime));
        assert!(!DataType::Null.evaluate(&DataType::Any));

        assert!(
            !DataType::Union((Box::new(DataType::String), Box::new(DataType::Int))).evaluate(
                &DataType::Union((Box::new(DataType::Int), Box::new(DataType::Float)))
            )
        );

        assert!(!DataType::Option(Box::new(DataType::String))
            .evaluate(&DataType::Option(Box::new(DataType::Int))));
    }
}
