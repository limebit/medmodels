mod attribute;
mod value;

pub use self::{attribute::MedRecordAttribute, value::MedRecordValue};
use crate::errors::MedRecordError;
use std::ops::Range;

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
