use super::evaluate::EvaluateOperand;
use crate::MedRecord;
use std::{cell::RefCell, rc::Rc};

pub(crate) trait DeepClone {
    fn deep_clone(&self) -> Self;
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct Wrapper<T>(pub(crate) Rc<RefCell<T>>);

impl<T> From<T> for Wrapper<T> {
    fn from(value: T) -> Self {
        Self(Rc::new(RefCell::new(value)))
    }
}

impl<T> EvaluateOperand for Wrapper<T>
where
    T: EvaluateOperand,
{
    type Index = T::Index;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        self.0.borrow().evaluate(medrecord)
    }
}

impl<T> DeepClone for Wrapper<T>
where
    T: DeepClone,
{
    fn deep_clone(&self) -> Self {
        self.0.borrow().deep_clone().into()
    }
}

#[derive(Debug, Clone)]
pub enum CardinalityWrapper<T> {
    Single(T),
    Multiple(Vec<T>),
}

impl<T> From<T> for CardinalityWrapper<T> {
    fn from(value: T) -> Self {
        Self::Single(value)
    }
}

impl<T> From<Vec<T>> for CardinalityWrapper<T> {
    fn from(value: Vec<T>) -> Self {
        Self::Multiple(value)
    }
}

impl<T: Clone, const N: usize> From<[T; N]> for CardinalityWrapper<T> {
    fn from(value: [T; N]) -> Self {
        Self::Multiple(value.to_vec())
    }
}
