use super::evaluate::{EvaluateOperand, EvaluateOperandContext};
use crate::MedRecord;
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

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

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct WeakWrapper<T>(pub(crate) Weak<RefCell<T>>);

impl<T> From<Wrapper<T>> for WeakWrapper<T> {
    fn from(value: Wrapper<T>) -> Self {
        Self(Rc::downgrade(&value.0))
    }
}

impl<T> EvaluateOperand for WeakWrapper<T>
where
    T: EvaluateOperand,
{
    type Index = T::Index;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        self.0.upgrade().unwrap().borrow().evaluate(medrecord)
    }
}

#[derive(Debug)]
pub struct OperandContext<T> {
    operand: Wrapper<T>,
}

impl<T: Clone> Clone for OperandContext<T> {
    fn clone(&self) -> Self {
        Self {
            operand: self.operand.clone(),
        }
    }
}

impl<T> EvaluateOperandContext for OperandContext<T>
where
    T: EvaluateOperand,
{
    type Index = T::Index;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        self.operand.evaluate(medrecord)
    }
}

impl<T> OperandContext<T> {
    pub(crate) fn new(operand: Wrapper<T>) -> Self {
        Self { operand }
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
