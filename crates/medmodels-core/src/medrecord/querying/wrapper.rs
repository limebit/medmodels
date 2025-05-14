use super::{DeepClone, ReadWriteOrPanic};
use std::sync::{Arc, RwLock};

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct Wrapper<T>(pub(crate) Arc<RwLock<T>>);

impl<T> From<T> for Wrapper<T> {
    fn from(value: T) -> Self {
        Self(Arc::new(RwLock::new(value)))
    }
}

impl<T> DeepClone for Wrapper<T>
where
    T: DeepClone,
{
    fn deep_clone(&self) -> Self {
        self.0.read_or_panic().deep_clone().into()
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
