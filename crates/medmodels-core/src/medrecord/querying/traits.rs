use crate::MedRecord;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

pub(crate) trait EvaluateOperation {
    type Index;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = &'a Self::Index> + 'a,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a>;
}

pub(crate) trait EvaluateOperand {
    type Index;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a>;
}

pub(crate) trait DeepClone {
    fn deep_clone(&self) -> Self;
}

pub(crate) trait ReadWriteOrPanic<T> {
    fn read_or_panic(&self) -> RwLockReadGuard<'_, T>;

    fn write_or_panic(&self) -> RwLockWriteGuard<'_, T>;
}

impl<T> ReadWriteOrPanic<T> for RwLock<T> {
    fn read_or_panic(&self) -> RwLockReadGuard<'_, T> {
        self.read().unwrap()
    }

    fn write_or_panic(&self) -> RwLockWriteGuard<'_, T> {
        self.write().unwrap()
    }
}
