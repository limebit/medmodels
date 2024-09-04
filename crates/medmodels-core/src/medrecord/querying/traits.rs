use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

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
