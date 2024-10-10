pub mod attributes;
pub mod edges;
pub mod nodes;
mod traits;
pub mod values;
pub mod wrapper;

pub(crate) type BoxedIterator<'a, T> = Box<dyn Iterator<Item = T> + 'a>;
