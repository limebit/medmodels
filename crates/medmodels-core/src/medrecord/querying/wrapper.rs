#[derive(Debug, Clone)]
pub enum Wrapper<T> {
    Single(T),
    Multiple(Vec<T>),
}

impl<T> From<T> for Wrapper<T> {
    fn from(value: T) -> Self {
        Self::Single(value)
    }
}

impl<T> From<Vec<T>> for Wrapper<T> {
    fn from(value: Vec<T>) -> Self {
        Self::Multiple(value)
    }
}

impl<T: Clone, const N: usize> From<[T; N]> for Wrapper<T> {
    fn from(value: [T; N]) -> Self {
        Self::Multiple(value.to_vec())
    }
}
