#[macro_export]
macro_rules! implement_from_for_wrapper {
    ($self: ty, $type: ty, $variant: ident) => {
        impl From<$type> for $self {
            fn from(value: $type) -> Self {
                Self::$variant(value)
            }
        }
    };
}
