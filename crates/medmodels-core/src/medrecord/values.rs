#[derive(Debug, Clone)]
pub enum MedRecordValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

macro_rules! implement_from {
    ($type: ty, $variant: ident) => {
        impl From<$type> for MedRecordValue {
            fn from(value: $type) -> Self {
                Self::$variant(value)
            }
        }
    };
}

implement_from!(String, String);
implement_from!(i64, Int);
implement_from!(f64, Float);
implement_from!(bool, Bool);
