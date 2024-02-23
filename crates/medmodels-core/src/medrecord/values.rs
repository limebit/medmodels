#[derive(Debug, Clone)]
pub enum MedRecordValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl From<&str> for MedRecordValue {
    fn from(value: &str) -> Self {
        value.to_string().into()
    }
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

impl PartialEq for MedRecordValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::String(l0), Self::String(r0)) => l0 == r0,
            (Self::Int(l0), Self::Int(r0)) => l0 == r0,
            (Self::Float(l0), Self::Float(r0)) => l0 == r0,
            (Self::Bool(l0), Self::Bool(r0)) => l0 == r0,
            _ => false,
        }
    }
}

#[cfg(test)]
mod test {
    use super::MedRecordValue;

    #[test]
    fn test_from_string() {
        let value = MedRecordValue::from("value".to_string());

        assert_eq!(MedRecordValue::String("value".to_string()), value);
    }

    #[test]
    fn test_from_i64() {
        let value = MedRecordValue::from(0_i64);

        assert_eq!(MedRecordValue::Int(0), value);
    }

    #[test]
    fn test_from_f64() {
        let value = MedRecordValue::from(0_f64);

        assert_eq!(MedRecordValue::Float(0.0), value);
    }

    #[test]
    fn test_from_bool() {
        let value = MedRecordValue::from(false);

        assert_eq!(MedRecordValue::Bool(false), value);
    }

    #[test]
    fn test_partial_eq() {
        assert!(MedRecordValue::from(0_i64) == MedRecordValue::from(0_i64));
        assert!(MedRecordValue::from(1_i64) != MedRecordValue::from(0_i64));

        assert!(MedRecordValue::from(0_f64) == MedRecordValue::from(0_f64));
        assert!(MedRecordValue::from(1_f64) != MedRecordValue::from(0_f64));

        assert!(
            MedRecordValue::from("value".to_string()) == MedRecordValue::from("value".to_string())
        );
        assert!(
            MedRecordValue::from("value2".to_string()) != MedRecordValue::from("value".to_string())
        );

        assert!(MedRecordValue::from(false) == MedRecordValue::from(false));
        assert!(MedRecordValue::from(true) != MedRecordValue::from(false));
    }
}
