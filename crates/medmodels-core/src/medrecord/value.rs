use medmodels_utils::implement_from_for_wrapper;

#[derive(Debug, Clone, PartialEq)]
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

implement_from_for_wrapper!(MedRecordValue, String, String);
implement_from_for_wrapper!(MedRecordValue, i64, Int);
implement_from_for_wrapper!(MedRecordValue, f64, Float);
implement_from_for_wrapper!(MedRecordValue, bool, Bool);

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
