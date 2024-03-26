use medmodels_utils::implement_from_for_wrapper;
use std::{cmp::Ordering, fmt::Display};

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

implement_from_for_wrapper!(MedRecordValue, String, String);
implement_from_for_wrapper!(MedRecordValue, i64, Int);
implement_from_for_wrapper!(MedRecordValue, f64, Float);
implement_from_for_wrapper!(MedRecordValue, bool, Bool);

impl PartialEq for MedRecordValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => value == other,
            (MedRecordValue::String(_), MedRecordValue::Int(_)) => todo!(),
            (MedRecordValue::String(_), MedRecordValue::Float(_)) => todo!(),
            (MedRecordValue::String(_), MedRecordValue::Bool(_)) => todo!(),
            (MedRecordValue::Int(_), MedRecordValue::String(_)) => todo!(),
            (MedRecordValue::Int(value), MedRecordValue::Int(other)) => value == other,
            (MedRecordValue::Int(_), MedRecordValue::Float(_)) => todo!(),
            (MedRecordValue::Int(_), MedRecordValue::Bool(_)) => todo!(),
            (MedRecordValue::Float(_), MedRecordValue::String(_)) => todo!(),
            (MedRecordValue::Float(_), MedRecordValue::Int(_)) => todo!(),
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => value == other,
            (MedRecordValue::Float(_), MedRecordValue::Bool(_)) => todo!(),
            (MedRecordValue::Bool(_), MedRecordValue::String(_)) => todo!(),
            (MedRecordValue::Bool(_), MedRecordValue::Int(_)) => todo!(),
            (MedRecordValue::Bool(_), MedRecordValue::Float(_)) => todo!(),
            (MedRecordValue::Bool(value), MedRecordValue::Bool(other)) => value == other,
        }
    }
}

impl PartialOrd for MedRecordValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (MedRecordValue::String(value), MedRecordValue::String(other)) => {
                Some(value.cmp(other))
            }
            (MedRecordValue::String(value), MedRecordValue::Int(other)) => {
                match value.parse::<i64>() {
                    Ok(value) => Some(value.cmp(other)),
                    Err(_) => Some(value.cmp(&other.to_string())),
                }
            }
            (MedRecordValue::String(value), MedRecordValue::Float(other)) => {
                match value.parse::<f64>() {
                    Ok(value) => value.partial_cmp(other),
                    Err(_) => Some(value.cmp(&other.to_string())),
                }
            }
            (MedRecordValue::String(value), MedRecordValue::Bool(other)) => {
                match value.parse::<bool>() {
                    Ok(value) => Some(value.cmp(other)),
                    Err(_) => Some(value.cmp(&other.to_string())),
                }
            }
            (MedRecordValue::Int(value), MedRecordValue::String(other)) => {
                match other.parse::<i64>() {
                    Ok(other) => Some(other.cmp(value)),
                    Err(_) => Some(other.cmp(&value.to_string())),
                }
            }
            (MedRecordValue::Int(value), MedRecordValue::Int(other)) => Some(value.cmp(other)),
            (MedRecordValue::Int(value), MedRecordValue::Float(other)) => {
                match other.to_string().parse::<i64>() {
                    Ok(other) => Some(other.cmp(value)),
                    Err(_) => other.partial_cmp(&(*value as f64)), // TODO: can overflow
                }
            }
            (MedRecordValue::Int(value), MedRecordValue::Bool(other)) => match value {
                0 => Some(other.cmp(&false)),
                _ => Some(other.cmp(&true)),
            },
            (MedRecordValue::Float(value), MedRecordValue::String(other)) => {
                match other.parse::<f64>() {
                    Ok(other) => other.partial_cmp(value),
                    Err(_) => Some(other.cmp(&value.to_string())),
                }
            }
            (MedRecordValue::Float(value), MedRecordValue::Int(other)) => {
                match value.to_string().parse::<i64>() {
                    Ok(value) => Some(value.cmp(other)),
                    Err(_) => value.partial_cmp(&(*other as f64)), // TODO: can overflow
                }
            }
            (MedRecordValue::Float(value), MedRecordValue::Float(other)) => {
                value.partial_cmp(other)
            }
            (MedRecordValue::Float(value), MedRecordValue::Bool(other)) => {
                if *value == 0.0 {
                    Some(other.cmp(&false))
                } else {
                    Some(other.cmp(&true))
                }
            }
            (MedRecordValue::Bool(value), MedRecordValue::String(other)) => {
                match other.parse::<bool>() {
                    Ok(other) => Some(other.cmp(value)),
                    Err(_) => Some(other.cmp(&value.to_string())),
                }
            }
            (MedRecordValue::Bool(value), MedRecordValue::Int(other)) => match other {
                0 => Some(value.cmp(&false)),
                _ => Some(value.cmp(&true)),
            },
            (MedRecordValue::Bool(value), MedRecordValue::Float(other)) => {
                if *other == 0.0 {
                    Some(value.cmp(&false))
                } else {
                    Some(value.cmp(&true))
                }
            }
            (MedRecordValue::Bool(value), MedRecordValue::Bool(other)) => Some(value.cmp(other)),
        }
    }
}

impl Display for MedRecordValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::String(value) => write!(f, "{}", value),
            Self::Int(value) => write!(f, "{}", value),
            Self::Float(value) => write!(f, "{}", value),
            Self::Bool(value) => write!(f, "{}", value),
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
    fn test_from_str() {
        let value = MedRecordValue::from("value");

        assert_eq!(MedRecordValue::String("value".to_string()), value)
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

    #[test]
    fn test_partial_ord() {
        // assert!(MedRecordValue::from(0_f64) < MedRecordValue::from(1_f64));
        // assert!(MedRecordValue::from(1_i64)  MedRecordValue::from(0_i64));
        let test = "0.1".to_string().parse::<f64>();
        println!("{:?}", test)
    }

    #[test]
    fn test_display() {
        assert_eq!(
            "value",
            MedRecordValue::from("value".to_string()).to_string()
        );

        assert_eq!("0", MedRecordValue::from(0_i64).to_string());

        assert_eq!("0.5", MedRecordValue::from(0.5).to_string());

        assert_eq!("false", MedRecordValue::from(false).to_string());
    }
}
