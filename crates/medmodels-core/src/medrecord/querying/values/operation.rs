use std::cmp::Ordering;

use super::operand::{MedRecordValueComparisonOperand, MedRecordValueOperand, ValueKind};
use crate::{
    errors::{MedRecordError, MedRecordResult},
    medrecord::{
        querying::traits::{DeepClone, ReadWriteOrPanic},
        DataType, MedRecordValue, Wrapper,
    },
    MedRecord,
};

#[derive(Debug, Clone)]
pub enum MedRecordValuesOperation {
    ValueOperand {
        operand: Wrapper<MedRecordValueOperand>,
    },

    LessThan {
        value: MedRecordValueComparisonOperand,
    },
}

impl DeepClone for MedRecordValuesOperation {
    fn deep_clone(&self) -> Self {
        match self {
            Self::ValueOperand { operand } => Self::ValueOperand {
                operand: operand.deep_clone(),
            },
            Self::LessThan { value } => Self::LessThan {
                value: value.deep_clone(),
            },
        }
    }
}

impl MedRecordValuesOperation {
    pub(crate) fn evaluate<'a, T: 'a>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a T, &'a MedRecordValue)> + 'a,
    ) -> MedRecordResult<Box<dyn Iterator<Item = (&'a T, &'a MedRecordValue)> + 'a>> {
        match self {
            Self::ValueOperand { operand } => {
                Self::evaluate_value_operand(medrecord, values, operand)
            }
            Self::LessThan { value } => Self::evaluate_less_than(medrecord, values, value.clone()),
        }
    }

    #[inline]
    pub(crate) fn get_max<'a, T: 'a>(
        mut values: impl Iterator<Item = (&'a T, &'a MedRecordValue)>,
    ) -> MedRecordResult<(&'a T, &'a MedRecordValue)> {
        let max_value = values.next().ok_or(MedRecordError::QueryError(
            "No values to compare".to_string(),
        ))?;

        values.try_fold(max_value, |max_value, value| {
            match value.1.partial_cmp(max_value.1) {
                Some(Ordering::Greater) => Ok(value),
                None => {
                    let first_dtype = DataType::from(value.1);
                    let second_dtype = DataType::from(max_value.1);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare values of data types {} and {}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool() or .is_datetime()",
                        first_dtype, second_dtype
                    )))
                }
                _ => Ok(max_value),
            }
        })
    }

    #[inline]
    pub(crate) fn get_min<'a, T: 'a>(
        mut values: impl Iterator<Item = (&'a T, &'a MedRecordValue)>,
    ) -> MedRecordResult<(&'a T, &'a MedRecordValue)> {
        let min_value = values.next().ok_or(MedRecordError::QueryError(
            "No values to compare".to_string(),
        ))?;

        values.try_fold(min_value, |min_value, value| {
            match value.1.partial_cmp(min_value.1) {
                Some(Ordering::Less) => Ok(value),
                None => {
                    let first_dtype = DataType::from(value.1);
                    let second_dtype = DataType::from(min_value.1);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare values of data types {} and {}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool() or .is_datetime()",
                        first_dtype, second_dtype
                    )))
                }
                _ => Ok(min_value),
            }
        })
    }

    #[inline]
    fn evaluate_value_operand<'a, T: 'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a T, &'a MedRecordValue)>,
        operand: &Wrapper<MedRecordValueOperand>,
    ) -> MedRecordResult<Box<dyn Iterator<Item = (&'a T, &'a MedRecordValue)> + 'a>> {
        let kind = &operand.0.read_or_panic().kind;

        let value = match kind {
            ValueKind::Max => Self::get_max(values),
            ValueKind::Min => Self::get_min(values),
        }?;

        Ok(match operand.evaluate(medrecord, value.1)? {
            true => Box::new(std::iter::once(value)),
            false => Box::new(std::iter::empty()),
        })
    }

    #[inline]
    fn evaluate_less_than<'a, T: 'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a T, &'a MedRecordValue)> + 'a,
        comparison: MedRecordValueComparisonOperand,
    ) -> MedRecordResult<Box<dyn Iterator<Item = (&'a T, &'a MedRecordValue)> + 'a>> {
        match comparison {
            MedRecordValueComparisonOperand::SingleOperand(comparison_operand) => {
                let context = &comparison_operand.context.context;
                let attribute = comparison_operand.context.attribute;
                let kind = &comparison_operand.kind;

                let comparison_values = context
                    .get_values(medrecord, attribute)?
                    .map(|value| (&0, value));

                let comparison_value = match kind {
                    ValueKind::Max => Self::get_max(comparison_values),
                    ValueKind::Min => Self::get_min(comparison_values),
                }?;

                Ok(Box::new(
                    values.filter(|value| value.1 < comparison_value.1),
                ))
            }
            MedRecordValueComparisonOperand::SingleValue(comparison_value) => Ok(Box::new(
                values.filter(move |value| value.1 < &comparison_value),
            )),
            MedRecordValueComparisonOperand::MultipleOperand(comparison_operand) => {
                let context = &comparison_operand.context;
                let attribute = comparison_operand.attribute;

                let mut comparison_values = context.get_values(medrecord, attribute)?;

                Ok(Box::new(values.filter(move |value| {
                    comparison_values.all(|comparison_value| value.1 < comparison_value)
                })))
            }
            MedRecordValueComparisonOperand::MultipleValues(comparison_values) => {
                Ok(Box::new(values.filter(move |value| {
                    comparison_values
                        .iter()
                        .all(|comparison_value| value.1 < comparison_value)
                })))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum MedRecordValueOperation {
    LessThan {
        value: MedRecordValueComparisonOperand,
    },
}

impl DeepClone for MedRecordValueOperation {
    fn deep_clone(&self) -> Self {
        // TODO
        self.clone()
    }
}

impl MedRecordValueOperation {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        value: &'a MedRecordValue,
    ) -> MedRecordResult<bool> {
        match self {
            Self::LessThan { value: operand } => {
                Self::evaluate_less_than(medrecord, value, operand)
            }
        }
    }

    fn evaluate_less_than(
        medrecord: &MedRecord,
        value: &MedRecordValue,
        comparison_operand: &MedRecordValueComparisonOperand,
    ) -> MedRecordResult<bool> {
        match comparison_operand {
            MedRecordValueComparisonOperand::SingleOperand(comparison_operand) => {
                let context = &comparison_operand.context.context;
                let attribute = comparison_operand.context.attribute.clone();
                let kind = &comparison_operand.kind;

                let values = context
                    .get_values(medrecord, attribute)?
                    .map(|value| (&0, value));

                let comparison_value = match kind {
                    ValueKind::Max => MedRecordValuesOperation::get_max(values),
                    ValueKind::Min => MedRecordValuesOperation::get_min(values),
                }?;

                Ok(value < comparison_value.1)
            }
            MedRecordValueComparisonOperand::SingleValue(comparison_value) => {
                Ok(value < comparison_value)
            }
            MedRecordValueComparisonOperand::MultipleOperand(comparison_operand) => {
                let context = &comparison_operand.context;
                let attribute = comparison_operand.attribute.clone();

                let mut values = context.get_values(medrecord, attribute)?;

                Ok(values.all(|comparison_value| value < comparison_value))
            }
            MedRecordValueComparisonOperand::MultipleValues(comparison_values) => {
                Ok(comparison_values
                    .iter()
                    .all(|comparison_value| value < comparison_value))
            }
        }
    }
}
