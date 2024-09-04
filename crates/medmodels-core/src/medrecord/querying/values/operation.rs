use super::operand::{MedRecordValueComparisonOperand, MedRecordValueOperand, ValueKind};
use crate::{
    medrecord::{
        querying::traits::{DeepClone, ReadWriteOrPanic},
        MedRecordValue, Wrapper,
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
    ) -> Box<dyn Iterator<Item = (&'a T, &'a MedRecordValue)> + 'a> {
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
    ) -> Option<(&'a T, &'a MedRecordValue)> {
        let max_value = values.next()?;

        Some(values.fold(max_value, |max_value, value| {
            if value.1 > max_value.1 {
                value
            } else {
                max_value
            }
        }))
    }

    #[inline]
    pub(crate) fn get_min<'a, T: 'a>(
        mut values: impl Iterator<Item = (&'a T, &'a MedRecordValue)>,
    ) -> Option<(&'a T, &'a MedRecordValue)> {
        let min_value = values.next()?;

        Some(values.fold(min_value, |min_value, value| {
            if value.1 < min_value.1 {
                value
            } else {
                min_value
            }
        }))
    }

    #[inline]
    fn evaluate_value_operand<'a, T: 'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a T, &'a MedRecordValue)>,
        operand: &Wrapper<MedRecordValueOperand>,
    ) -> Box<dyn Iterator<Item = (&'a T, &'a MedRecordValue)> + 'a> {
        let kind = &operand.0.read_or_panic().kind;

        let value = match kind {
            ValueKind::Max => Self::get_max(values),
            ValueKind::Min => Self::get_min(values),
        };

        let Some(value) = value else {
            return Box::new(std::iter::empty());
        };

        match operand.evaluate(medrecord, value.1) {
            true => Box::new(std::iter::once(value)),
            false => Box::new(std::iter::empty()),
        }
    }

    #[inline]
    fn evaluate_less_than<'a, T: 'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a T, &'a MedRecordValue)> + 'a,
        comparison: MedRecordValueComparisonOperand,
    ) -> Box<dyn Iterator<Item = (&'a T, &'a MedRecordValue)> + 'a> {
        match comparison {
            MedRecordValueComparisonOperand::SingleOperand(comparison_operand) => {
                let context = &comparison_operand.context.context;
                let attribute = comparison_operand.context.attribute;
                let kind = &comparison_operand.kind;

                let comparison_values = context
                    .get_values(medrecord, attribute)
                    .map(|value| (&0, value));

                let comparison_value = match kind {
                    ValueKind::Max => Self::get_max(comparison_values),
                    ValueKind::Min => Self::get_min(comparison_values),
                };

                match comparison_value {
                    Some(comparison_value) => {
                        Box::new(values.filter(|value| value.1 < comparison_value.1))
                    }
                    None => Box::new(std::iter::empty()),
                }
            }
            MedRecordValueComparisonOperand::SingleValue(comparison_value) => {
                Box::new(values.filter(move |value| value.1 < &comparison_value))
            }
            MedRecordValueComparisonOperand::MultipleOperand(comparison_operand) => {
                let context = &comparison_operand.context;
                let attribute = comparison_operand.attribute;

                let mut comparison_values = context.get_values(medrecord, attribute);

                Box::new(values.filter(move |value| {
                    comparison_values.all(|comparison_value| value.1 < comparison_value)
                }))
            }
            MedRecordValueComparisonOperand::MultipleValues(comparison_values) => {
                Box::new(values.filter(move |value| {
                    comparison_values
                        .iter()
                        .all(|comparison_value| value.1 < comparison_value)
                }))
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
    pub(crate) fn evaluate<'a>(&self, medrecord: &'a MedRecord, value: &'a MedRecordValue) -> bool {
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
    ) -> bool {
        match comparison_operand {
            MedRecordValueComparisonOperand::SingleOperand(comparison_operand) => {
                let context = &comparison_operand.context.context;
                let attribute = comparison_operand.context.attribute.clone();
                let kind = &comparison_operand.kind;

                let values = context
                    .get_values(medrecord, attribute)
                    .map(|value| (&0, value));

                let comparison_value = match kind {
                    ValueKind::Max => MedRecordValuesOperation::get_max(values),
                    ValueKind::Min => MedRecordValuesOperation::get_min(values),
                };

                match comparison_value {
                    Some(comparison_value) => value < comparison_value.1,
                    None => false,
                }
            }
            MedRecordValueComparisonOperand::SingleValue(comparison_value) => {
                value < comparison_value
            }
            MedRecordValueComparisonOperand::MultipleOperand(comparison_operand) => {
                let context = &comparison_operand.context;
                let attribute = comparison_operand.attribute.clone();

                let mut values = context.get_values(medrecord, attribute);

                values.all(|comparison_value| value < comparison_value)
            }
            MedRecordValueComparisonOperand::MultipleValues(comparison_values) => comparison_values
                .iter()
                .all(|comparison_value| value < comparison_value),
        }
    }
}
