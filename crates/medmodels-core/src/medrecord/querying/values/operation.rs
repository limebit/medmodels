use super::{
    operand::{
        MultipleValuesComparisonOperand, MultipleValuesOperand, SingleValueComparisonOperand,
        SingleValueOperand,
    },
    BinaryArithmeticKind, MultipleComparisonKind, SingleComparisonKind, SingleKind,
    UnaryArithmeticKind,
};
use crate::{
    errors::{MedRecordError, MedRecordResult},
    medrecord::{
        datatypes::{
            Abs, Ceil, Contains, EndsWith, Floor, Lowercase, Mod, Pow, Round, Slice, Sqrt,
            StartsWith, Trim, TrimEnd, TrimStart, Uppercase,
        },
        querying::{
            traits::{DeepClone, ReadWriteOrPanic},
            BoxedIterator, Index, Operand, OptionalIndexWrapper,
        },
        DataType, MedRecordValue, Wrapper,
    },
    MedRecord,
};
use itertools::Itertools;
use medmodels_utils::aliases::MrHashSet;
use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Range, Sub},
};

macro_rules! get_median {
    ($values:ident, $variant:ident) => {
        if $values.len() % 2 == 0 {
            let middle = $values.len() / 2;

            let first = $values.get(middle - 1).unwrap();
            let second = $values.get(middle).unwrap();

            let first = MedRecordValue::$variant(*first);
            let second = MedRecordValue::$variant(*second);

            first.add(second).unwrap().div(MedRecordValue::Int(2))
        } else {
            let middle = $values.len() / 2;

            Ok(MedRecordValue::$variant(
                $values.get(middle).unwrap().clone(),
            ))
        }
    };
}

#[derive(Debug, Clone)]
pub enum MultipleValuesOperation<O: Operand> {
    ValueOperation {
        operand: Wrapper<SingleValueOperand<O>>,
    },
    SingleValueComparisonOperation {
        operand: SingleValueComparisonOperand,
        kind: SingleComparisonKind,
    },
    MultipleValuesComparisonOperation {
        operand: MultipleValuesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOpration {
        operand: SingleValueComparisonOperand,
        kind: BinaryArithmeticKind,
    },
    UnaryArithmeticOperation {
        kind: UnaryArithmeticKind,
    },

    Slice(Range<usize>),

    IsString,
    IsInt,
    IsFloat,
    IsBool,
    IsDateTime,
    IsDuration,
    IsNull,

    IsMax,
    IsMin,

    EitherOr {
        either: Wrapper<MultipleValuesOperand<O>>,
        or: Wrapper<MultipleValuesOperand<O>>,
    },
    Exclude {
        operand: Wrapper<MultipleValuesOperand<O>>,
    },
}

impl<O: Operand> DeepClone for MultipleValuesOperation<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::ValueOperation { operand } => Self::ValueOperation {
                operand: operand.deep_clone(),
            },
            Self::SingleValueComparisonOperation { operand, kind } => {
                Self::SingleValueComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::MultipleValuesComparisonOperation { operand, kind } => {
                Self::MultipleValuesComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::BinaryArithmeticOpration { operand, kind } => Self::BinaryArithmeticOpration {
                operand: operand.deep_clone(),
                kind: kind.clone(),
            },
            Self::UnaryArithmeticOperation { kind } => {
                Self::UnaryArithmeticOperation { kind: kind.clone() }
            }
            Self::Slice(range) => Self::Slice(range.clone()),
            Self::IsString => Self::IsString,
            Self::IsInt => Self::IsInt,
            Self::IsFloat => Self::IsFloat,
            Self::IsBool => Self::IsBool,
            Self::IsDateTime => Self::IsDateTime,
            Self::IsDuration => Self::IsDuration,
            Self::IsNull => Self::IsNull,
            Self::IsMax => Self::IsMax,
            Self::IsMin => Self::IsMin,
            Self::EitherOr { either, or } => Self::EitherOr {
                either: either.deep_clone(),
                or: or.deep_clone(),
            },
            Self::Exclude { operand } => Self::Exclude {
                operand: operand.deep_clone(),
            },
        }
    }
}

impl<O: Operand> MultipleValuesOperation<O> {
    pub(crate) fn evaluate<'a, I: Index + 'a>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (I, MedRecordValue)> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, (I, MedRecordValue)>>
    where
        O: 'a,
    {
        match self {
            Self::ValueOperation { operand } => {
                Self::evaluate_value_operation(medrecord, values, operand)
            }
            Self::SingleValueComparisonOperation { operand, kind } => {
                Self::evaluate_single_value_comparison_operation(medrecord, values, operand, kind)
            }
            Self::MultipleValuesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_values_comparison_operation(
                    medrecord, values, operand, kind,
                )
            }
            Self::BinaryArithmeticOpration { operand, kind } => Ok(Box::new(
                Self::evaluate_binary_arithmetic_operation(medrecord, values, operand, kind)?,
            )),
            Self::UnaryArithmeticOperation { kind } => Ok(Box::new(
                Self::evaluate_unary_arithmetic_operation(values, kind.clone()),
            )),
            Self::Slice(range) => Ok(Box::new(Self::evaluate_slice(values, range.clone()))),
            Self::IsString => {
                Ok(Box::new(values.filter(|(_, value)| {
                    matches!(value, MedRecordValue::String(_))
                })))
            }
            Self::IsInt => {
                Ok(Box::new(values.filter(|(_, value)| {
                    matches!(value, MedRecordValue::Int(_))
                })))
            }
            Self::IsFloat => {
                Ok(Box::new(values.filter(|(_, value)| {
                    matches!(value, MedRecordValue::Float(_))
                })))
            }
            Self::IsBool => {
                Ok(Box::new(values.filter(|(_, value)| {
                    matches!(value, MedRecordValue::Bool(_))
                })))
            }
            Self::IsDateTime => {
                Ok(Box::new(values.filter(|(_, value)| {
                    matches!(value, MedRecordValue::DateTime(_))
                })))
            }
            Self::IsDuration => {
                Ok(Box::new(values.filter(|(_, value)| {
                    matches!(value, MedRecordValue::Duration(_))
                })))
            }
            Self::IsNull => {
                Ok(Box::new(values.filter(|(_, value)| {
                    matches!(value, MedRecordValue::Null)
                })))
            }
            Self::IsMax => {
                let max_value = Self::get_max(values)?;

                Ok(Box::new(std::iter::once(max_value)))
            }
            Self::IsMin => {
                let min_value = Self::get_min(values)?;

                Ok(Box::new(std::iter::once(min_value)))
            }
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, values, either, or)
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, values, operand),
        }
    }

    #[inline]
    pub(crate) fn get_max<I>(
        mut values: impl Iterator<Item = (I, MedRecordValue)>,
    ) -> MedRecordResult<(I, MedRecordValue)> {
        let max_value = values.next().ok_or(MedRecordError::QueryError(
            "No values to compare".to_string(),
        ))?;

        values.try_fold(max_value, |max_value, value| {
            match value.1.partial_cmp(&max_value.1) {
                Some(Ordering::Greater) => Ok(value),
                None => {
                    let first_dtype = DataType::from(value.1);
                    let second_dtype = DataType::from(max_value.1);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare values of data types {} and {}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool(), .is_datetime() or .is_duration()",
                        first_dtype, second_dtype
                    )))
                }
                _ => Ok(max_value),
            }
        })
    }

    #[inline]
    pub(crate) fn get_min<I>(
        mut values: impl Iterator<Item = (I, MedRecordValue)>,
    ) -> MedRecordResult<(I, MedRecordValue)> {
        let min_value = values.next().ok_or(MedRecordError::QueryError(
            "No values to compare".to_string(),
        ))?;

        values.try_fold(min_value, |min_value, value| {
            match value.1.partial_cmp(&min_value.1) {
                Some(Ordering::Less) => Ok(value),
                None => {
                    let first_dtype = DataType::from(value.1);
                    let second_dtype = DataType::from(min_value.1);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare values of data types {} and {}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool(), .is_datetime() or .is_duration()",
                        first_dtype, second_dtype
                    )))
                }
                _ => Ok(min_value),
            }
        })
    }

    #[inline]
    pub(crate) fn get_mean<I>(
        mut values: impl Iterator<Item = (I, MedRecordValue)>,
    ) -> MedRecordResult<MedRecordValue> {
        let first_value = values.next().ok_or(MedRecordError::QueryError(
            "No values to compare".to_string(),
        ))?;

        let (sum, count) = values.try_fold((first_value.1, 1), |(sum, count), (_, value)| {
            let first_dtype = DataType::from(&sum);
            let second_dtype = DataType::from(&value);

            match sum.add(value) {
                Ok(sum) => Ok((sum, count + 1)),
                Err(_) => Err(MedRecordError::QueryError(format!(
                    "Cannot add values of data types {} and {}. Consider narrowing down the values using .is_int(), .is_float(), .is_datetime() or .is_duration()",
                    first_dtype, second_dtype
                ))),
            }
        })?;

        sum.div(MedRecordValue::Int(count as i64))
    }

    // TODO: This is a temporary solution. It should be optimized.
    #[inline]
    pub(crate) fn get_median<I>(
        mut values: impl Iterator<Item = (I, MedRecordValue)>,
    ) -> MedRecordResult<MedRecordValue> {
        let first_value = values.next().ok_or(MedRecordError::QueryError(
            "No values to compare".to_string(),
        ))?;

        let first_data_type = DataType::from(&first_value.1);

        match first_value.1 {
            MedRecordValue::Int(value) => {
                let mut values = values.map(|(_, value)| {
                    let data_type = DataType::from(&value);

                    match value {
                        MedRecordValue::Int(value) => Ok(value as f64),
                        MedRecordValue::Float(value) => Ok(value),
                        _ => Err(MedRecordError::QueryError(format!(
                            "Cannot calculate median of mixed data types {} and {}. Consider narrowing down the values using .is_int(), .is_float() , .is_datetime() or .is_duration()",
                            first_data_type, data_type
                        ))),
                    }
                }).collect::<MedRecordResult<Vec<_>>>()?;
                values.push(value as f64);
                values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

                get_median!(values, Float)
            }
            MedRecordValue::Float(value) => {
                let mut values = values.map(|(_, value)| {
                    let data_type = DataType::from(&value);

                    match value {
                        MedRecordValue::Int(value) => Ok(value as f64),
                        MedRecordValue::Float(value) => Ok(value),
                        _ => Err(MedRecordError::QueryError(format!(
                            "Cannot calculate median of mixed data types {} and {}. Consider narrowing down the values using .is_int(), .is_float(), .is_datetime() or .is_duration()",
                            first_data_type, data_type
                        ))),
                    }
                }).collect::<MedRecordResult<Vec<_>>>()?;
                values.push(value);
                values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

                get_median!(values, Float)
            }
            MedRecordValue::DateTime(value) => {
                let mut values = values.map(|(_, value)| {
                    let data_type = DataType::from(&value);

                    match value {
                        MedRecordValue::DateTime(naive_date_time) => Ok(naive_date_time),
                        _ => Err(MedRecordError::QueryError(format!(
                            "Cannot calculate median of mixed data types {} and {}. Consider narrowing down the values using .is_int(), .is_float(), .is_datetime() or .is_duration()",
                            first_data_type, data_type
                        ))),
                    }
                }).collect::<MedRecordResult<Vec<_>>>()?;
                values.push(value);
                values.sort();

                get_median!(values, DateTime)
            }
            MedRecordValue::Duration(value) => {
                let mut values = values.map(|(_, value)| {
                    let data_type = DataType::from(&value);

                    match value {
                        MedRecordValue::Duration(naive_date_time) => Ok(naive_date_time),
                        _ => Err(MedRecordError::QueryError(format!(
                            "Cannot calculate median of mixed data types {} and {}. Consider narrowing down the values using .is_int(), .is_float(), .is_datetime() or .is_duration()",
                            first_data_type, data_type
                        ))),
                    }
                }).collect::<MedRecordResult<Vec<_>>>()?;
                values.push(value);
                values.sort();

                get_median!(values, Duration)
            }
            _ => Err(MedRecordError::QueryError(format!(
                "Cannot calculate median of data type {}",
                first_data_type
            )))?,
        }
    }

    // TODO: This is a temporary solution. It should be optimized.
    #[inline]
    pub(crate) fn get_mode<I>(
        values: impl Iterator<Item = (I, MedRecordValue)>,
    ) -> MedRecordResult<MedRecordValue> {
        let values: Vec<_> = values.map(|(_, value)| value).collect();

        let most_common_value = values
            .first()
            .ok_or(MedRecordError::QueryError(
                "No values to compare".to_string(),
            ))?
            .clone();
        let most_common_count = values
            .iter()
            .filter(|value| **value == most_common_value)
            .count();

        let (_, most_common_value) = values.clone().into_iter().fold(
            (most_common_count, most_common_value),
            |acc, value| {
                let count = values.iter().filter(|v| **v == value).count();

                if count > acc.0 {
                    (count, value)
                } else {
                    acc
                }
            },
        );

        Ok(most_common_value)
    }

    #[inline]
    // 👀
    pub(crate) fn get_std<I: Index>(
        values: impl Iterator<Item = (I, MedRecordValue)>,
    ) -> MedRecordResult<MedRecordValue> {
        let variance = Self::get_var(values)?;

        let MedRecordValue::Float(variance) = variance else {
            unreachable!()
        };

        Ok(MedRecordValue::Float(variance.sqrt()))
    }

    // TODO: This is a temporary solution. It should be optimized.
    #[inline]
    pub(crate) fn get_var<I: Index>(
        values: impl Iterator<Item = (I, MedRecordValue)>,
    ) -> MedRecordResult<MedRecordValue> {
        let values: Vec<_> = values.collect();

        let mean = Self::get_mean(values.clone().into_iter())?;

        let MedRecordValue::Float(mean) = mean else {
            let data_type = DataType::from(mean);

            return Err(MedRecordError::QueryError(
                format!("Cannot calculate variance of data type {}. Consider narrowing down the values using .is_int() or .is_float()", data_type),
            ));
        };

        let values = values
            .into_iter()
            .map(|value| {
                let data_type = DataType::from(&value.1);

                match value.1 {
                MedRecordValue::Int(value) => Ok(value as f64),
                MedRecordValue::Float(value) => Ok(value),
                _ => Err(MedRecordError::QueryError(
                    format!("Cannot calculate variance of data type {}. Consider narrowing down the values using .is_int() or .is_float()", data_type),
                )),
            }})
            .collect::<MedRecordResult<Vec<_>>>()?;

        let values_length = values.len();

        let variance = values
            .into_iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values_length as f64;

        Ok(MedRecordValue::Float(variance))
    }

    #[inline]
    pub(crate) fn get_count<I>(
        values: impl Iterator<Item = (I, MedRecordValue)>,
    ) -> MedRecordValue {
        MedRecordValue::Int(values.count() as i64)
    }

    #[inline]
    // 🥊💥
    pub(crate) fn get_sum<I>(
        mut values: impl Iterator<Item = (I, MedRecordValue)>,
    ) -> MedRecordResult<MedRecordValue> {
        let first_value = values.next().ok_or(MedRecordError::QueryError(
            "No values to compare".to_string(),
        ))?;

        values.try_fold(first_value.1, |sum, (_, value)| {
            let first_dtype = DataType::from(&sum);
            let second_dtype = DataType::from(&value);

            sum.add(value).map_err(|_| {
                MedRecordError::QueryError(format!(
                    "Cannot add values of data types {} and {}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool(), .is_datetime() or .is_duration()",
                    first_dtype, second_dtype
                ))
            })
        })
    }

    #[inline]
    pub(crate) fn get_first<I>(
        mut values: impl Iterator<Item = (I, MedRecordValue)>,
    ) -> MedRecordResult<(I, MedRecordValue)> {
        values.next().ok_or(MedRecordError::QueryError(
            "No values to compare".to_string(),
        ))
    }

    #[inline]
    pub(crate) fn get_last<I>(
        values: impl Iterator<Item = (I, MedRecordValue)>,
    ) -> MedRecordResult<(I, MedRecordValue)> {
        values.last().ok_or(MedRecordError::QueryError(
            "No values to compare".to_string(),
        ))
    }

    #[inline]
    fn evaluate_value_operation<'a, I: Index + 'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (I, MedRecordValue)> + 'a,
        operand: &Wrapper<SingleValueOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, (I, MedRecordValue)>> {
        let (values_1, values_2) = Itertools::tee(values);

        let kind = &operand.0.read_or_panic().kind;

        let value: OptionalIndexWrapper<_, _> = match kind {
            SingleKind::Max => MultipleValuesOperation::<O>::get_max(values_1)?.into(),
            SingleKind::Min => MultipleValuesOperation::<O>::get_min(values_1)?.into(),
            SingleKind::Mean => MultipleValuesOperation::<O>::get_mean(values_1)?.into(),
            SingleKind::Median => MultipleValuesOperation::<O>::get_median(values_1)?.into(),
            SingleKind::Mode => MultipleValuesOperation::<O>::get_mode(values_1)?.into(),
            SingleKind::Std => MultipleValuesOperation::<O>::get_std(values_1)?.into(),
            SingleKind::Var => MultipleValuesOperation::<O>::get_var(values_1)?.into(),
            SingleKind::Count => MultipleValuesOperation::<O>::get_count(values_1).into(),
            SingleKind::Sum => MultipleValuesOperation::<O>::get_sum(values_1)?.into(),
            SingleKind::First => MultipleValuesOperation::<O>::get_first(values_1)?.into(),
            SingleKind::Last => MultipleValuesOperation::<O>::get_last(values_1)?.into(),
        };

        Ok(match operand.evaluate_forward(medrecord, value)? {
            Some(_) => Box::new(values_2),
            None => Box::new(std::iter::empty()),
        })
    }

    #[inline]
    fn evaluate_single_value_comparison_operation<'a, I>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (I, MedRecordValue)> + 'a,
        comparison_operand: &SingleValueComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, (I, MedRecordValue)>> {
        let comparison_value =
            comparison_operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No value to compare".to_string(),
                ))?;

        match kind {
            SingleComparisonKind::GreaterThan => Ok(Box::new(
                values.filter(move |(_, value)| value > &comparison_value),
            )),
            SingleComparisonKind::GreaterThanOrEqualTo => Ok(Box::new(
                values.filter(move |(_, value)| value >= &comparison_value),
            )),
            SingleComparisonKind::LessThan => Ok(Box::new(
                values.filter(move |(_, value)| value < &comparison_value),
            )),
            SingleComparisonKind::LessThanOrEqualTo => Ok(Box::new(
                values.filter(move |(_, value)| value <= &comparison_value),
            )),
            SingleComparisonKind::EqualTo => Ok(Box::new(
                values.filter(move |(_, value)| value == &comparison_value),
            )),
            SingleComparisonKind::NotEqualTo => Ok(Box::new(
                values.filter(move |(_, value)| value != &comparison_value),
            )),
            SingleComparisonKind::StartsWith => {
                Ok(Box::new(values.filter(move |(_, value)| {
                    value.starts_with(&comparison_value)
                })))
            }
            SingleComparisonKind::EndsWith => {
                Ok(Box::new(values.filter(move |(_, value)| {
                    value.ends_with(&comparison_value)
                })))
            }
            SingleComparisonKind::Contains => {
                Ok(Box::new(values.filter(move |(_, value)| {
                    value.contains(&comparison_value)
                })))
            }
        }
    }

    #[inline]
    fn evaluate_multiple_values_comparison_operation<'a, I: Index>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (I, MedRecordValue)> + 'a,
        comparison_operand: &MultipleValuesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, (I, MedRecordValue)>> {
        let comparison_values = comparison_operand.evaluate_backward(medrecord)?;

        match kind {
            MultipleComparisonKind::IsIn => {
                Ok(Box::new(values.filter(move |(_, value)| {
                    comparison_values.contains(value)
                })))
            }
            MultipleComparisonKind::IsNotIn => {
                Ok(Box::new(values.filter(move |(_, value)| {
                    !comparison_values.contains(value)
                })))
            }
        }
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation<I>(
        medrecord: &MedRecord,
        values: impl Iterator<Item = (I, MedRecordValue)>,
        operand: &SingleValueComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<impl Iterator<Item = (I, MedRecordValue)>> {
        let arithmetic_value =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No value to compare".to_string(),
                ))?;

        let values = values
            .map(move |(t, value)| {
                match kind {
                    BinaryArithmeticKind::Add => value.add(arithmetic_value.clone()),
                    BinaryArithmeticKind::Sub => value.sub(arithmetic_value.clone()),
                    BinaryArithmeticKind::Mul => {
                        value.clone().mul(arithmetic_value.clone())
                    }
                    BinaryArithmeticKind::Div => {
                        value.clone().div(arithmetic_value.clone())
                    }
                    BinaryArithmeticKind::Pow => {
                        value.clone().pow(arithmetic_value.clone())
                    }
                    BinaryArithmeticKind::Mod => {
                        value.clone().r#mod(arithmetic_value.clone())
                    }
                }
                .map_err(|_| {
                    MedRecordError::QueryError(format!(
                        "Failed arithmetic operation {}. Consider narrowing down the values using .is_int() or .is_float()",
                        kind,
                    ))
                }).map(|result| (t, result))
            });

        // TODO: This is a temporary solution. It should be optimized.
        Ok(values.collect::<MedRecordResult<Vec<_>>>()?.into_iter())
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation<I>(
        values: impl Iterator<Item = (I, MedRecordValue)>,
        kind: UnaryArithmeticKind,
    ) -> impl Iterator<Item = (I, MedRecordValue)> {
        values.map(move |(t, value)| {
            let value = match kind {
                UnaryArithmeticKind::Round => value.round(),
                UnaryArithmeticKind::Ceil => value.ceil(),
                UnaryArithmeticKind::Floor => value.floor(),
                UnaryArithmeticKind::Abs => value.abs(),
                UnaryArithmeticKind::Sqrt => value.sqrt(),
                UnaryArithmeticKind::Trim => value.trim(),
                UnaryArithmeticKind::TrimStart => value.trim_start(),
                UnaryArithmeticKind::TrimEnd => value.trim_end(),
                UnaryArithmeticKind::Lowercase => value.lowercase(),
                UnaryArithmeticKind::Uppercase => value.uppercase(),
            };
            (t, value)
        })
    }

    #[inline]
    fn evaluate_slice<I>(
        values: impl Iterator<Item = (I, MedRecordValue)>,
        range: Range<usize>,
    ) -> impl Iterator<Item = (I, MedRecordValue)> {
        values.map(move |(t, value)| (t, value.slice(range.clone())))
    }

    #[inline]
    fn evaluate_either_or<'a, I: Index + 'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (I, MedRecordValue)>,
        either: &Wrapper<MultipleValuesOperand<O>>,
        or: &Wrapper<MultipleValuesOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, (I, MedRecordValue)>>
    where
        O: 'a,
    {
        let values: Vec<_> = values.collect();

        let either_values = either.evaluate_forward(medrecord, values.clone().into_iter())?;
        let or_values = or.evaluate_forward(medrecord, values.into_iter())?;

        Ok(Box::new(
            either_values
                .chain(or_values)
                .unique_by(|value| value.0.clone()),
        ))
    }

    #[inline]
    fn evaluate_exclude<'a, I: Index + 'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (I, MedRecordValue)>,
        operand: &Wrapper<MultipleValuesOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, (I, MedRecordValue)>> {
        let values: Vec<_> = values.collect();

        let result: MrHashSet<_> = operand
            .evaluate_forward(medrecord, values.clone().into_iter())?
            .map(|(t, _)| t)
            .collect();

        Ok(Box::new(
            values.into_iter().filter(move |(t, _)| !result.contains(t)),
        ))
    }
}

#[derive(Debug, Clone)]
pub enum SingleValueOperation<O: Operand> {
    SingleValueComparisonOperation {
        operand: SingleValueComparisonOperand,
        kind: SingleComparisonKind,
    },
    MultipleValuesComparisonOperation {
        operand: MultipleValuesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOpration {
        operand: SingleValueComparisonOperand,
        kind: BinaryArithmeticKind,
    },
    UnaryArithmeticOperation {
        kind: UnaryArithmeticKind,
    },

    Slice(Range<usize>),

    IsString,
    IsInt,
    IsFloat,
    IsBool,
    IsDateTime,
    IsDuration,
    IsNull,

    EitherOr {
        either: Wrapper<SingleValueOperand<O>>,
        or: Wrapper<SingleValueOperand<O>>,
    },
    Exclude {
        operand: Wrapper<SingleValueOperand<O>>,
    },
}

impl<O: Operand> DeepClone for SingleValueOperation<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::SingleValueComparisonOperation { operand, kind } => {
                Self::SingleValueComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::MultipleValuesComparisonOperation { operand, kind } => {
                Self::MultipleValuesComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::BinaryArithmeticOpration { operand, kind } => Self::BinaryArithmeticOpration {
                operand: operand.deep_clone(),
                kind: kind.clone(),
            },
            Self::UnaryArithmeticOperation { kind } => {
                Self::UnaryArithmeticOperation { kind: kind.clone() }
            }
            Self::Slice(range) => Self::Slice(range.clone()),
            Self::IsString => Self::IsString,
            Self::IsInt => Self::IsInt,
            Self::IsFloat => Self::IsFloat,
            Self::IsBool => Self::IsBool,
            Self::IsDateTime => Self::IsDateTime,
            Self::IsDuration => Self::IsDuration,
            Self::IsNull => Self::IsNull,
            Self::EitherOr { either, or } => Self::EitherOr {
                either: either.deep_clone(),
                or: or.deep_clone(),
            },
            Self::Exclude { operand } => Self::Exclude {
                operand: operand.deep_clone(),
            },
        }
    }
}

impl<O: Operand> SingleValueOperation<O> {
    pub(crate) fn evaluate<I: Index>(
        &self,
        medrecord: &MedRecord,
        value: OptionalIndexWrapper<I, MedRecordValue>,
    ) -> MedRecordResult<Option<OptionalIndexWrapper<I, MedRecordValue>>> {
        match self {
            Self::SingleValueComparisonOperation { operand, kind } => {
                Self::evaluate_single_value_comparison_operation(medrecord, value, operand, kind)
            }
            Self::MultipleValuesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_values_comparison_operation(medrecord, value, operand, kind)
            }
            Self::BinaryArithmeticOpration { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, value, operand, kind)
            }
            Self::UnaryArithmeticOperation { kind } => Ok(Some(match kind {
                UnaryArithmeticKind::Round => value.map(|value| value.round()),
                UnaryArithmeticKind::Ceil => value.map(|value| value.ceil()),
                UnaryArithmeticKind::Floor => value.map(|value| value.floor()),
                UnaryArithmeticKind::Abs => value.map(|value| value.abs()),
                UnaryArithmeticKind::Sqrt => value.map(|value| value.sqrt()),
                UnaryArithmeticKind::Trim => value.map(|value| value.trim()),
                UnaryArithmeticKind::TrimStart => value.map(|value| value.trim_start()),
                UnaryArithmeticKind::TrimEnd => value.map(|value| value.trim_end()),
                UnaryArithmeticKind::Lowercase => value.map(|value| value.lowercase()),
                UnaryArithmeticKind::Uppercase => value.map(|value| value.uppercase()),
            })),
            Self::Slice(range) => Ok(Some(value.map(|value| value.slice(range.clone())))),
            Self::IsString => Ok(match value.get_value() {
                MedRecordValue::String(_) => Some(value),
                _ => None,
            }),
            Self::IsInt => Ok(match value.get_value() {
                MedRecordValue::Int(_) => Some(value),
                _ => None,
            }),
            Self::IsFloat => Ok(match value.get_value() {
                MedRecordValue::Float(_) => Some(value),
                _ => None,
            }),
            Self::IsBool => Ok(match value.get_value() {
                MedRecordValue::Bool(_) => Some(value),
                _ => None,
            }),
            Self::IsDateTime => Ok(match value.get_value() {
                MedRecordValue::DateTime(_) => Some(value),
                _ => None,
            }),
            Self::IsDuration => Ok(match value.get_value() {
                MedRecordValue::Duration(_) => Some(value),
                _ => None,
            }),
            Self::IsNull => Ok(match value.get_value() {
                MedRecordValue::Null => Some(value),
                _ => None,
            }),
            Self::EitherOr { either, or } => Self::evaluate_either_or(medrecord, value, either, or),
            Self::Exclude { operand } => {
                Ok(match operand.evaluate_forward(medrecord, value.clone())? {
                    Some(_) => None,
                    None => Some(value),
                })
            }
        }
    }

    #[inline]
    fn evaluate_single_value_comparison_operation<I: Index>(
        medrecord: &MedRecord,
        value: OptionalIndexWrapper<I, MedRecordValue>,
        comparison_operand: &SingleValueComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<Option<OptionalIndexWrapper<I, MedRecordValue>>> {
        let comparison_value =
            comparison_operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No value to compare".to_string(),
                ))?;

        let comparison_result = match kind {
            SingleComparisonKind::GreaterThan => *value.get_value() > comparison_value,
            SingleComparisonKind::GreaterThanOrEqualTo => *value.get_value() >= comparison_value,
            SingleComparisonKind::LessThan => *value.get_value() < comparison_value,
            SingleComparisonKind::LessThanOrEqualTo => *value.get_value() <= comparison_value,
            SingleComparisonKind::EqualTo => *value.get_value() == comparison_value,
            SingleComparisonKind::NotEqualTo => *value.get_value() != comparison_value,
            SingleComparisonKind::StartsWith => value.get_value().starts_with(&comparison_value),
            SingleComparisonKind::EndsWith => value.get_value().ends_with(&comparison_value),
            SingleComparisonKind::Contains => value.get_value().contains(&comparison_value),
        };

        Ok(if comparison_result { Some(value) } else { None })
    }

    #[inline]
    fn evaluate_multiple_values_comparison_operation<I: Index>(
        medrecord: &MedRecord,
        value: OptionalIndexWrapper<I, MedRecordValue>,
        comparison_operand: &MultipleValuesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<Option<OptionalIndexWrapper<I, MedRecordValue>>> {
        let comparison_values = comparison_operand.evaluate_backward(medrecord)?;

        let comparison_result = match kind {
            MultipleComparisonKind::IsIn => comparison_values.contains(value.get_value()),
            MultipleComparisonKind::IsNotIn => !comparison_values.contains(value.get_value()),
        };

        Ok(if comparison_result { Some(value) } else { None })
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation<I: Index>(
        medrecord: &MedRecord,
        value: OptionalIndexWrapper<I, MedRecordValue>,
        operand: &SingleValueComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<Option<OptionalIndexWrapper<I, MedRecordValue>>> {
        let arithmetic_value =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No value to compare".to_string(),
                ))?;

        // Refactor once Try trait is stabilized
        Ok(Some(match value {
            OptionalIndexWrapper::WithIndex((index, value)) => match kind {
                BinaryArithmeticKind::Add => (index, value.add(arithmetic_value)?).into(),
                BinaryArithmeticKind::Sub => (index, value.sub(arithmetic_value)?).into(),
                BinaryArithmeticKind::Mul => (index, value.mul(arithmetic_value)?).into(),
                BinaryArithmeticKind::Div => (index, value.div(arithmetic_value)?).into(),
                BinaryArithmeticKind::Pow => (index, value.pow(arithmetic_value)?).into(),
                BinaryArithmeticKind::Mod => (index, value.r#mod(arithmetic_value)?).into(),
            },
            OptionalIndexWrapper::WithoutIndex(attribute) => match kind {
                BinaryArithmeticKind::Add => attribute.add(arithmetic_value)?.into(),
                BinaryArithmeticKind::Sub => attribute.sub(arithmetic_value)?.into(),
                BinaryArithmeticKind::Mul => attribute.mul(arithmetic_value)?.into(),
                BinaryArithmeticKind::Div => attribute.div(arithmetic_value)?.into(),
                BinaryArithmeticKind::Pow => attribute.pow(arithmetic_value)?.into(),
                BinaryArithmeticKind::Mod => attribute.r#mod(arithmetic_value)?.into(),
            },
        }))
    }

    #[inline]
    fn evaluate_either_or<I: Index>(
        medrecord: &MedRecord,
        value: OptionalIndexWrapper<I, MedRecordValue>,
        either: &Wrapper<SingleValueOperand<O>>,
        or: &Wrapper<SingleValueOperand<O>>,
    ) -> MedRecordResult<Option<OptionalIndexWrapper<I, MedRecordValue>>> {
        let either_result = either.evaluate_forward(medrecord, value.clone())?;
        let or_result = or.evaluate_forward(medrecord, value)?;

        match (either_result, or_result) {
            (Some(either_result), _) => Ok(Some(either_result)),
            (None, Some(or_result)) => Ok(Some(or_result)),
            _ => Ok(None),
        }
    }
}
