use super::{
    operand::{
        MultipleValuesComparisonOperand, MultipleValuesWithIndexOperand,
        SingleValueComparisonOperand, SingleValueWithIndexOperand,
    },
    BinaryArithmeticKind, MultipleComparisonKind, SingleComparisonKind, SingleKindWithIndex,
    UnaryArithmeticKind,
};
use crate::{
    errors::{MedRecordError, MedRecordResult},
    medrecord::{
        datatypes::{
            Abs, Ceil, Contains, DataType, EndsWith, Floor, Lowercase, Mod, Pow, Round, Slice,
            Sqrt, StartsWith, Trim, TrimEnd, TrimStart, Uppercase,
        },
        querying::{
            tee_grouped_iterator,
            values::{
                operand::{MultipleValuesWithoutIndexOperand, SingleValueWithoutIndexOperand},
                SingleKindWithoutIndex,
            },
            BoxedIterator, DeepClone, EvaluateForward, EvaluateForwardGrouped, GroupedIterator,
            ReadWriteOrPanic, RootOperand,
        },
        MedRecordValue, Wrapper,
    },
    MedRecord,
};
use itertools::Itertools;
use medmodels_utils::aliases::MrHashSet;
use rand::{rng, seq::IteratorRandom};
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
pub enum MultipleValuesWithIndexOperation<O: RootOperand> {
    ValueWithIndexOperation {
        operand: Wrapper<SingleValueWithIndexOperand<O>>,
    },
    ValueWithoutIndexOperation {
        operand: Wrapper<SingleValueWithoutIndexOperand<O>>,
    },
    SingleValueComparisonOperation {
        operand: SingleValueComparisonOperand,
        kind: SingleComparisonKind,
    },
    MultipleValuesComparisonOperation {
        operand: MultipleValuesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOperation {
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
        either: Wrapper<MultipleValuesWithIndexOperand<O>>,
        or: Wrapper<MultipleValuesWithIndexOperand<O>>,
    },
    Exclude {
        operand: Wrapper<MultipleValuesWithIndexOperand<O>>,
    },

    Merge {
        operand: Wrapper<MultipleValuesWithIndexOperand<O>>,
    },
}

impl<O: RootOperand> DeepClone for MultipleValuesWithIndexOperation<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::ValueWithIndexOperation { operand } => Self::ValueWithIndexOperation {
                operand: operand.deep_clone(),
            },
            Self::ValueWithoutIndexOperation { operand } => Self::ValueWithoutIndexOperation {
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
            Self::BinaryArithmeticOperation { operand, kind } => Self::BinaryArithmeticOperation {
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
            Self::Merge { operand } => Self::Merge {
                operand: operand.deep_clone(),
            },
        }
    }
}

impl<O: RootOperand> MultipleValuesWithIndexOperation<O> {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::ValueWithIndexOperation { operand } => {
                Self::evaluate_value_with_index_operation(medrecord, values, operand)?
            }
            Self::ValueWithoutIndexOperation { operand } => {
                Self::evaluate_value_without_index_operation(medrecord, values, operand)?
            }
            Self::SingleValueComparisonOperation { operand, kind } => {
                Self::evaluate_single_value_comparison_operation(medrecord, values, operand, kind)?
            }
            Self::MultipleValuesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_values_comparison_operation(
                    medrecord, values, operand, kind,
                )?
            }
            Self::BinaryArithmeticOperation { operand, kind } => Box::new(
                Self::evaluate_binary_arithmetic_operation(medrecord, values, operand, kind)?,
            ),
            Self::UnaryArithmeticOperation { kind } => Box::new(
                Self::evaluate_unary_arithmetic_operation(values, kind.clone()),
            ),
            Self::Slice(range) => Box::new(Self::evaluate_slice(values, range.clone())),
            Self::IsString => Box::new(Self::evaluate_is_string(values)),
            Self::IsInt => Box::new(Self::evaluate_is_int(values)),
            Self::IsFloat => Box::new(Self::evaluate_is_float(values)),
            Self::IsBool => Box::new(Self::evaluate_is_bool(values)),
            Self::IsDateTime => Box::new(Self::evaluate_is_datetime(values)),
            Self::IsDuration => Box::new(Self::evaluate_is_duration(values)),
            Self::IsNull => Box::new(Self::evaluate_is_null(values)),
            Self::IsMax => Self::evaluate_is_max(values)?,
            Self::IsMin => Self::evaluate_is_min(values)?,
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, values, either, or)?
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, values, operand)?,
            Self::Merge { operand: _ } => unreachable!(),
        })
    }

    #[inline]
    pub(crate) fn get_max<'a>(
        mut values: impl Iterator<Item = (&'a O::Index, MedRecordValue)>,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordValue)>> {
        let max_value = values.next();

        let Some(max_value) = max_value else {
            return Ok(None);
        };

        let max_value = values.try_fold(max_value, |max_value, value| {
            match value.1.partial_cmp(&max_value.1) {
                Some(Ordering::Greater) => Ok(value),
                None => {
                    let first_dtype = DataType::from(value.1);
                    let second_dtype = DataType::from(max_value.1);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare values of data types {first_dtype} and {second_dtype}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool(), .is_datetime() or .is_duration()"
                    )))
                }
                _ => Ok(max_value),
            }
        })?;

        Ok(Some(max_value))
    }

    #[inline]
    pub(crate) fn get_min<'a>(
        mut values: impl Iterator<Item = (&'a O::Index, MedRecordValue)>,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordValue)>> {
        let min_value = values.next();

        let Some(min_value) = min_value else {
            return Ok(None);
        };

        let min_value = values.try_fold(min_value, |min_value, value| {
            match value.1.partial_cmp(&min_value.1) {
                Some(Ordering::Less) => Ok(value),
                None => {
                    let first_dtype = DataType::from(value.1);
                    let second_dtype = DataType::from(min_value.1);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare values of data types {first_dtype} and {second_dtype}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool(), .is_datetime() or .is_duration()"
                    )))
                }
                _ => Ok(min_value),
            }
        })?;

        Ok(Some(min_value))
    }

    #[inline]
    pub(crate) fn get_random<'a>(
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)>,
    ) -> Option<(&'a O::Index, MedRecordValue)> {
        values.choose(&mut rng())
    }

    #[inline]
    fn evaluate_value_with_index_operation<'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)> + 'a,
        operand: &Wrapper<SingleValueWithIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);

        let kind = &operand.0.read_or_panic().kind;

        let value = match kind {
            SingleKindWithIndex::Max => MultipleValuesWithIndexOperation::<O>::get_max(values_1)?,
            SingleKindWithIndex::Min => MultipleValuesWithIndexOperation::<O>::get_min(values_1)?,
            SingleKindWithIndex::Random => {
                MultipleValuesWithIndexOperation::<O>::get_random(values_1)
            }
        };

        Ok(match operand.evaluate_forward(medrecord, value)? {
            Some(_) => Box::new(values_2),
            None => Box::new(std::iter::empty()),
        })
    }

    #[inline]
    fn evaluate_value_without_index_operation<'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)> + 'a,
        operand: &Wrapper<SingleValueWithoutIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);
        let values_1 = values_1.map(|(_, value)| value);

        let kind = &operand.0.read_or_panic().kind;

        let value = match kind {
            SingleKindWithoutIndex::Max => {
                MultipleValuesWithoutIndexOperation::<O>::get_max(values_1)?
            }
            SingleKindWithoutIndex::Min => {
                MultipleValuesWithoutIndexOperation::<O>::get_min(values_1)?
            }
            SingleKindWithoutIndex::Mean => {
                MultipleValuesWithoutIndexOperation::<O>::get_mean(values_1)?
            }
            SingleKindWithoutIndex::Median => {
                MultipleValuesWithoutIndexOperation::<O>::get_median(values_1)?
            }
            SingleKindWithoutIndex::Mode => {
                MultipleValuesWithoutIndexOperation::<O>::get_mode(values_1)?
            }
            SingleKindWithoutIndex::Std => {
                MultipleValuesWithoutIndexOperation::<O>::get_std(values_1)?
            }
            SingleKindWithoutIndex::Var => {
                MultipleValuesWithoutIndexOperation::<O>::get_var(values_1)?
            }
            SingleKindWithoutIndex::Count => Some(
                MultipleValuesWithoutIndexOperation::<O>::get_count(values_1),
            ),
            SingleKindWithoutIndex::Sum => {
                MultipleValuesWithoutIndexOperation::<O>::get_sum(values_1)?
            }
            SingleKindWithoutIndex::Random => {
                MultipleValuesWithoutIndexOperation::<O>::get_random(values_1)
            }
        };

        Ok(match operand.evaluate_forward(medrecord, value)? {
            Some(_) => Box::new(values_2),
            None => Box::new(std::iter::empty()),
        })
    }

    #[inline]
    fn evaluate_single_value_comparison_operation<'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)> + 'a,
        comparison_operand: &SingleValueComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordValue)>> {
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
    fn evaluate_multiple_values_comparison_operation<'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)> + 'a,
        comparison_operand: &MultipleValuesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordValue)>> {
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
    fn evaluate_binary_arithmetic_operation<'a>(
        medrecord: &MedRecord,
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)>,
        operand: &SingleValueComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordValue)>>
    where
        O: 'a,
    {
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
                        "Failed arithmetic operation {kind}. Consider narrowing down the values using .is_int() or .is_float()",
                    ))
                }).map(|result| (t, result))
            });

        Ok(values.collect::<MedRecordResult<Vec<_>>>()?.into_iter())
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation<'a>(
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)>,
        kind: UnaryArithmeticKind,
    ) -> impl Iterator<Item = (&'a O::Index, MedRecordValue)>
    where
        O: 'a,
    {
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
    fn evaluate_slice<'a>(
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)>,
        range: Range<usize>,
    ) -> impl Iterator<Item = (&'a O::Index, MedRecordValue)>
    where
        O: 'a,
    {
        values.map(move |(t, value)| (t, value.slice(range.clone())))
    }

    #[inline]
    fn evaluate_is_string<'a>(
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)>,
    ) -> impl Iterator<Item = (&'a O::Index, MedRecordValue)>
    where
        O: 'a,
    {
        values.filter(|(_, value)| matches!(value, MedRecordValue::String(_)))
    }

    #[inline]
    fn evaluate_is_int<'a>(
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)>,
    ) -> impl Iterator<Item = (&'a O::Index, MedRecordValue)>
    where
        O: 'a,
    {
        values.filter(|(_, value)| matches!(value, MedRecordValue::Int(_)))
    }

    #[inline]
    fn evaluate_is_float<'a>(
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)>,
    ) -> impl Iterator<Item = (&'a O::Index, MedRecordValue)>
    where
        O: 'a,
    {
        values.filter(|(_, value)| matches!(value, MedRecordValue::Float(_)))
    }

    #[inline]
    fn evaluate_is_bool<'a>(
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)>,
    ) -> impl Iterator<Item = (&'a O::Index, MedRecordValue)>
    where
        O: 'a,
    {
        values.filter(|(_, value)| matches!(value, MedRecordValue::Bool(_)))
    }

    #[inline]
    fn evaluate_is_datetime<'a>(
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)>,
    ) -> impl Iterator<Item = (&'a O::Index, MedRecordValue)>
    where
        O: 'a,
    {
        values.filter(|(_, value)| matches!(value, MedRecordValue::DateTime(_)))
    }

    #[inline]
    fn evaluate_is_duration<'a>(
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)>,
    ) -> impl Iterator<Item = (&'a O::Index, MedRecordValue)>
    where
        O: 'a,
    {
        values.filter(|(_, value)| matches!(value, MedRecordValue::Duration(_)))
    }

    #[inline]
    fn evaluate_is_null<'a>(
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)>,
    ) -> impl Iterator<Item = (&'a O::Index, MedRecordValue)>
    where
        O: 'a,
    {
        values.filter(|(_, value)| matches!(value, MedRecordValue::Null))
    }

    #[inline]
    fn evaluate_is_max<'a>(
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);

        let max_value = Self::get_max(values_1)?;

        let Some(max_value) = max_value else {
            return Ok(Box::new(std::iter::empty()));
        };

        Ok(Box::new(
            values_2.filter(move |(_, value)| *value == max_value.1),
        ))
    }

    #[inline]
    fn evaluate_is_min<'a>(
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);

        let min_value = Self::get_min(values_1)?;

        let Some(min_value) = min_value else {
            return Ok(Box::new(std::iter::empty()));
        };

        Ok(Box::new(
            values_2.filter(move |(_, value)| *value == min_value.1),
        ))
    }

    #[inline]
    fn evaluate_either_or<'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)> + 'a,
        either: &Wrapper<MultipleValuesWithIndexOperand<O>>,
        or: &Wrapper<MultipleValuesWithIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);

        let either_values = either.evaluate_forward(medrecord, Box::new(values_1))?;
        let or_values = or.evaluate_forward(medrecord, Box::new(values_2))?;

        Ok(Box::new(
            either_values
                .chain(or_values)
                .unique_by(|value| value.0.clone()),
        ))
    }

    #[inline]
    fn evaluate_exclude<'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)> + 'a,
        operand: &Wrapper<MultipleValuesWithIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);

        let result: MrHashSet<_> = operand
            .evaluate_forward(medrecord, Box::new(values_1))?
            .map(|(t, _)| t)
            .collect();

        Ok(Box::new(values_2.filter(move |(t, _)| !result.contains(t))))
    }
}

impl<O: RootOperand> MultipleValuesWithIndexOperation<O> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::ValueWithIndexOperation { operand } => Box::new(
                Self::evaluate_value_with_index_operation_grouped(medrecord, values, operand)?,
            ),
            Self::ValueWithoutIndexOperation { operand } => Box::new(
                Self::evaluate_value_without_index_operation_grouped(medrecord, values, operand)?,
            ),
            Self::SingleValueComparisonOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_single_value_comparison_operation(
                                medrecord, values, operand, kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::MultipleValuesComparisonOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_multiple_values_comparison_operation(
                                medrecord, values, operand, kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::BinaryArithmeticOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_binary_arithmetic_operation(
                                medrecord, values, operand, kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::UnaryArithmeticOperation { kind } => {
                let kind = kind.clone();

                Box::new(values.map(move |(key, values)| {
                    (
                        key,
                        Box::new(Self::evaluate_unary_arithmetic_operation(
                            values,
                            kind.clone(),
                        )) as BoxedIterator<_>,
                    )
                }))
            }
            Self::Slice(range) => {
                let range = range.clone();

                Box::new(values.map(move |(key, values)| {
                    (
                        key,
                        Box::new(Self::evaluate_slice(values, range.clone())) as BoxedIterator<_>,
                    )
                }))
            }
            Self::IsString => Box::new(values.map(move |(key, values)| {
                (
                    key,
                    Box::new(Self::evaluate_is_string(values)) as BoxedIterator<_>,
                )
            })),
            Self::IsInt => Box::new(values.map(move |(key, values)| {
                (
                    key,
                    Box::new(Self::evaluate_is_int(values)) as BoxedIterator<_>,
                )
            })),
            Self::IsFloat => Box::new(values.map(move |(key, values)| {
                (
                    key,
                    Box::new(Self::evaluate_is_float(values)) as BoxedIterator<_>,
                )
            })),
            Self::IsBool => Box::new(values.map(move |(key, values)| {
                (
                    key,
                    Box::new(Self::evaluate_is_bool(values)) as BoxedIterator<_>,
                )
            })),
            Self::IsDateTime => Box::new(values.map(move |(key, values)| {
                (
                    key,
                    Box::new(Self::evaluate_is_datetime(values)) as BoxedIterator<_>,
                )
            })),
            Self::IsDuration => Box::new(values.map(move |(key, values)| {
                (
                    key,
                    Box::new(Self::evaluate_is_duration(values)) as BoxedIterator<_>,
                )
            })),
            Self::IsNull => Box::new(values.map(move |(key, values)| {
                (
                    key,
                    Box::new(Self::evaluate_is_null(values)) as BoxedIterator<_>,
                )
            })),
            Self::IsMax => Box::new(
                values
                    .map(move |(key, values)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_is_max(values)?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::IsMin => Box::new(
                values
                    .map(move |(key, values)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_is_min(values)?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or_grouped(medrecord, values, either, or)?
            }
            Self::Exclude { operand } => {
                Self::evaluate_exclude_grouped(medrecord, values, operand)?
            }
            Self::Merge { operand } => {
                let (values_1, values_2) = tee_grouped_iterator(values);

                let values_1 = values_1.flat_map(|(_, value)| value);

                let values_1: Vec<_> = operand
                    .evaluate_forward(medrecord, Box::new(values_1))?
                    .collect();

                Box::new(values_2.map(move |(key, values)| {
                    let values: Vec<_> = values.filter(|value| values_1.contains(value)).collect();

                    let values: BoxedIterator<_> = Box::new(values.into_iter());

                    (key, values)
                }))
            }
        })
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_value_with_index_operation_grouped<'a>(
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>,
        operand: &Wrapper<SingleValueWithIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>>
    where
        O: 'a,
    {
        let (values_1, values_2) = tee_grouped_iterator(values);
        let mut values_2 = values_2.collect::<Vec<_>>();

        let kind = &operand.0.read_or_panic().kind;

        let values_1: Vec<_> = values_1
            .map(|(key, values)| {
                let value = match kind {
                    SingleKindWithIndex::Max => {
                        MultipleValuesWithIndexOperation::<O>::get_max(values)?
                    }
                    SingleKindWithIndex::Min => {
                        MultipleValuesWithIndexOperation::<O>::get_min(values)?
                    }
                    SingleKindWithIndex::Random => {
                        MultipleValuesWithIndexOperation::<O>::get_random(values)
                    }
                };

                Ok((key, value))
            })
            .collect::<MedRecordResult<_>>()?;

        let values_1 =
            operand.evaluate_forward_grouped(medrecord, Box::new(values_1.into_iter()))?;

        Ok(Box::new(values_1.map(move |(key, value)| match value {
            Some(_) => {
                let values_position = values_2
                    .iter()
                    .position(|(k, _)| k == &key)
                    .expect("Entry must exist");

                values_2.remove(values_position)
            }
            None => (key, Box::new(std::iter::empty()) as BoxedIterator<_>),
        })))
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_value_without_index_operation_grouped<'a>(
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>,
        operand: &Wrapper<SingleValueWithoutIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>>
    where
        O: 'a,
    {
        let (values_1, values_2) = tee_grouped_iterator(values);
        let mut values_2: Vec<_> = values_2.collect();

        let kind = &operand.0.read_or_panic().kind;

        let values_1: Vec<_> = values_1
            .map(|(key, values)| {
                let values = values.map(|(_, value)| value);

                let value = match kind {
                    SingleKindWithoutIndex::Max => {
                        MultipleValuesWithoutIndexOperation::<O>::get_max(values)?
                    }
                    SingleKindWithoutIndex::Min => {
                        MultipleValuesWithoutIndexOperation::<O>::get_min(values)?
                    }
                    SingleKindWithoutIndex::Mean => {
                        MultipleValuesWithoutIndexOperation::<O>::get_mean(values)?
                    }
                    SingleKindWithoutIndex::Median => {
                        MultipleValuesWithoutIndexOperation::<O>::get_median(values)?
                    }
                    SingleKindWithoutIndex::Mode => {
                        MultipleValuesWithoutIndexOperation::<O>::get_mode(values)?
                    }
                    SingleKindWithoutIndex::Std => {
                        MultipleValuesWithoutIndexOperation::<O>::get_std(values)?
                    }
                    SingleKindWithoutIndex::Var => {
                        MultipleValuesWithoutIndexOperation::<O>::get_var(values)?
                    }
                    SingleKindWithoutIndex::Count => {
                        Some(MultipleValuesWithoutIndexOperation::<O>::get_count(values))
                    }
                    SingleKindWithoutIndex::Sum => {
                        MultipleValuesWithoutIndexOperation::<O>::get_sum(values)?
                    }
                    SingleKindWithoutIndex::Random => {
                        MultipleValuesWithoutIndexOperation::<O>::get_random(values)
                    }
                };

                Ok((key, value))
            })
            .collect::<MedRecordResult<_>>()?;

        let values_1 =
            operand.evaluate_forward_grouped(medrecord, Box::new(values_1.into_iter()))?;

        Ok(Box::new(values_1.map(move |(key, value)| match value {
            Some(_) => {
                let values_position = values_2
                    .iter()
                    .position(|(k, _)| k == &key)
                    .expect("Entry must exist");

                values_2.remove(values_position)
            }
            None => (key, Box::new(std::iter::empty()) as BoxedIterator<_>),
        })))
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_either_or_grouped<'a>(
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>,
        either: &Wrapper<MultipleValuesWithIndexOperand<O>>,
        or: &Wrapper<MultipleValuesWithIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>>
    where
        O: 'a,
    {
        let (values_1, values_2) = tee_grouped_iterator(values);

        let either_values = either.evaluate_forward_grouped(medrecord, values_1)?;
        let mut or_values: Vec<_> = or.evaluate_forward_grouped(medrecord, values_2)?.collect();

        let values = either_values.map(move |(key, either_values)| {
            let values_position = or_values
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let or_values = or_values.remove(values_position).1;

            let values: BoxedIterator<_> = Box::new(
                either_values
                    .chain(or_values)
                    .unique_by(|value| value.0.clone()),
            );

            (key, values)
        });

        Ok(Box::new(values))
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_exclude_grouped<'a>(
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>,
        operand: &Wrapper<MultipleValuesWithIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>>
    where
        O: 'a,
    {
        let (values_1, values_2) = tee_grouped_iterator(values);

        let mut result: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, values_1)?
            .collect();

        let values = values_2.map(move |(key, values)| {
            let values_position = result
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let excluded_values: Vec<_> = result.remove(values_position).1.collect();

            let values: BoxedIterator<_> =
                Box::new(values.filter(move |value| !excluded_values.contains(value)));

            (key, values)
        });

        Ok(Box::new(values))
    }
}

#[derive(Debug, Clone)]
pub enum MultipleValuesWithoutIndexOperation<O: RootOperand> {
    ValueOperation {
        operand: Wrapper<SingleValueWithoutIndexOperand<O>>,
    },
    SingleValueComparisonOperation {
        operand: SingleValueComparisonOperand,
        kind: SingleComparisonKind,
    },
    MultipleValuesComparisonOperation {
        operand: MultipleValuesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOperation {
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
        either: Wrapper<MultipleValuesWithoutIndexOperand<O>>,
        or: Wrapper<MultipleValuesWithoutIndexOperand<O>>,
    },
    Exclude {
        operand: Wrapper<MultipleValuesWithoutIndexOperand<O>>,
    },
}

impl<O: RootOperand> DeepClone for MultipleValuesWithoutIndexOperation<O> {
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
            Self::BinaryArithmeticOperation { operand, kind } => Self::BinaryArithmeticOperation {
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

impl<O: RootOperand> MultipleValuesWithoutIndexOperation<O> {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = MedRecordValue> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordValue>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::ValueOperation { operand } => {
                Self::evaluate_value_operation(medrecord, values, operand)?
            }
            Self::SingleValueComparisonOperation { operand, kind } => {
                Self::evaluate_single_value_comparison_operation(medrecord, values, operand, kind)?
            }
            Self::MultipleValuesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_values_comparison_operation(
                    medrecord, values, operand, kind,
                )?
            }
            Self::BinaryArithmeticOperation { operand, kind } => Box::new(
                Self::evaluate_binary_arithmetic_operation(medrecord, values, operand, kind)?,
            ),
            Self::UnaryArithmeticOperation { kind } => Box::new(
                Self::evaluate_unary_arithmetic_operation(values, kind.clone()),
            ),
            Self::Slice(range) => Box::new(Self::evaluate_slice(values, range.clone())),
            Self::IsString => Box::new(Self::evaluate_is_string(values)),
            Self::IsInt => Box::new(Self::evaluate_is_int(values)),
            Self::IsFloat => Box::new(Self::evaluate_is_float(values)),
            Self::IsBool => Box::new(Self::evaluate_is_bool(values)),
            Self::IsDateTime => Box::new(Self::evaluate_is_datetime(values)),
            Self::IsDuration => Box::new(Self::evaluate_is_duration(values)),
            Self::IsNull => Box::new(Self::evaluate_is_null(values)),
            Self::IsMax => Self::evaluate_is_max(values)?,
            Self::IsMin => Self::evaluate_is_min(values)?,
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, values, either, or)?
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, values, operand)?,
        })
    }

    #[inline]
    pub(crate) fn get_max(
        mut values: impl Iterator<Item = MedRecordValue>,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        let max_value = values.next();

        let Some(max_value) = max_value else {
            return Ok(None);
        };

        let max_value = values.try_fold(max_value, |max_value, value| {
            match value.partial_cmp(&max_value) {
                Some(Ordering::Greater) => Ok(value),
                None => {
                    let first_dtype = DataType::from(value);
                    let second_dtype = DataType::from(max_value);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare values of data types {first_dtype} and {second_dtype}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool(), .is_datetime() or .is_duration()"
                    )))
                }
                _ => Ok(max_value),
            }
        })?;

        Ok(Some(max_value))
    }

    #[inline]
    pub(crate) fn get_min(
        mut values: impl Iterator<Item = MedRecordValue>,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        let min_value = values.next();

        let Some(min_value) = min_value else {
            return Ok(None);
        };

        let min_value = values.try_fold(min_value, |min_value, value| {
            match value.partial_cmp(&min_value) {
                Some(Ordering::Less) => Ok(value),
                None => {
                    let first_dtype = DataType::from(value);
                    let second_dtype = DataType::from(min_value);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare values of data types {first_dtype} and {second_dtype}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool(), .is_datetime() or .is_duration()"
                    )))
                }
                _ => Ok(min_value),
            }
        })?;

        Ok(Some(min_value))
    }

    #[inline]
    pub(crate) fn get_mean(
        mut values: impl Iterator<Item = MedRecordValue>,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        let first_value = values.next();

        let Some(first_value) = first_value else {
            return Ok(None);
        };

        let (sum, count) = values.try_fold((first_value, 1), |(sum, count), value| {
            let first_dtype = DataType::from(&sum);
            let second_dtype = DataType::from(&value);

            match sum.add(value) {
                Ok(sum) => Ok((sum, count + 1)),
                Err(_) => Err(MedRecordError::QueryError(format!(
                    "Cannot add values of data types {first_dtype} and {second_dtype}. Consider narrowing down the values using .is_int(), .is_float(), .is_datetime() or .is_duration()"
                ))),
            }
        })?;

        Ok(Some(sum.div(MedRecordValue::Int(count as i64))?))
    }

    // TODO: This is a temporary solution. It should be optimized.
    #[inline]
    pub(crate) fn get_median(
        mut values: impl Iterator<Item = MedRecordValue>,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        let first_value = values.next();

        let Some(first_value) = first_value else {
            return Ok(None);
        };

        let first_data_type = DataType::from(&first_value);

        let median = match first_value {
            MedRecordValue::Int(value) => {
                let mut values: Vec<_> = values.map(|value| {
                    let data_type = DataType::from(&value);

                    match value {
                        MedRecordValue::Int(value) => Ok(value as f64),
                        MedRecordValue::Float(value) => Ok(value),
                        _ => Err(MedRecordError::QueryError(format!(
                            "Cannot calculate median of mixed data types {first_data_type} and {data_type}. Consider narrowing down the values using .is_int(), .is_float() , .is_datetime() or .is_duration()"
                        ))),
                    }
                }).collect::<MedRecordResult<_>>()?;
                values.push(value as f64);
                values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

                get_median!(values, Float)
            }
            MedRecordValue::Float(value) => {
                let mut values: Vec<_> = values.map(|value| {
                    let data_type = DataType::from(&value);

                    match value {
                        MedRecordValue::Int(value) => Ok(value as f64),
                        MedRecordValue::Float(value) => Ok(value),
                        _ => Err(MedRecordError::QueryError(format!(
                            "Cannot calculate median of mixed data types {first_data_type} and {data_type}. Consider narrowing down the values using .is_int(), .is_float(), .is_datetime() or .is_duration()"
                        ))),
                    }
                }).collect::<MedRecordResult<_>>()?;
                values.push(value);
                values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

                get_median!(values, Float)
            }
            MedRecordValue::DateTime(value) => {
                let mut values: Vec<_> = values.map(|value| {
                    let data_type = DataType::from(&value);

                    match value {
                        MedRecordValue::DateTime(naive_date_time) => Ok(naive_date_time),
                        _ => Err(MedRecordError::QueryError(format!(
                            "Cannot calculate median of mixed data types {first_data_type} and {data_type}. Consider narrowing down the values using .is_int(), .is_float(), .is_datetime() or .is_duration()"
                        ))),
                    }
                }).collect::<MedRecordResult<_>>()?;
                values.push(value);
                values.sort();

                get_median!(values, DateTime)
            }
            MedRecordValue::Duration(value) => {
                let mut values: Vec<_> = values.map(|value| {
                    let data_type = DataType::from(&value);

                    match value {
                        MedRecordValue::Duration(naive_date_time) => Ok(naive_date_time),
                        _ => Err(MedRecordError::QueryError(format!(
                            "Cannot calculate median of mixed data types {first_data_type} and {data_type}. Consider narrowing down the values using .is_int(), .is_float(), .is_datetime() or .is_duration()"
                        ))),
                    }
                }).collect::<MedRecordResult<_>>()?;
                values.push(value);
                values.sort();

                get_median!(values, Duration)
            }
            _ => Err(MedRecordError::QueryError(format!(
                "Cannot calculate median of data type {first_data_type}"
            )))?,
        }?;

        Ok(Some(median))
    }

    // TODO: This is a temporary solution. It should be optimized.
    #[inline]
    pub(crate) fn get_mode(
        values: impl Iterator<Item = MedRecordValue>,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        let values: Vec<_> = values.collect();

        let most_common_value = values.first();

        let Some(most_common_value) = most_common_value else {
            return Ok(None);
        };

        let most_common_value = most_common_value.clone();

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

        Ok(Some(most_common_value.clone()))
    }

    #[inline]
    // 👀
    pub(crate) fn get_std(
        values: impl Iterator<Item = MedRecordValue>,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        let variance = Self::get_var(values)?;

        let Some(variance) = variance else {
            return Ok(None);
        };

        let MedRecordValue::Float(variance) = variance else {
            unreachable!()
        };

        Ok(Some(MedRecordValue::Float(variance.sqrt())))
    }

    // TODO: This is a temporary solution. It should be optimized.
    #[inline]
    pub(crate) fn get_var(
        values: impl Iterator<Item = MedRecordValue>,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        let (values_1, values_2) = Itertools::tee(values);

        let mean = Self::get_mean(values_1)?;

        let Some(mean) = mean else {
            return Ok(None);
        };

        let MedRecordValue::Float(mean) = mean else {
            let data_type = DataType::from(mean);

            return Err(MedRecordError::QueryError(
                format!("Cannot calculate variance of data type {data_type}. Consider narrowing down the values using .is_int() or .is_float()"),
            ));
        };

        let values = values_2
            .into_iter()
            .map(|value| {
                let data_type = DataType::from(&value);

                match value {
                MedRecordValue::Int(value) => Ok(value as f64),
                MedRecordValue::Float(value) => Ok(value),
                _ => Err(MedRecordError::QueryError(
                    format!("Cannot calculate variance of data type {data_type}. Consider narrowing down the values using .is_int() or .is_float()"),
                )),
            }})
            .collect::<MedRecordResult<Vec<_>>>()?;

        let values_length = values.len();

        let variance = values
            .into_iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values_length as f64;

        Ok(Some(MedRecordValue::Float(variance)))
    }

    #[inline]
    pub(crate) fn get_count(values: impl Iterator<Item = MedRecordValue>) -> MedRecordValue {
        MedRecordValue::Int(values.count() as i64)
    }

    #[inline]
    // 🥊💥
    pub(crate) fn get_sum(
        mut values: impl Iterator<Item = MedRecordValue>,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        let first_value = values.next();

        let Some(first_value) = first_value else {
            return Ok(None);
        };

        let sum = values.try_fold(first_value, |sum, value| {
            let first_dtype = DataType::from(&sum);
            let second_dtype = DataType::from(&value);

            sum.add(value).map_err(|_| {
                MedRecordError::QueryError(format!(
                    "Cannot add values of data types {first_dtype} and {second_dtype}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool(), .is_datetime() or .is_duration()"
                ))
            })
        })?;

        Ok(Some(sum))
    }

    #[inline]
    pub(crate) fn get_random(
        values: impl Iterator<Item = MedRecordValue>,
    ) -> Option<MedRecordValue> {
        values.choose(&mut rng())
    }

    #[inline]
    fn evaluate_value_operation<'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = MedRecordValue> + 'a,
        operand: &Wrapper<SingleValueWithoutIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordValue>> {
        let (values_1, values_2) = Itertools::tee(values);

        let kind = &operand.0.read_or_panic().kind;

        let value = match kind {
            SingleKindWithoutIndex::Max => {
                MultipleValuesWithoutIndexOperation::<O>::get_max(values_1)?
            }
            SingleKindWithoutIndex::Min => {
                MultipleValuesWithoutIndexOperation::<O>::get_min(values_1)?
            }
            SingleKindWithoutIndex::Mean => {
                MultipleValuesWithoutIndexOperation::<O>::get_mean(values_1)?
            }
            SingleKindWithoutIndex::Median => {
                MultipleValuesWithoutIndexOperation::<O>::get_median(values_1)?
            }
            SingleKindWithoutIndex::Mode => {
                MultipleValuesWithoutIndexOperation::<O>::get_mode(values_1)?
            }
            SingleKindWithoutIndex::Std => {
                MultipleValuesWithoutIndexOperation::<O>::get_std(values_1)?
            }
            SingleKindWithoutIndex::Var => {
                MultipleValuesWithoutIndexOperation::<O>::get_var(values_1)?
            }
            SingleKindWithoutIndex::Count => Some(
                MultipleValuesWithoutIndexOperation::<O>::get_count(values_1),
            ),
            SingleKindWithoutIndex::Sum => {
                MultipleValuesWithoutIndexOperation::<O>::get_sum(values_1)?
            }
            SingleKindWithoutIndex::Random => {
                MultipleValuesWithoutIndexOperation::<O>::get_random(values_1)
            }
        };

        Ok(match operand.evaluate_forward(medrecord, value)? {
            Some(_) => Box::new(values_2),
            None => Box::new(std::iter::empty()),
        })
    }

    #[inline]
    fn evaluate_single_value_comparison_operation<'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = MedRecordValue> + 'a,
        comparison_operand: &SingleValueComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordValue>> {
        let comparison_value =
            comparison_operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No value to compare".to_string(),
                ))?;

        match kind {
            SingleComparisonKind::GreaterThan => Ok(Box::new(
                values.filter(move |value| value > &comparison_value),
            )),
            SingleComparisonKind::GreaterThanOrEqualTo => Ok(Box::new(
                values.filter(move |value| value >= &comparison_value),
            )),
            SingleComparisonKind::LessThan => Ok(Box::new(
                values.filter(move |value| value < &comparison_value),
            )),
            SingleComparisonKind::LessThanOrEqualTo => Ok(Box::new(
                values.filter(move |value| value <= &comparison_value),
            )),
            SingleComparisonKind::EqualTo => Ok(Box::new(
                values.filter(move |value| value == &comparison_value),
            )),
            SingleComparisonKind::NotEqualTo => Ok(Box::new(
                values.filter(move |value| value != &comparison_value),
            )),
            SingleComparisonKind::StartsWith => Ok(Box::new(
                values.filter(move |value| value.starts_with(&comparison_value)),
            )),
            SingleComparisonKind::EndsWith => Ok(Box::new(
                values.filter(move |value| value.ends_with(&comparison_value)),
            )),
            SingleComparisonKind::Contains => Ok(Box::new(
                values.filter(move |value| value.contains(&comparison_value)),
            )),
        }
    }

    #[inline]
    fn evaluate_multiple_values_comparison_operation<'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = MedRecordValue> + 'a,
        comparison_operand: &MultipleValuesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordValue>> {
        let comparison_values = comparison_operand.evaluate_backward(medrecord)?;

        match kind {
            MultipleComparisonKind::IsIn => Ok(Box::new(
                values.filter(move |value| comparison_values.contains(value)),
            )),
            MultipleComparisonKind::IsNotIn => Ok(Box::new(
                values.filter(move |value| !comparison_values.contains(value)),
            )),
        }
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation(
        medrecord: &MedRecord,
        values: impl Iterator<Item = MedRecordValue>,
        operand: &SingleValueComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<impl Iterator<Item = MedRecordValue>> {
        let arithmetic_value =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No value to compare".to_string(),
                ))?;

        let values = values
            .map(move |value| {
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
                        "Failed arithmetic operation {kind}. Consider narrowing down the values using .is_int() or .is_float()",
                    ))
                })
            });

        Ok(values.collect::<MedRecordResult<Vec<_>>>()?.into_iter())
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation<'a>(
        values: impl Iterator<Item = MedRecordValue>,
        kind: UnaryArithmeticKind,
    ) -> impl Iterator<Item = MedRecordValue>
    where
        O: 'a,
    {
        values.map(move |value| match kind {
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
        })
    }

    #[inline]
    fn evaluate_slice<'a>(
        values: impl Iterator<Item = MedRecordValue>,
        range: Range<usize>,
    ) -> impl Iterator<Item = MedRecordValue>
    where
        O: 'a,
    {
        values.map(move |value| value.slice(range.clone()))
    }

    #[inline]
    fn evaluate_is_string<'a>(
        values: impl Iterator<Item = MedRecordValue>,
    ) -> impl Iterator<Item = MedRecordValue>
    where
        O: 'a,
    {
        values.filter(|value| matches!(value, MedRecordValue::String(_)))
    }

    #[inline]
    fn evaluate_is_int<'a>(
        values: impl Iterator<Item = MedRecordValue>,
    ) -> impl Iterator<Item = MedRecordValue>
    where
        O: 'a,
    {
        values.filter(|value| matches!(value, MedRecordValue::Int(_)))
    }

    #[inline]
    fn evaluate_is_float<'a>(
        values: impl Iterator<Item = MedRecordValue>,
    ) -> impl Iterator<Item = MedRecordValue>
    where
        O: 'a,
    {
        values.filter(|value| matches!(value, MedRecordValue::Float(_)))
    }

    #[inline]
    fn evaluate_is_bool<'a>(
        values: impl Iterator<Item = MedRecordValue>,
    ) -> impl Iterator<Item = MedRecordValue>
    where
        O: 'a,
    {
        values.filter(|value| matches!(value, MedRecordValue::Bool(_)))
    }

    #[inline]
    fn evaluate_is_datetime<'a>(
        values: impl Iterator<Item = MedRecordValue>,
    ) -> impl Iterator<Item = MedRecordValue>
    where
        O: 'a,
    {
        values.filter(|value| matches!(value, MedRecordValue::DateTime(_)))
    }

    #[inline]
    fn evaluate_is_duration<'a>(
        values: impl Iterator<Item = MedRecordValue>,
    ) -> impl Iterator<Item = MedRecordValue>
    where
        O: 'a,
    {
        values.filter(|value| matches!(value, MedRecordValue::Duration(_)))
    }

    #[inline]
    fn evaluate_is_null<'a>(
        values: impl Iterator<Item = MedRecordValue>,
    ) -> impl Iterator<Item = MedRecordValue>
    where
        O: 'a,
    {
        values.filter(|value| matches!(value, MedRecordValue::Null))
    }

    #[inline]
    fn evaluate_is_max<'a>(
        values: impl Iterator<Item = MedRecordValue> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordValue>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);

        let max_value = Self::get_max(values_1)?;

        let Some(max_value) = max_value else {
            return Ok(Box::new(std::iter::empty()));
        };

        Ok(Box::new(values_2.filter(move |value| *value == max_value)))
    }

    #[inline]
    fn evaluate_is_min<'a>(
        values: impl Iterator<Item = MedRecordValue> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordValue>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);

        let min_value = Self::get_min(values_1)?;

        let Some(min_value) = min_value else {
            return Ok(Box::new(std::iter::empty()));
        };

        Ok(Box::new(values_2.filter(move |value| *value == min_value)))
    }

    #[inline]
    fn evaluate_either_or<'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = MedRecordValue> + 'a,
        either: &Wrapper<MultipleValuesWithoutIndexOperand<O>>,
        or: &Wrapper<MultipleValuesWithoutIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordValue>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);

        let either_values = either.evaluate_forward(medrecord, Box::new(values_1))?;
        let or_values = or.evaluate_forward(medrecord, Box::new(values_2))?;

        // TODO: Maybe add unique_by
        Ok(Box::new(either_values.chain(or_values)))
    }

    #[inline]
    fn evaluate_exclude<'a>(
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = MedRecordValue> + 'a,
        operand: &Wrapper<MultipleValuesWithoutIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordValue>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);

        let result: Vec<_> = operand
            .evaluate_forward(medrecord, Box::new(values_1))?
            .collect();

        Ok(Box::new(
            values_2.filter(move |value| !result.contains(value)),
        ))
    }
}

#[derive(Debug, Clone)]
pub enum SingleValueWithIndexOperation<O: RootOperand> {
    SingleValueComparisonOperation {
        operand: SingleValueComparisonOperand,
        kind: SingleComparisonKind,
    },
    MultipleValuesComparisonOperation {
        operand: MultipleValuesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOperation {
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
        either: Wrapper<SingleValueWithIndexOperand<O>>,
        or: Wrapper<SingleValueWithIndexOperand<O>>,
    },
    Exclude {
        operand: Wrapper<SingleValueWithIndexOperand<O>>,
    },

    Merge {
        operand: Wrapper<MultipleValuesWithIndexOperand<O>>,
    },
}

impl<O: RootOperand> DeepClone for SingleValueWithIndexOperation<O> {
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
            Self::BinaryArithmeticOperation { operand, kind } => Self::BinaryArithmeticOperation {
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
            Self::Merge { operand } => Self::Merge {
                operand: operand.deep_clone(),
            },
        }
    }
}

impl<O: RootOperand> SingleValueWithIndexOperation<O> {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        value: Option<(&'a O::Index, MedRecordValue)>,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordValue)>>
    where
        O: 'a,
    {
        let Some(value) = value else {
            return Ok(None);
        };

        Ok(match self {
            Self::SingleValueComparisonOperation { operand, kind } => {
                Self::evaluate_single_value_comparison_operation(medrecord, value, operand, kind)?
            }
            Self::MultipleValuesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_values_comparison_operation(
                    medrecord, value, operand, kind,
                )?
            }
            Self::BinaryArithmeticOperation { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, value, operand, kind)?
            }
            Self::UnaryArithmeticOperation { kind } => {
                Some(Self::evaluate_unary_arithmetic_operation(value, kind))
            }
            Self::Slice(range) => Some(Self::evaluate_slice(value, range)),
            Self::IsString => Self::evaluate_is_string(value),
            Self::IsInt => Self::evaluate_is_int(value),
            Self::IsFloat => Self::evaluate_is_float(value),
            Self::IsBool => Self::evaluate_is_bool(value),
            Self::IsDateTime => Self::evaluate_is_datetime(value),
            Self::IsDuration => Self::evaluate_is_duration(value),
            Self::IsNull => Self::evaluate_is_null(value),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, value, either, or)?
            }
            Self::Exclude { operand } => {
                match operand.evaluate_forward(medrecord, Some(value.clone()))? {
                    Some(_) => None,
                    None => Some(value),
                }
            }
            Self::Merge { operand: _ } => {
                unreachable!()
            }
        })
    }

    #[inline]
    fn evaluate_single_value_comparison_operation<'a>(
        medrecord: &MedRecord,
        value: (&'a O::Index, MedRecordValue),
        comparison_operand: &SingleValueComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordValue)>> {
        let comparison_value =
            comparison_operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No value to compare".to_string(),
                ))?;

        let comparison_result = match kind {
            SingleComparisonKind::GreaterThan => value.1 > comparison_value,
            SingleComparisonKind::GreaterThanOrEqualTo => value.1 >= comparison_value,
            SingleComparisonKind::LessThan => value.1 < comparison_value,
            SingleComparisonKind::LessThanOrEqualTo => value.1 <= comparison_value,
            SingleComparisonKind::EqualTo => value.1 == comparison_value,
            SingleComparisonKind::NotEqualTo => value.1 != comparison_value,
            SingleComparisonKind::StartsWith => value.1.starts_with(&comparison_value),
            SingleComparisonKind::EndsWith => value.1.ends_with(&comparison_value),
            SingleComparisonKind::Contains => value.1.contains(&comparison_value),
        };

        Ok(if comparison_result { Some(value) } else { None })
    }

    #[inline]
    fn evaluate_multiple_values_comparison_operation<'a>(
        medrecord: &MedRecord,
        value: (&'a O::Index, MedRecordValue),
        comparison_operand: &MultipleValuesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordValue)>> {
        let comparison_values = comparison_operand.evaluate_backward(medrecord)?;

        let comparison_result = match kind {
            MultipleComparisonKind::IsIn => comparison_values.contains(&value.1),
            MultipleComparisonKind::IsNotIn => !comparison_values.contains(&value.1),
        };

        Ok(if comparison_result { Some(value) } else { None })
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation<'a>(
        medrecord: &MedRecord,
        value: (&'a O::Index, MedRecordValue),
        operand: &SingleValueComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordValue)>> {
        let arithmetic_value =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No value to compare".to_string(),
                ))?;

        Ok(Some(match kind {
            BinaryArithmeticKind::Add => (value.0, value.1.add(arithmetic_value)?),
            BinaryArithmeticKind::Sub => (value.0, value.1.sub(arithmetic_value)?),
            BinaryArithmeticKind::Mul => (value.0, value.1.mul(arithmetic_value)?),
            BinaryArithmeticKind::Div => (value.0, value.1.div(arithmetic_value)?),
            BinaryArithmeticKind::Pow => (value.0, value.1.pow(arithmetic_value)?),
            BinaryArithmeticKind::Mod => (value.0, value.1.r#mod(arithmetic_value)?),
        }))
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation<'a>(
        value: (&'a O::Index, MedRecordValue),
        kind: &UnaryArithmeticKind,
    ) -> (&'a O::Index, MedRecordValue) {
        match kind {
            UnaryArithmeticKind::Round => (value.0, value.1.round()),
            UnaryArithmeticKind::Ceil => (value.0, value.1.ceil()),
            UnaryArithmeticKind::Floor => (value.0, value.1.floor()),
            UnaryArithmeticKind::Abs => (value.0, value.1.abs()),
            UnaryArithmeticKind::Sqrt => (value.0, value.1.sqrt()),
            UnaryArithmeticKind::Trim => (value.0, value.1.trim()),
            UnaryArithmeticKind::TrimStart => (value.0, value.1.trim_start()),
            UnaryArithmeticKind::TrimEnd => (value.0, value.1.trim_end()),
            UnaryArithmeticKind::Lowercase => (value.0, value.1.lowercase()),
            UnaryArithmeticKind::Uppercase => (value.0, value.1.uppercase()),
        }
    }

    #[inline]
    fn evaluate_slice<'a>(
        value: (&'a O::Index, MedRecordValue),
        range: &Range<usize>,
    ) -> (&'a O::Index, MedRecordValue) {
        (value.0, value.1.slice(range.clone()))
    }

    #[inline]
    fn evaluate_is_string(
        value: (&O::Index, MedRecordValue),
    ) -> Option<(&O::Index, MedRecordValue)> {
        match value.1 {
            MedRecordValue::String(_) => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_int(value: (&O::Index, MedRecordValue)) -> Option<(&O::Index, MedRecordValue)> {
        match value.1 {
            MedRecordValue::Int(_) => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_float(
        value: (&O::Index, MedRecordValue),
    ) -> Option<(&O::Index, MedRecordValue)> {
        match value.1 {
            MedRecordValue::Float(_) => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_bool(value: (&O::Index, MedRecordValue)) -> Option<(&O::Index, MedRecordValue)> {
        match value.1 {
            MedRecordValue::Bool(_) => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_datetime(
        value: (&O::Index, MedRecordValue),
    ) -> Option<(&O::Index, MedRecordValue)> {
        match value.1 {
            MedRecordValue::DateTime(_) => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_duration(
        value: (&O::Index, MedRecordValue),
    ) -> Option<(&O::Index, MedRecordValue)> {
        match value.1 {
            MedRecordValue::Duration(_) => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_null(value: (&O::Index, MedRecordValue)) -> Option<(&O::Index, MedRecordValue)> {
        match value.1 {
            MedRecordValue::Null => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_either_or<'a>(
        medrecord: &'a MedRecord,
        value: (&'a O::Index, MedRecordValue),
        either: &Wrapper<SingleValueWithIndexOperand<O>>,
        or: &Wrapper<SingleValueWithIndexOperand<O>>,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordValue)>>
    where
        O: 'a,
    {
        let either_result = either.evaluate_forward(medrecord, Some(value.clone()))?;
        let or_result = or.evaluate_forward(medrecord, Some(value))?;

        match (either_result, or_result) {
            (Some(either_result), _) => Ok(Some(either_result)),
            (None, Some(or_result)) => Ok(Some(or_result)),
            _ => Ok(None),
        }
    }
}

impl<O: RootOperand> SingleValueWithIndexOperation<O> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Option<(&'a O::Index, MedRecordValue)>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<(&'a O::Index, MedRecordValue)>>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::SingleValueComparisonOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, value)| {
                        let Some(value) = value else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_single_value_comparison_operation(
                                medrecord, value, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::MultipleValuesComparisonOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, value)| {
                        let Some(value) = value else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_multiple_values_comparison_operation(
                                medrecord, value, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::BinaryArithmeticOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, value)| {
                        let Some(value) = value else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_binary_arithmetic_operation(
                                medrecord, value, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::UnaryArithmeticOperation { kind } => {
                let kind = kind.clone();

                Box::new(values.map(move |(key, value)| {
                    let Some(value) = value else {
                        return (key, None);
                    };

                    (
                        key,
                        Some(Self::evaluate_unary_arithmetic_operation(value, &kind)),
                    )
                }))
            }
            Self::Slice(range) => {
                let range = range.clone();

                Box::new(values.map(move |(key, value)| {
                    let Some(value) = value else {
                        return (key, None);
                    };

                    (key, Some(Self::evaluate_slice(value, &range)))
                }))
            }
            Self::IsString => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_string(value))
            })),
            Self::IsInt => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_int(value))
            })),
            Self::IsFloat => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_float(value))
            })),
            Self::IsBool => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_bool(value))
            })),
            Self::IsDateTime => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_datetime(value))
            })),
            Self::IsDuration => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_duration(value))
            })),
            Self::IsNull => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_null(value))
            })),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or_grouped(medrecord, values, either, or)?
            }
            Self::Exclude { operand } => {
                Self::evaluate_exclude_grouped(medrecord, values, operand)?
            }
            Self::Merge { operand } => {
                let (values_1, values_2) = Itertools::tee(values);

                let values_1 = values_1.filter_map(|(_, value)| value);

                let values_1: Vec<_> = operand
                    .evaluate_forward(medrecord, Box::new(values_1))?
                    .collect();

                Box::new(values_2.map(move |(key, value)| {
                    let value = value.filter(|value| values_1.contains(value));

                    (key, value)
                }))
            }
        })
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_either_or_grouped<'a>(
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Option<(&'a O::Index, MedRecordValue)>>,
        either: &Wrapper<SingleValueWithIndexOperand<O>>,
        or: &Wrapper<SingleValueWithIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<(&'a O::Index, MedRecordValue)>>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);

        let either_values = either.evaluate_forward_grouped(medrecord, Box::new(values_1))?;
        let mut or_values: Vec<_> = or
            .evaluate_forward_grouped(medrecord, Box::new(values_2))?
            .collect();

        let values = either_values.map(move |(key, either_value)| {
            let value_position = or_values
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let or_value = or_values.remove(value_position).1;

            let value = match (either_value, or_value) {
                (Some(either_result), _) => Some(either_result),
                (None, Some(or_result)) => Some(or_result),
                _ => None,
            };

            (key, value)
        });

        Ok(Box::new(values))
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_exclude_grouped<'a>(
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Option<(&'a O::Index, MedRecordValue)>>,
        operand: &Wrapper<SingleValueWithIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<(&'a O::Index, MedRecordValue)>>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);

        let mut result: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, Box::new(values_1))?
            .collect();

        let values = values_2.map(move |(key, value)| {
            let value_position = result
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let excluded_value = result.remove(value_position).1;

            let value = match excluded_value {
                Some(_) => None,
                None => value,
            };

            (key, value)
        });

        Ok(Box::new(values))
    }
}

#[derive(Debug, Clone)]
pub enum SingleValueWithoutIndexOperation<O: RootOperand> {
    SingleValueComparisonOperation {
        operand: SingleValueComparisonOperand,
        kind: SingleComparisonKind,
    },
    MultipleValuesComparisonOperation {
        operand: MultipleValuesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOperation {
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
        either: Wrapper<SingleValueWithoutIndexOperand<O>>,
        or: Wrapper<SingleValueWithoutIndexOperand<O>>,
    },
    Exclude {
        operand: Wrapper<SingleValueWithoutIndexOperand<O>>,
    },

    Merge {
        operand: Wrapper<MultipleValuesWithoutIndexOperand<O>>,
    },
}

impl<O: RootOperand> DeepClone for SingleValueWithoutIndexOperation<O> {
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
            Self::BinaryArithmeticOperation { operand, kind } => Self::BinaryArithmeticOperation {
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
            Self::Merge { operand } => Self::Merge {
                operand: operand.deep_clone(),
            },
        }
    }
}

impl<O: RootOperand> SingleValueWithoutIndexOperation<O> {
    pub(crate) fn evaluate(
        &self,
        medrecord: &MedRecord,
        value: Option<MedRecordValue>,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        let Some(value) = value else {
            return Ok(None);
        };

        match self {
            Self::SingleValueComparisonOperation { operand, kind } => {
                Self::evaluate_single_value_comparison_operation(medrecord, value, operand, kind)
            }
            Self::MultipleValuesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_values_comparison_operation(medrecord, value, operand, kind)
            }
            Self::BinaryArithmeticOperation { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, value, operand, kind)
            }
            Self::UnaryArithmeticOperation { kind } => {
                Ok(Some(Self::evaluate_unary_arithmetic_operation(value, kind)))
            }
            Self::Slice(range) => Ok(Some(Self::evaluate_slice(value, range))),
            Self::IsString => Ok(Self::evaluate_is_string(value)),
            Self::IsInt => Ok(Self::evaluate_is_int(value)),
            Self::IsFloat => Ok(Self::evaluate_is_float(value)),
            Self::IsBool => Ok(Self::evaluate_is_bool(value)),
            Self::IsDateTime => Ok(Self::evaluate_is_datetime(value)),
            Self::IsDuration => Ok(Self::evaluate_is_duration(value)),
            Self::IsNull => Ok(Self::evaluate_is_null(value)),
            Self::EitherOr { either, or } => Self::evaluate_either_or(medrecord, value, either, or),
            Self::Exclude { operand } => Ok(
                match operand.evaluate_forward(medrecord, Some(value.clone()))? {
                    Some(_) => None,
                    None => Some(value),
                },
            ),
            Self::Merge { operand: _ } => {
                unreachable!()
            }
        }
    }

    #[inline]
    fn evaluate_single_value_comparison_operation(
        medrecord: &MedRecord,
        value: MedRecordValue,
        comparison_operand: &SingleValueComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        let comparison_value =
            comparison_operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No value to compare".to_string(),
                ))?;

        let comparison_result = match kind {
            SingleComparisonKind::GreaterThan => value > comparison_value,
            SingleComparisonKind::GreaterThanOrEqualTo => value >= comparison_value,
            SingleComparisonKind::LessThan => value < comparison_value,
            SingleComparisonKind::LessThanOrEqualTo => value <= comparison_value,
            SingleComparisonKind::EqualTo => value == comparison_value,
            SingleComparisonKind::NotEqualTo => value != comparison_value,
            SingleComparisonKind::StartsWith => value.starts_with(&comparison_value),
            SingleComparisonKind::EndsWith => value.ends_with(&comparison_value),
            SingleComparisonKind::Contains => value.contains(&comparison_value),
        };

        Ok(if comparison_result { Some(value) } else { None })
    }

    #[inline]
    fn evaluate_multiple_values_comparison_operation(
        medrecord: &MedRecord,
        value: MedRecordValue,
        comparison_operand: &MultipleValuesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        let comparison_values = comparison_operand.evaluate_backward(medrecord)?;

        let comparison_result = match kind {
            MultipleComparisonKind::IsIn => comparison_values.contains(&value),
            MultipleComparisonKind::IsNotIn => !comparison_values.contains(&value),
        };

        Ok(if comparison_result { Some(value) } else { None })
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation(
        medrecord: &MedRecord,
        value: MedRecordValue,
        operand: &SingleValueComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        let arithmetic_value =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No value to compare".to_string(),
                ))?;

        Ok(Some(match kind {
            BinaryArithmeticKind::Add => value.add(arithmetic_value)?,
            BinaryArithmeticKind::Sub => value.sub(arithmetic_value)?,
            BinaryArithmeticKind::Mul => value.mul(arithmetic_value)?,
            BinaryArithmeticKind::Div => value.div(arithmetic_value)?,
            BinaryArithmeticKind::Pow => value.pow(arithmetic_value)?,
            BinaryArithmeticKind::Mod => value.r#mod(arithmetic_value)?,
        }))
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation(
        value: MedRecordValue,
        kind: &UnaryArithmeticKind,
    ) -> MedRecordValue {
        match kind {
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
        }
    }

    #[inline]
    fn evaluate_slice(value: MedRecordValue, range: &Range<usize>) -> MedRecordValue {
        value.slice(range.clone())
    }

    #[inline]
    fn evaluate_is_string(value: MedRecordValue) -> Option<MedRecordValue> {
        match value {
            MedRecordValue::String(_) => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_int(value: MedRecordValue) -> Option<MedRecordValue> {
        match value {
            MedRecordValue::Int(_) => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_float(value: MedRecordValue) -> Option<MedRecordValue> {
        match value {
            MedRecordValue::Float(_) => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_bool(value: MedRecordValue) -> Option<MedRecordValue> {
        match value {
            MedRecordValue::Bool(_) => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_datetime(value: MedRecordValue) -> Option<MedRecordValue> {
        match value {
            MedRecordValue::DateTime(_) => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_duration(value: MedRecordValue) -> Option<MedRecordValue> {
        match value {
            MedRecordValue::Duration(_) => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_null(value: MedRecordValue) -> Option<MedRecordValue> {
        match value {
            MedRecordValue::Null => Some(value),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_either_or(
        medrecord: &MedRecord,
        value: MedRecordValue,
        either: &Wrapper<SingleValueWithoutIndexOperand<O>>,
        or: &Wrapper<SingleValueWithoutIndexOperand<O>>,
    ) -> MedRecordResult<Option<MedRecordValue>> {
        let either_result = either.evaluate_forward(medrecord, Some(value.clone()))?;
        let or_result = or.evaluate_forward(medrecord, Some(value))?;

        match (either_result, or_result) {
            (Some(either_result), _) => Ok(Some(either_result)),
            (None, Some(or_result)) => Ok(Some(or_result)),
            _ => Ok(None),
        }
    }
}

impl<O: RootOperand> SingleValueWithoutIndexOperation<O> {
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Option<MedRecordValue>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<MedRecordValue>>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::SingleValueComparisonOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, value)| {
                        let Some(value) = value else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_single_value_comparison_operation(
                                medrecord, value, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::MultipleValuesComparisonOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, value)| {
                        let Some(value) = value else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_multiple_values_comparison_operation(
                                medrecord, value, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::BinaryArithmeticOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, value)| {
                        let Some(value) = value else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_binary_arithmetic_operation(
                                medrecord, value, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::UnaryArithmeticOperation { kind } => {
                let kind = kind.clone();

                Box::new(values.map(move |(key, value)| {
                    let Some(value) = value else {
                        return (key, None);
                    };

                    (
                        key,
                        Some(Self::evaluate_unary_arithmetic_operation(value, &kind)),
                    )
                }))
            }
            Self::Slice(range) => {
                let range = range.clone();

                Box::new(values.map(move |(key, value)| {
                    let Some(value) = value else {
                        return (key, None);
                    };

                    (key, Some(Self::evaluate_slice(value, &range)))
                }))
            }
            Self::IsString => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_string(value))
            })),
            Self::IsInt => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_int(value))
            })),
            Self::IsFloat => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_float(value))
            })),
            Self::IsBool => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_bool(value))
            })),
            Self::IsDateTime => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_datetime(value))
            })),
            Self::IsDuration => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_duration(value))
            })),
            Self::IsNull => Box::new(values.map(move |(key, value)| {
                let Some(value) = value else {
                    return (key, None);
                };

                (key, Self::evaluate_is_null(value))
            })),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or_grouped(medrecord, values, either, or)?
            }
            Self::Exclude { operand } => {
                Self::evaluate_exclude_grouped(medrecord, values, operand)?
            }
            Self::Merge { operand } => {
                let (values_1, values_2) = Itertools::tee(values);

                let values_1 = values_1.filter_map(|(_, value)| value);

                let values_1: Vec<_> = operand
                    .evaluate_forward(medrecord, Box::new(values_1))?
                    .collect();

                Box::new(values_2.map(move |(key, value)| {
                    let value = value.filter(|value| values_1.contains(value));

                    (key, value)
                }))
            }
        })
    }

    #[inline]
    fn evaluate_either_or_grouped<'a>(
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Option<MedRecordValue>>,
        either: &Wrapper<SingleValueWithoutIndexOperand<O>>,
        or: &Wrapper<SingleValueWithoutIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<MedRecordValue>>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);

        let either_values = either.evaluate_forward_grouped(medrecord, Box::new(values_1))?;
        let mut or_value: Vec<_> = or
            .evaluate_forward_grouped(medrecord, Box::new(values_2))?
            .collect();

        let values = either_values.map(move |(key, either_value)| {
            let value_position = or_value
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let or_value = or_value.remove(value_position).1;

            let value = match (either_value, or_value) {
                (Some(either_result), _) => Some(either_result),
                (None, Some(or_result)) => Some(or_result),
                _ => None,
            };

            (key, value)
        });

        Ok(Box::new(values))
    }

    #[inline]
    fn evaluate_exclude_grouped<'a>(
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Option<MedRecordValue>>,
        operand: &Wrapper<SingleValueWithoutIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<MedRecordValue>>>
    where
        O: 'a,
    {
        let (values_1, values_2) = Itertools::tee(values);

        let mut result: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, Box::new(values_1))?
            .collect();

        let values = values_2.map(move |(key, value)| {
            let value_position = result
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let excluded_value = result.remove(value_position).1;

            let value = match excluded_value {
                Some(_) => None,
                None => value,
            };

            (key, value)
        });

        Ok(Box::new(values))
    }
}
