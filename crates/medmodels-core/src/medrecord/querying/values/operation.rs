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
pub enum MultipleValuesOperationWithIndex<O: RootOperand> {
    ValueOperationWithIndex {
        operand: Wrapper<SingleValueWithIndexOperand<O>>,
    },
    ValueOperationWithoutIndex {
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
        either: Wrapper<MultipleValuesWithIndexOperand<O>>,
        or: Wrapper<MultipleValuesWithIndexOperand<O>>,
    },
    Exclude {
        operand: Wrapper<MultipleValuesWithIndexOperand<O>>,
    },
}

impl<O: RootOperand> DeepClone for MultipleValuesOperationWithIndex<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::ValueOperationWithIndex { operand } => Self::ValueOperationWithIndex {
                operand: operand.deep_clone(),
            },
            Self::ValueOperationWithoutIndex { operand } => Self::ValueOperationWithoutIndex {
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

impl<O: RootOperand> MultipleValuesOperationWithIndex<O> {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a O::Index, MedRecordValue)> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>
    where
        O: 'a,
    {
        match self {
            Self::ValueOperationWithIndex { operand } => {
                Self::evaluate_value_operation_with_index(medrecord, values, operand)
            }
            Self::ValueOperationWithoutIndex { operand } => {
                Self::evaluate_value_operation_without_index(medrecord, values, operand)
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
                let (values_1, values_2) = Itertools::tee(values);

                let max_value = Self::get_max(values_1)?;

                let Some(max_value) = max_value else {
                    return Ok(Box::new(std::iter::empty()));
                };

                Ok(Box::new(
                    values_2.filter(move |(_, value)| *value == max_value.1),
                ))
            }
            Self::IsMin => {
                let (values_1, values_2) = Itertools::tee(values);

                let min_value = Self::get_min(values_1)?;

                let Some(min_value) = min_value else {
                    return Ok(Box::new(std::iter::empty()));
                };

                Ok(Box::new(
                    values_2.filter(move |(_, value)| *value == min_value.1),
                ))
            }
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, values, either, or)
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, values, operand),
        }
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
                        "Cannot compare values of data types {} and {}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool(), .is_datetime() or .is_duration()",
                        first_dtype, second_dtype
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
                        "Cannot compare values of data types {} and {}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool(), .is_datetime() or .is_duration()",
                        first_dtype, second_dtype
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
    fn evaluate_value_operation_with_index<'a>(
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
            SingleKindWithIndex::Max => MultipleValuesOperationWithIndex::<O>::get_max(values_1)?,
            SingleKindWithIndex::Min => MultipleValuesOperationWithIndex::<O>::get_min(values_1)?,
            SingleKindWithIndex::Random => {
                MultipleValuesOperationWithIndex::<O>::get_random(values_1)
            }
        };

        Ok(match operand.evaluate_forward(medrecord, value)? {
            Some(_) => Box::new(values_2),
            None => Box::new(std::iter::empty()),
        })
    }

    #[inline]
    fn evaluate_value_operation_without_index<'a>(
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
                MultipleValuesOperationWithoutIndex::<O>::get_max(values_1)?
            }
            SingleKindWithoutIndex::Min => {
                MultipleValuesOperationWithoutIndex::<O>::get_min(values_1)?
            }
            SingleKindWithoutIndex::Mean => {
                MultipleValuesOperationWithoutIndex::<O>::get_mean(values_1)?
            }
            SingleKindWithoutIndex::Median => {
                MultipleValuesOperationWithoutIndex::<O>::get_median(values_1)?
            }
            SingleKindWithoutIndex::Mode => {
                MultipleValuesOperationWithoutIndex::<O>::get_mode(values_1)?
            }
            SingleKindWithoutIndex::Std => {
                MultipleValuesOperationWithoutIndex::<O>::get_std(values_1)?
            }
            SingleKindWithoutIndex::Var => {
                MultipleValuesOperationWithoutIndex::<O>::get_var(values_1)?
            }
            SingleKindWithoutIndex::Count => Some(
                MultipleValuesOperationWithoutIndex::<O>::get_count(values_1),
            ),
            SingleKindWithoutIndex::Sum => {
                MultipleValuesOperationWithoutIndex::<O>::get_sum(values_1)?
            }
            SingleKindWithoutIndex::Random => {
                MultipleValuesOperationWithoutIndex::<O>::get_random(values_1)
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
                        "Failed arithmetic operation {}. Consider narrowing down the values using .is_int() or .is_float()",
                        kind,
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

impl<O: RootOperand> MultipleValuesOperationWithIndex<O> {
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
            Self::ValueOperationWithIndex { operand } => Box::new(
                Self::evaluate_value_operation_with_index_grouped(medrecord, values, operand)?,
            ),
            Self::ValueOperationWithoutIndex { operand } => Box::new(
                Self::evaluate_value_operation_without_index_grouped(medrecord, values, operand)?,
            ),
            Self::SingleValueComparisonOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_single_value_comparison_operation(
                                medrecord, values, operand, kind,
                            )?)
                                as BoxedIterator<'a, (&'a O::Index, MedRecordValue)>,
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
                            )?)
                                as BoxedIterator<'a, (&'a O::Index, MedRecordValue)>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::BinaryArithmeticOpration { operand, kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_binary_arithmetic_operation(
                                medrecord, values, operand, kind,
                            )?)
                                as BoxedIterator<'a, (&'a O::Index, MedRecordValue)>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::UnaryArithmeticOperation { kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_unary_arithmetic_operation(
                                values,
                                kind.clone(),
                            ))
                                as BoxedIterator<'a, (&'a O::Index, MedRecordValue)>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::Slice(range) => Box::new(
                values
                    .map(move |(key, values)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_slice(values, range.clone()))
                                as BoxedIterator<'a, (&'a O::Index, MedRecordValue)>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::IsString => todo!(),
            Self::IsInt => todo!(),
            Self::IsFloat => todo!(),
            Self::IsBool => todo!(),
            Self::IsDateTime => todo!(),
            Self::IsDuration => todo!(),
            Self::IsNull => todo!(),
            Self::IsMax => todo!(),
            Self::IsMin => todo!(),
            Self::EitherOr { either: _, or: _ } => todo!(),
            Self::Exclude { operand: _ } => todo!(),
        })
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_value_operation_with_index_grouped<'a>(
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
                        MultipleValuesOperationWithIndex::<O>::get_max(values)?
                    }
                    SingleKindWithIndex::Min => {
                        MultipleValuesOperationWithIndex::<O>::get_min(values)?
                    }
                    SingleKindWithIndex::Random => {
                        MultipleValuesOperationWithIndex::<O>::get_random(values)
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
            None => (
                key,
                Box::new(std::iter::empty()) as BoxedIterator<'a, (&'a O::Index, MedRecordValue)>,
            ),
        })))
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_value_operation_without_index_grouped<'a>(
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>,
        operand: &Wrapper<SingleValueWithoutIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordValue)>>>
    where
        O: 'a,
    {
        let (values_1, values_2) = tee_grouped_iterator(values);
        let mut values_2 = values_2.collect::<Vec<_>>();

        let kind = &operand.0.read_or_panic().kind;

        let values_1: Vec<_> = values_1
            .map(|(key, values)| {
                let values = values.map(|(_, value)| value);

                let value = match kind {
                    SingleKindWithoutIndex::Max => {
                        MultipleValuesOperationWithoutIndex::<O>::get_max(values)?
                    }
                    SingleKindWithoutIndex::Min => {
                        MultipleValuesOperationWithoutIndex::<O>::get_min(values)?
                    }
                    SingleKindWithoutIndex::Mean => {
                        MultipleValuesOperationWithoutIndex::<O>::get_mean(values)?
                    }
                    SingleKindWithoutIndex::Median => {
                        MultipleValuesOperationWithoutIndex::<O>::get_median(values)?
                    }
                    SingleKindWithoutIndex::Mode => {
                        MultipleValuesOperationWithoutIndex::<O>::get_mode(values)?
                    }
                    SingleKindWithoutIndex::Std => {
                        MultipleValuesOperationWithoutIndex::<O>::get_std(values)?
                    }
                    SingleKindWithoutIndex::Var => {
                        MultipleValuesOperationWithoutIndex::<O>::get_var(values)?
                    }
                    SingleKindWithoutIndex::Count => {
                        Some(MultipleValuesOperationWithoutIndex::<O>::get_count(values))
                    }
                    SingleKindWithoutIndex::Sum => {
                        MultipleValuesOperationWithoutIndex::<O>::get_sum(values)?
                    }
                    SingleKindWithoutIndex::Random => {
                        MultipleValuesOperationWithoutIndex::<O>::get_random(values)
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
            None => (
                key,
                Box::new(std::iter::empty()) as BoxedIterator<'a, (&'a O::Index, MedRecordValue)>,
            ),
        })))
    }
}

#[derive(Debug, Clone)]
pub enum MultipleValuesOperationWithoutIndex<O: RootOperand> {
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
        either: Wrapper<MultipleValuesWithoutIndexOperand<O>>,
        or: Wrapper<MultipleValuesWithoutIndexOperand<O>>,
    },
    Exclude {
        operand: Wrapper<MultipleValuesWithoutIndexOperand<O>>,
    },
}

impl<O: RootOperand> DeepClone for MultipleValuesOperationWithoutIndex<O> {
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

impl<O: RootOperand> MultipleValuesOperationWithoutIndex<O> {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = MedRecordValue> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordValue>>
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
                Ok(Box::new(values.filter(|value| {
                    matches!(value, MedRecordValue::String(_))
                })))
            }
            Self::IsInt => Ok(Box::new(
                values.filter(|value| matches!(value, MedRecordValue::Int(_))),
            )),
            Self::IsFloat => Ok(Box::new(
                values.filter(|value| matches!(value, MedRecordValue::Float(_))),
            )),
            Self::IsBool => Ok(Box::new(
                values.filter(|value| matches!(value, MedRecordValue::Bool(_))),
            )),
            Self::IsDateTime => {
                Ok(Box::new(values.filter(|value| {
                    matches!(value, MedRecordValue::DateTime(_))
                })))
            }
            Self::IsDuration => {
                Ok(Box::new(values.filter(|value| {
                    matches!(value, MedRecordValue::Duration(_))
                })))
            }
            Self::IsNull => Ok(Box::new(
                values.filter(|value| matches!(value, MedRecordValue::Null)),
            )),
            Self::IsMax => {
                let (values_1, values_2) = Itertools::tee(values);

                let max_value = Self::get_max(values_1)?;

                let Some(max_value) = max_value else {
                    return Ok(Box::new(std::iter::empty()));
                };

                Ok(Box::new(values_2.filter(move |value| *value == max_value)))
            }
            Self::IsMin => {
                let (values_1, values_2) = Itertools::tee(values);

                let min_value = Self::get_min(values_1)?;

                let Some(min_value) = min_value else {
                    return Ok(Box::new(std::iter::empty()));
                };

                Ok(Box::new(values_2.filter(move |value| *value == min_value)))
            }
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, values, either, or)
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, values, operand),
        }
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
                        "Cannot compare values of data types {} and {}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool(), .is_datetime() or .is_duration()",
                        first_dtype, second_dtype
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
                        "Cannot compare values of data types {} and {}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool(), .is_datetime() or .is_duration()",
                        first_dtype, second_dtype
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
                    "Cannot add values of data types {} and {}. Consider narrowing down the values using .is_int(), .is_float(), .is_datetime() or .is_duration()",
                    first_dtype, second_dtype
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
                            "Cannot calculate median of mixed data types {} and {}. Consider narrowing down the values using .is_int(), .is_float() , .is_datetime() or .is_duration()",
                            first_data_type, data_type
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
                            "Cannot calculate median of mixed data types {} and {}. Consider narrowing down the values using .is_int(), .is_float(), .is_datetime() or .is_duration()",
                            first_data_type, data_type
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
                            "Cannot calculate median of mixed data types {} and {}. Consider narrowing down the values using .is_int(), .is_float(), .is_datetime() or .is_duration()",
                            first_data_type, data_type
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
                            "Cannot calculate median of mixed data types {} and {}. Consider narrowing down the values using .is_int(), .is_float(), .is_datetime() or .is_duration()",
                            first_data_type, data_type
                        ))),
                    }
                }).collect::<MedRecordResult<_>>()?;
                values.push(value);
                values.sort();

                get_median!(values, Duration)
            }
            _ => Err(MedRecordError::QueryError(format!(
                "Cannot calculate median of data type {}",
                first_data_type
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
                format!("Cannot calculate variance of data type {}. Consider narrowing down the values using .is_int() or .is_float()", data_type),
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
                    "Cannot add values of data types {} and {}. Consider narrowing down the values using .is_string(), .is_int(), .is_float(), .is_bool(), .is_datetime() or .is_duration()",
                    first_dtype, second_dtype
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
                MultipleValuesOperationWithoutIndex::<O>::get_max(values_1)?
            }
            SingleKindWithoutIndex::Min => {
                MultipleValuesOperationWithoutIndex::<O>::get_min(values_1)?
            }
            SingleKindWithoutIndex::Mean => {
                MultipleValuesOperationWithoutIndex::<O>::get_mean(values_1)?
            }
            SingleKindWithoutIndex::Median => {
                MultipleValuesOperationWithoutIndex::<O>::get_median(values_1)?
            }
            SingleKindWithoutIndex::Mode => {
                MultipleValuesOperationWithoutIndex::<O>::get_mode(values_1)?
            }
            SingleKindWithoutIndex::Std => {
                MultipleValuesOperationWithoutIndex::<O>::get_std(values_1)?
            }
            SingleKindWithoutIndex::Var => {
                MultipleValuesOperationWithoutIndex::<O>::get_var(values_1)?
            }
            SingleKindWithoutIndex::Count => Some(
                MultipleValuesOperationWithoutIndex::<O>::get_count(values_1),
            ),
            SingleKindWithoutIndex::Sum => {
                MultipleValuesOperationWithoutIndex::<O>::get_sum(values_1)?
            }
            SingleKindWithoutIndex::Random => {
                MultipleValuesOperationWithoutIndex::<O>::get_random(values_1)
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
                        "Failed arithmetic operation {}. Consider narrowing down the values using .is_int() or .is_float()",
                        kind,
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

        let _either_values = either.evaluate_forward(medrecord, Box::new(values_1))?;
        let _or_values = or.evaluate_forward(medrecord, Box::new(values_2))?;

        todo!()
    }

    #[inline]
    fn evaluate_exclude<'a>(
        _medrecord: &'a MedRecord,
        values: impl Iterator<Item = MedRecordValue> + 'a,
        _operand: &Wrapper<MultipleValuesWithoutIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordValue>>
    where
        O: 'a,
    {
        let (_values_1, _values_2) = Itertools::tee(values);

        todo!()
    }
}

impl<O: RootOperand> MultipleValuesOperationWithoutIndex<O> {
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, BoxedIterator<'a, MedRecordValue>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, MedRecordValue>>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::ValueOperation { operand } => Box::new(Self::evaluate_value_operation_grouped(
                medrecord, values, operand,
            )?),
            Self::SingleValueComparisonOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_single_value_comparison_operation(
                                medrecord, values, operand, kind,
                            )?) as BoxedIterator<'a, MedRecordValue>,
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
                            )?) as BoxedIterator<'a, MedRecordValue>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::BinaryArithmeticOpration { operand, kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_binary_arithmetic_operation(
                                medrecord, values, operand, kind,
                            )?) as BoxedIterator<'a, MedRecordValue>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::UnaryArithmeticOperation { kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_unary_arithmetic_operation(
                                values,
                                kind.clone(),
                            )) as BoxedIterator<'a, MedRecordValue>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::Slice(range) => Box::new(
                values
                    .map(move |(key, values)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_slice(values, range.clone()))
                                as BoxedIterator<'a, MedRecordValue>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::IsString => todo!(),
            Self::IsInt => todo!(),
            Self::IsFloat => todo!(),
            Self::IsBool => todo!(),
            Self::IsDateTime => todo!(),
            Self::IsDuration => todo!(),
            Self::IsNull => todo!(),
            Self::IsMax => todo!(),
            Self::IsMin => todo!(),
            Self::EitherOr { either: _, or: _ } => todo!(),
            Self::Exclude { operand: _ } => todo!(),
        })
    }

    #[inline]
    fn evaluate_value_operation_grouped<'a>(
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, BoxedIterator<'a, MedRecordValue>>,
        operand: &Wrapper<SingleValueWithoutIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, MedRecordValue>>>
    where
        O: 'a,
    {
        let (values_1, values_2) = tee_grouped_iterator(values);
        let mut values_2 = values_2.collect::<Vec<_>>();

        let kind = &operand.0.read_or_panic().kind;

        let values_1: Vec<_> = values_1
            .map(|(key, values)| {
                let value = match kind {
                    SingleKindWithoutIndex::Max => {
                        MultipleValuesOperationWithoutIndex::<O>::get_max(values)?
                    }
                    SingleKindWithoutIndex::Min => {
                        MultipleValuesOperationWithoutIndex::<O>::get_min(values)?
                    }
                    SingleKindWithoutIndex::Mean => {
                        MultipleValuesOperationWithoutIndex::<O>::get_mean(values)?
                    }
                    SingleKindWithoutIndex::Median => {
                        MultipleValuesOperationWithoutIndex::<O>::get_median(values)?
                    }
                    SingleKindWithoutIndex::Mode => {
                        MultipleValuesOperationWithoutIndex::<O>::get_mode(values)?
                    }
                    SingleKindWithoutIndex::Std => {
                        MultipleValuesOperationWithoutIndex::<O>::get_std(values)?
                    }
                    SingleKindWithoutIndex::Var => {
                        MultipleValuesOperationWithoutIndex::<O>::get_var(values)?
                    }
                    SingleKindWithoutIndex::Count => {
                        Some(MultipleValuesOperationWithoutIndex::<O>::get_count(values))
                    }
                    SingleKindWithoutIndex::Sum => {
                        MultipleValuesOperationWithoutIndex::<O>::get_sum(values)?
                    }
                    SingleKindWithoutIndex::Random => {
                        MultipleValuesOperationWithoutIndex::<O>::get_random(values)
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
            None => (
                key,
                Box::new(std::iter::empty()) as BoxedIterator<'a, MedRecordValue>,
            ),
        })))
    }
}

#[derive(Debug, Clone)]
pub enum SingleValueOperationWithIndex<O: RootOperand> {
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

impl<O: RootOperand> DeepClone for SingleValueOperationWithIndex<O> {
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
            Self::Merge { operand } => Self::Merge {
                operand: operand.deep_clone(),
            },
        }
    }
}

impl<O: RootOperand> SingleValueOperationWithIndex<O> {
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
            })),
            Self::Slice(range) => Ok(Some((value.0, value.1.slice(range.clone())))),
            Self::IsString => Ok(match value.1 {
                MedRecordValue::String(_) => Some(value),
                _ => None,
            }),
            Self::IsInt => Ok(match value.1 {
                MedRecordValue::Int(_) => Some(value),
                _ => None,
            }),
            Self::IsFloat => Ok(match value.1 {
                MedRecordValue::Float(_) => Some(value),
                _ => None,
            }),
            Self::IsBool => Ok(match value.1 {
                MedRecordValue::Bool(_) => Some(value),
                _ => None,
            }),
            Self::IsDateTime => Ok(match value.1 {
                MedRecordValue::DateTime(_) => Some(value),
                _ => None,
            }),
            Self::IsDuration => Ok(match value.1 {
                MedRecordValue::Duration(_) => Some(value),
                _ => None,
            }),
            Self::IsNull => Ok(match value.1 {
                MedRecordValue::Null => Some(value),
                _ => None,
            }),
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

impl<O: RootOperand> SingleValueOperationWithIndex<O> {
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
                    .map(move |(key, values)| {
                        let Some(values) = values else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_single_value_comparison_operation(
                                medrecord, values, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::MultipleValuesComparisonOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        let Some(values) = values else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_multiple_values_comparison_operation(
                                medrecord, values, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::BinaryArithmeticOpration { operand, kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        let Some(values) = values else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_binary_arithmetic_operation(
                                medrecord, values, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::UnaryArithmeticOperation { kind: _ } => todo!(),
            Self::Slice(_range) => todo!(),
            Self::IsString => todo!(),
            Self::IsInt => todo!(),
            Self::IsFloat => todo!(),
            Self::IsBool => todo!(),
            Self::IsDateTime => todo!(),
            Self::IsDuration => todo!(),
            Self::IsNull => todo!(),
            Self::EitherOr { either: _, or: _ } => todo!(),
            Self::Exclude { operand: _ } => todo!(),
            Self::Merge { operand } => {
                let (values_1, values_2) = Itertools::tee(values);

                let values_1 = values_1.filter_map(|(_, value)| value);

                let values_1 = operand
                    .evaluate_forward(medrecord, Box::new(values_1))?
                    .collect::<Vec<_>>();

                Box::new(values_2.map(move |(key, value)| {
                    let value = value.filter(|value| values_1.contains(value));

                    (key, value)
                }))
            }
        })
    }
}

#[derive(Debug, Clone)]
pub enum SingleValueOperationWithoutIndex<O: RootOperand> {
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

impl<O: RootOperand> DeepClone for SingleValueOperationWithoutIndex<O> {
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
            Self::Merge { operand } => Self::Merge {
                operand: operand.deep_clone(),
            },
        }
    }
}

impl<O: RootOperand> SingleValueOperationWithoutIndex<O> {
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
            Self::BinaryArithmeticOpration { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, value, operand, kind)
            }
            Self::UnaryArithmeticOperation { kind } => Ok(Some(match kind {
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
            })),
            Self::Slice(range) => Ok(Some(value.slice(range.clone()))),
            Self::IsString => Ok(match value {
                MedRecordValue::String(_) => Some(value),
                _ => None,
            }),
            Self::IsInt => Ok(match value {
                MedRecordValue::Int(_) => Some(value),
                _ => None,
            }),
            Self::IsFloat => Ok(match value {
                MedRecordValue::Float(_) => Some(value),
                _ => None,
            }),
            Self::IsBool => Ok(match value {
                MedRecordValue::Bool(_) => Some(value),
                _ => None,
            }),
            Self::IsDateTime => Ok(match value {
                MedRecordValue::DateTime(_) => Some(value),
                _ => None,
            }),
            Self::IsDuration => Ok(match value {
                MedRecordValue::Duration(_) => Some(value),
                _ => None,
            }),
            Self::IsNull => Ok(match value {
                MedRecordValue::Null => Some(value),
                _ => None,
            }),
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

impl<O: RootOperand> SingleValueOperationWithoutIndex<O> {
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Option<MedRecordValue>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<MedRecordValue>>> {
        Ok(match self {
            Self::SingleValueComparisonOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        let Some(values) = values else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_single_value_comparison_operation(
                                medrecord, values, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::MultipleValuesComparisonOperation { operand, kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        let Some(values) = values else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_multiple_values_comparison_operation(
                                medrecord, values, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::BinaryArithmeticOpration { operand, kind } => Box::new(
                values
                    .map(move |(key, values)| {
                        let Some(values) = values else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_binary_arithmetic_operation(
                                medrecord, values, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::UnaryArithmeticOperation { kind: _ } => todo!(),
            Self::Slice(_range) => todo!(),
            Self::IsString => todo!(),
            Self::IsInt => todo!(),
            Self::IsFloat => todo!(),
            Self::IsBool => todo!(),
            Self::IsDateTime => todo!(),
            Self::IsDuration => todo!(),
            Self::IsNull => todo!(),
            Self::EitherOr { either: _, or: _ } => todo!(),
            Self::Exclude { operand: _ } => todo!(),
            Self::Merge { operand: _ } => {
                let (_values_1, _values_2) = Itertools::tee(values);

                todo!()

                // let values_1 = operand
                //     .evaluate_forward(medrecord, Box::new(values_1))?
                //     .map(|(index, _)| index)
                //     .collect::<Vec<_>>();

                // Box::new(values_2.map(move |(key, value)| {
                //     (
                //         key,
                //         match value {
                //             Some(value) => match value {
                //                 OptionalIndexWrapper::WithIndex((index, value)) => {
                //                     if values_1.contains(&index) {
                //                         Some(OptionalIndexWrapper::WithIndex((index, value)))
                //                     } else {
                //                         None
                //                     }
                //                 }
                //                 OptionalIndexWrapper::WithoutIndex(_) => todo!(),
                //             },
                //             None => None,
                //         },
                //     )
                // }))
            }
        })
    }
}
