use super::{
    operand::{
        MultipleAttributesComparisonOperand, MultipleAttributesWithIndexOperand,
        SingleAttributeComparisonOperand, SingleAttributeWithIndexOperand,
    },
    AttributesTreeOperand, BinaryArithmeticKind, GetAttributes, MultipleComparisonKind,
    SingleComparisonKind, SingleKindWithIndex, UnaryArithmeticKind,
};
use crate::{
    errors::{MedRecordError, MedRecordResult},
    medrecord::{
        datatypes::{
            Abs, Contains, DataType, EndsWith, Lowercase, Mod, Pow, Slice, StartsWith, Trim,
            TrimEnd, TrimStart, Uppercase,
        },
        querying::{
            attributes::{
                operand::SingleAttributeWithoutIndexOperand, MultipleAttributesWithIndexContext,
                MultipleAttributesWithoutIndexOperand, MultipleKind, SingleKindWithoutIndex,
            },
            tee_grouped_iterator,
            values::MultipleValuesWithIndexOperand,
            BoxedIterator, DeepClone, EvaluateForward, EvaluateForwardGrouped, GroupedIterator,
            ReadWriteOrPanic, RootOperand,
        },
        MedRecordAttribute, MedRecordValue, Wrapper,
    },
    MedRecord,
};
use itertools::Itertools;
use medmodels_utils::aliases::{MrHashMap, MrHashSet};
use rand::{rng, seq::IteratorRandom};
use std::{
    cmp::Ordering,
    collections::HashMap,
    ops::{Add, Mul, Range, Sub},
};

#[derive(Debug, Clone)]
pub enum AttributesTreeOperation<O: RootOperand> {
    AttributesOperation {
        operand: Wrapper<MultipleAttributesWithIndexOperand<O>>,
    },
    SingleAttributeComparisonOperation {
        operand: SingleAttributeComparisonOperand,
        kind: SingleComparisonKind,
    },
    MultipleAttributesComparisonOperation {
        operand: MultipleAttributesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOperation {
        operand: SingleAttributeComparisonOperand,
        kind: BinaryArithmeticKind,
    },
    UnaryArithmeticOperation {
        kind: UnaryArithmeticKind,
    },

    Slice(Range<usize>),

    IsString,
    IsInt,

    IsMax,
    IsMin,

    EitherOr {
        either: Wrapper<AttributesTreeOperand<O>>,
        or: Wrapper<AttributesTreeOperand<O>>,
    },
    Exclude {
        operand: Wrapper<AttributesTreeOperand<O>>,
    },

    Merge {
        operand: Wrapper<AttributesTreeOperand<O>>,
    },
}

impl<O: RootOperand> DeepClone for AttributesTreeOperation<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::AttributesOperation { operand } => Self::AttributesOperation {
                operand: operand.deep_clone(),
            },
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::SingleAttributeComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::MultipleAttributesComparisonOperation {
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

impl<O: RootOperand> AttributesTreeOperation<O> {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::AttributesOperation { operand } => Box::new(Self::evaluate_attributes_operation(
                medrecord, attributes, operand,
            )?),
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::evaluate_single_attribute_comparison_operation(
                    medrecord, attributes, operand, kind,
                )?
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_attributes_comparison_operation(
                    medrecord, attributes, operand, kind,
                )?
            }
            Self::BinaryArithmeticOperation { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, attributes, operand, kind)?
            }
            Self::UnaryArithmeticOperation { kind } => Box::new(
                Self::evaluate_unary_arithmetic_operation(attributes, kind.clone()),
            ),
            Self::Slice(range) => Box::new(Self::evaluate_slice(attributes, range.clone())),
            Self::IsString => Box::new(Self::evaluate_is_string(attributes)),
            Self::IsInt => Box::new(Self::evaluate_is_int(attributes)),
            Self::IsMax => Box::new(Self::evaluate_is_max(attributes)?),
            Self::IsMin => Box::new(Self::evaluate_is_min(attributes)?),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, attributes, either, or)?
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, attributes, operand)?,
            Self::Merge { operand: _ } => unreachable!(),
        })
    }

    #[inline]
    pub(crate) fn get_max<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        Ok(attributes.map(|(index, attributes)| {
            let mut attributes = attributes.into_iter();

            let first_attribute = attributes.next().ok_or(MedRecordError::QueryError(
                "No attributes to compare".to_string(),
            ))?;

            let attribute = attributes.try_fold(first_attribute, |max, attribute| {
                match attribute.partial_cmp(&max) {
                    Some(Ordering::Greater) => Ok(attribute),
                    None => {
                        let first_dtype = DataType::from(attribute);
                        let second_dtype = DataType::from(max);

                        Err(MedRecordError::QueryError(format!(
                            "Cannot compare attributes of data types {first_dtype} and {second_dtype}. Consider narrowing down the attributes using .is_string() or .is_int()"
                        )))
                    }
                    _ => Ok(max),
                }
            })?;

            Ok((index, attribute))
        }).collect::<MedRecordResult<Vec<_>>>()?.into_iter())
    }

    #[inline]
    pub(crate) fn get_min<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        Ok(attributes.map(|(index, attributes)| {
            let mut attributes = attributes.into_iter();

            let first_attribute = attributes.next().ok_or(MedRecordError::QueryError(
                "No attributes to compare".to_string(),
            ))?;

            let attribute = attributes.try_fold(first_attribute, |max, attribute| {
                match attribute.partial_cmp(&max) {
                    Some(Ordering::Less) => Ok(attribute),
                    None => {
                        let first_dtype = DataType::from(attribute);
                        let second_dtype = DataType::from(max);

                        Err(MedRecordError::QueryError(format!(
                            "Cannot compare attributes of data types {first_dtype} and {second_dtype}. Consider narrowing down the attributes using .is_string() or .is_int()"
                        )))
                    }
                    _ => Ok(max),
                }
            })?;

            Ok((index, attribute))
        }).collect::<MedRecordResult<Vec<_>>>()?.into_iter())
    }

    #[inline]
    pub(crate) fn get_count<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        Ok(attributes
            .map(|(index, attribute)| (index, MedRecordAttribute::Int(attribute.len() as i64))))
    }

    #[inline]
    pub(crate) fn get_sum<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        Ok(attributes.map(|(index, attributes)| {
            let mut attributes = attributes.into_iter();

            let first_attribute = attributes.next().ok_or(MedRecordError::QueryError(
                "No attributes to compare".to_string(),
            ))?;

            let attribute = attributes.try_fold(first_attribute, |sum, attribute| {
                let first_dtype = DataType::from(&sum);
                let second_dtype = DataType::from(&attribute);

                sum.add(attribute).map_err(|_| {
                    MedRecordError::QueryError(format!(
                        "Cannot add attributes of data types {first_dtype} and {second_dtype}. Consider narrowing down the attributes using .is_string() or .is_int()"
                    ))
                })
            })?;

            Ok((index, attribute))
        }).collect::<MedRecordResult<Vec<_>>>()?.into_iter())
    }

    #[inline]
    pub(crate) fn get_random<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        Ok(attributes
            .map(|(index, attributes)| {
                let first_attribute =
                    attributes
                        .into_iter()
                        .choose(&mut rng())
                        .ok_or(MedRecordError::QueryError(
                            "No attributes to compare".to_string(),
                        ))?;

                Ok((index, first_attribute))
            })
            .collect::<MedRecordResult<Vec<_>>>()?
            .into_iter())
    }

    #[inline]
    fn evaluate_attributes_operation<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)> + 'a,
        operand: &Wrapper<MultipleAttributesWithIndexOperand<O>>,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)> + 'a>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let MultipleAttributesWithIndexContext::AttributesTree {
            operand: _,
            ref kind,
        } = operand.0.read_or_panic().context
        else {
            unreachable!()
        };

        let multiple_operand_attributes: BoxedIterator<_> = match kind {
            MultipleKind::Max => Box::new(AttributesTreeOperation::<O>::get_max(attributes_1)?),
            MultipleKind::Min => Box::new(AttributesTreeOperation::<O>::get_min(attributes_1)?),
            MultipleKind::Count => Box::new(AttributesTreeOperation::<O>::get_count(attributes_1)?),
            MultipleKind::Sum => Box::new(AttributesTreeOperation::<O>::get_sum(attributes_1)?),
            MultipleKind::Random => {
                Box::new(AttributesTreeOperation::<O>::get_random(attributes_1)?)
            }
        };

        let result = operand.evaluate_forward(medrecord, multiple_operand_attributes)?;

        let mut attributes: MrHashMap<_, _> = attributes_2.into_iter().collect();

        Ok(result
            .map(move |(index, _)| (index, attributes.remove(&index).expect("Index must exist"))))
    }

    #[inline]
    fn evaluate_single_attribute_comparison_operation<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)> + 'a,
        comparison_operand: &SingleAttributeComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>> {
        let comparison_attribute =
            comparison_operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No attribute to compare".to_string(),
                ))?;

        match kind {
            SingleComparisonKind::GreaterThan => {
                Ok(Box::new(attributes.map(move |(index, attributes)| {
                    (
                        index,
                        attributes
                            .into_iter()
                            .filter(|attribute| attribute > &comparison_attribute)
                            .collect(),
                    )
                })))
            }
            SingleComparisonKind::GreaterThanOrEqualTo => {
                Ok(Box::new(attributes.map(move |(index, attributes)| {
                    (
                        index,
                        attributes
                            .into_iter()
                            .filter(|attribute| attribute >= &comparison_attribute)
                            .collect(),
                    )
                })))
            }
            SingleComparisonKind::LessThan => {
                Ok(Box::new(attributes.map(move |(index, attributes)| {
                    (
                        index,
                        attributes
                            .into_iter()
                            .filter(|attribute| attribute < &comparison_attribute)
                            .collect(),
                    )
                })))
            }
            SingleComparisonKind::LessThanOrEqualTo => {
                Ok(Box::new(attributes.map(move |(index, attributes)| {
                    (
                        index,
                        attributes
                            .into_iter()
                            .filter(|attribute| attribute <= &comparison_attribute)
                            .collect(),
                    )
                })))
            }
            SingleComparisonKind::EqualTo => {
                Ok(Box::new(attributes.map(move |(index, attributes)| {
                    (
                        index,
                        attributes
                            .into_iter()
                            .filter(|attribute| attribute == &comparison_attribute)
                            .collect(),
                    )
                })))
            }
            SingleComparisonKind::NotEqualTo => {
                Ok(Box::new(attributes.map(move |(index, attributes)| {
                    (
                        index,
                        attributes
                            .into_iter()
                            .filter(|attribute| attribute != &comparison_attribute)
                            .collect(),
                    )
                })))
            }
            SingleComparisonKind::StartsWith => {
                Ok(Box::new(attributes.map(move |(index, attributes)| {
                    (
                        index,
                        attributes
                            .into_iter()
                            .filter(|attribute| attribute.starts_with(&comparison_attribute))
                            .collect(),
                    )
                })))
            }
            SingleComparisonKind::EndsWith => {
                Ok(Box::new(attributes.map(move |(index, attributes)| {
                    (
                        index,
                        attributes
                            .into_iter()
                            .filter(|attribute| attribute.ends_with(&comparison_attribute))
                            .collect(),
                    )
                })))
            }
            SingleComparisonKind::Contains => {
                Ok(Box::new(attributes.map(move |(index, attributes)| {
                    (
                        index,
                        attributes
                            .into_iter()
                            .filter(|attribute| attribute.contains(&comparison_attribute))
                            .collect(),
                    )
                })))
            }
        }
    }

    #[inline]
    fn evaluate_multiple_attributes_comparison_operation<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)> + 'a,
        comparison_operand: &MultipleAttributesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>> {
        let comparison_attributes = comparison_operand.evaluate_backward(medrecord)?;

        match kind {
            MultipleComparisonKind::IsIn => {
                Ok(Box::new(attributes.map(move |(index, attributes)| {
                    (
                        index,
                        attributes
                            .into_iter()
                            .filter(|attribute| comparison_attributes.contains(attribute))
                            .collect(),
                    )
                })))
            }
            MultipleComparisonKind::IsNotIn => {
                Ok(Box::new(attributes.map(move |(index, attributes)| {
                    (
                        index,
                        attributes
                            .into_iter()
                            .filter(|attribute| !comparison_attributes.contains(attribute))
                            .collect(),
                    )
                })))
            }
        }
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation<'a, I: 'a>(
        medrecord: &MedRecord,
        attributes: impl Iterator<Item = (I, Vec<MedRecordAttribute>)>,
        operand: &SingleAttributeComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<BoxedIterator<'a, (I, Vec<MedRecordAttribute>)>> {
        let arithmetic_attribute =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No attribute to compare".to_string(),
                ))?;

        let attributes: Box<dyn Iterator<Item = MedRecordResult<(I, Vec<MedRecordAttribute>)>>> =
            match kind {
                BinaryArithmeticKind::Add => {
                    Box::new(attributes.map(move |(index, attributes)| {
                        Ok((
                            index,
                            attributes
                                .into_iter()
                                .map(|attribute| attribute.add(arithmetic_attribute.clone()))
                                .collect::<MedRecordResult<Vec<_>>>()?,
                        ))
                    }))
                }
                BinaryArithmeticKind::Sub => {
                    Box::new(attributes.map(move |(index, attributes)| {
                        Ok((
                            index,
                            attributes
                                .into_iter()
                                .map(|attribute| attribute.sub(arithmetic_attribute.clone()))
                                .collect::<MedRecordResult<Vec<_>>>()?,
                        ))
                    }))
                }
                BinaryArithmeticKind::Mul => {
                    Box::new(attributes.map(move |(index, attributes)| {
                        Ok((
                            index,
                            attributes
                                .into_iter()
                                .map(|attribute| attribute.mul(arithmetic_attribute.clone()))
                                .collect::<MedRecordResult<Vec<_>>>()?,
                        ))
                    }))
                }
                BinaryArithmeticKind::Pow => {
                    Box::new(attributes.map(move |(index, attributes)| {
                        Ok((
                            index,
                            attributes
                                .into_iter()
                                .map(|attribute| attribute.pow(arithmetic_attribute.clone()))
                                .collect::<MedRecordResult<Vec<_>>>()?,
                        ))
                    }))
                }
                BinaryArithmeticKind::Mod => {
                    Box::new(attributes.map(move |(index, attributes)| {
                        Ok((
                            index,
                            attributes
                                .into_iter()
                                .map(|attribute| attribute.r#mod(arithmetic_attribute.clone()))
                                .collect::<MedRecordResult<Vec<_>>>()?,
                        ))
                    }))
                }
            };

        Ok(Box::new(
            attributes.collect::<MedRecordResult<Vec<_>>>()?.into_iter(),
        ))
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>,
        kind: UnaryArithmeticKind,
    ) -> impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>
    where
        O: 'a,
    {
        attributes.map(move |(index, attributes)| {
            (
                index,
                attributes
                    .into_iter()
                    .map(|attribute| match kind {
                        UnaryArithmeticKind::Abs => attribute.abs(),
                        UnaryArithmeticKind::Trim => attribute.trim(),
                        UnaryArithmeticKind::TrimStart => attribute.trim_start(),
                        UnaryArithmeticKind::TrimEnd => attribute.trim_end(),
                        UnaryArithmeticKind::Lowercase => attribute.lowercase(),
                        UnaryArithmeticKind::Uppercase => attribute.uppercase(),
                    })
                    .collect(),
            )
        })
    }

    #[inline]
    fn evaluate_slice<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>,
        range: Range<usize>,
    ) -> impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>
    where
        O: 'a,
    {
        attributes.map(move |(index, attributes)| {
            (
                index,
                attributes
                    .into_iter()
                    .map(|attribute| attribute.slice(range.clone()))
                    .collect(),
            )
        })
    }

    #[inline]
    fn evaluate_is_string<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>,
    ) -> impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>
    where
        O: 'a,
    {
        attributes.map(|(index, attribute)| {
            (
                index,
                attribute
                    .into_iter()
                    .filter(|attribute| matches!(attribute, MedRecordAttribute::String(_)))
                    .collect(),
            )
        })
    }

    #[inline]
    fn evaluate_is_int<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>,
    ) -> impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>
    where
        O: 'a,
    {
        attributes.map(|(index, attribute)| {
            (
                index,
                attribute
                    .into_iter()
                    .filter(|attribute| matches!(attribute, MedRecordAttribute::Int(_)))
                    .collect(),
            )
        })
    }

    #[inline]
    fn evaluate_is_max<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let max_attributes: MrHashMap<_, _> = Self::get_max(attributes_1)?.collect();

        Ok(Box::new(attributes_2.map(move |(index, attributes)| {
            let max_attribute = max_attributes.get(&index).expect("Index must exist");

            (
                index,
                attributes
                    .into_iter()
                    .filter(|attribute| attribute == max_attribute)
                    .collect(),
            )
        })))
    }

    #[inline]
    fn evaluate_is_min<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let min_attributes: MrHashMap<_, _> = Self::get_min(attributes_1)?.collect();

        Ok(Box::new(attributes_2.map(move |(index, attributes)| {
            let min_attribute = min_attributes.get(&index).expect("Index must exist");

            (
                index,
                attributes
                    .into_iter()
                    .filter(|attribute| attribute == min_attribute)
                    .collect(),
            )
        })))
    }

    #[inline]
    fn evaluate_either_or<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)> + 'a,
        either: &Wrapper<AttributesTreeOperand<O>>,
        or: &Wrapper<AttributesTreeOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let either_attributes = either.evaluate_forward(medrecord, Box::new(attributes_1))?;
        let or_attributes = or.evaluate_forward(medrecord, Box::new(attributes_2))?;

        Ok(Box::new(
            either_attributes
                .chain(or_attributes)
                .into_group_map_by(|(k, _)| *k)
                .into_iter()
                .map(|(idx, group)| {
                    let attrs = group.into_iter().flat_map(|(_, v)| v).unique().collect();
                    (idx, attrs)
                }),
        ))
    }

    #[inline]
    fn evaluate_exclude<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, Vec<MedRecordAttribute>)> + 'a,
        operand: &Wrapper<AttributesTreeOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let mut result: MrHashMap<_, _> = operand
            .evaluate_forward(medrecord, Box::new(attributes_1))?
            .collect();

        Ok(Box::new(attributes_2.map(move |(index, attributes)| {
            let entry = result.remove(&index).unwrap_or(Vec::new());

            (
                index,
                attributes
                    .into_iter()
                    .filter(|attr| !entry.contains(attr))
                    .collect(),
            )
        })))
    }
}

impl<O: RootOperand> AttributesTreeOperation<O> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>>,
    ) -> MedRecordResult<
        GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>>,
    >
    where
        O: 'a,
    {
        Ok(match self {
            Self::AttributesOperation { operand } => {
                Self::evaluate_attributes_operation_grouped(self, medrecord, attributes, operand)?
            }
            Self::SingleAttributeComparisonOperation { operand, kind } => Box::new(
                attributes
                    .map(move |(key, attributes)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_single_attribute_comparison_operation(
                                medrecord, attributes, operand, kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::MultipleAttributesComparisonOperation { operand, kind } => Box::new(
                attributes
                    .map(move |(key, attributes)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_multiple_attributes_comparison_operation(
                                medrecord, attributes, operand, kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::BinaryArithmeticOperation { operand, kind } => Box::new(
                attributes
                    .map(move |(key, attributes)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_binary_arithmetic_operation(
                                medrecord, attributes, operand, kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::UnaryArithmeticOperation { kind } => {
                let kind = kind.clone();

                Box::new(attributes.map(move |(key, attributes)| {
                    (
                        key,
                        Box::new(Self::evaluate_unary_arithmetic_operation(
                            attributes,
                            kind.clone(),
                        )) as BoxedIterator<_>,
                    )
                }))
            }
            Self::Slice(range) => {
                let range = range.clone();

                Box::new(attributes.map(move |(key, attributes)| {
                    (
                        key,
                        Box::new(Self::evaluate_slice(attributes, range.clone()))
                            as BoxedIterator<_>,
                    )
                }))
            }
            Self::IsString => Box::new(attributes.map(move |(key, attributes)| {
                (
                    key,
                    Box::new(Self::evaluate_is_string(attributes)) as BoxedIterator<_>,
                )
            })),
            Self::IsInt => Box::new(attributes.map(move |(key, attributes)| {
                (
                    key,
                    Box::new(Self::evaluate_is_int(attributes)) as BoxedIterator<_>,
                )
            })),
            Self::IsMax => Box::new(
                attributes
                    .map(move |(key, attributes)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_is_max(attributes)?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::IsMin => Box::new(
                attributes
                    .map(move |(key, attributes)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_is_min(attributes)?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or_grouped(medrecord, attributes, either, or)?
            }
            Self::Exclude { operand } => {
                Self::evaluate_exclude_grouped(medrecord, attributes, operand)?
            }
            Self::Merge { operand } => {
                let (attributes_1, attributes_2) = tee_grouped_iterator(attributes);

                let attributes_1 = attributes_1.flat_map(|(_, attribute)| attribute);

                let attributes_1: MrHashSet<_> = operand
                    .evaluate_forward(medrecord, Box::new(attributes_1))?
                    .collect();

                Box::new(attributes_2.map(move |(key, attributes)| {
                    let attributes: Vec<_> = attributes
                        .filter(|attributes| attributes_1.contains(attributes))
                        .collect();

                    let attributes: BoxedIterator<_> = Box::new(attributes.into_iter());

                    (key, attributes)
                }))
            }
        })
    }

    #[allow(clippy::type_complexity)]
    pub(crate) fn evaluate_attributes_operation_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>>,
        operand: &Wrapper<MultipleAttributesWithIndexOperand<O>>,
    ) -> MedRecordResult<
        GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>>,
    >
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = tee_grouped_iterator(attributes);
        let mut attributes_2: Vec<_> = attributes_2.collect();

        let MultipleAttributesWithIndexContext::AttributesTree {
            operand: _,
            ref kind,
        } = operand.0.read_or_panic().context
        else {
            unreachable!()
        };

        let attributes_1: Vec<_> = attributes_1
            .map(|(key, attributes)| {
                let attributes: BoxedIterator<_> = match kind {
                    MultipleKind::Max => {
                        Box::new(AttributesTreeOperation::<O>::get_max(attributes)?)
                    }
                    MultipleKind::Min => {
                        Box::new(AttributesTreeOperation::<O>::get_min(attributes)?)
                    }
                    MultipleKind::Count => {
                        Box::new(AttributesTreeOperation::<O>::get_count(attributes)?)
                    }
                    MultipleKind::Sum => {
                        Box::new(AttributesTreeOperation::<O>::get_sum(attributes)?)
                    }
                    MultipleKind::Random => {
                        Box::new(AttributesTreeOperation::<O>::get_random(attributes)?)
                    }
                };

                Ok((key, attributes))
            })
            .collect::<MedRecordResult<_>>()?;

        let attibutes_1 =
            operand.evaluate_forward_grouped(medrecord, Box::new(attributes_1.into_iter()))?;

        let attributes = attibutes_1.map(move |(key, attributes_1)| {
            let attributes_position = attributes_2
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let mut attributes_2 = attributes_2
                .remove(attributes_position)
                .1
                .collect::<HashMap<_, _>>();

            let attributes: BoxedIterator<_> = Box::new(attributes_1.map(move |(index, _)| {
                (
                    index,
                    attributes_2.remove(&index).expect("Attribute must exist"),
                )
            }));

            (key, attributes)
        });

        Ok(Box::new(attributes))
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_either_or_grouped<'a>(
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>>,
        either: &Wrapper<AttributesTreeOperand<O>>,
        or: &Wrapper<AttributesTreeOperand<O>>,
    ) -> MedRecordResult<
        GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>>,
    >
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = tee_grouped_iterator(attributes);

        let either_attributes = either.evaluate_forward_grouped(medrecord, attributes_1)?;
        let mut or_attributes: Vec<_> = or
            .evaluate_forward_grouped(medrecord, attributes_2)?
            .collect();

        let attributes = either_attributes.map(move |(key, either_attributes)| {
            let attributes_position = or_attributes
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let or_attributes = or_attributes.remove(attributes_position).1;

            let attributes: BoxedIterator<_> = Box::new(
                either_attributes
                    .chain(or_attributes)
                    .into_group_map_by(|(k, _)| *k)
                    .into_iter()
                    .map(|(idx, group)| {
                        let attrs = group.into_iter().flat_map(|(_, v)| v).unique().collect();
                        (idx, attrs)
                    }),
            );

            (key, attributes)
        });

        Ok(Box::new(attributes))
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_exclude_grouped<'a>(
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>>,
        operand: &Wrapper<AttributesTreeOperand<O>>,
    ) -> MedRecordResult<
        GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>>,
    >
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = tee_grouped_iterator(attributes);

        let mut result: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, attributes_1)?
            .collect();

        let attributes = attributes_2.map(move |(key, attributes)| {
            let attributes_position = result
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let mut excluded_attributes: MrHashMap<_, _> =
                result.remove(attributes_position).1.collect();

            let attributes: BoxedIterator<_> =
                Box::new(attributes.map(move |(index, attributes)| {
                    let entry = excluded_attributes.remove(&index).unwrap_or(Vec::new());

                    (
                        index,
                        attributes
                            .into_iter()
                            .filter(|attr| !entry.contains(attr))
                            .collect(),
                    )
                }));

            (key, attributes)
        });

        Ok(Box::new(attributes))
    }
}

#[derive(Debug, Clone)]
pub enum MultipleAttributesWithIndexOperation<O: RootOperand> {
    AttributeWithIndexOperation {
        operand: Wrapper<SingleAttributeWithIndexOperand<O>>,
    },
    AttributeWithoutIndexOperation {
        operand: Wrapper<SingleAttributeWithoutIndexOperand<O>>,
    },
    SingleAttributeComparisonOperation {
        operand: SingleAttributeComparisonOperand,
        kind: SingleComparisonKind,
    },
    MultipleAttributesComparisonOperation {
        operand: MultipleAttributesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOperation {
        operand: SingleAttributeComparisonOperand,
        kind: BinaryArithmeticKind,
    },
    UnaryArithmeticOperation {
        kind: UnaryArithmeticKind,
    },

    ToValues {
        operand: Wrapper<MultipleValuesWithIndexOperand<O>>,
    },

    Slice(Range<usize>),

    IsString,
    IsInt,

    IsMax,
    IsMin,

    EitherOr {
        either: Wrapper<MultipleAttributesWithIndexOperand<O>>,
        or: Wrapper<MultipleAttributesWithIndexOperand<O>>,
    },
    Exclude {
        operand: Wrapper<MultipleAttributesWithIndexOperand<O>>,
    },

    Merge {
        operand: Wrapper<MultipleAttributesWithIndexOperand<O>>,
    },
}

impl<O: RootOperand> DeepClone for MultipleAttributesWithIndexOperation<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::AttributeWithIndexOperation { operand } => Self::AttributeWithIndexOperation {
                operand: operand.deep_clone(),
            },
            Self::AttributeWithoutIndexOperation { operand } => {
                Self::AttributeWithoutIndexOperation {
                    operand: operand.deep_clone(),
                }
            }
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::SingleAttributeComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::MultipleAttributesComparisonOperation {
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
            Self::ToValues { operand } => Self::ToValues {
                operand: operand.deep_clone(),
            },
            Self::Slice(range) => Self::Slice(range.clone()),
            Self::IsString => Self::IsString,
            Self::IsInt => Self::IsInt,
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

impl<O: RootOperand> MultipleAttributesWithIndexOperation<O> {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::AttributeWithIndexOperation { operand } => {
                Self::evaluate_attribute_with_index_operation(medrecord, attributes, operand)?
            }
            Self::AttributeWithoutIndexOperation { operand } => {
                Self::evaluate_attribute_without_index_operation(medrecord, attributes, operand)?
            }
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::evaluate_single_attribute_comparison_operation(
                    medrecord, attributes, operand, kind,
                )?
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_attributes_comparison_operation(
                    medrecord, attributes, operand, kind,
                )?
            }
            Self::BinaryArithmeticOperation { operand, kind } => Box::new(
                Self::evaluate_binary_arithmetic_operation(medrecord, attributes, operand, kind)?,
            ),
            Self::UnaryArithmeticOperation { kind } => Box::new(
                Self::evaluate_unary_arithmetic_operation(attributes, kind.clone()),
            ),
            Self::ToValues { operand } => {
                Box::new(Self::evaluate_to_values(medrecord, attributes, operand)?)
            }
            Self::Slice(range) => Box::new(Self::evaluate_slice(attributes, range.clone())),
            Self::IsString => Box::new(Self::evaluate_is_string(attributes)),
            Self::IsInt => Box::new(Self::evaluate_is_int(attributes)),
            Self::IsMax => Box::new(Self::evaluate_is_max(attributes)?),
            Self::IsMin => Box::new(Self::evaluate_is_min(attributes)?),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, attributes, either, or)?
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, attributes, operand)?,
            Self::Merge { operand: _ } => unreachable!(),
        })
    }

    #[inline]
    pub(crate) fn get_max<'a>(
        mut attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordAttribute)>> {
        let max_attribute = attributes.next();

        let Some(max_attribute) = max_attribute else {
            return Ok(None);
        };

        let max_attribute = attributes.try_fold(max_attribute, |max_attribute, attribute| {
            match attribute.1.partial_cmp(&max_attribute.1) {
                Some(Ordering::Greater) => Ok(attribute),
                None => {
                    let first_dtype = DataType::from(attribute.1);
                    let second_dtype = DataType::from(max_attribute.1);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare attributes of data types {first_dtype} and {second_dtype}. Consider narrowing down the attributes using .is_string() or .is_int()"
                    )))
                }
                _ => Ok(max_attribute),
            }
        })?;

        Ok(Some(max_attribute))
    }

    #[inline]
    pub(crate) fn get_min<'a>(
        mut attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordAttribute)>> {
        let min_attribute = attributes.next();

        let Some(min_attribute) = min_attribute else {
            return Ok(None);
        };

        let min_attribute = attributes.try_fold(min_attribute, |min_attribute, attribute| {
            match attribute.1.partial_cmp(&min_attribute.1) {
                Some(Ordering::Less) => Ok(attribute),
                None => {
                    let first_dtype = DataType::from(attribute.1);
                    let second_dtype = DataType::from(min_attribute.1);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare attributes of data types {first_dtype} and {second_dtype}. Consider narrowing down the attributes using .is_string() or .is_int()"
                    )))
                }
                _ => Ok(min_attribute),
            }
        })?;

        Ok(Some(min_attribute))
    }

    #[inline]
    pub(crate) fn get_random<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>,
    ) -> Option<(&'a O::Index, MedRecordAttribute)> {
        attributes.choose(&mut rng())
    }

    #[inline]
    fn evaluate_attribute_with_index_operation<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a,
        operand: &Wrapper<SingleAttributeWithIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let kind = &operand.0.read_or_panic().kind;

        let attribute = match kind {
            SingleKindWithIndex::Max => {
                MultipleAttributesWithIndexOperation::<O>::get_max(attributes_1)?
            }
            SingleKindWithIndex::Min => {
                MultipleAttributesWithIndexOperation::<O>::get_min(attributes_1)?
            }
            SingleKindWithIndex::Random => {
                MultipleAttributesWithIndexOperation::<O>::get_random(attributes_1)
            }
        };

        Ok(match operand.evaluate_forward(medrecord, attribute)? {
            Some(_) => Box::new(attributes_2),
            None => Box::new(std::iter::empty()),
        })
    }

    #[inline]
    fn evaluate_attribute_without_index_operation<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a,
        operand: &Wrapper<SingleAttributeWithoutIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);
        let attributes_1 = attributes_1.map(|(_, attribute)| attribute);

        let kind = &operand.0.read_or_panic().kind;

        let attribute = match kind {
            SingleKindWithoutIndex::Max => {
                MultipleAttributesWithoutIndexOperation::<O>::get_max(attributes_1)?
            }
            SingleKindWithoutIndex::Min => {
                MultipleAttributesWithoutIndexOperation::<O>::get_min(attributes_1)?
            }
            SingleKindWithoutIndex::Count => Some(
                MultipleAttributesWithoutIndexOperation::<O>::get_count(attributes_1),
            ),
            SingleKindWithoutIndex::Sum => {
                MultipleAttributesWithoutIndexOperation::<O>::get_sum(attributes_1)?
            }
            SingleKindWithoutIndex::Random => {
                MultipleAttributesWithoutIndexOperation::<O>::get_random(attributes_1)
            }
        };

        Ok(match operand.evaluate_forward(medrecord, attribute)? {
            Some(_) => Box::new(attributes_2),
            None => Box::new(std::iter::empty()),
        })
    }

    #[inline]
    fn evaluate_single_attribute_comparison_operation<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a,
        comparison_operand: &SingleAttributeComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>> {
        let comparison_attribute =
            comparison_operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No attribute to compare".to_string(),
                ))?;

        match kind {
            SingleComparisonKind::GreaterThan => {
                Ok(Box::new(attributes.filter(move |(_, attribute)| {
                    attribute > &comparison_attribute
                })))
            }
            SingleComparisonKind::GreaterThanOrEqualTo => {
                Ok(Box::new(attributes.filter(move |(_, attribute)| {
                    attribute >= &comparison_attribute
                })))
            }
            SingleComparisonKind::LessThan => {
                Ok(Box::new(attributes.filter(move |(_, attribute)| {
                    attribute < &comparison_attribute
                })))
            }
            SingleComparisonKind::LessThanOrEqualTo => {
                Ok(Box::new(attributes.filter(move |(_, attribute)| {
                    attribute <= &comparison_attribute
                })))
            }
            SingleComparisonKind::EqualTo => {
                Ok(Box::new(attributes.filter(move |(_, attribute)| {
                    attribute == &comparison_attribute
                })))
            }
            SingleComparisonKind::NotEqualTo => {
                Ok(Box::new(attributes.filter(move |(_, attribute)| {
                    attribute != &comparison_attribute
                })))
            }
            SingleComparisonKind::StartsWith => {
                Ok(Box::new(attributes.filter(move |(_, attribute)| {
                    attribute.starts_with(&comparison_attribute)
                })))
            }
            SingleComparisonKind::EndsWith => {
                Ok(Box::new(attributes.filter(move |(_, attribute)| {
                    attribute.ends_with(&comparison_attribute)
                })))
            }
            SingleComparisonKind::Contains => {
                Ok(Box::new(attributes.filter(move |(_, attribute)| {
                    attribute.contains(&comparison_attribute)
                })))
            }
        }
    }

    #[inline]
    fn evaluate_multiple_attributes_comparison_operation<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a,
        comparison_operand: &MultipleAttributesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>> {
        let comparison_attributes = comparison_operand.evaluate_backward(medrecord)?;

        match kind {
            MultipleComparisonKind::IsIn => {
                Ok(Box::new(attributes.filter(move |(_, attribute)| {
                    comparison_attributes.contains(attribute)
                })))
            }
            MultipleComparisonKind::IsNotIn => {
                Ok(Box::new(attributes.filter(move |(_, attribute)| {
                    !comparison_attributes.contains(attribute)
                })))
            }
        }
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation<'a>(
        medrecord: &MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>,
        operand: &SingleAttributeComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        let arithmetic_attribute =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No attribute to compare".to_string(),
                ))?;

        let attributes = attributes
            .map(move |(t, attribute)| {
                match kind {
                    BinaryArithmeticKind::Add => attribute.add(arithmetic_attribute.clone()),
                    BinaryArithmeticKind::Sub => attribute.sub(arithmetic_attribute.clone()),
                    BinaryArithmeticKind::Mul => {
                        attribute.clone().mul(arithmetic_attribute.clone())
                    }
                    BinaryArithmeticKind::Pow => {
                        attribute.clone().pow(arithmetic_attribute.clone())
                    }
                    BinaryArithmeticKind::Mod => {
                        attribute.clone().r#mod(arithmetic_attribute.clone())
                    }
                }
                .map_err(|_| {
                    MedRecordError::QueryError(format!(
                        "Failed arithmetic operation {kind}. Consider narrowing down the attributes using .is_int() or .is_float()",
                    ))
                }).map(|result| (t, result))
            });

        Ok(attributes.collect::<MedRecordResult<Vec<_>>>()?.into_iter())
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>,
        kind: UnaryArithmeticKind,
    ) -> impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>
    where
        O: 'a,
    {
        attributes.map(move |(t, attribute)| {
            let attribute = match kind {
                UnaryArithmeticKind::Abs => attribute.abs(),
                UnaryArithmeticKind::Trim => attribute.trim(),
                UnaryArithmeticKind::TrimStart => attribute.trim_start(),
                UnaryArithmeticKind::TrimEnd => attribute.trim_end(),
                UnaryArithmeticKind::Lowercase => attribute.lowercase(),
                UnaryArithmeticKind::Uppercase => attribute.uppercase(),
            };
            (t, attribute)
        })
    }

    pub(crate) fn get_values<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordValue)>>
    where
        O: 'a,
    {
        Ok(attributes
            .map(|(index, attribute)| {
                let value = index.get_attributes(medrecord)?.get(&attribute).ok_or(
                    MedRecordError::QueryError(format!(
                        "Cannot find attribute {attribute} for index {index}"
                    )),
                )?;

                Ok((index, value.clone()))
            })
            .collect::<MedRecordResult<Vec<_>>>()?
            .into_iter())
    }

    #[inline]
    fn evaluate_to_values<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a,
        operand: &Wrapper<MultipleValuesWithIndexOperand<O>>,
    ) -> MedRecordResult<impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let values = Self::get_values(medrecord, attributes_1)?;

        let mut attributes: HashMap<_, _> = attributes_2.collect();

        let values = operand.evaluate_forward(medrecord, Box::new(values.into_iter()))?;

        Ok(values.map(move |(index, _)| {
            (
                index,
                attributes.remove(&index).expect("Attribute must exist"),
            )
        }))
    }

    #[inline]
    fn evaluate_slice<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>,
        range: Range<usize>,
    ) -> impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>
    where
        O: 'a,
    {
        attributes.map(move |(t, attribute)| (t, attribute.slice(range.clone())))
    }

    #[inline]
    fn evaluate_is_string<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>,
    ) -> impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>
    where
        O: 'a,
    {
        attributes.filter(|(_, attribute)| matches!(attribute, MedRecordAttribute::String(_)))
    }

    #[inline]
    fn evaluate_is_int<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>,
    ) -> impl Iterator<Item = (&'a O::Index, MedRecordAttribute)>
    where
        O: 'a,
    {
        attributes.filter(|(_, attribute)| matches!(attribute, MedRecordAttribute::Int(_)))
    }

    #[inline]
    fn evaluate_is_max<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let max_attribute = Self::get_max(attributes_1)?;

        let Some(max_attribute) = max_attribute else {
            return Ok(Box::new(std::iter::empty()));
        };

        Ok(Box::new(attributes_2.filter(move |(_, attribute)| {
            *attribute == max_attribute.1
        })))
    }

    #[inline]
    fn evaluate_is_min<'a>(
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let min_attribute = Self::get_min(attributes_1)?;

        let Some(min_attribute) = min_attribute else {
            return Ok(Box::new(std::iter::empty()));
        };

        Ok(Box::new(attributes_2.filter(move |(_, attribute)| {
            *attribute == min_attribute.1
        })))
    }

    #[inline]
    fn evaluate_either_or<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a,
        either: &Wrapper<MultipleAttributesWithIndexOperand<O>>,
        or: &Wrapper<MultipleAttributesWithIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let either_attributes = either.evaluate_forward(medrecord, Box::new(attributes_1))?;
        let or_attributes = or.evaluate_forward(medrecord, Box::new(attributes_2))?;

        Ok(Box::new(
            either_attributes
                .chain(or_attributes)
                .unique_by(|attribute| attribute.0.clone()),
        ))
    }

    #[inline]
    fn evaluate_exclude<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a,
        operand: &Wrapper<MultipleAttributesWithIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let result: MrHashSet<_> = operand
            .evaluate_forward(medrecord, Box::new(attributes_1))?
            .map(|(index, _)| index)
            .collect();

        Ok(Box::new(
            attributes_2.filter(move |(index, _)| !result.contains(index)),
        ))
    }
}

impl<O: RootOperand> MultipleAttributesWithIndexOperation<O> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::AttributeWithIndexOperation { operand } => {
                Self::evaluate_attribute_with_index_operation_grouped(
                    medrecord, attributes, operand,
                )?
            }
            Self::AttributeWithoutIndexOperation { operand } => {
                Self::evaluate_attribute_without_index_operation_grouped(
                    medrecord, attributes, operand,
                )?
            }
            Self::SingleAttributeComparisonOperation { operand, kind } => Box::new(
                attributes
                    .map(move |(key, attributes)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_single_attribute_comparison_operation(
                                medrecord, attributes, operand, kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::MultipleAttributesComparisonOperation { operand, kind } => Box::new(
                attributes
                    .map(move |(key, attributes)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_multiple_attributes_comparison_operation(
                                medrecord, attributes, operand, kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::BinaryArithmeticOperation { operand, kind } => Box::new(
                attributes
                    .map(move |(key, attributes)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_binary_arithmetic_operation(
                                medrecord, attributes, operand, kind,
                            )?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::UnaryArithmeticOperation { kind } => {
                let kind = kind.clone();

                Box::new(attributes.map(move |(key, attributes)| {
                    (
                        key,
                        Box::new(Self::evaluate_unary_arithmetic_operation(
                            attributes,
                            kind.clone(),
                        )) as BoxedIterator<_>,
                    )
                }))
            }
            Self::ToValues { operand } => Box::new(Self::evaluate_to_values_grouped(
                medrecord, attributes, operand,
            )?),
            Self::Slice(range) => {
                let range = range.clone();

                Box::new(attributes.map(move |(key, attributes)| {
                    (
                        key,
                        Box::new(Self::evaluate_slice(attributes, range.clone()))
                            as BoxedIterator<_>,
                    )
                }))
            }
            Self::IsString => Box::new(attributes.map(move |(key, attributes)| {
                (
                    key,
                    Box::new(Self::evaluate_is_string(attributes)) as BoxedIterator<_>,
                )
            })),
            Self::IsInt => Box::new(attributes.map(move |(key, attributes)| {
                (
                    key,
                    Box::new(Self::evaluate_is_int(attributes)) as BoxedIterator<_>,
                )
            })),
            Self::IsMax => Box::new(
                attributes
                    .map(move |(key, attributes)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_is_max(attributes)?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::IsMin => Box::new(
                attributes
                    .map(move |(key, attributes)| {
                        Ok((
                            key,
                            Box::new(Self::evaluate_is_min(attributes)?) as BoxedIterator<_>,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or_grouped(medrecord, attributes, either, or)?
            }
            Self::Exclude { operand } => {
                Self::evaluate_exclude_grouped(medrecord, attributes, operand)?
            }
            Self::Merge { operand } => {
                let (attributes_1, attributes_2) = tee_grouped_iterator(attributes);

                let attributes_1 = attributes_1.flat_map(|(_, attribute)| attribute);

                let attributes_1: MrHashSet<_> = operand
                    .evaluate_forward(medrecord, Box::new(attributes_1))?
                    .collect();

                Box::new(attributes_2.map(move |(key, attributes)| {
                    let attributes: Vec<_> = attributes
                        .filter(|attribute| attributes_1.contains(attribute))
                        .collect();

                    let attributes: BoxedIterator<_> = Box::new(attributes.into_iter());

                    (key, attributes)
                }))
            }
        })
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_attribute_with_index_operation_grouped<'a>(
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>,
        operand: &Wrapper<SingleAttributeWithIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = tee_grouped_iterator(attributes);
        let mut attributes_2 = attributes_2.collect::<Vec<_>>();

        let kind = &operand.0.read_or_panic().kind;

        let attributes_1: Vec<_> = attributes_1
            .map(|(key, attributes)| {
                let attribute = match kind {
                    SingleKindWithIndex::Max => {
                        MultipleAttributesWithIndexOperation::<O>::get_max(attributes)?
                    }
                    SingleKindWithIndex::Min => {
                        MultipleAttributesWithIndexOperation::<O>::get_min(attributes)?
                    }
                    SingleKindWithIndex::Random => {
                        MultipleAttributesWithIndexOperation::<O>::get_random(attributes)
                    }
                };

                Ok((key, attribute))
            })
            .collect::<MedRecordResult<_>>()?;

        let attributes_1 =
            operand.evaluate_forward_grouped(medrecord, Box::new(attributes_1.into_iter()))?;

        Ok(Box::new(attributes_1.map(
            move |(key, attribute)| match attribute {
                Some(_) => {
                    let attributes_position = attributes_2
                        .iter()
                        .position(|(k, _)| k == &key)
                        .expect("Entry must exist");

                    attributes_2.remove(attributes_position)
                }
                None => (key, Box::new(std::iter::empty()) as BoxedIterator<_>),
            },
        )))
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_attribute_without_index_operation_grouped<'a>(
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>,
        operand: &Wrapper<SingleAttributeWithoutIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = tee_grouped_iterator(attributes);
        let mut attributes_2: Vec<_> = attributes_2.collect();

        let kind = &operand.0.read_or_panic().kind;

        let attributes_1: Vec<_> = attributes_1
            .map(|(key, attributes)| {
                let attributes = attributes.map(|(_, attribute)| attribute);

                let attribute = match kind {
                    SingleKindWithoutIndex::Max => {
                        MultipleAttributesWithoutIndexOperation::<O>::get_max(attributes)?
                    }
                    SingleKindWithoutIndex::Min => {
                        MultipleAttributesWithoutIndexOperation::<O>::get_min(attributes)?
                    }
                    SingleKindWithoutIndex::Count => Some(
                        MultipleAttributesWithoutIndexOperation::<O>::get_count(attributes),
                    ),
                    SingleKindWithoutIndex::Sum => {
                        MultipleAttributesWithoutIndexOperation::<O>::get_sum(attributes)?
                    }
                    SingleKindWithoutIndex::Random => {
                        MultipleAttributesWithoutIndexOperation::<O>::get_random(attributes)
                    }
                };

                Ok((key, attribute))
            })
            .collect::<MedRecordResult<_>>()?;

        let attributes_1 =
            operand.evaluate_forward_grouped(medrecord, Box::new(attributes_1.into_iter()))?;

        Ok(Box::new(attributes_1.map(
            move |(key, attribute)| match attribute {
                Some(_) => {
                    let attributes_position = attributes_2
                        .iter()
                        .position(|(k, _)| k == &key)
                        .expect("Entry must exist");

                    attributes_2.remove(attributes_position)
                }
                None => (key, Box::new(std::iter::empty()) as BoxedIterator<_>),
            },
        )))
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_to_values_grouped<'a>(
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>,
        operand: &Wrapper<MultipleValuesWithIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = tee_grouped_iterator(attributes);
        let mut attributes_2: Vec<_> = attributes_2.collect();

        let values: Vec<_> = attributes_1
            .map(|(key, attributes)| {
                let values: BoxedIterator<_> = Box::new(Self::get_values(medrecord, attributes)?);

                Ok((key, values))
            })
            .collect::<MedRecordResult<_>>()?;

        let values = operand.evaluate_forward_grouped(medrecord, Box::new(values.into_iter()))?;

        let attributes = values.map(move |(key, values)| {
            let attributes_position = attributes_2
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let mut attributes = attributes_2
                .remove(attributes_position)
                .1
                .collect::<HashMap<_, _>>();

            let attributes: BoxedIterator<_> = Box::new(values.map(move |(index, _)| {
                (
                    index,
                    attributes.remove(&index).expect("Attribute must exist"),
                )
            }));

            (key, attributes)
        });

        Ok(Box::new(attributes))
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_either_or_grouped<'a>(
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>,
        either: &Wrapper<MultipleAttributesWithIndexOperand<O>>,
        or: &Wrapper<MultipleAttributesWithIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = tee_grouped_iterator(attributes);

        let either_attributes = either.evaluate_forward_grouped(medrecord, attributes_1)?;
        let mut or_attributes: Vec<_> = or
            .evaluate_forward_grouped(medrecord, attributes_2)?
            .collect();

        let attributes = either_attributes.map(move |(key, either_attributes)| {
            let attributes_position = or_attributes
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let or_attributes = or_attributes.remove(attributes_position).1;

            let attributes: BoxedIterator<_> = Box::new(
                either_attributes
                    .chain(or_attributes)
                    .unique_by(|attributes| attributes.0.clone()),
            );

            (key, attributes)
        });

        Ok(Box::new(attributes))
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_exclude_grouped<'a>(
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>,
        operand: &Wrapper<MultipleAttributesWithIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = tee_grouped_iterator(attributes);

        let mut result: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, attributes_1)?
            .collect();

        let attributes = attributes_2.map(move |(key, attributes)| {
            let attributes_position = result
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let excluded_attributes: MrHashSet<_> = result.remove(attributes_position).1.collect();

            let attributes: BoxedIterator<_> = Box::new(
                attributes.filter(move |attributes| !excluded_attributes.contains(attributes)),
            );

            (key, attributes)
        });

        Ok(Box::new(attributes))
    }
}
#[derive(Debug, Clone)]
pub enum MultipleAttributesWithoutIndexOperation<O: RootOperand> {
    AttributeOperation {
        operand: Wrapper<SingleAttributeWithoutIndexOperand<O>>,
    },
    SingleAttributeComparisonOperation {
        operand: SingleAttributeComparisonOperand,
        kind: SingleComparisonKind,
    },
    MultipleAttributesComparisonOperation {
        operand: MultipleAttributesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOperation {
        operand: SingleAttributeComparisonOperand,
        kind: BinaryArithmeticKind,
    },
    UnaryArithmeticOperation {
        kind: UnaryArithmeticKind,
    },

    Slice(Range<usize>),

    IsString,
    IsInt,

    IsMax,
    IsMin,

    EitherOr {
        either: Wrapper<MultipleAttributesWithoutIndexOperand<O>>,
        or: Wrapper<MultipleAttributesWithoutIndexOperand<O>>,
    },
    Exclude {
        operand: Wrapper<MultipleAttributesWithoutIndexOperand<O>>,
    },
}

impl<O: RootOperand> DeepClone for MultipleAttributesWithoutIndexOperation<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::AttributeOperation { operand } => Self::AttributeOperation {
                operand: operand.deep_clone(),
            },
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::SingleAttributeComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::MultipleAttributesComparisonOperation {
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

impl<O: RootOperand> MultipleAttributesWithoutIndexOperation<O> {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = MedRecordAttribute> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordAttribute>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::AttributeOperation { operand } => {
                Self::evaluate_attribute_operation(medrecord, attributes, operand)?
            }
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::evaluate_single_attribute_comparison_operation(
                    medrecord, attributes, operand, kind,
                )?
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_attributes_comparison_operation(
                    medrecord, attributes, operand, kind,
                )?
            }
            Self::BinaryArithmeticOperation { operand, kind } => Box::new(
                Self::evaluate_binary_arithmetic_operation(medrecord, attributes, operand, kind)?,
            ),
            Self::UnaryArithmeticOperation { kind } => Box::new(
                Self::evaluate_unary_arithmetic_operation(attributes, kind.clone()),
            ),
            Self::Slice(range) => Box::new(Self::evaluate_slice(attributes, range.clone())),
            Self::IsString => Box::new(
                attributes.filter(|attribute| matches!(attribute, MedRecordAttribute::String(_))),
            ),
            Self::IsInt => Box::new(
                attributes.filter(|attribute| matches!(attribute, MedRecordAttribute::Int(_))),
            ),
            Self::IsMax => {
                let (attributes_1, attributes_2) = Itertools::tee(attributes);

                let max_attribute = Self::get_max(attributes_1)?;

                let Some(max_attribute) = max_attribute else {
                    return Ok(Box::new(std::iter::empty()));
                };

                Box::new(attributes_2.filter(move |attribute| *attribute == max_attribute))
            }
            Self::IsMin => {
                let (attributes_1, attributes_2) = Itertools::tee(attributes);

                let min_attribute = Self::get_min(attributes_1)?;

                let Some(min_attribute) = min_attribute else {
                    return Ok(Box::new(std::iter::empty()));
                };

                Box::new(attributes_2.filter(move |attribute| *attribute == min_attribute))
            }
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, attributes, either, or)?
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, attributes, operand)?,
        })
    }

    #[inline]
    pub(crate) fn get_max(
        mut attributes: impl Iterator<Item = MedRecordAttribute>,
    ) -> MedRecordResult<Option<MedRecordAttribute>> {
        let max_attribute = attributes.next();

        let Some(max_attribute) = max_attribute else {
            return Ok(None);
        };

        let max_attribute = attributes.try_fold(max_attribute, |max_attribute, attribute| {
            match attribute.partial_cmp(&max_attribute) {
                Some(Ordering::Greater) => Ok(attribute),
                None => {
                    let first_dtype = DataType::from(attribute);
                    let second_dtype = DataType::from(max_attribute);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare attributes of data types {first_dtype} and {second_dtype}. Consider narrowing down the attributes using .is_string() or .is_int()"
                    )))
                }
                _ => Ok(max_attribute),
            }
        })?;

        Ok(Some(max_attribute))
    }

    #[inline]
    pub(crate) fn get_min(
        mut attributes: impl Iterator<Item = MedRecordAttribute>,
    ) -> MedRecordResult<Option<MedRecordAttribute>> {
        let min_attribute = attributes.next();

        let Some(min_attribute) = min_attribute else {
            return Ok(None);
        };

        let min_attribute = attributes.try_fold(min_attribute, |min_attribute, attribute| {
            match attribute.partial_cmp(&min_attribute) {
                Some(Ordering::Less) => Ok(attribute),
                None => {
                    let first_dtype = DataType::from(attribute);
                    let second_dtype = DataType::from(min_attribute);

                    Err(MedRecordError::QueryError(format!(
                        "Cannot compare attributes of data types {first_dtype} and {second_dtype}. Consider narrowing down the attributes using .is_string() or .is_int()"
                    )))
                }
                _ => Ok(min_attribute),
            }
        })?;

        Ok(Some(min_attribute))
    }

    #[inline]
    pub(crate) fn get_count(
        attributes: impl Iterator<Item = MedRecordAttribute>,
    ) -> MedRecordAttribute {
        MedRecordAttribute::Int(attributes.count() as i64)
    }

    #[inline]
    // 🥊💥
    pub(crate) fn get_sum(
        mut attributes: impl Iterator<Item = MedRecordAttribute>,
    ) -> MedRecordResult<Option<MedRecordAttribute>> {
        let first_attribute = attributes.next();

        let Some(first_attribute) = first_attribute else {
            return Ok(None);
        };

        let sum = attributes.try_fold(first_attribute, |sum, attribute| {
            let first_dtype = DataType::from(&sum);
            let second_dtype = DataType::from(&attribute);

            sum.add(attribute).map_err(|_| {
                MedRecordError::QueryError(format!(
                    "Cannot add attributes of data types {first_dtype} and {second_dtype}. Consider narrowing down the attributes using .is_string() or .is_int()"
                ))
            })
        })?;

        Ok(Some(sum))
    }

    #[inline]
    pub(crate) fn get_random(
        attributes: impl Iterator<Item = MedRecordAttribute>,
    ) -> Option<MedRecordAttribute> {
        attributes.choose(&mut rng())
    }

    #[inline]
    fn evaluate_attribute_operation<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = MedRecordAttribute> + 'a,
        operand: &Wrapper<SingleAttributeWithoutIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordAttribute>> {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let kind = &operand.0.read_or_panic().kind;

        let attribute = match kind {
            SingleKindWithoutIndex::Max => Self::get_max(attributes_1)?,
            SingleKindWithoutIndex::Min => Self::get_min(attributes_1)?,
            SingleKindWithoutIndex::Count => Some(Self::get_count(attributes_1)),
            SingleKindWithoutIndex::Sum => Self::get_sum(attributes_1)?,
            SingleKindWithoutIndex::Random => Self::get_random(attributes_1),
        };

        Ok(match operand.evaluate_forward(medrecord, attribute)? {
            Some(_) => Box::new(attributes_2),
            None => Box::new(std::iter::empty()),
        })
    }

    #[inline]
    fn evaluate_single_attribute_comparison_operation<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = MedRecordAttribute> + 'a,
        comparison_operand: &SingleAttributeComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordAttribute>> {
        let comparison_attribute =
            comparison_operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No attribute to compare".to_string(),
                ))?;

        match kind {
            SingleComparisonKind::GreaterThan => {
                Ok(Box::new(attributes.filter(move |attribute| {
                    attribute > &comparison_attribute
                })))
            }
            SingleComparisonKind::GreaterThanOrEqualTo => {
                Ok(Box::new(attributes.filter(move |attribute| {
                    attribute >= &comparison_attribute
                })))
            }
            SingleComparisonKind::LessThan => {
                Ok(Box::new(attributes.filter(move |attribute| {
                    attribute < &comparison_attribute
                })))
            }
            SingleComparisonKind::LessThanOrEqualTo => {
                Ok(Box::new(attributes.filter(move |attribute| {
                    attribute <= &comparison_attribute
                })))
            }
            SingleComparisonKind::EqualTo => {
                Ok(Box::new(attributes.filter(move |attribute| {
                    attribute == &comparison_attribute
                })))
            }
            SingleComparisonKind::NotEqualTo => {
                Ok(Box::new(attributes.filter(move |attribute| {
                    attribute != &comparison_attribute
                })))
            }
            SingleComparisonKind::StartsWith => {
                Ok(Box::new(attributes.filter(move |attribute| {
                    attribute.starts_with(&comparison_attribute)
                })))
            }
            SingleComparisonKind::EndsWith => {
                Ok(Box::new(attributes.filter(move |attribute| {
                    attribute.ends_with(&comparison_attribute)
                })))
            }
            SingleComparisonKind::Contains => {
                Ok(Box::new(attributes.filter(move |attribute| {
                    attribute.contains(&comparison_attribute)
                })))
            }
        }
    }

    #[inline]
    fn evaluate_multiple_attributes_comparison_operation<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = MedRecordAttribute> + 'a,
        comparison_operand: &MultipleAttributesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordAttribute>> {
        let comparison_attributes = comparison_operand.evaluate_backward(medrecord)?;

        match kind {
            MultipleComparisonKind::IsIn => {
                Ok(Box::new(attributes.filter(move |attribute| {
                    comparison_attributes.contains(attribute)
                })))
            }
            MultipleComparisonKind::IsNotIn => {
                Ok(Box::new(attributes.filter(move |attribute| {
                    !comparison_attributes.contains(attribute)
                })))
            }
        }
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation<'a>(
        medrecord: &MedRecord,
        attributes: impl Iterator<Item = MedRecordAttribute>,
        operand: &SingleAttributeComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<impl Iterator<Item = MedRecordAttribute>>
    where
        O: 'a,
    {
        let arithmetic_attribute =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No attribute to compare".to_string(),
                ))?;

        let attributes = attributes
            .map(move |attribute| {
                match kind {
                    BinaryArithmeticKind::Add => attribute.add(arithmetic_attribute.clone()),
                    BinaryArithmeticKind::Sub => attribute.sub(arithmetic_attribute.clone()),
                    BinaryArithmeticKind::Mul => {
                        attribute.clone().mul(arithmetic_attribute.clone())
                    }
                    BinaryArithmeticKind::Pow => {
                        attribute.clone().pow(arithmetic_attribute.clone())
                    }
                    BinaryArithmeticKind::Mod => {
                        attribute.clone().r#mod(arithmetic_attribute.clone())
                    }
                }
                .map_err(|_| {
                    MedRecordError::QueryError(format!(
                        "Failed arithmetic operation {kind}. Consider narrowing down the attributes using .is_int() or .is_float()",
                    ))
                })
            });

        Ok(attributes.collect::<MedRecordResult<Vec<_>>>()?.into_iter())
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation<'a>(
        attributes: impl Iterator<Item = MedRecordAttribute>,
        kind: UnaryArithmeticKind,
    ) -> impl Iterator<Item = MedRecordAttribute>
    where
        O: 'a,
    {
        attributes.map(move |attribute| match kind {
            UnaryArithmeticKind::Abs => attribute.abs(),
            UnaryArithmeticKind::Trim => attribute.trim(),
            UnaryArithmeticKind::TrimStart => attribute.trim_start(),
            UnaryArithmeticKind::TrimEnd => attribute.trim_end(),
            UnaryArithmeticKind::Lowercase => attribute.lowercase(),
            UnaryArithmeticKind::Uppercase => attribute.uppercase(),
        })
    }

    #[inline]
    fn evaluate_slice(
        attributes: impl Iterator<Item = MedRecordAttribute>,
        range: Range<usize>,
    ) -> impl Iterator<Item = MedRecordAttribute> {
        attributes.map(move |attribute| attribute.slice(range.clone()))
    }

    #[inline]
    fn evaluate_either_or<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = MedRecordAttribute> + 'a,
        either: &Wrapper<MultipleAttributesWithoutIndexOperand<O>>,
        or: &Wrapper<MultipleAttributesWithoutIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordAttribute>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let either_attributes = either.evaluate_forward(medrecord, Box::new(attributes_1))?;
        let or_attributes = or.evaluate_forward(medrecord, Box::new(attributes_2))?;

        Ok(Box::new(
            either_attributes
                .chain(or_attributes)
                .unique_by(|attribute| attribute.clone()),
        ))
    }

    #[inline]
    fn evaluate_exclude<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = MedRecordAttribute> + 'a,
        operand: &Wrapper<MultipleAttributesWithoutIndexOperand<O>>,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordAttribute>> {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let result: MrHashSet<_> = operand
            .evaluate_forward(medrecord, Box::new(attributes_1))?
            .collect();

        Ok(Box::new(
            attributes_2.filter(move |index| !result.contains(index)),
        ))
    }
}

#[derive(Debug, Clone)]
pub enum SingleAttributeWithIndexOperation<O: RootOperand> {
    SingleAttributeComparisonOperation {
        operand: SingleAttributeComparisonOperand,
        kind: SingleComparisonKind,
    },
    MultipleAttributesComparisonOperation {
        operand: MultipleAttributesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOperation {
        operand: SingleAttributeComparisonOperand,
        kind: BinaryArithmeticKind,
    },
    UnaryArithmeticOperation {
        kind: UnaryArithmeticKind,
    },

    Slice(Range<usize>),

    IsString,
    IsInt,

    EitherOr {
        either: Wrapper<SingleAttributeWithIndexOperand<O>>,
        or: Wrapper<SingleAttributeWithIndexOperand<O>>,
    },
    Exclude {
        operand: Wrapper<SingleAttributeWithIndexOperand<O>>,
    },

    Merge {
        operand: Wrapper<MultipleAttributesWithIndexOperand<O>>,
    },
}

impl<O: RootOperand> DeepClone for SingleAttributeWithIndexOperation<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::SingleAttributeComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::MultipleAttributesComparisonOperation {
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

impl<O: RootOperand> SingleAttributeWithIndexOperation<O> {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        attribute: Option<(&'a O::Index, MedRecordAttribute)>,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        let Some(attribute) = attribute else {
            return Ok(None);
        };

        Ok(match self {
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::evaluate_single_attribute_comparison_operation(
                    medrecord, attribute, operand, kind,
                )?
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_attributes_comparison_operation(
                    medrecord, attribute, operand, kind,
                )?
            }
            Self::BinaryArithmeticOperation { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, attribute, operand, kind)?
            }
            Self::UnaryArithmeticOperation { kind } => {
                Some(Self::evaluate_unary_arithmetic_operation(attribute, kind))
            }
            Self::Slice(range) => Some(Self::evaluate_slice(attribute, range)),
            Self::IsString => Self::evaluate_is_string(attribute),
            Self::IsInt => Self::evaluate_is_int(attribute),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, attribute, either, or)?
            }
            Self::Exclude { operand } => {
                match operand.evaluate_forward(medrecord, Some(attribute.clone()))? {
                    Some(_) => None,
                    None => Some(attribute),
                }
            }
            Self::Merge { operand: _ } => unreachable!(),
        })
    }

    #[inline]
    fn evaluate_single_attribute_comparison_operation<'a>(
        medrecord: &MedRecord,
        attribute: (&'a O::Index, MedRecordAttribute),
        comparison_operand: &SingleAttributeComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordAttribute)>> {
        let comparison_attribute =
            comparison_operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No attribute to compare".to_string(),
                ))?;

        let comparison_result = match kind {
            SingleComparisonKind::GreaterThan => attribute.1 > comparison_attribute,
            SingleComparisonKind::GreaterThanOrEqualTo => attribute.1 >= comparison_attribute,
            SingleComparisonKind::LessThan => attribute.1 < comparison_attribute,
            SingleComparisonKind::LessThanOrEqualTo => attribute.1 <= comparison_attribute,
            SingleComparisonKind::EqualTo => attribute.1 == comparison_attribute,
            SingleComparisonKind::NotEqualTo => attribute.1 != comparison_attribute,
            SingleComparisonKind::StartsWith => attribute.1.starts_with(&comparison_attribute),
            SingleComparisonKind::EndsWith => attribute.1.ends_with(&comparison_attribute),
            SingleComparisonKind::Contains => attribute.1.contains(&comparison_attribute),
        };

        Ok(if comparison_result {
            Some(attribute)
        } else {
            None
        })
    }

    #[inline]
    fn evaluate_multiple_attributes_comparison_operation<'a>(
        medrecord: &MedRecord,
        attribute: (&'a O::Index, MedRecordAttribute),
        comparison_operand: &MultipleAttributesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordAttribute)>> {
        let comparison_attributes = comparison_operand.evaluate_backward(medrecord)?;

        let comparison_result = match kind {
            MultipleComparisonKind::IsIn => comparison_attributes.contains(&attribute.1),
            MultipleComparisonKind::IsNotIn => !comparison_attributes.contains(&attribute.1),
        };

        Ok(if comparison_result {
            Some(attribute)
        } else {
            None
        })
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation<'a>(
        medrecord: &MedRecord,
        attribute: (&'a O::Index, MedRecordAttribute),
        operand: &SingleAttributeComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordAttribute)>> {
        let arithmetic_attribute =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No attribute to compare".to_string(),
                ))?;

        Ok(Some(match kind {
            BinaryArithmeticKind::Add => (attribute.0, attribute.1.add(arithmetic_attribute)?),
            BinaryArithmeticKind::Sub => (attribute.0, attribute.1.sub(arithmetic_attribute)?),
            BinaryArithmeticKind::Mul => (attribute.0, attribute.1.mul(arithmetic_attribute)?),
            BinaryArithmeticKind::Pow => (attribute.0, attribute.1.pow(arithmetic_attribute)?),
            BinaryArithmeticKind::Mod => (attribute.0, attribute.1.r#mod(arithmetic_attribute)?),
        }))
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation<'a>(
        attribute: (&'a O::Index, MedRecordAttribute),
        kind: &UnaryArithmeticKind,
    ) -> (&'a O::Index, MedRecordAttribute) {
        match kind {
            UnaryArithmeticKind::Abs => (attribute.0, attribute.1.abs()),
            UnaryArithmeticKind::Trim => (attribute.0, attribute.1.trim()),
            UnaryArithmeticKind::TrimStart => (attribute.0, attribute.1.trim_start()),
            UnaryArithmeticKind::TrimEnd => (attribute.0, attribute.1.trim_end()),
            UnaryArithmeticKind::Lowercase => (attribute.0, attribute.1.lowercase()),
            UnaryArithmeticKind::Uppercase => (attribute.0, attribute.1.uppercase()),
        }
    }

    #[inline]
    fn evaluate_slice<'a>(
        attribute: (&'a O::Index, MedRecordAttribute),
        range: &Range<usize>,
    ) -> (&'a O::Index, MedRecordAttribute) {
        (attribute.0, attribute.1.slice(range.clone()))
    }

    #[inline]
    fn evaluate_is_string(
        attribute: (&O::Index, MedRecordAttribute),
    ) -> Option<(&O::Index, MedRecordAttribute)> {
        match attribute.1 {
            MedRecordAttribute::String(_) => Some(attribute),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_int(
        attribute: (&O::Index, MedRecordAttribute),
    ) -> Option<(&O::Index, MedRecordAttribute)> {
        match attribute.1 {
            MedRecordAttribute::Int(_) => Some(attribute),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_either_or<'a>(
        medrecord: &'a MedRecord,
        attribute: (&'a O::Index, MedRecordAttribute),
        either: &Wrapper<SingleAttributeWithIndexOperand<O>>,
        or: &Wrapper<SingleAttributeWithIndexOperand<O>>,
    ) -> MedRecordResult<Option<(&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        let either_result = either.evaluate_forward(medrecord, Some(attribute.clone()))?;
        let or_result = or.evaluate_forward(medrecord, Some(attribute))?;

        match (either_result, or_result) {
            (Some(either_result), _) => Ok(Some(either_result)),
            (None, Some(or_result)) => Ok(Some(or_result)),
            _ => Ok(None),
        }
    }
}

impl<O: RootOperand> SingleAttributeWithIndexOperation<O> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, Option<(&'a O::Index, MedRecordAttribute)>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<(&'a O::Index, MedRecordAttribute)>>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::SingleAttributeComparisonOperation { operand, kind } => Box::new(
                attributes
                    .map(move |(key, attribute)| {
                        let Some(attribute) = attribute else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_single_attribute_comparison_operation(
                                medrecord, attribute, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::MultipleAttributesComparisonOperation { operand, kind } => Box::new(
                attributes
                    .map(move |(key, attribute)| {
                        let Some(attribute) = attribute else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_multiple_attributes_comparison_operation(
                                medrecord, attribute, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::BinaryArithmeticOperation { operand, kind } => Box::new(
                attributes
                    .map(move |(key, attribute)| {
                        let Some(attribute) = attribute else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_binary_arithmetic_operation(
                                medrecord, attribute, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::UnaryArithmeticOperation { kind } => {
                let kind = kind.clone();

                Box::new(attributes.map(move |(key, attribute)| {
                    let Some(attribute) = attribute else {
                        return (key, None);
                    };

                    (
                        key,
                        Some(Self::evaluate_unary_arithmetic_operation(attribute, &kind)),
                    )
                }))
            }
            Self::Slice(range) => {
                let range = range.clone();

                Box::new(attributes.map(move |(key, attribute)| {
                    let Some(attribute) = attribute else {
                        return (key, None);
                    };

                    (key, Some(Self::evaluate_slice(attribute, &range)))
                }))
            }
            Self::IsString => Box::new(attributes.map(move |(key, attribute)| {
                let Some(attribute) = attribute else {
                    return (key, None);
                };

                (key, Self::evaluate_is_string(attribute))
            })),
            Self::IsInt => Box::new(attributes.map(move |(key, attribute)| {
                let Some(attribute) = attribute else {
                    return (key, None);
                };

                (key, Self::evaluate_is_int(attribute))
            })),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or_grouped(medrecord, attributes, either, or)?
            }
            Self::Exclude { operand } => {
                Self::evaluate_exclude_grouped(medrecord, attributes, operand)?
            }
            Self::Merge { operand } => {
                let (attributes_1, attributes_2) = Itertools::tee(attributes);

                let attributes_1 = attributes_1.filter_map(|(_, value)| value);

                let attributes_1: MrHashSet<_> = operand
                    .evaluate_forward(medrecord, Box::new(attributes_1))?
                    .collect();

                Box::new(attributes_2.map(move |(key, attribute)| {
                    let attribute = attribute.filter(|value| attributes_1.contains(value));

                    (key, attribute)
                }))
            }
        })
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_either_or_grouped<'a>(
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, Option<(&'a O::Index, MedRecordAttribute)>>,
        either: &Wrapper<SingleAttributeWithIndexOperand<O>>,
        or: &Wrapper<SingleAttributeWithIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<(&'a O::Index, MedRecordAttribute)>>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let either_attributes =
            either.evaluate_forward_grouped(medrecord, Box::new(attributes_1))?;
        let mut or_attributes: Vec<_> = or
            .evaluate_forward_grouped(medrecord, Box::new(attributes_2))?
            .collect();

        let attributes = either_attributes.map(move |(key, either_attribute)| {
            let attribute_position = or_attributes
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let or_attribute = or_attributes.remove(attribute_position).1;

            let attribute = match (either_attribute, or_attribute) {
                (Some(either_result), _) => Some(either_result),
                (None, Some(or_result)) => Some(or_result),
                _ => None,
            };

            (key, attribute)
        });

        Ok(Box::new(attributes))
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_exclude_grouped<'a>(
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Option<(&'a O::Index, MedRecordAttribute)>>,
        operand: &Wrapper<SingleAttributeWithIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<(&'a O::Index, MedRecordAttribute)>>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(values);

        let mut result: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, Box::new(attributes_1))?
            .collect();

        let attributes = attributes_2.map(move |(key, attribute)| {
            let attribute_position = result
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let excluded_attribute = result.remove(attribute_position).1;

            let attribute = match excluded_attribute {
                Some(_) => None,
                None => attribute,
            };

            (key, attribute)
        });

        Ok(Box::new(attributes))
    }
}

#[derive(Debug, Clone)]
pub enum SingleAttributeWithoutIndexOperation<O: RootOperand> {
    SingleAttributeComparisonOperation {
        operand: SingleAttributeComparisonOperand,
        kind: SingleComparisonKind,
    },
    MultipleAttributesComparisonOperation {
        operand: MultipleAttributesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOperation {
        operand: SingleAttributeComparisonOperand,
        kind: BinaryArithmeticKind,
    },
    UnaryArithmeticOperation {
        kind: UnaryArithmeticKind,
    },

    Slice(Range<usize>),

    IsString,
    IsInt,

    EitherOr {
        either: Wrapper<SingleAttributeWithoutIndexOperand<O>>,
        or: Wrapper<SingleAttributeWithoutIndexOperand<O>>,
    },
    Exclude {
        operand: Wrapper<SingleAttributeWithoutIndexOperand<O>>,
    },

    Merge {
        operand: Wrapper<MultipleAttributesWithoutIndexOperand<O>>,
    },
}

impl<O: RootOperand> DeepClone for SingleAttributeWithoutIndexOperation<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::SingleAttributeComparisonOperation {
                    operand: operand.deep_clone(),
                    kind: kind.clone(),
                }
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::MultipleAttributesComparisonOperation {
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

impl<O: RootOperand> SingleAttributeWithoutIndexOperation<O> {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        attribute: Option<MedRecordAttribute>,
    ) -> MedRecordResult<Option<MedRecordAttribute>>
    where
        O: 'a,
    {
        let Some(attribute) = attribute else {
            return Ok(None);
        };

        Ok(match self {
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::evaluate_single_attribute_comparison_operation(
                    medrecord, attribute, operand, kind,
                )?
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_attributes_comparison_operation(
                    medrecord, attribute, operand, kind,
                )?
            }
            Self::BinaryArithmeticOperation { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, attribute, operand, kind)?
            }
            Self::UnaryArithmeticOperation { kind } => {
                Some(Self::evaluate_unary_arithmetic_operation(attribute, kind))
            }
            Self::Slice(range) => Some(Self::evaluate_slice(attribute, range)),
            Self::IsString => Self::evaluate_is_string(attribute),
            Self::IsInt => Self::evaluate_is_int(attribute),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, attribute, either, or)?
            }
            Self::Exclude { operand } => {
                match operand.evaluate_forward(medrecord, Some(attribute.clone()))? {
                    Some(_) => None,
                    None => Some(attribute),
                }
            }
            Self::Merge { operand: _ } => unreachable!(),
        })
    }

    #[inline]
    fn evaluate_single_attribute_comparison_operation(
        medrecord: &MedRecord,
        attribute: MedRecordAttribute,
        comparison_operand: &SingleAttributeComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<Option<MedRecordAttribute>> {
        let comparison_attribute =
            comparison_operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No attribute to compare".to_string(),
                ))?;

        let comparison_result = match kind {
            SingleComparisonKind::GreaterThan => attribute > comparison_attribute,
            SingleComparisonKind::GreaterThanOrEqualTo => attribute >= comparison_attribute,
            SingleComparisonKind::LessThan => attribute < comparison_attribute,
            SingleComparisonKind::LessThanOrEqualTo => attribute <= comparison_attribute,
            SingleComparisonKind::EqualTo => attribute == comparison_attribute,
            SingleComparisonKind::NotEqualTo => attribute != comparison_attribute,
            SingleComparisonKind::StartsWith => attribute.starts_with(&comparison_attribute),
            SingleComparisonKind::EndsWith => attribute.ends_with(&comparison_attribute),
            SingleComparisonKind::Contains => attribute.contains(&comparison_attribute),
        };

        Ok(if comparison_result {
            Some(attribute)
        } else {
            None
        })
    }

    #[inline]
    fn evaluate_multiple_attributes_comparison_operation(
        medrecord: &MedRecord,
        attribute: MedRecordAttribute,
        comparison_operand: &MultipleAttributesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<Option<MedRecordAttribute>> {
        let comparison_attributes = comparison_operand.evaluate_backward(medrecord)?;

        let comparison_result = match kind {
            MultipleComparisonKind::IsIn => comparison_attributes.contains(&attribute),
            MultipleComparisonKind::IsNotIn => !comparison_attributes.contains(&attribute),
        };

        Ok(if comparison_result {
            Some(attribute)
        } else {
            None
        })
    }

    #[inline]
    fn evaluate_binary_arithmetic_operation(
        medrecord: &MedRecord,
        attribute: MedRecordAttribute,
        operand: &SingleAttributeComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<Option<MedRecordAttribute>> {
        let arithmetic_attribute =
            operand
                .evaluate_backward(medrecord)?
                .ok_or(MedRecordError::QueryError(
                    "No attribute to compare".to_string(),
                ))?;

        Ok(Some(match kind {
            BinaryArithmeticKind::Add => attribute.add(arithmetic_attribute)?,
            BinaryArithmeticKind::Sub => attribute.sub(arithmetic_attribute)?,
            BinaryArithmeticKind::Mul => attribute.mul(arithmetic_attribute)?,
            BinaryArithmeticKind::Pow => attribute.pow(arithmetic_attribute)?,
            BinaryArithmeticKind::Mod => attribute.r#mod(arithmetic_attribute)?,
        }))
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation(
        attribute: MedRecordAttribute,
        kind: &UnaryArithmeticKind,
    ) -> MedRecordAttribute {
        match kind {
            UnaryArithmeticKind::Abs => attribute.abs(),
            UnaryArithmeticKind::Trim => attribute.trim(),
            UnaryArithmeticKind::TrimStart => attribute.trim_start(),
            UnaryArithmeticKind::TrimEnd => attribute.trim_end(),
            UnaryArithmeticKind::Lowercase => attribute.lowercase(),
            UnaryArithmeticKind::Uppercase => attribute.uppercase(),
        }
    }

    #[inline]
    fn evaluate_slice(attribute: MedRecordAttribute, range: &Range<usize>) -> MedRecordAttribute {
        attribute.slice(range.clone())
    }

    #[inline]
    fn evaluate_is_string(attribute: MedRecordAttribute) -> Option<MedRecordAttribute> {
        match attribute {
            MedRecordAttribute::String(_) => Some(attribute),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_is_int(attribute: MedRecordAttribute) -> Option<MedRecordAttribute> {
        match attribute {
            MedRecordAttribute::Int(_) => Some(attribute),
            _ => None,
        }
    }

    #[inline]
    fn evaluate_either_or<'a>(
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
        either: &Wrapper<SingleAttributeWithoutIndexOperand<O>>,
        or: &Wrapper<SingleAttributeWithoutIndexOperand<O>>,
    ) -> MedRecordResult<Option<MedRecordAttribute>>
    where
        O: 'a,
    {
        let either_result = either.evaluate_forward(medrecord, Some(attribute.clone()))?;
        let or_result = or.evaluate_forward(medrecord, Some(attribute))?;

        match (either_result, or_result) {
            (Some(either_result), _) => Ok(Some(either_result)),
            (None, Some(or_result)) => Ok(Some(or_result)),
            _ => Ok(None),
        }
    }
}

impl<O: RootOperand> SingleAttributeWithoutIndexOperation<O> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, Option<MedRecordAttribute>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<MedRecordAttribute>>>
    where
        O: 'a,
    {
        Ok(match self {
            Self::SingleAttributeComparisonOperation { operand, kind } => Box::new(
                attributes
                    .map(move |(key, attribute)| {
                        let Some(attribute) = attribute else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_single_attribute_comparison_operation(
                                medrecord, attribute, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::MultipleAttributesComparisonOperation { operand, kind } => Box::new(
                attributes
                    .map(move |(key, attribute)| {
                        let Some(attribute) = attribute else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_multiple_attributes_comparison_operation(
                                medrecord, attribute, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::BinaryArithmeticOperation { operand, kind } => Box::new(
                attributes
                    .map(move |(key, attribute)| {
                        let Some(attribute) = attribute else {
                            return Ok((key, None));
                        };

                        Ok((
                            key,
                            Self::evaluate_binary_arithmetic_operation(
                                medrecord, attribute, operand, kind,
                            )?,
                        ))
                    })
                    .collect::<MedRecordResult<Vec<_>>>()?
                    .into_iter(),
            ),
            Self::UnaryArithmeticOperation { kind } => {
                let kind = kind.clone();

                Box::new(attributes.map(move |(key, attribute)| {
                    let Some(attribute) = attribute else {
                        return (key, None);
                    };

                    (
                        key,
                        Some(Self::evaluate_unary_arithmetic_operation(attribute, &kind)),
                    )
                }))
            }
            Self::Slice(range) => {
                let range = range.clone();

                Box::new(attributes.map(move |(key, attribute)| {
                    let Some(attribute) = attribute else {
                        return (key, None);
                    };

                    (key, Some(Self::evaluate_slice(attribute, &range)))
                }))
            }
            Self::IsString => Box::new(attributes.map(move |(key, attribute)| {
                let Some(attribute) = attribute else {
                    return (key, None);
                };

                (key, Self::evaluate_is_string(attribute))
            })),
            Self::IsInt => Box::new(attributes.map(move |(key, attribute)| {
                let Some(attribute) = attribute else {
                    return (key, None);
                };

                (key, Self::evaluate_is_int(attribute))
            })),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or_grouped(medrecord, attributes, either, or)?
            }
            Self::Exclude { operand } => {
                Self::evaluate_exclude_grouped(medrecord, attributes, operand)?
            }
            Self::Merge { operand } => {
                let (attributes_1, attributes_2) = Itertools::tee(attributes);

                let attributes_1 = attributes_1.filter_map(|(_, attribute)| attribute);

                let attributes_1: MrHashSet<_> = operand
                    .evaluate_forward(medrecord, Box::new(attributes_1))?
                    .collect();

                Box::new(attributes_2.map(move |(key, attribute)| {
                    let attribute = attribute.filter(|attribute| attributes_1.contains(attribute));

                    (key, attribute)
                }))
            }
        })
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_either_or_grouped<'a>(
        medrecord: &'a MedRecord,
        attributes: GroupedIterator<'a, Option<MedRecordAttribute>>,
        either: &Wrapper<SingleAttributeWithoutIndexOperand<O>>,
        or: &Wrapper<SingleAttributeWithoutIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<MedRecordAttribute>>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(attributes);

        let either_attributes =
            either.evaluate_forward_grouped(medrecord, Box::new(attributes_1))?;
        let mut or_attributes: Vec<_> = or
            .evaluate_forward_grouped(medrecord, Box::new(attributes_2))?
            .collect();

        let attributes = either_attributes.map(move |(key, either_attribute)| {
            let attribute_position = or_attributes
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let or_attribute = or_attributes.remove(attribute_position).1;

            let attribute = match (either_attribute, or_attribute) {
                (Some(either_result), _) => Some(either_result),
                (None, Some(or_result)) => Some(or_result),
                _ => None,
            };

            (key, attribute)
        });

        Ok(Box::new(attributes))
    }

    #[allow(clippy::type_complexity)]
    #[inline]
    fn evaluate_exclude_grouped<'a>(
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Option<MedRecordAttribute>>,
        operand: &Wrapper<SingleAttributeWithoutIndexOperand<O>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<MedRecordAttribute>>>
    where
        O: 'a,
    {
        let (attributes_1, attributes_2) = Itertools::tee(values);

        let mut result: Vec<_> = operand
            .evaluate_forward_grouped(medrecord, Box::new(attributes_1))?
            .collect();

        let attributes = attributes_2.map(move |(key, attribute)| {
            let attribute_position = result
                .iter()
                .position(|(k, _)| k == &key)
                .expect("Entry must exist");

            let excluded_attribute = result.remove(attribute_position).1;

            let attribute = match excluded_attribute {
                Some(_) => None,
                None => attribute,
            };

            (key, attribute)
        });

        Ok(Box::new(attributes))
    }
}
