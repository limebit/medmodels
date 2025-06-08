use super::{
    operand::{
        MultipleAttributesComparisonOperand, MultipleAttributesWithIndexOperand,
        SingleAttributeComparisonOperand, SingleAttributeWithIndexOperand,
    },
    AttributesTreeOperand, BinaryArithmeticKind, GetAttributes, MultipleComparisonKind,
    MultipleKind, SingleComparisonKind, SingleKindWithIndex, UnaryArithmeticKind,
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
                operand::SingleAttributeWithoutIndexOperand, MultipleAttributesWithoutIndexOperand,
                SingleKindWithoutIndex,
            },
            values::MultipleValuesWithIndexOperand,
            BoxedIterator, DeepClone, EvaluateForward, GroupedIterator, ReadWriteOrPanic,
            RootOperand,
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
        match self {
            Self::AttributesOperation { operand } => Ok(Box::new(
                Self::evaluate_attributes_operation(medrecord, attributes, operand)?,
            )),
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::evaluate_single_attribute_comparison_operation(
                    medrecord, attributes, operand, kind,
                )
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_attributes_comparison_operation(
                    medrecord, attributes, operand, kind,
                )
            }
            Self::BinaryArithmeticOperation { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, attributes, operand, kind)
            }
            Self::UnaryArithmeticOperation { kind } => Ok(Box::new(
                Self::evaluate_unary_arithmetic_operation(attributes, kind.clone()),
            )),
            Self::Slice(range) => Ok(Box::new(Self::evaluate_slice(attributes, range.clone()))),
            Self::IsString => Ok(Box::new(attributes.map(|(index, attribute)| {
                (
                    index,
                    attribute
                        .into_iter()
                        .filter(|attribute| matches!(attribute, MedRecordAttribute::String(_)))
                        .collect(),
                )
            }))),
            Self::IsInt => Ok(Box::new(attributes.map(|(index, attribute)| {
                (
                    index,
                    attribute
                        .into_iter()
                        .filter(|attribute| matches!(attribute, MedRecordAttribute::Int(_)))
                        .collect(),
                )
            }))),
            Self::IsMax => {
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
            Self::IsMin => {
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
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, attributes, either, or)
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, attributes, operand),
        }
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
                            "Cannot compare attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                            first_dtype, second_dtype
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
                            "Cannot compare attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                            first_dtype, second_dtype
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
                        "Cannot add attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                        first_dtype, second_dtype
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

        let kind = &operand.0.read_or_panic().kind;

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
        _medrecord: &'a MedRecord,
        _attributes: GroupedIterator<
            'a,
            BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>,
        >,
    ) -> MedRecordResult<
        GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, Vec<MedRecordAttribute>)>>,
    >
    where
        O: 'a,
    {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub enum MultipleAttributesOperationWithIndex<O: RootOperand> {
    AttributeOperation {
        operand: Wrapper<SingleAttributeWithIndexOperand<O>>,
    },
    AttributeOperationWithoutIndex {
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
}

impl<O: RootOperand> DeepClone for MultipleAttributesOperationWithIndex<O> {
    fn deep_clone(&self) -> Self {
        match self {
            Self::AttributeOperation { operand } => Self::AttributeOperation {
                operand: operand.deep_clone(),
            },
            Self::AttributeOperationWithoutIndex { operand } => {
                Self::AttributeOperationWithoutIndex {
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
        }
    }
}

impl<O: RootOperand> MultipleAttributesOperationWithIndex<O> {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (&'a O::Index, MedRecordAttribute)> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>
    where
        O: 'a,
    {
        match self {
            Self::AttributeOperation { operand } => {
                Self::evaluate_attribute_operation(medrecord, attributes, operand)
            }
            Self::AttributeOperationWithoutIndex { operand } => {
                Self::evaluate_attribute_operation_without_index(medrecord, attributes, operand)
            }
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::evaluate_single_attribute_comparison_operation(
                    medrecord, attributes, operand, kind,
                )
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_attributes_comparison_operation(
                    medrecord, attributes, operand, kind,
                )
            }
            Self::BinaryArithmeticOperation { operand, kind } => Ok(Box::new(
                Self::evaluate_binary_arithmetic_operation(medrecord, attributes, operand, kind)?,
            )),
            Self::UnaryArithmeticOperation { kind } => Ok(Box::new(
                Self::evaluate_unary_arithmetic_operation(attributes, kind.clone()),
            )),
            Self::ToValues { operand } => Ok(Box::new(Self::evaluate_to_values(
                medrecord, attributes, operand,
            )?)),
            Self::Slice(range) => Ok(Box::new(Self::evaluate_slice(attributes, range.clone()))),
            Self::IsString => {
                Ok(Box::new(attributes.filter(|(_, attribute)| {
                    matches!(attribute, MedRecordAttribute::String(_))
                })))
            }
            Self::IsInt => {
                Ok(Box::new(attributes.filter(|(_, attribute)| {
                    matches!(attribute, MedRecordAttribute::Int(_))
                })))
            }
            Self::IsMax => {
                let (attributes_1, attributes_2) = Itertools::tee(attributes);

                let max_attribute = Self::get_max(attributes_1)?;

                let Some(max_attribute) = max_attribute else {
                    return Ok(Box::new(std::iter::empty()));
                };

                Ok(Box::new(attributes_2.filter(move |(_, attribute)| {
                    *attribute == max_attribute.1
                })))
            }
            Self::IsMin => {
                let (attributes_1, attributes_2) = Itertools::tee(attributes);

                let min_attribute = Self::get_min(attributes_1)?;

                let Some(min_attribute) = min_attribute else {
                    return Ok(Box::new(std::iter::empty()));
                };

                Ok(Box::new(attributes_2.filter(move |(_, attribute)| {
                    *attribute == min_attribute.1
                })))
            }
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, attributes, either, or)
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, attributes, operand),
        }
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
                        "Cannot compare attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                        first_dtype, second_dtype
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
                        "Cannot compare attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                        first_dtype, second_dtype
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
    fn evaluate_attribute_operation<'a>(
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
                MultipleAttributesOperationWithIndex::<O>::get_max(attributes_1)?
            }
            SingleKindWithIndex::Min => {
                MultipleAttributesOperationWithIndex::<O>::get_min(attributes_1)?
            }
            SingleKindWithIndex::Random => {
                MultipleAttributesOperationWithIndex::<O>::get_random(attributes_1)
            }
        };

        Ok(match operand.evaluate_forward(medrecord, attribute)? {
            Some(_) => Box::new(attributes_2),
            None => Box::new(std::iter::empty()),
        })
    }

    #[inline]
    fn evaluate_attribute_operation_without_index<'a>(
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
                MultipleAttributesOperationWithoutIndex::<O>::get_max(attributes_1)?
            }
            SingleKindWithoutIndex::Min => {
                MultipleAttributesOperationWithoutIndex::<O>::get_min(attributes_1)?
            }
            SingleKindWithoutIndex::Count => Some(
                MultipleAttributesOperationWithoutIndex::<O>::get_count(attributes_1),
            ),
            SingleKindWithoutIndex::Sum => {
                MultipleAttributesOperationWithoutIndex::<O>::get_sum(attributes_1)?
            }
            SingleKindWithoutIndex::Random => {
                MultipleAttributesOperationWithoutIndex::<O>::get_random(attributes_1)
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
                        "Failed arithmetic operation {}. Consider narrowing down the attributes using .is_int() or .is_float()",
                        kind,
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
                        "Cannot find attribute {} for index {}",
                        attribute, index
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
        let attributes: Vec<_> = attributes.collect();

        let values = Self::get_values(medrecord, attributes.clone().into_iter())?;

        let mut attributes: HashMap<_, _> = attributes.into_iter().collect();

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

impl<O: RootOperand> MultipleAttributesOperationWithIndex<O> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        _medrecord: &'a MedRecord,
        _attributes: GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, (&'a O::Index, MedRecordAttribute)>>>
    where
        O: 'a,
    {
        todo!()
    }
}
#[derive(Debug, Clone)]
pub enum MultipleAttributesOperationWithoutIndex<O: RootOperand> {
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

impl<O: RootOperand> DeepClone for MultipleAttributesOperationWithoutIndex<O> {
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

impl<O: RootOperand> MultipleAttributesOperationWithoutIndex<O> {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = MedRecordAttribute> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, MedRecordAttribute>>
    where
        O: 'a,
    {
        match self {
            Self::AttributeOperation { operand } => {
                Self::evaluate_attribute_operation(medrecord, attributes, operand)
            }
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::evaluate_single_attribute_comparison_operation(
                    medrecord, attributes, operand, kind,
                )
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_attributes_comparison_operation(
                    medrecord, attributes, operand, kind,
                )
            }
            Self::BinaryArithmeticOperation { operand, kind } => Ok(Box::new(
                Self::evaluate_binary_arithmetic_operation(medrecord, attributes, operand, kind)?,
            )),
            Self::UnaryArithmeticOperation { kind } => Ok(Box::new(
                Self::evaluate_unary_arithmetic_operation(attributes, kind.clone()),
            )),
            Self::Slice(range) => Ok(Box::new(Self::evaluate_slice(attributes, range.clone()))),
            Self::IsString => {
                Ok(Box::new(attributes.filter(|attribute| {
                    matches!(attribute, MedRecordAttribute::String(_))
                })))
            }
            Self::IsInt => {
                Ok(Box::new(attributes.filter(|attribute| {
                    matches!(attribute, MedRecordAttribute::Int(_))
                })))
            }
            Self::IsMax => {
                let (attributes_1, attributes_2) = Itertools::tee(attributes);

                let max_attribute = Self::get_max(attributes_1)?;

                let Some(max_attribute) = max_attribute else {
                    return Ok(Box::new(std::iter::empty()));
                };

                Ok(Box::new(
                    attributes_2.filter(move |attribute| *attribute == max_attribute),
                ))
            }
            Self::IsMin => {
                let (attributes_1, attributes_2) = Itertools::tee(attributes);

                let min_attribute = Self::get_min(attributes_1)?;

                let Some(min_attribute) = min_attribute else {
                    return Ok(Box::new(std::iter::empty()));
                };

                Ok(Box::new(
                    attributes_2.filter(move |attribute| *attribute == min_attribute),
                ))
            }
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, attributes, either, or)
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, attributes, operand),
        }
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
                        "Cannot compare attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                        first_dtype, second_dtype
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
                        "Cannot compare attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                        first_dtype, second_dtype
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
                    "Cannot add attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                    first_dtype, second_dtype
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
                        "Failed arithmetic operation {}. Consider narrowing down the attributes using .is_int() or .is_float()",
                        kind,
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

impl<O: RootOperand> MultipleAttributesOperationWithoutIndex<O> {
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        _medrecord: &'a MedRecord,
        _attributes: GroupedIterator<'a, BoxedIterator<'a, MedRecordAttribute>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, MedRecordAttribute>>>
    where
        O: 'a,
    {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub enum SingleAttributeOperationWithIndex<O: RootOperand> {
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
}

impl<O: RootOperand> DeepClone for SingleAttributeOperationWithIndex<O> {
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
        }
    }
}

impl<O: RootOperand> SingleAttributeOperationWithIndex<O> {
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

        match self {
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::evaluate_single_attribute_comparison_operation(
                    medrecord, attribute, operand, kind,
                )
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_attribute_comparison_operation(
                    medrecord, attribute, operand, kind,
                )
            }
            Self::BinaryArithmeticOperation { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, attribute, operand, kind)
            }
            Self::UnaryArithmeticOperation { kind } => Ok(Some(match kind {
                UnaryArithmeticKind::Abs => (attribute.0, attribute.1.abs()),
                UnaryArithmeticKind::Trim => (attribute.0, attribute.1.trim()),
                UnaryArithmeticKind::TrimStart => (attribute.0, attribute.1.trim_start()),
                UnaryArithmeticKind::TrimEnd => (attribute.0, attribute.1.trim_end()),
                UnaryArithmeticKind::Lowercase => (attribute.0, attribute.1.lowercase()),
                UnaryArithmeticKind::Uppercase => (attribute.0, attribute.1.uppercase()),
            })),
            Self::Slice(range) => Ok(Some((attribute.0, attribute.1.slice(range.clone())))),
            Self::IsString => Ok(match attribute.1 {
                MedRecordAttribute::String(_) => Some(attribute),
                _ => None,
            }),
            Self::IsInt => Ok(match attribute.1 {
                MedRecordAttribute::Int(_) => Some(attribute),
                _ => None,
            }),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, attribute, either, or)
            }
            Self::Exclude { operand } => Ok(
                match operand.evaluate_forward(medrecord, Some(attribute.clone()))? {
                    Some(_) => None,
                    None => Some(attribute),
                },
            ),
        }
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
    fn evaluate_multiple_attribute_comparison_operation<'a>(
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

impl<O: RootOperand> SingleAttributeOperationWithIndex<O> {
    #[allow(clippy::type_complexity)]
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        _medrecord: &'a MedRecord,
        _attributes: GroupedIterator<'a, Option<(&'a O::Index, MedRecordAttribute)>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<(&'a O::Index, MedRecordAttribute)>>>
    where
        O: 'a,
    {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub enum SingleAttributeOperationWithoutIndex<O: RootOperand> {
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
}

impl<O: RootOperand> DeepClone for SingleAttributeOperationWithoutIndex<O> {
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
        }
    }
}

impl<O: RootOperand> SingleAttributeOperationWithoutIndex<O> {
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

        match self {
            Self::SingleAttributeComparisonOperation { operand, kind } => {
                Self::evaluate_single_attribute_comparison_operation(
                    medrecord, attribute, operand, kind,
                )
            }
            Self::MultipleAttributesComparisonOperation { operand, kind } => {
                Self::evaluate_multiple_attribute_comparison_operation(
                    medrecord, attribute, operand, kind,
                )
            }
            Self::BinaryArithmeticOperation { operand, kind } => {
                Self::evaluate_binary_arithmetic_operation(medrecord, attribute, operand, kind)
            }
            Self::UnaryArithmeticOperation { kind } => Ok(Some(match kind {
                UnaryArithmeticKind::Abs => attribute.abs(),
                UnaryArithmeticKind::Trim => attribute.trim(),
                UnaryArithmeticKind::TrimStart => attribute.trim_start(),
                UnaryArithmeticKind::TrimEnd => attribute.trim_end(),
                UnaryArithmeticKind::Lowercase => attribute.lowercase(),
                UnaryArithmeticKind::Uppercase => attribute.uppercase(),
            })),
            Self::Slice(range) => Ok(Some(attribute.slice(range.clone()))),
            Self::IsString => Ok(match attribute {
                MedRecordAttribute::String(_) => Some(attribute),
                _ => None,
            }),
            Self::IsInt => Ok(match attribute {
                MedRecordAttribute::Int(_) => Some(attribute),
                _ => None,
            }),
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, attribute, either, or)
            }
            Self::Exclude { operand } => Ok(
                match operand.evaluate_forward(medrecord, Some(attribute.clone()))? {
                    Some(_) => None,
                    None => Some(attribute),
                },
            ),
        }
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
    fn evaluate_multiple_attribute_comparison_operation(
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

impl<O: RootOperand> SingleAttributeOperationWithoutIndex<O> {
    pub(crate) fn evaluate_grouped<'a>(
        &self,
        _medrecord: &'a MedRecord,
        _attributes: GroupedIterator<'a, Option<MedRecordAttribute>>,
    ) -> MedRecordResult<GroupedIterator<'a, Option<MedRecordAttribute>>>
    where
        O: 'a,
    {
        todo!()
    }
}
