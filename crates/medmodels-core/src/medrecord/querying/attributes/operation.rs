use super::{
    operand::{
        MultipleAttributesComparisonOperand, MultipleAttributesOperand,
        SingleAttributeComparisonOperand, SingleAttributeOperand,
    },
    AttributesTreeOperand, BinaryArithmeticKind, GetAttributes, MultipleComparisonKind,
    SingleComparisonKind, UnaryArithmeticKind,
};
use crate::{
    errors::{MedRecordError, MedRecordResult},
    medrecord::{
        datatypes::{
            Abs, Contains, EndsWith, Lowercase, Mod, Pow, Slice, StartsWith, Trim, TrimEnd,
            TrimStart, Uppercase,
        },
        querying::{
            traits::{DeepClone, ReadWriteOrPanic},
            values::MultipleValuesOperand,
            BoxedIterator, Operand, OptionalIndexWrapper,
        },
        DataType, MedRecordAttribute, MedRecordValue, Wrapper,
    },
    MedRecord,
};
use itertools::Itertools;
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    fmt::Display,
    hash::Hash,
    ops::{Add, Mul, Range, Sub},
};

#[derive(Debug, Clone)]
pub enum AttributesTreeOperation<O: Operand> {
    AttributesOperation {
        operand: Wrapper<MultipleAttributesOperand<O>>,
    },
    SingleAttributeComparisonOperation {
        operand: SingleAttributeComparisonOperand<O>,
        kind: SingleComparisonKind,
    },
    MultipleAttributesComparisonOperation {
        operand: MultipleAttributesComparisonOperand<O>,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOpration {
        operand: SingleAttributeComparisonOperand<O>,
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

impl<O: Operand> DeepClone for AttributesTreeOperation<O> {
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

impl<O: Operand> AttributesTreeOperation<O> {
    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (O::Index, Vec<MedRecordAttribute>)> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, (O::Index, Vec<MedRecordAttribute>)>> {
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
            Self::BinaryArithmeticOpration { operand, kind } => {
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
                        .filter(|attribute| matches!(attribute, MedRecordAttribute::String(_)))
                        .collect(),
                )
            }))),
            Self::IsMax => {
                let max_attributes = Self::get_max(attributes)?;

                Ok(Box::new(
                    max_attributes.map(|(index, attribute)| (index, vec![attribute])),
                ))
            }
            Self::IsMin => {
                let min_attributes = Self::get_min(attributes)?;

                Ok(Box::new(
                    min_attributes.map(|(index, attribute)| (index, vec![attribute])),
                ))
            }
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, attributes, either, or)
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, attributes, operand),
        }
    }

    #[inline]
    pub(crate) fn get_max<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)>> {
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
    pub(crate) fn get_min<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)>> {
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
    pub(crate) fn get_count<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)>> {
        Ok(attributes
            .map(|(index, attribute)| (index, MedRecordAttribute::Int(attribute.len() as i64))))
    }

    #[inline]
    pub(crate) fn get_sum<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)>> {
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
    pub(crate) fn get_first<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)>> {
        Ok(attributes
            .map(|(index, attributes)| {
                let first_attribute =
                    attributes
                        .into_iter()
                        .next()
                        .ok_or(MedRecordError::QueryError(
                            "No attributes to compare".to_string(),
                        ))?;

                Ok((index, first_attribute))
            })
            .collect::<MedRecordResult<Vec<_>>>()?
            .into_iter())
    }

    #[inline]
    pub(crate) fn get_last<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)>> {
        Ok(attributes
            .map(|(index, attributes)| {
                let first_attribute =
                    attributes
                        .into_iter()
                        .next_back()
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
        attributes: impl Iterator<Item = (O::Index, Vec<MedRecordAttribute>)> + 'a,
        operand: &Wrapper<MultipleAttributesOperand<O>>,
    ) -> MedRecordResult<impl Iterator<Item = (O::Index, Vec<MedRecordAttribute>)> + 'a>
    where
        O: 'a,
    {
        let mut attributes = attributes.collect::<HashMap<_, _>>();

        let result = operand.evaluate(medrecord)?;

        Ok(result.map(move |(index, _)| {
            (
                index.clone(),
                attributes.remove(&index).expect("Index must exist"),
            )
        }))
    }

    #[inline]
    fn evaluate_single_attribute_comparison_operation<'a>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (O::Index, Vec<MedRecordAttribute>)> + 'a,
        comparison_operand: &SingleAttributeComparisonOperand<O>,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, (O::Index, Vec<MedRecordAttribute>)>>
    where
        O: 'a,
    {
        let comparison_attribute = comparison_operand
            .evaluate(medrecord)?
            .ok_or(MedRecordError::QueryError(
                "No attribute to compare".to_string(),
            ))?
            .unpack()
            .1;

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
        attributes: impl Iterator<Item = (O::Index, Vec<MedRecordAttribute>)> + 'a,
        comparison_operand: &MultipleAttributesComparisonOperand<O>,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, (O::Index, Vec<MedRecordAttribute>)>> {
        let comparison_attributes = match comparison_operand {
            MultipleAttributesComparisonOperand::Operand(operand) => {
                let context = &operand.context.context;
                let kind = &operand.kind;

                let comparison_attributes = context.get_attributes(medrecord)?;

                // TODO: Does this need to be passed through operand.context.evaluate as well?

                let comparison_attributes: Box<dyn Iterator<Item = (_, MedRecordAttribute)>> =
                    get_multiple_operand_attributes!(kind, comparison_attributes);

                operand
                    .evaluate(medrecord, comparison_attributes)?
                    .map(|(_, attribute)| attribute)
                    .collect::<Vec<_>>()
            }
            MultipleAttributesComparisonOperand::Attributes(attributes) => attributes.clone(),
        };

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
    fn evaluate_binary_arithmetic_operation<'a>(
        medrecord: &MedRecord,
        attributes: impl Iterator<Item = (O::Index, Vec<MedRecordAttribute>)>,
        operand: &SingleAttributeComparisonOperand<O>,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<BoxedIterator<'a, (O::Index, Vec<MedRecordAttribute>)>>
    where
        O: 'a,
    {
        let arithmetic_attribute = operand
            .evaluate(medrecord)?
            .ok_or(MedRecordError::QueryError(
                "No attribute to compare".to_string(),
            ))?
            .unpack()
            .1;

        let attributes: Box<
            dyn Iterator<Item = MedRecordResult<(O::Index, Vec<MedRecordAttribute>)>>,
        > = match kind {
            BinaryArithmeticKind::Add => Box::new(attributes.map(move |(index, attributes)| {
                Ok((
                    index,
                    attributes
                        .into_iter()
                        .map(|attribute| attribute.add(arithmetic_attribute.clone()))
                        .collect::<MedRecordResult<Vec<_>>>()?,
                ))
            })),
            BinaryArithmeticKind::Sub => Box::new(attributes.map(move |(index, attributes)| {
                Ok((
                    index,
                    attributes
                        .into_iter()
                        .map(|attribute| attribute.sub(arithmetic_attribute.clone()))
                        .collect::<MedRecordResult<Vec<_>>>()?,
                ))
            })),
            BinaryArithmeticKind::Mul => Box::new(attributes.map(move |(index, attributes)| {
                Ok((
                    index,
                    attributes
                        .into_iter()
                        .map(|attribute| attribute.mul(arithmetic_attribute.clone()))
                        .collect::<MedRecordResult<Vec<_>>>()?,
                ))
            })),
            BinaryArithmeticKind::Pow => Box::new(attributes.map(move |(index, attributes)| {
                Ok((
                    index,
                    attributes
                        .into_iter()
                        .map(|attribute| attribute.pow(arithmetic_attribute.clone()))
                        .collect::<MedRecordResult<Vec<_>>>()?,
                ))
            })),
            BinaryArithmeticKind::Mod => Box::new(attributes.map(move |(index, attributes)| {
                Ok((
                    index,
                    attributes
                        .into_iter()
                        .map(|attribute| attribute.r#mod(arithmetic_attribute.clone()))
                        .collect::<MedRecordResult<Vec<_>>>()?,
                ))
            })),
        };

        Ok(Box::new(
            attributes.collect::<MedRecordResult<Vec<_>>>()?.into_iter(),
        ))
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
        kind: UnaryArithmeticKind,
    ) -> impl Iterator<Item = (T, Vec<MedRecordAttribute>)> {
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
    fn evaluate_slice<T>(
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
        range: Range<usize>,
    ) -> impl Iterator<Item = (T, Vec<MedRecordAttribute>)> {
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
    fn evaluate_either_or<'a, T: 'a + Eq + Clone + Hash + GetAttributes + Display>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
        either: &Wrapper<AttributesTreeOperand>,
        or: &Wrapper<AttributesTreeOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, (T, Vec<MedRecordAttribute>)>> {
        let attributes = attributes.collect::<Vec<_>>();

        let either_attributes = either.evaluate(medrecord, attributes.clone().into_iter())?;
        let or_attributes = or.evaluate(medrecord, attributes.into_iter())?;

        Ok(Box::new(
            either_attributes
                .chain(or_attributes)
                .unique_by(|attribute| attribute.0.clone()),
        ))
    }

    #[inline]
    fn evaluate_exclude<'a, T: 'a + Eq + Clone + Hash + GetAttributes + Display>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (T, Vec<MedRecordAttribute>)>,
        operand: &Wrapper<AttributesTreeOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, (T, Vec<MedRecordAttribute>)>> {
        let attributes = attributes.collect::<Vec<_>>();

        let result = operand
            .evaluate(medrecord, attributes.clone().into_iter())?
            .map(|(index, _)| index)
            .collect::<HashSet<_>>();

        Ok(Box::new(
            attributes
                .into_iter()
                .filter(move |(index, _)| !result.contains(index)),
        ))
    }
}

#[derive(Debug, Clone)]
pub enum MultipleAttributesOperation {
    AttributeOperation {
        operand: Wrapper<SingleAttributeOperand>,
    },
    SingleAttributeComparisonOperation {
        operand: SingleAttributeComparisonOperand,
        kind: SingleComparisonKind,
    },
    MultipleAttributesComparisonOperation {
        operand: MultipleAttributesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOpration {
        operand: SingleAttributeComparisonOperand,
        kind: BinaryArithmeticKind,
    },
    UnaryArithmeticOperation {
        kind: UnaryArithmeticKind,
    },

    ToValues {
        operand: Wrapper<MultipleValuesOperand>,
    },

    Slice(Range<usize>),

    IsString,
    IsInt,

    IsMax,
    IsMin,

    EitherOr {
        either: Wrapper<MultipleAttributesOperand>,
        or: Wrapper<MultipleAttributesOperand>,
    },
    Exclude {
        operand: Wrapper<MultipleAttributesOperand>,
    },
}

impl DeepClone for MultipleAttributesOperation {
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
            Self::BinaryArithmeticOpration { operand, kind } => Self::BinaryArithmeticOpration {
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

impl MultipleAttributesOperation {
    pub(crate) fn evaluate<'a, T: 'a + Eq + Clone + Hash + GetAttributes + Display>(
        &self,
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (T, MedRecordAttribute)> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, (T, MedRecordAttribute)>> {
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
            Self::BinaryArithmeticOpration { operand, kind } => Ok(Box::new(
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
                let max_attribute = Self::get_max(attributes)?;

                Ok(Box::new(std::iter::once(max_attribute)))
            }
            Self::IsMin => {
                let min_attribute = Self::get_min(attributes)?;

                Ok(Box::new(std::iter::once(min_attribute)))
            }
            Self::EitherOr { either, or } => {
                Self::evaluate_either_or(medrecord, attributes, either, or)
            }
            Self::Exclude { operand } => Self::evaluate_exclude(medrecord, attributes, operand),
        }
    }

    #[inline]
    pub(crate) fn get_max<T>(
        mut attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
    ) -> MedRecordResult<(T, MedRecordAttribute)> {
        let max_attribute = attributes.next().ok_or(MedRecordError::QueryError(
            "No attributes to compare".to_string(),
        ))?;

        attributes.try_fold(max_attribute, |max_attribute, attribute| {
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
        })
    }

    #[inline]
    pub(crate) fn get_min<T>(
        mut attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
    ) -> MedRecordResult<(T, MedRecordAttribute)> {
        let min_attribute = attributes.next().ok_or(MedRecordError::QueryError(
            "No attributes to compare".to_string(),
        ))?;

        attributes.try_fold(min_attribute, |min_attribute, attribute| {
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
        })
    }

    #[inline]
    pub(crate) fn get_count<T>(
        attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
    ) -> MedRecordAttribute {
        MedRecordAttribute::Int(attributes.count() as i64)
    }

    #[inline]
    // 🥊💥
    pub(crate) fn get_sum<T>(
        mut attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
    ) -> MedRecordResult<MedRecordAttribute> {
        let first_attribute = attributes.next().ok_or(MedRecordError::QueryError(
            "No attributes to compare".to_string(),
        ))?;

        attributes.try_fold(first_attribute.1, |sum, (_, attribute)| {
            let first_dtype = DataType::from(&sum);
            let second_dtype = DataType::from(&attribute);

            sum.add(attribute).map_err(|_| {
                MedRecordError::QueryError(format!(
                    "Cannot add attributes of data types {} and {}. Consider narrowing down the attributes using .is_string() or .is_int()",
                    first_dtype, second_dtype
                ))
            })
        })
    }

    #[inline]
    pub(crate) fn get_first<T>(
        mut attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
    ) -> MedRecordResult<(T, MedRecordAttribute)> {
        attributes.next().ok_or(MedRecordError::QueryError(
            "No attributes to get the first".to_string(),
        ))
    }

    #[inline]
    pub(crate) fn get_last<T>(
        attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
    ) -> MedRecordResult<(T, MedRecordAttribute)> {
        attributes.last().ok_or(MedRecordError::QueryError(
            "No attributes to get the first".to_string(),
        ))
    }

    #[inline]
    fn evaluate_attribute_operation<'a, T: 'a + Clone>(
        medrecord: &'a MedRecord,
        attribtues: impl Iterator<Item = (T, MedRecordAttribute)>,
        operand: &Wrapper<SingleAttributeOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, (T, MedRecordAttribute)>> {
        let kind = &operand.0.read_or_panic().kind;

        let attributes = attribtues.collect::<Vec<_>>();

        let attribute = get_single_operand_attribute!(kind, attributes.clone().into_iter());

        Ok(match operand.evaluate(medrecord, attribute)? {
            Some(_) => Box::new(attributes.into_iter()),
            None => Box::new(std::iter::empty()),
        })
    }

    #[inline]
    fn evaluate_single_attribute_comparison_operation<'a, T>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (T, MedRecordAttribute)> + 'a,
        comparison_operand: &SingleAttributeComparisonOperand,
        kind: &SingleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, (T, MedRecordAttribute)>> {
        let comparison_attribute =
            get_single_attribute_comparison_operand_attribute!(comparison_operand, medrecord);

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
    fn evaluate_multiple_attributes_comparison_operation<'a, T>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (T, MedRecordAttribute)> + 'a,
        comparison_operand: &MultipleAttributesComparisonOperand,
        kind: &MultipleComparisonKind,
    ) -> MedRecordResult<BoxedIterator<'a, (T, MedRecordAttribute)>> {
        let comparison_attributes = match comparison_operand {
            MultipleAttributesComparisonOperand::Operand(operand) => {
                let context = &operand.context.context;
                let kind = &operand.kind;

                let comparison_attributes = context.get_attributes(medrecord)?;

                let comparison_attributes: Box<dyn Iterator<Item = (_, MedRecordAttribute)>> =
                    get_multiple_operand_attributes!(kind, comparison_attributes);

                operand
                    .evaluate(medrecord, comparison_attributes)?
                    .map(|(_, attribute)| attribute)
                    .collect::<Vec<_>>()
            }
            MultipleAttributesComparisonOperand::Attributes(attributes) => attributes.clone(),
        };

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
    fn evaluate_binary_arithmetic_operation<T>(
        medrecord: &MedRecord,
        attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
        operand: &SingleAttributeComparisonOperand,
        kind: &BinaryArithmeticKind,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)>> {
        let arithmetic_attribute =
            get_single_attribute_comparison_operand_attribute!(operand, medrecord);

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

        // TODO: This is a temporary solution. It should be optimized.
        Ok(attributes.collect::<MedRecordResult<Vec<_>>>()?.into_iter())
    }

    #[inline]
    fn evaluate_unary_arithmetic_operation<T>(
        attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
        kind: UnaryArithmeticKind,
    ) -> impl Iterator<Item = (T, MedRecordAttribute)> {
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

    pub(crate) fn get_values<'a, T: 'a + Eq + Clone + Hash + GetAttributes + Display>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordValue)>> {
        Ok(attributes
            .map(|(index, attribute)| {
                let value = index.get_attributes(medrecord)?.get(&attribute).ok_or(
                    MedRecordError::QueryError(format!(
                        "Cannot find attribute {} for index {}",
                        attribute, index
                    )),
                )?;

                Ok((index.clone(), value.clone()))
            })
            .collect::<MedRecordResult<Vec<_>>>()?
            .into_iter())
    }

    #[inline]
    fn evaluate_to_values<'a, T: 'a + Eq + Clone + Hash + GetAttributes + Display>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (T, MedRecordAttribute)> + 'a,
        operand: &Wrapper<MultipleValuesOperand>,
    ) -> MedRecordResult<impl Iterator<Item = (T, MedRecordAttribute)> + 'a> {
        let attributes = attributes.collect::<Vec<_>>();

        let values = Self::get_values(medrecord, attributes.clone().into_iter())?;

        let mut attributes = attributes.into_iter().collect::<HashMap<_, _>>();

        let values = operand.evaluate(medrecord, values.into_iter())?;

        Ok(values.map(move |(index, _)| {
            (
                index.clone(),
                attributes.remove(&index).expect("Attribute must exist"),
            )
        }))
    }

    #[inline]
    fn evaluate_slice<T>(
        attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
        range: Range<usize>,
    ) -> impl Iterator<Item = (T, MedRecordAttribute)> {
        attributes.map(move |(t, attribute)| (t, attribute.slice(range.clone())))
    }

    #[inline]
    fn evaluate_either_or<'a, T: 'a + Eq + Clone + Hash + GetAttributes + Display>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
        either: &Wrapper<MultipleAttributesOperand>,
        or: &Wrapper<MultipleAttributesOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, (T, MedRecordAttribute)>> {
        let attributes = attributes.collect::<Vec<_>>();

        let either_attributes = either.evaluate(medrecord, attributes.clone().into_iter())?;
        let or_attributes = or.evaluate(medrecord, attributes.into_iter())?;

        Ok(Box::new(
            either_attributes
                .chain(or_attributes)
                .unique_by(|attribute| attribute.0.clone()),
        ))
    }

    #[inline]
    fn evaluate_exclude<'a, T: 'a + Eq + Clone + Hash + GetAttributes + Display>(
        medrecord: &'a MedRecord,
        attributes: impl Iterator<Item = (T, MedRecordAttribute)>,
        operand: &Wrapper<MultipleAttributesOperand>,
    ) -> MedRecordResult<BoxedIterator<'a, (T, MedRecordAttribute)>> {
        let attributes = attributes.collect::<Vec<_>>();

        let result = operand
            .evaluate(medrecord, attributes.clone().into_iter())?
            .map(|(index, _)| index)
            .collect::<HashSet<_>>();

        Ok(Box::new(
            attributes
                .into_iter()
                .filter(move |(index, _)| !result.contains(index)),
        ))
    }
}

#[derive(Debug, Clone)]
pub enum SingleAttributeOperation {
    SingleAttributeComparisonOperation {
        operand: SingleAttributeComparisonOperand,
        kind: SingleComparisonKind,
    },
    MultipleAttributesComparisonOperation {
        operand: MultipleAttributesComparisonOperand,
        kind: MultipleComparisonKind,
    },
    BinaryArithmeticOpration {
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
        either: Wrapper<SingleAttributeOperand>,
        or: Wrapper<SingleAttributeOperand>,
    },
    Exclude {
        operand: Wrapper<SingleAttributeOperand>,
    },
}

impl DeepClone for SingleAttributeOperation {
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

impl SingleAttributeOperation {
    pub(crate) fn evaluate<'a, T>(
        &self,
        medrecord: &MedRecord,
        attribute: OptionalIndexWrapper<T, MedRecordAttribute>,
    ) -> MedRecordResult<Option<OptionalIndexWrapper<T, MedRecordAttribute>>> {
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
            Self::BinaryArithmeticOpration { operand, kind } => {
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
            Self::Exclude { operand } => {
                Ok(match operand.evaluate(medrecord, attribute.clone())? {
                    Some(_) => None,
                    None => Some(attribute),
                })
            }
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
            get_single_attribute_comparison_operand_attribute!(comparison_operand, medrecord);

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
        let comparison_attributes = match comparison_operand {
            MultipleAttributesComparisonOperand::Operand(operand) => {
                let context = &operand.context.context;
                let kind = &operand.kind;

                let comparison_attributes = context.get_attributes(medrecord)?;

                let comparison_attributes: Box<dyn Iterator<Item = (_, MedRecordAttribute)>> =
                    get_multiple_operand_attributes!(kind, comparison_attributes);

                operand
                    .evaluate(medrecord, comparison_attributes)?
                    .map(|(_, attribute)| attribute)
                    .collect::<Vec<_>>()
            }
            MultipleAttributesComparisonOperand::Attributes(attributes) => attributes.clone(),
        };

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
            get_single_attribute_comparison_operand_attribute!(operand, medrecord);

        match kind {
            BinaryArithmeticKind::Add => attribute.add(arithmetic_attribute),
            BinaryArithmeticKind::Sub => attribute.sub(arithmetic_attribute),
            BinaryArithmeticKind::Mul => attribute.mul(arithmetic_attribute),
            BinaryArithmeticKind::Pow => attribute.pow(arithmetic_attribute),
            BinaryArithmeticKind::Mod => attribute.r#mod(arithmetic_attribute),
        }
        .map(Some)
    }

    #[inline]
    fn evaluate_either_or(
        medrecord: &MedRecord,
        attribute: MedRecordAttribute,
        either: &Wrapper<SingleAttributeOperand>,
        or: &Wrapper<SingleAttributeOperand>,
    ) -> MedRecordResult<Option<MedRecordAttribute>> {
        let either_result = either.evaluate(medrecord, attribute.clone())?;
        let or_result = or.evaluate(medrecord, attribute)?;

        match (either_result, or_result) {
            (Some(either_result), _) => Ok(Some(either_result)),
            (None, Some(or_result)) => Ok(Some(or_result)),
            _ => Ok(None),
        }
    }
}
