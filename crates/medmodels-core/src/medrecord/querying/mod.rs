pub mod attributes;
pub mod edges;
pub mod nodes;
pub mod traits;
pub mod values;
pub mod wrapper;

use super::{
    AttributesTreeOperand, EdgeIndex, MedRecord, MedRecordAttribute, MedRecordValue,
    MultipleAttributesOperand, MultipleValuesOperand, NodeIndex, SingleAttributeOperand,
    SingleValueOperand, Wrapper,
};
use crate::errors::MedRecordResult;
use attributes::{AttributesTreeOperation, GetAttributes, MultipleAttributesOperation};
use edges::{EdgeIndexOperand, EdgeIndicesOperand, EdgeIndicesOperation, EdgeOperand};
use nodes::{NodeIndexOperand, NodeIndicesOperand, NodeIndicesOperation, NodeOperand};
use std::{fmt::Display, hash::Hash};
use traits::ReadWriteOrPanic;
use values::{GetValues, MultipleValuesOperation};

pub(crate) type BoxedIterator<'a, T> = Box<dyn Iterator<Item = T> + 'a>;

pub trait Index: Eq + Hash + Display + Clone {}

impl Index for NodeIndex {}

impl Index for EdgeIndex {}

pub trait Operand: GetAttributes<Self::Index> + GetValues<Self::Index> + Clone {
    type Index: Index;
}

impl Operand for NodeOperand {
    type Index = NodeIndex;
}

impl Operand for EdgeOperand {
    type Index = EdgeIndex;
}

pub enum OptionalIndexWrapper<I, T> {
    WithIndex((I, T)),
    WithoutIndex(T),
}

#[derive(Debug, Clone)]
pub struct Selection<'a> {
    medrecord: &'a MedRecord,
    return_operand: ReturnOperand,
}

impl<'a> Selection<'a> {
    pub fn new_node<Q, R>(medrecord: &'a MedRecord, query: Q) -> Self
    where
        Q: FnOnce(&mut Wrapper<NodeOperand>) -> R,
        R: Into<ReturnOperand>,
    {
        let mut operand = Wrapper::<NodeOperand>::new();

        Self {
            medrecord,
            return_operand: query(&mut operand).into(),
        }
    }

    pub fn new_edge<Q, R>(medrecord: &'a MedRecord, query: Q) -> Self
    where
        Q: FnOnce(&mut Wrapper<EdgeOperand>) -> R,
        R: Into<ReturnOperand>,
    {
        let mut operand = Wrapper::<EdgeOperand>::new();

        Self {
            medrecord,
            return_operand: query(&mut operand).into(),
        }
    }

    pub fn evaluate(self) -> MedRecordResult<ReturnValue<'a>> {
        let result = match self.return_operand {
            ReturnOperand::AttributesTree(operand) => {
                ReturnValue::AttributesTree(Box::new(operand.evaluate(self.medrecord)?))
            }
            ReturnOperand::MultipleAttributes(operand) => {
                ReturnValue::MultipleAttributes(Box::new(operand.evaluate(self.medrecord)?))
            }
            ReturnOperand::SingleAttribute(operand) => {
                let operand = operand.0.read_or_panic();

                let context_context_attributes = operand
                    .context
                    .context
                    .context
                    .get_attributes(self.medrecord)?;
                let context_attributes = operand
                    .context
                    .context
                    .evaluate(self.medrecord, context_context_attributes)?;
                let attributes: Box<dyn Iterator<Item = (_, MedRecordAttribute)>> =
                    match operand.context.kind {
                        attributes::MultipleKind::Max => {
                            Box::new(AttributesTreeOperation::get_max(context_attributes)?)
                        }
                        attributes::MultipleKind::Min => {
                            Box::new(AttributesTreeOperation::get_min(context_attributes)?)
                        }
                        attributes::MultipleKind::Count => {
                            Box::new(AttributesTreeOperation::get_count(context_attributes)?)
                        }
                        attributes::MultipleKind::Sum => {
                            Box::new(AttributesTreeOperation::get_sum(context_attributes)?)
                        }
                        attributes::MultipleKind::First => {
                            Box::new(AttributesTreeOperation::get_first(context_attributes)?)
                        }
                        attributes::MultipleKind::Last => {
                            Box::new(AttributesTreeOperation::get_last(context_attributes)?)
                        }
                    };

                let attributes = operand.context.evaluate(self.medrecord, attributes)?;

                match operand.kind {
                    attributes::SingleKind::Max => {
                        let attribute = MultipleAttributesOperation::get_max(attributes)?;

                        ReturnValue::SingleAttributeWithIndex(
                            operand
                                .evaluate(self.medrecord, attribute.1)?
                                .map(|result| (attribute.0, result)),
                        )
                    }
                    attributes::SingleKind::Min => {
                        let attribute = MultipleAttributesOperation::get_min(attributes)?;

                        ReturnValue::SingleAttributeWithIndex(
                            operand
                                .evaluate(self.medrecord, attribute.1)?
                                .map(|result| (attribute.0, result)),
                        )
                    }
                    attributes::SingleKind::Count => {
                        ReturnValue::SingleAttribute(operand.evaluate(
                            self.medrecord,
                            MultipleAttributesOperation::get_count(attributes),
                        )?)
                    }
                    attributes::SingleKind::Sum => ReturnValue::SingleAttribute(operand.evaluate(
                        self.medrecord,
                        MultipleAttributesOperation::get_sum(attributes)?,
                    )?),
                    attributes::SingleKind::First => {
                        let attribute = MultipleAttributesOperation::get_first(attributes)?;

                        ReturnValue::SingleAttributeWithIndex(
                            operand
                                .evaluate(self.medrecord, attribute.1)?
                                .map(|result| (attribute.0, result)),
                        )
                    }
                    attributes::SingleKind::Last => {
                        let attribute = MultipleAttributesOperation::get_last(attributes)?;

                        ReturnValue::SingleAttributeWithIndex(
                            operand
                                .evaluate(self.medrecord, attribute.1)?
                                .map(|result| (attribute.0, result)),
                        )
                    }
                }
            }
            ReturnOperand::EdgeIndices(operand) => {
                let operand = operand.0.read_or_panic();

                // TODO: This is a temporary solution. It should be optimized.
                let indices = operand.context.evaluate(self.medrecord)?.cloned();

                ReturnValue::EdgeIndices(Box::new(operand.evaluate(self.medrecord, indices)?))
            }
            ReturnOperand::EdgeIndex(operand) => {
                let operand = operand.0.read_or_panic();

                // TODO: This is a temporary solution. It should be optimized.
                let context_indices = operand.context.context.evaluate(self.medrecord)?.cloned();
                let indices = operand.context.evaluate(self.medrecord, context_indices)?;

                let index = match operand.kind {
                    edges::SingleKind::Max => EdgeIndicesOperation::get_max(indices)?,
                    edges::SingleKind::Min => EdgeIndicesOperation::get_min(indices)?,
                    edges::SingleKind::Count => EdgeIndicesOperation::get_count(indices),
                    edges::SingleKind::Sum => EdgeIndicesOperation::get_sum(indices),
                    edges::SingleKind::First => EdgeIndicesOperation::get_first(indices)?,
                    edges::SingleKind::Last => EdgeIndicesOperation::get_last(indices)?,
                };

                ReturnValue::EdgeIndex(operand.evaluate(self.medrecord, index)?)
            }
            ReturnOperand::NodeIndices(operand) => {
                let operand = operand.0.read_or_panic();

                let indices = operand.context.evaluate(self.medrecord)?.cloned();

                ReturnValue::NodeIndices(Box::new(operand.evaluate(self.medrecord, indices)?))
            }
            ReturnOperand::NodeIndex(operand) => {
                let operand = operand.0.read_or_panic();

                // TODO: This is a temporary solution. It should be optimized.
                let context_indices = operand.context.context.evaluate(self.medrecord)?.cloned();
                let indices = operand.context.evaluate(self.medrecord, context_indices)?;

                let index = match operand.kind {
                    nodes::SingleKind::Max => NodeIndicesOperation::get_max(indices)?.clone(),
                    nodes::SingleKind::Min => NodeIndicesOperation::get_min(indices)?.clone(),
                    nodes::SingleKind::Count => NodeIndicesOperation::get_count(indices),
                    nodes::SingleKind::Sum => NodeIndicesOperation::get_sum(indices)?,
                    nodes::SingleKind::First => NodeIndicesOperation::get_first(indices)?,
                    nodes::SingleKind::Last => NodeIndicesOperation::get_last(indices)?,
                };

                ReturnValue::NodeIndex(operand.evaluate(self.medrecord, index)?)
            }
            ReturnOperand::MultipleValues(operand) => {
                let operand = operand.0.read_or_panic();
                let attribute = operand.attribute.clone();

                let values = operand.context.get_values(self.medrecord, attribute)?;

                ReturnValue::MultipleValues(Box::new(operand.evaluate(self.medrecord, values)?))
            }
            ReturnOperand::SingleValue(operand) => {
                let operand = operand.0.read_or_panic();

                let attribute = operand.context.attribute.clone();

                let context_values = operand
                    .context
                    .context
                    .get_values(self.medrecord, attribute)?;
                let values = operand.context.evaluate(self.medrecord, context_values)?;

                match operand.kind {
                    values::SingleKind::Max => {
                        let value = MultipleValuesOperation::get_max(values)?;

                        ReturnValue::SingleValueWithIndex(
                            operand
                                .evaluate(self.medrecord, value.1)?
                                .map(|result| (value.0, result)),
                        )
                    }
                    values::SingleKind::Min => {
                        let value = MultipleValuesOperation::get_min(values)?;

                        ReturnValue::SingleValueWithIndex(
                            operand
                                .evaluate(self.medrecord, value.1)?
                                .map(|result| (value.0, result)),
                        )
                    }
                    values::SingleKind::Mean => ReturnValue::SingleValue(
                        operand
                            .evaluate(self.medrecord, MultipleValuesOperation::get_mean(values)?)?,
                    ),
                    values::SingleKind::Median => {
                        ReturnValue::SingleValue(operand.evaluate(
                            self.medrecord,
                            MultipleValuesOperation::get_median(values)?,
                        )?)
                    }
                    values::SingleKind::Mode => ReturnValue::SingleValue(
                        operand
                            .evaluate(self.medrecord, MultipleValuesOperation::get_mode(values)?)?,
                    ),
                    values::SingleKind::Std => ReturnValue::SingleValue(
                        operand
                            .evaluate(self.medrecord, MultipleValuesOperation::get_std(values)?)?,
                    ),
                    values::SingleKind::Var => ReturnValue::SingleValue(
                        operand
                            .evaluate(self.medrecord, MultipleValuesOperation::get_var(values)?)?,
                    ),
                    values::SingleKind::Count => ReturnValue::SingleValue(
                        operand
                            .evaluate(self.medrecord, MultipleValuesOperation::get_count(values))?,
                    ),
                    values::SingleKind::Sum => ReturnValue::SingleValue(
                        operand
                            .evaluate(self.medrecord, MultipleValuesOperation::get_sum(values)?)?,
                    ),
                    values::SingleKind::First => {
                        let value = MultipleValuesOperation::get_first(values)?;

                        ReturnValue::SingleValueWithIndex(
                            operand
                                .evaluate(self.medrecord, value.1)?
                                .map(|result| (value.0, result)),
                        )
                    }
                    values::SingleKind::Last => {
                        let value = MultipleValuesOperation::get_last(values)?;

                        ReturnValue::SingleValueWithIndex(
                            operand
                                .evaluate(self.medrecord, value.1)?
                                .map(|result| (value.0, result)),
                        )
                    }
                }
            }
        };

        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub enum ReturnOperand {
    AttributesTree(Wrapper<AttributesTreeOperand>),
    MultipleAttributes(Wrapper<MultipleAttributesOperand>),
    SingleAttribute(Wrapper<SingleAttributeOperand>),
    EdgeIndices(Wrapper<EdgeIndicesOperand>),
    EdgeIndex(Wrapper<EdgeIndexOperand>),
    NodeIndices(Wrapper<NodeIndicesOperand>),
    NodeIndex(Wrapper<NodeIndexOperand>),
    MultipleValues(Wrapper<MultipleValuesOperand>),
    SingleValue(Wrapper<SingleValueOperand>),
}

impl From<Wrapper<AttributesTreeOperand>> for ReturnOperand {
    fn from(operand: Wrapper<AttributesTreeOperand>) -> Self {
        Self::AttributesTree(operand)
    }
}

impl From<Wrapper<MultipleAttributesOperand>> for ReturnOperand {
    fn from(operand: Wrapper<MultipleAttributesOperand>) -> Self {
        Self::MultipleAttributes(operand)
    }
}

impl From<Wrapper<SingleAttributeOperand>> for ReturnOperand {
    fn from(operand: Wrapper<SingleAttributeOperand>) -> Self {
        Self::SingleAttribute(operand)
    }
}

impl From<Wrapper<EdgeIndicesOperand>> for ReturnOperand {
    fn from(operand: Wrapper<EdgeIndicesOperand>) -> Self {
        Self::EdgeIndices(operand)
    }
}

impl From<Wrapper<EdgeIndexOperand>> for ReturnOperand {
    fn from(operand: Wrapper<EdgeIndexOperand>) -> Self {
        Self::EdgeIndex(operand)
    }
}

impl From<Wrapper<NodeIndicesOperand>> for ReturnOperand {
    fn from(operand: Wrapper<NodeIndicesOperand>) -> Self {
        Self::NodeIndices(operand)
    }
}

impl From<Wrapper<NodeIndexOperand>> for ReturnOperand {
    fn from(operand: Wrapper<NodeIndexOperand>) -> Self {
        Self::NodeIndex(operand)
    }
}

impl From<Wrapper<MultipleValuesOperand>> for ReturnOperand {
    fn from(operand: Wrapper<MultipleValuesOperand>) -> Self {
        Self::MultipleValues(operand)
    }
}

impl From<Wrapper<SingleValueOperand>> for ReturnOperand {
    fn from(operand: Wrapper<SingleValueOperand>) -> Self {
        Self::SingleValue(operand)
    }
}

pub enum ReturnValue<'a> {
    AttributesTree(Box<dyn Iterator<Item = (Index<'a>, Vec<MedRecordAttribute>)> + 'a>),
    MultipleAttributes(Box<dyn Iterator<Item = (Index<'a>, MedRecordAttribute)> + 'a>),
    SingleAttributeWithIndex(Option<(Index<'a>, MedRecordAttribute)>),
    SingleAttribute(Option<MedRecordAttribute>),
    EdgeIndices(Box<dyn Iterator<Item = EdgeIndex> + 'a>),
    EdgeIndex(Option<EdgeIndex>),
    NodeIndices(Box<dyn Iterator<Item = NodeIndex> + 'a>),
    NodeIndex(Option<NodeIndex>),
    MultipleValues(Box<dyn Iterator<Item = (Index<'a>, MedRecordValue)> + 'a>),
    SingleValueWithIndex(Option<(Index<'a>, MedRecordValue)>),
    SingleValue(Option<MedRecordValue>),
}
