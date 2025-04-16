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
use attributes::GetAttributes;
use edges::{EdgeIndexOperand, EdgeIndicesOperand, EdgeOperand};
use nodes::{NodeIndexOperand, NodeIndicesOperand, NodeOperand};
use std::{fmt::Display, hash::Hash};
use values::GetValues;

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

impl<I, T> OptionalIndexWrapper<I, T> {
    pub fn get_value(&self) -> &T {
        match self {
            OptionalIndexWrapper::WithIndex((_, value)) => value,
            OptionalIndexWrapper::WithoutIndex(value) => value,
        }
    }

    pub fn get_index(&self) -> Option<&I> {
        match self {
            OptionalIndexWrapper::WithIndex((index, _)) => Some(index),
            OptionalIndexWrapper::WithoutIndex(_) => None,
        }
    }

    pub fn unpack(self) -> (Option<I>, T) {
        match self {
            OptionalIndexWrapper::WithIndex((index, value)) => (Some(index), value),
            OptionalIndexWrapper::WithoutIndex(value) => (None, value),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Selection<'a, O: Operand> {
    medrecord: &'a MedRecord,
    return_operand: ReturnOperand<O>,
}

impl<'a, O: Operand> Selection<'a, O> {
    pub fn new_node<Q, R>(medrecord: &'a MedRecord, query: Q) -> Self
    where
        Q: FnOnce(&mut Wrapper<NodeOperand>) -> R,
        R: Into<ReturnOperand<O>>,
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
        R: Into<ReturnOperand<O>>,
    {
        let mut operand = Wrapper::<EdgeOperand>::new();

        Self {
            medrecord,
            return_operand: query(&mut operand).into(),
        }
    }

    pub fn evaluate(self) -> MedRecordResult<ReturnValue<'a, O>>
    where
        O: 'a,
    {
        Ok(match self.return_operand {
            ReturnOperand::AttributesTree(operand) => {
                ReturnValue::AttributesTree(Box::new(operand.evaluate(self.medrecord)?))
            }
            ReturnOperand::MultipleAttributes(operand) => {
                ReturnValue::MultipleAttributes(Box::new(operand.evaluate(self.medrecord)?))
            }
            ReturnOperand::SingleAttribute(operand) => {
                ReturnValue::SingleAttribute(operand.evaluate(self.medrecord)?)
            }
            ReturnOperand::EdgeIndices(operand) => {
                ReturnValue::EdgeIndices(Box::new(operand.evaluate(self.medrecord)?))
            }
            ReturnOperand::EdgeIndex(operand) => {
                ReturnValue::EdgeIndex(operand.evaluate(self.medrecord)?)
            }
            ReturnOperand::NodeIndices(operand) => {
                ReturnValue::NodeIndices(Box::new(operand.evaluate(self.medrecord)?))
            }
            ReturnOperand::NodeIndex(operand) => {
                ReturnValue::NodeIndex(operand.evaluate(self.medrecord)?)
            }
            ReturnOperand::MultipleValues(operand) => {
                ReturnValue::MultipleValues(Box::new(operand.evaluate(self.medrecord)?))
            }
            ReturnOperand::SingleValue(operand) => {
                ReturnValue::SingleValue(operand.evaluate(self.medrecord)?)
            }
        })
    }
}

#[derive(Debug, Clone)]
pub enum ReturnOperand<O: Operand> {
    AttributesTree(Wrapper<AttributesTreeOperand<O>>),
    MultipleAttributes(Wrapper<MultipleAttributesOperand<O>>),
    SingleAttribute(Wrapper<SingleAttributeOperand<O>>),
    EdgeIndices(Wrapper<EdgeIndicesOperand>),
    EdgeIndex(Wrapper<EdgeIndexOperand>),
    NodeIndices(Wrapper<NodeIndicesOperand>),
    NodeIndex(Wrapper<NodeIndexOperand>),
    MultipleValues(Wrapper<MultipleValuesOperand<O>>),
    SingleValue(Wrapper<SingleValueOperand<O>>),
}

impl<O: Operand> From<Wrapper<AttributesTreeOperand<O>>> for ReturnOperand<O> {
    fn from(operand: Wrapper<AttributesTreeOperand<O>>) -> Self {
        Self::AttributesTree(operand)
    }
}

impl<O: Operand> From<Wrapper<MultipleAttributesOperand<O>>> for ReturnOperand<O> {
    fn from(operand: Wrapper<MultipleAttributesOperand<O>>) -> Self {
        Self::MultipleAttributes(operand)
    }
}

impl<O: Operand> From<Wrapper<SingleAttributeOperand<O>>> for ReturnOperand<O> {
    fn from(operand: Wrapper<SingleAttributeOperand<O>>) -> Self {
        Self::SingleAttribute(operand)
    }
}

impl<O: Operand> From<Wrapper<EdgeIndicesOperand>> for ReturnOperand<O> {
    fn from(operand: Wrapper<EdgeIndicesOperand>) -> Self {
        Self::EdgeIndices(operand)
    }
}

impl<O: Operand> From<Wrapper<EdgeIndexOperand>> for ReturnOperand<O> {
    fn from(operand: Wrapper<EdgeIndexOperand>) -> Self {
        Self::EdgeIndex(operand)
    }
}

impl<O: Operand> From<Wrapper<NodeIndicesOperand>> for ReturnOperand<O> {
    fn from(operand: Wrapper<NodeIndicesOperand>) -> Self {
        Self::NodeIndices(operand)
    }
}

impl<O: Operand> From<Wrapper<NodeIndexOperand>> for ReturnOperand<O> {
    fn from(operand: Wrapper<NodeIndexOperand>) -> Self {
        Self::NodeIndex(operand)
    }
}

impl<O: Operand> From<Wrapper<MultipleValuesOperand<O>>> for ReturnOperand<O> {
    fn from(operand: Wrapper<MultipleValuesOperand<O>>) -> Self {
        Self::MultipleValues(operand)
    }
}

impl<O: Operand> From<Wrapper<SingleValueOperand<O>>> for ReturnOperand<O> {
    fn from(operand: Wrapper<SingleValueOperand<O>>) -> Self {
        Self::SingleValue(operand)
    }
}

pub enum ReturnValue<'a, O: Operand> {
    AttributesTree(BoxedIterator<'a, (O::Index, Vec<MedRecordAttribute>)>),
    MultipleAttributes(BoxedIterator<'a, (O::Index, MedRecordAttribute)>),
    SingleAttribute(Option<OptionalIndexWrapper<O::Index, MedRecordAttribute>>),
    EdgeIndices(BoxedIterator<'a, EdgeIndex>),
    EdgeIndex(Option<EdgeIndex>),
    NodeIndices(BoxedIterator<'a, NodeIndex>),
    NodeIndex(Option<NodeIndex>),
    MultipleValues(BoxedIterator<'a, (O::Index, MedRecordValue)>),
    SingleValue(Option<OptionalIndexWrapper<O::Index, MedRecordValue>>),
}
