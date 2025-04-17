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
use std::{
    fmt::{Debug, Display},
    hash::Hash,
};
use traits::ReadWriteOrPanic;
use values::{GetValues, MultipleValuesOperation};

pub(crate) type BoxedIterator<'a, T> = Box<dyn Iterator<Item = T> + 'a>;

pub trait Index: Eq + Hash + Display + Clone {}

impl Index for NodeIndex {}

impl Index for EdgeIndex {}

pub trait Operand: GetAttributes<Self::Index> + GetValues<Self::Index> + Debug + Clone {
    type Index: Index;
}

impl Operand for NodeOperand {
    type Index = NodeIndex;
}

impl Operand for EdgeOperand {
    type Index = EdgeIndex;
}

impl<'a, I: Index> Index for &'a I {}

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

    pub fn evaluate<I: Index>(self) -> MedRecordResult<ReturnValue<I>> {
        todo!()
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

pub enum ReturnValue<I: Index> {
    AttributesTree(Box<dyn Iterator<Item = (I, Vec<MedRecordAttribute>)>>),
    MultipleAttributes(Box<dyn Iterator<Item = (I, MedRecordAttribute)>>),
    SingleAttributeWithIndex(Option<(I, MedRecordAttribute)>),
    SingleAttribute(Option<MedRecordAttribute>),
    EdgeIndices(Box<dyn Iterator<Item = EdgeIndex>>),
    EdgeIndex(Option<EdgeIndex>),
    NodeIndices(Box<dyn Iterator<Item = NodeIndex>>),
    NodeIndex(Option<NodeIndex>),
    MultipleValues(Box<dyn Iterator<Item = (I, MedRecordValue)>>),
    SingleValueWithIndex(Option<(I, MedRecordValue)>),
    SingleValue(Option<MedRecordValue>),
}
