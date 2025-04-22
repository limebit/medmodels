pub mod attributes;
pub mod edges;
pub mod nodes;
pub mod traits;
pub mod values;
pub mod wrapper;

use super::{EdgeIndex, MedRecord, MedRecordAttribute, MedRecordValue, NodeIndex, Wrapper};
use crate::errors::MedRecordResult;
use attributes::{
    EdgeAttributesTreeOperand, EdgeMultipleAttributesOperand, EdgeSingleAttributeOperand,
    GetAllAttributes, GetAttributes, NodeAttributesTreeOperand, NodeMultipleAttributesOperand,
    NodeSingleAttributeOperand,
};
use edges::{EdgeIndexOperand, EdgeIndicesOperand, EdgeOperand};
use nodes::{NodeIndexOperand, NodeIndicesOperand, NodeOperand};
use std::{
    fmt::{Debug, Display},
    hash::Hash,
};
use values::{
    EdgeMultipleValuesOperand, EdgeSingleValueOperand, GetValues, NodeMultipleValuesOperand,
    NodeSingleValueOperand,
};

pub trait Index: Eq + Clone + Hash + Display + GetAttributes {}

impl Index for NodeIndex {}

impl Index for EdgeIndex {}

impl<I: Index> Index for &I {}

pub trait Operand: GetAllAttributes<Self::Index> + GetValues<Self::Index> + Debug + Clone {
    type Index: Index;
}

impl Operand for NodeOperand {
    type Index = NodeIndex;
}

impl Operand for EdgeOperand {
    type Index = EdgeIndex;
}

pub(crate) type BoxedIterator<'a, T> = Box<dyn Iterator<Item = T> + 'a>;

#[derive(Debug, Clone)]
pub enum OptionalIndexWrapper<I: Index, T> {
    WithIndex((I, T)),
    WithoutIndex(T),
}

impl<I: Index, T> OptionalIndexWrapper<I, T> {
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

    pub fn map<U, F>(self, f: F) -> OptionalIndexWrapper<I, U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            OptionalIndexWrapper::WithIndex((index, value)) => {
                OptionalIndexWrapper::WithIndex((index, f(value)))
            }
            OptionalIndexWrapper::WithoutIndex(value) => {
                OptionalIndexWrapper::WithoutIndex(f(value))
            }
        }
    }
}

impl<I: Index, T> From<T> for OptionalIndexWrapper<I, T> {
    fn from(value: T) -> Self {
        OptionalIndexWrapper::WithoutIndex(value)
    }
}

impl<I: Index, T> From<(I, T)> for OptionalIndexWrapper<I, T> {
    fn from(value: (I, T)) -> Self {
        OptionalIndexWrapper::WithIndex(value)
    }
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
        let mut operand = Wrapper::<NodeOperand>::new(None);

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
        let mut operand = Wrapper::<EdgeOperand>::new(None);

        Self {
            medrecord,
            return_operand: query(&mut operand).into(),
        }
    }

    pub fn evaluate(self) -> MedRecordResult<ReturnValue<'a>> {
        let result = match self.return_operand {
            ReturnOperand::NodeAttributesTree(operand) => ReturnValue::NodeAttributesTree(
                Box::new(operand.evaluate_backward(self.medrecord)?),
            ),
            ReturnOperand::EdgeAttributesTree(operand) => ReturnValue::EdgeAttributesTree(
                Box::new(operand.evaluate_backward(self.medrecord)?),
            ),
            ReturnOperand::NodeMultipleAttributes(operand) => ReturnValue::NodeMultipleAttributes(
                Box::new(operand.evaluate_backward(self.medrecord)?),
            ),
            ReturnOperand::EdgeMultipleAttributes(operand) => ReturnValue::EdgeMultipleAttributes(
                Box::new(operand.evaluate_backward(self.medrecord)?),
            ),
            ReturnOperand::NodeSingleAttribute(operand) => {
                ReturnValue::NodeSingleAttribute(operand.evaluate_backward(self.medrecord)?)
            }
            ReturnOperand::EdgeSingleAttribute(operand) => {
                ReturnValue::EdgeSingleAttribute(operand.evaluate_backward(self.medrecord)?)
            }
            ReturnOperand::EdgeIndices(operand) => {
                ReturnValue::EdgeIndices(Box::new(operand.evaluate_backward(self.medrecord)?))
            }
            ReturnOperand::EdgeIndex(operand) => {
                ReturnValue::EdgeIndex(operand.evaluate_backward(self.medrecord)?)
            }
            ReturnOperand::NodeIndices(operand) => {
                ReturnValue::NodeIndices(Box::new(operand.evaluate_backward(self.medrecord)?))
            }
            ReturnOperand::NodeIndex(operand) => {
                ReturnValue::NodeIndex(operand.evaluate_backward(self.medrecord)?)
            }
            ReturnOperand::NodeMultipleValues(operand) => ReturnValue::NodeMultipleValues(
                Box::new(operand.evaluate_backward(self.medrecord)?),
            ),
            ReturnOperand::EdgeMultipleValues(operand) => ReturnValue::EdgeMultipleValues(
                Box::new(operand.evaluate_backward(self.medrecord)?),
            ),
            ReturnOperand::NodeSingleValue(operand) => {
                ReturnValue::NodeSingleValue(operand.evaluate_backward(self.medrecord)?)
            }
            ReturnOperand::EdgeSingleValue(operand) => {
                ReturnValue::EdgeSingleValue(operand.evaluate_backward(self.medrecord)?)
            }
        };

        Ok(result)
    }
}

#[derive(Debug, Clone)]
pub enum ReturnOperand {
    NodeAttributesTree(Wrapper<NodeAttributesTreeOperand>),
    EdgeAttributesTree(Wrapper<EdgeAttributesTreeOperand>),
    NodeMultipleAttributes(Wrapper<NodeMultipleAttributesOperand>),
    EdgeMultipleAttributes(Wrapper<EdgeMultipleAttributesOperand>),
    NodeSingleAttribute(Wrapper<NodeSingleAttributeOperand>),
    EdgeSingleAttribute(Wrapper<EdgeSingleAttributeOperand>),
    EdgeIndices(Wrapper<EdgeIndicesOperand>),
    EdgeIndex(Wrapper<EdgeIndexOperand>),
    NodeIndices(Wrapper<NodeIndicesOperand>),
    NodeIndex(Wrapper<NodeIndexOperand>),
    NodeMultipleValues(Wrapper<NodeMultipleValuesOperand>),
    EdgeMultipleValues(Wrapper<EdgeMultipleValuesOperand>),
    NodeSingleValue(Wrapper<NodeSingleValueOperand>),
    EdgeSingleValue(Wrapper<EdgeSingleValueOperand>),
}

impl From<Wrapper<NodeAttributesTreeOperand>> for ReturnOperand {
    fn from(operand: Wrapper<NodeAttributesTreeOperand>) -> Self {
        Self::NodeAttributesTree(operand)
    }
}

impl From<Wrapper<EdgeAttributesTreeOperand>> for ReturnOperand {
    fn from(operand: Wrapper<EdgeAttributesTreeOperand>) -> Self {
        Self::EdgeAttributesTree(operand)
    }
}

impl From<Wrapper<NodeMultipleAttributesOperand>> for ReturnOperand {
    fn from(operand: Wrapper<NodeMultipleAttributesOperand>) -> Self {
        Self::NodeMultipleAttributes(operand)
    }
}

impl From<Wrapper<EdgeMultipleAttributesOperand>> for ReturnOperand {
    fn from(operand: Wrapper<EdgeMultipleAttributesOperand>) -> Self {
        Self::EdgeMultipleAttributes(operand)
    }
}

impl From<Wrapper<NodeSingleAttributeOperand>> for ReturnOperand {
    fn from(operand: Wrapper<NodeSingleAttributeOperand>) -> Self {
        Self::NodeSingleAttribute(operand)
    }
}

impl From<Wrapper<EdgeSingleAttributeOperand>> for ReturnOperand {
    fn from(operand: Wrapper<EdgeSingleAttributeOperand>) -> Self {
        Self::EdgeSingleAttribute(operand)
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

impl From<Wrapper<NodeMultipleValuesOperand>> for ReturnOperand {
    fn from(operand: Wrapper<NodeMultipleValuesOperand>) -> Self {
        Self::NodeMultipleValues(operand)
    }
}

impl From<Wrapper<EdgeMultipleValuesOperand>> for ReturnOperand {
    fn from(operand: Wrapper<EdgeMultipleValuesOperand>) -> Self {
        Self::EdgeMultipleValues(operand)
    }
}

impl From<Wrapper<NodeSingleValueOperand>> for ReturnOperand {
    fn from(operand: Wrapper<NodeSingleValueOperand>) -> Self {
        Self::NodeSingleValue(operand)
    }
}

impl From<Wrapper<EdgeSingleValueOperand>> for ReturnOperand {
    fn from(operand: Wrapper<EdgeSingleValueOperand>) -> Self {
        Self::EdgeSingleValue(operand)
    }
}

pub enum ReturnValue<'a> {
    NodeAttributesTree(Box<dyn Iterator<Item = (&'a NodeIndex, Vec<MedRecordAttribute>)> + 'a>),
    EdgeAttributesTree(Box<dyn Iterator<Item = (&'a EdgeIndex, Vec<MedRecordAttribute>)> + 'a>),
    NodeMultipleAttributes(Box<dyn Iterator<Item = (&'a NodeIndex, MedRecordAttribute)> + 'a>),
    EdgeMultipleAttributes(Box<dyn Iterator<Item = (&'a EdgeIndex, MedRecordAttribute)> + 'a>),
    NodeSingleAttribute(Option<OptionalIndexWrapper<&'a NodeIndex, MedRecordAttribute>>),
    EdgeSingleAttribute(Option<OptionalIndexWrapper<&'a EdgeIndex, MedRecordAttribute>>),
    EdgeIndices(Box<dyn Iterator<Item = EdgeIndex> + 'a>),
    EdgeIndex(Option<EdgeIndex>),
    NodeIndices(Box<dyn Iterator<Item = NodeIndex> + 'a>),
    NodeIndex(Option<NodeIndex>),
    NodeMultipleValues(Box<dyn Iterator<Item = (&'a NodeIndex, MedRecordValue)> + 'a>),
    EdgeMultipleValues(Box<dyn Iterator<Item = (&'a EdgeIndex, MedRecordValue)> + 'a>),
    NodeSingleValue(Option<OptionalIndexWrapper<&'a NodeIndex, MedRecordValue>>),
    EdgeSingleValue(Option<OptionalIndexWrapper<&'a EdgeIndex, MedRecordValue>>),
}
