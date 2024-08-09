#![allow(dead_code)]
// TODO: Remove this once the file is complete

use super::operation::{EdgeOperation, EdgeValueOperation, EdgeValuesOperation};
use crate::{
    medrecord::{
        querying::{
            evaluate::{EvaluateOperand, EvaluateOperation},
            nodes::NodeOperand,
            traits::{DeepClone, ReadWriteOrPanic},
            values::ComparisonOperand,
            wrapper::Wrapper,
        },
        EdgeIndex, MedRecordAttribute,
    },
    MedRecord,
};
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct EdgeOperand {
    pub(crate) operations: Vec<EdgeOperation>,
}

impl EvaluateOperand for EdgeOperand {
    type Index = EdgeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        let edge_indices =
            Box::new(medrecord.edge_indices()) as Box<dyn Iterator<Item = &'a EdgeIndex>>;

        self.operations
            .iter()
            .fold(Box::new(edge_indices), |edge_indices, operation| {
                operation.evaluate(medrecord, edge_indices)
            })
    }
}

impl DeepClone for EdgeOperand {
    fn deep_clone(&self) -> Self {
        Self {
            operations: self
                .operations
                .iter()
                .map(|operation| operation.deep_clone())
                .collect(),
        }
    }
}

impl EdgeOperand {
    pub(crate) fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn connects_to<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<NodeOperand>),
    {
        let mut node_operand = Wrapper::<NodeOperand>::new();

        query(&mut node_operand);

        self.operations.push(EdgeOperation::ConnectsTo {
            operand: node_operand,
        });
    }

    pub fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<EdgeValuesOperand> {
        let operand = Wrapper::<EdgeValuesOperand>::new(self.deep_clone(), attribute);

        self.operations.push(EdgeOperation::Attribute {
            operand: operand.clone(),
        });

        operand
    }
}

impl Wrapper<EdgeOperand> {
    pub(crate) fn new() -> Self {
        EdgeOperand::new().into()
    }

    pub fn connects_to<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<NodeOperand>),
    {
        self.0.write_or_panic().connects_to(query);
    }

    pub fn attribute<A>(&self, attribute: A) -> Wrapper<EdgeValuesOperand>
    where
        A: Into<MedRecordAttribute>,
    {
        self.0.write_or_panic().attribute(attribute.into())
    }
}

#[derive(Debug, Clone)]
pub struct EdgeValuesOperand {
    context: EdgeOperand,
    pub(crate) attribute: MedRecordAttribute,
    operations: Vec<EdgeValuesOperation>,
}

impl EvaluateOperand for EdgeValuesOperand {
    type Index = EdgeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        let edge_indices = self.context.evaluate(medrecord);

        self.operations
            .iter()
            .fold(Box::new(edge_indices), |edge_indices, operation| {
                operation.evaluate(medrecord, edge_indices)
            })
    }
}

impl DeepClone for EdgeValuesOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            attribute: self.attribute.clone(),
            operations: self
                .operations
                .iter()
                .map(|operation| operation.deep_clone())
                .collect(),
        }
    }
}

impl EdgeValuesOperand {
    pub(crate) fn new(context: EdgeOperand, attribute: MedRecordAttribute) -> Self {
        Self {
            context,
            attribute,
            operations: Vec::new(),
        }
    }

    pub fn max(&mut self) -> Wrapper<EdgeValueOperand> {
        let mut operand = EdgeValueOperand::new(self.attribute.clone());

        let context = EdgeValueOperation::MaxContext {
            context: self.deep_clone(),
            attribute: self.attribute.clone(),
        };

        operand.operations.push(context);

        let operand = Wrapper::from(operand);

        self.operations.push(EdgeValuesOperation::Max {
            operand: operand.clone(),
        });

        operand
    }
}

impl Wrapper<EdgeValuesOperand> {
    pub(crate) fn new(context: EdgeOperand, attribute: MedRecordAttribute) -> Self {
        EdgeValuesOperand::new(context, attribute).into()
    }

    pub fn max(&self) -> Wrapper<EdgeValueOperand> {
        self.0.write_or_panic().max()
    }
}
#[derive(Debug, Clone)]
pub struct EdgeValueOperand {
    pub(crate) attribute: MedRecordAttribute,
    operations: Vec<EdgeValueOperation>,
}

impl EvaluateOperand for EdgeValueOperand {
    type Index = EdgeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        let edge_indices = medrecord.edge_indices();

        self.operations
            .iter()
            .fold(Box::new(edge_indices), |edge_indices, operation| {
                operation.evaluate(medrecord, edge_indices)
            })
    }
}

impl DeepClone for EdgeValueOperand {
    fn deep_clone(&self) -> Self {
        Self {
            attribute: self.attribute.clone(),
            operations: self
                .operations
                .iter()
                .map(|operation| operation.deep_clone())
                .collect(),
        }
    }
}

impl EdgeValueOperand {
    pub(crate) fn new(attribute: MedRecordAttribute) -> Self {
        Self {
            attribute,
            operations: Vec::new(),
        }
    }

    pub fn less_than(&mut self, comparison: ComparisonOperand) {
        self.operations.push(EdgeValueOperation::LessThan {
            operand: comparison,
            attribute: self.attribute.clone(),
        });
    }
}

impl Wrapper<EdgeValueOperand> {
    pub(crate) fn new(attribute: MedRecordAttribute) -> Self {
        EdgeValueOperand::new(attribute).into()
    }

    pub fn less_than<O>(&self, comparison: O)
    where
        O: Into<ComparisonOperand>,
    {
        self.0.write_or_panic().less_than(comparison.into());
    }
}
