#![allow(dead_code)]
// TODO: Remove this once the file is complete

use super::operation::{EdgeOperation, EdgeValueOperation, EdgeValuesOperation};
use crate::{
    medrecord::{
        querying::{
            nodes::NodeOperand,
            traits::{DeepClone, EvaluateOperand, EvaluateOperation, ReadWriteOrPanic},
            values::{ComparisonOperand, ValueKind, ValuesKind},
            wrapper::Wrapper,
        },
        CardinalityWrapper, EdgeIndex, Group, MedRecordAttribute,
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

    pub fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<EdgeValuesOperand> {
        let operand =
            Wrapper::<EdgeValuesOperand>::new(self.deep_clone(), ValuesKind::Attribute(attribute));

        self.operations.push(EdgeOperation::Attribute {
            operand: operand.clone(),
        });

        operand
    }

    pub fn in_group<G>(&mut self, group: G)
    where
        G: Into<CardinalityWrapper<Group>>,
    {
        self.operations.push(EdgeOperation::InGroup {
            group: group.into(),
        });
    }

    pub fn has_attribute<A>(&mut self, attribute: A)
    where
        A: Into<CardinalityWrapper<MedRecordAttribute>>,
    {
        self.operations.push(EdgeOperation::HasAttribute {
            attribute: attribute.into(),
        });
    }

    pub fn source_node(&mut self) -> Wrapper<NodeOperand> {
        let operand = Wrapper::<NodeOperand>::new();

        self.operations.push(EdgeOperation::SourceNode {
            operand: operand.clone(),
        });

        operand
    }

    pub fn target_node(&mut self) -> Wrapper<NodeOperand> {
        let operand = Wrapper::<NodeOperand>::new();

        self.operations.push(EdgeOperation::TargetNode {
            operand: operand.clone(),
        });

        operand
    }
}

impl Wrapper<EdgeOperand> {
    pub(crate) fn new() -> Self {
        EdgeOperand::new().into()
    }

    pub fn attribute<A>(&self, attribute: A) -> Wrapper<EdgeValuesOperand>
    where
        A: Into<MedRecordAttribute>,
    {
        self.0.write_or_panic().attribute(attribute.into())
    }

    pub fn in_group<G>(&mut self, group: G)
    where
        G: Into<CardinalityWrapper<Group>>,
    {
        self.0.write_or_panic().in_group(group);
    }

    pub fn has_attribute<A>(&mut self, attribute: A)
    where
        A: Into<CardinalityWrapper<MedRecordAttribute>>,
    {
        self.0.write_or_panic().has_attribute(attribute);
    }

    pub fn source_node(&self) -> Wrapper<NodeOperand> {
        self.0.write_or_panic().source_node()
    }

    pub fn target_node(&self) -> Wrapper<NodeOperand> {
        self.0.write_or_panic().target_node()
    }
}

#[derive(Debug, Clone)]
pub struct EdgeValuesOperand {
    context: EdgeOperand,
    kind: ValuesKind,
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
            kind: self.kind.clone(),
            operations: self
                .operations
                .iter()
                .map(|operation| operation.deep_clone())
                .collect(),
        }
    }
}

impl EdgeValuesOperand {
    pub(crate) fn new(context: EdgeOperand, kind: ValuesKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    pub fn max(&mut self) -> Wrapper<EdgeValueOperand> {
        let operand = Wrapper::<EdgeValueOperand>::new(ValueKind::Max(self.kind.clone()));

        self.operations.push(EdgeValuesOperation::Max {
            operand: operand.clone(),
        });

        operand
    }
}

impl Wrapper<EdgeValuesOperand> {
    pub(crate) fn new(context: EdgeOperand, kind: ValuesKind) -> Self {
        EdgeValuesOperand::new(context, kind).into()
    }

    pub fn max(&self) -> Wrapper<EdgeValueOperand> {
        self.0.write_or_panic().max()
    }
}
#[derive(Debug, Clone)]
pub struct EdgeValueOperand {
    kind: ValueKind,
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
            kind: self.kind.clone(),
            operations: self
                .operations
                .iter()
                .map(|operation| operation.deep_clone())
                .collect(),
        }
    }
}

impl EdgeValueOperand {
    pub(crate) fn new(kind: ValueKind) -> Self {
        Self {
            kind,
            operations: Vec::new(),
        }
    }

    pub fn less_than(&mut self, comparison: ComparisonOperand) {
        self.operations.push(EdgeValueOperation::LessThan {
            operand: comparison,
            kind: self.kind.clone(),
        });
    }
}

impl Wrapper<EdgeValueOperand> {
    pub(crate) fn new(kind: ValueKind) -> Self {
        EdgeValueOperand::new(kind).into()
    }

    pub fn less_than<O>(&self, comparison: O)
    where
        O: Into<ComparisonOperand>,
    {
        self.0.write_or_panic().less_than(comparison.into());
    }
}
