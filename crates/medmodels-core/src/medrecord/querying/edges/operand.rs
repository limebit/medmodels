use super::operation::EdgeOperation;
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            nodes::NodeOperand,
            traits::{DeepClone, ReadWriteOrPanic},
            values::{Context, MedRecordValuesOperand},
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

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        let edge_indices =
            Box::new(medrecord.edge_indices()) as Box<dyn Iterator<Item = &'a EdgeIndex>>;

        self.operations
            .iter()
            .try_fold(edge_indices, |edge_indices, operation| {
                operation.evaluate(medrecord, edge_indices)
            })
    }

    pub fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<MedRecordValuesOperand> {
        let operand = Wrapper::<MedRecordValuesOperand>::new(
            Context::EdgeOperand(self.deep_clone()),
            attribute,
        );

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

    pub(crate) fn evaluate<'a>(
        &'a self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        self.0.read_or_panic().evaluate(medrecord)
    }

    pub fn attribute<A>(&self, attribute: A) -> Wrapper<MedRecordValuesOperand>
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
