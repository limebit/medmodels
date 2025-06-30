use super::NodeOperand;
use crate::{
    errors::MedRecordResult,
    medrecord::querying::{
        edges::EdgeOperand,
        group_by::{GroupBy, GroupOperand, GroupedOperand, PartitionGroups, Ungroup},
        nodes::{NodeIndexOperand, NodeIndicesOperand, NodeIndicesOperandContext},
        wrapper::Wrapper,
        DeepClone, EvaluateBackward, EvaluateForward, EvaluateForwardGrouped, GroupedIterator,
    },
    prelude::MedRecordAttribute,
    MedRecord,
};

#[derive(Debug, Clone)]
pub enum NodeOperandContext {
    Discriminator(<NodeOperand as GroupBy>::Discriminator),
    Nodes(Box<GroupOperand<NodeOperand>>),
    Edges(Box<GroupOperand<EdgeOperand>>),
}

impl DeepClone for NodeOperandContext {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Discriminator(discriminator) => Self::Discriminator(discriminator.clone()),
            Self::Nodes(operand) => Self::Nodes(operand.deep_clone()),
            Self::Edges(operand) => Self::Edges(operand.deep_clone()),
        }
    }
}

impl From<<NodeOperand as GroupBy>::Discriminator> for NodeOperandContext {
    fn from(discriminator: <NodeOperand as GroupBy>::Discriminator) -> Self {
        Self::Discriminator(discriminator)
    }
}

impl From<GroupOperand<NodeOperand>> for NodeOperandContext {
    fn from(operand: GroupOperand<NodeOperand>) -> Self {
        Self::Nodes(Box::new(operand))
    }
}

impl From<GroupOperand<EdgeOperand>> for NodeOperandContext {
    fn from(operand: GroupOperand<EdgeOperand>) -> Self {
        Self::Edges(Box::new(operand))
    }
}

impl GroupedOperand for NodeOperand {
    type Context = NodeOperandContext;
}

#[derive(Debug, Clone)]
pub enum NodeOperandGroupDiscriminator {
    Attribute(MedRecordAttribute),
}

impl DeepClone for NodeOperandGroupDiscriminator {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Attribute(attr) => Self::Attribute(attr.clone()),
        }
    }
}

impl<'a> EvaluateForward<'a> for GroupOperand<NodeOperand> {
    type InputValue = <NodeOperand as EvaluateForward<'a>>::InputValue;
    type ReturnValue = GroupedIterator<'a, <NodeOperand as EvaluateForward<'a>>::ReturnValue>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        indices: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        match &self.context {
            NodeOperandContext::Discriminator(discriminator) => {
                let partitions = NodeOperand::partition(medrecord, indices, discriminator);

                self.operand
                    .evaluate_forward_grouped(medrecord, Box::new(partitions))
            }
            NodeOperandContext::Nodes(_) => unreachable!(),
            NodeOperandContext::Edges(_) => unreachable!(),
        }
    }
}

impl GroupedOperand for NodeIndicesOperand {
    type Context = GroupOperand<NodeOperand>;
}

impl<'a> EvaluateBackward<'a> for GroupOperand<NodeIndicesOperand> {
    type ReturnValue =
        GroupedIterator<'a, <NodeIndicesOperand as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let partitions = self.context.evaluate_backward(medrecord)?;

        let indices = Box::new(partitions.map(|(key, partition)| {
            let reduced_partition = self.operand.reduce_input(partition)?;

            Ok((key, reduced_partition))
        }))
        .collect::<MedRecordResult<Vec<_>>>()?;

        self.operand
            .evaluate_forward_grouped(medrecord, Box::new(indices.into_iter()))
    }
}

impl Ungroup for GroupOperand<NodeIndicesOperand> {
    type OutputOperand = NodeIndicesOperand;

    fn ungroup(&self) -> Wrapper<Self::OutputOperand> {
        let operand = Wrapper::<Self::OutputOperand>::new(
            NodeIndicesOperandContext::NodeIndicesGroupByOperand(self.deep_clone()),
        );

        self.operand.push_merge_operation(operand.clone());

        operand
    }
}

impl GroupedOperand for NodeIndexOperand {
    type Context = GroupOperand<NodeIndicesOperand>;
}

impl<'a> EvaluateBackward<'a> for GroupOperand<NodeIndexOperand> {
    type ReturnValue = GroupedIterator<'a, <NodeIndexOperand as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let partitions = self.context.evaluate_backward(medrecord)?;

        let indices: Vec<_> = partitions
            .map(|(key, partition)| {
                let reduced_partition = self.operand.reduce_input(partition)?;

                Ok((key, reduced_partition))
            })
            .collect::<MedRecordResult<_>>()?;

        self.operand
            .evaluate_forward_grouped(medrecord, Box::new(indices.into_iter()))
    }
}

impl Ungroup for GroupOperand<NodeIndexOperand> {
    type OutputOperand = NodeIndicesOperand;

    fn ungroup(&self) -> Wrapper<Self::OutputOperand> {
        let operand = Wrapper::<Self::OutputOperand>::new(
            NodeIndicesOperandContext::NodeIndexGroupByOperand(self.deep_clone()),
        );

        self.operand.push_merge_operation(operand.clone());

        operand
    }
}
