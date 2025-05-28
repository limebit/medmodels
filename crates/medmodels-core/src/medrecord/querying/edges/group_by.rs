use super::{EdgeOperand, EdgeOperation};
use crate::{
    errors::MedRecordResult,
    medrecord::querying::{
        group_by::{GroupBy, GroupOperand, GroupedOperand, PartitionGroups},
        wrapper::Wrapper,
        BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, EvaluateForwardGrouped,
    },
    prelude::{EdgeIndex, MedRecordAttribute, MedRecordValue, NodeIndex},
    MedRecord,
};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum EdgeOperandContext {
    Discriminator(<EdgeOperand as GroupBy>::Discriminator),
}

impl DeepClone for EdgeOperandContext {
    fn deep_clone(&self) -> Self {
        match self {
            EdgeOperandContext::Discriminator(discriminator) => {
                EdgeOperandContext::Discriminator(discriminator.clone())
            }
        }
    }
}

impl From<<EdgeOperand as GroupBy>::Discriminator> for EdgeOperandContext {
    fn from(discriminator: <EdgeOperand as GroupBy>::Discriminator) -> Self {
        EdgeOperandContext::Discriminator(discriminator)
    }
}

impl GroupedOperand for EdgeOperand {
    type Context = EdgeOperandContext;
}

#[derive(Debug, Clone)]
pub enum EdgeOperandGroupDiscriminator {
    SourceNode,
    TargetNode,
    Attribute(MedRecordAttribute),
}

impl DeepClone for EdgeOperandGroupDiscriminator {
    fn deep_clone(&self) -> Self {
        match self {
            EdgeOperandGroupDiscriminator::SourceNode => EdgeOperandGroupDiscriminator::SourceNode,
            EdgeOperandGroupDiscriminator::TargetNode => EdgeOperandGroupDiscriminator::TargetNode,
            EdgeOperandGroupDiscriminator::Attribute(attr) => {
                EdgeOperandGroupDiscriminator::Attribute(attr.clone())
            }
        }
    }
}

impl GroupBy for EdgeOperand {
    type Discriminator = EdgeOperandGroupDiscriminator;

    fn group_by(&mut self, discriminator: Self::Discriminator) -> Wrapper<GroupOperand<Self>> {
        let operand =
            Wrapper::<GroupOperand<Self>>::new(discriminator.into(), self.deep_clone().into());

        self.operations.push(EdgeOperation::GroupBy {
            operand: operand.clone(),
        });

        operand
    }
}

pub enum EdgeOperandGroupKey<'a> {
    NodeIndex(&'a NodeIndex),
    Value(Option<&'a MedRecordValue>),
}

impl<'a> PartitionGroups<'a> for EdgeOperand {
    type GroupKey = EdgeOperandGroupKey<'a>;
    type Values = <Self as EvaluateForward<'a>>::InputValue;

    fn partition(
        medrecord: &'a MedRecord,
        edge_indices: Self::Values,
        discriminator: &Self::Discriminator,
    ) -> BoxedIterator<'a, (Self::GroupKey, Self::Values)> {
        match discriminator {
            EdgeOperandGroupDiscriminator::SourceNode => {
                let mut buckets: HashMap<&'a MedRecordAttribute, Vec<&'a EdgeIndex>> =
                    HashMap::new();

                for edge_index in edge_indices {
                    let source_node = medrecord
                        .edge_endpoints(edge_index)
                        .expect("Edge must exist")
                        .0;

                    buckets.entry(source_node).or_default().push(edge_index);
                }

                Box::new(buckets.into_iter().map(|(key, group)| {
                    (
                        EdgeOperandGroupKey::NodeIndex(key),
                        Box::new(group.into_iter()) as Self::Values,
                    )
                }))
            }

            EdgeOperandGroupDiscriminator::TargetNode => {
                let mut buckets: HashMap<&'a MedRecordAttribute, Vec<&'a EdgeIndex>> =
                    HashMap::new();

                for edge_index in edge_indices {
                    let target_node = medrecord
                        .edge_endpoints(edge_index)
                        .expect("Edge must exist")
                        .1;

                    buckets.entry(target_node).or_default().push(edge_index);
                }

                Box::new(buckets.into_iter().map(|(key, group)| {
                    (
                        EdgeOperandGroupKey::NodeIndex(key),
                        Box::new(group.into_iter()) as Self::Values,
                    )
                }))
            }

            EdgeOperandGroupDiscriminator::Attribute(attr) => {
                let mut buckets: Vec<(Option<&'a MedRecordValue>, Vec<&'a EdgeIndex>)> = Vec::new();

                for edge_index in edge_indices {
                    let value = medrecord
                        .edge_attributes(edge_index)
                        .expect("Edge must exist")
                        .get(attr);

                    if let Some((_, bucket)) = buckets.iter_mut().find(|(k, _)| *k == value) {
                        bucket.push(edge_index);
                    } else {
                        buckets.push((value, vec![edge_index]));
                    }
                }

                Box::new(buckets.into_iter().map(|(key, group)| {
                    (
                        EdgeOperandGroupKey::Value(key),
                        Box::new(group.into_iter()) as Self::Values,
                    )
                }))
            }
        }
    }

    fn merge(values: BoxedIterator<'a, (Self::GroupKey, Self::Values)>) -> Self::Values {
        Box::new(values.flat_map(|(_, value)| value))
    }
}

impl<'a> EvaluateForward<'a> for GroupOperand<EdgeOperand> {
    type InputValue = <EdgeOperand as EvaluateForward<'a>>::InputValue;
    // type ReturnValue = BoxedIterator<
    //     'a,
    //     (
    //         <EdgeOperand as PartitionGroups<'a>>::GroupKey,
    //         <EdgeOperand as EvaluateForward<'a>>::ReturnValue,
    //     ),
    // >;
    type ReturnValue = <EdgeOperand as EvaluateForward<'a>>::ReturnValue;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        indices: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        let EdgeOperandContext::Discriminator(discriminator) = &self.context;

        let partitions = EdgeOperand::partition(medrecord, indices, discriminator);

        let indices = self.operand.evaluate_forward_grouped(
            medrecord,
            Box::new(partitions.map(|(_, parition)| parition)),
        )?;

        // let indices: Vec<_> = partitions
        //     .map(|(key, partition)| Ok((key, self.operand.evaluate_forward(medrecord, partition)?)))
        //     .collect::<MedRecordResult<_>>()?;

        Ok(Box::new(indices.into_iter()))
    }
}

impl<'a> EvaluateBackward<'a> for GroupOperand<EdgeOperand> {
    type ReturnValue = BoxedIterator<'a, <EdgeOperand as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        match &self.context {
            EdgeOperandContext::Discriminator(discriminator) => {
                let values = self.operand.evaluate_backward(medrecord)?;

                Ok(Box::new(
                    EdgeOperand::partition(medrecord, values, discriminator)
                        .map(|(_, partition)| partition),
                ))
            }
        }
    }
}
