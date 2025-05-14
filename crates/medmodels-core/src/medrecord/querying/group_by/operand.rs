use super::Context;
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            edges::EdgeOperation,
            operand_traits::{Attribute, Max},
            BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, ReadWriteOrPanic,
        },
        EdgeOperand, MedRecordAttribute, Wrapper,
    },
    prelude::{EdgeIndex, MedRecordValue, NodeIndex},
    MedRecord,
};
use std::{collections::HashMap, fmt::Debug};

pub trait GroupableOperand: DeepClone {
    type Discriminator: Debug + Clone;

    fn group_by(
        &mut self,
        discriminator: Self::Discriminator,
    ) -> Wrapper<GroupByOperand<Self, Self>>
    where
        Self: Sized;
}

impl<O: GroupableOperand> Wrapper<O> {
    pub fn group_by(&self, discriminator: O::Discriminator) -> Wrapper<GroupByOperand<O, O>> {
        self.0.write_or_panic().group_by(discriminator)
    }
}

pub trait PartitionGroups<'a>: GroupableOperand + EvaluateForward<'a> {
    type GroupKey;

    fn partition(
        medrecord: &'a MedRecord,
        values: Self::InputValue,
        discriminator: &Self::Discriminator,
    ) -> BoxedIterator<'a, (Self::GroupKey, Self::InputValue)>;

    fn merge(values: BoxedIterator<'a, (Self::GroupKey, Self::InputValue)>) -> Self::ReturnValue;
}

#[derive(Debug, Clone)]
pub enum EdgeOperandGroupDiscriminator {
    SourceNode,
    TargetNode,
    Attribute(MedRecordAttribute),
}

impl GroupableOperand for EdgeOperand {
    type Discriminator = EdgeOperandGroupDiscriminator;

    fn group_by(
        &mut self,
        discriminator: Self::Discriminator,
    ) -> Wrapper<GroupByOperand<Self, Self>> {
        let operand = Wrapper::<GroupByOperand<Self, Self>>::new(
            Context::new(self.deep_clone(), discriminator),
            self.deep_clone().into(),
        );

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

    fn partition(
        medrecord: &'a MedRecord,
        edge_indices: Self::InputValue,
        discriminator: &Self::Discriminator,
    ) -> BoxedIterator<'a, (Self::GroupKey, Self::InputValue)> {
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
                        Box::new(group.into_iter()) as Self::InputValue,
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
                        Box::new(group.into_iter()) as Self::InputValue,
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
                        Box::new(group.into_iter()) as Self::InputValue,
                    )
                }))
            }
        }
    }

    fn merge(values: BoxedIterator<'a, (Self::GroupKey, Self::InputValue)>) -> Self::ReturnValue {
        Box::new(values.flat_map(|(_, value)| value))
    }
}

#[derive(Debug, Clone)]
pub struct GroupByOperand<CO: GroupableOperand, O> {
    context: Context<CO>,
    operand: Wrapper<O>,
}

impl<CO: GroupableOperand + DeepClone, O: DeepClone> DeepClone for GroupByOperand<CO, O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            operand: self.operand.deep_clone(),
        }
    }
}

impl<'a, CO: 'a + PartitionGroups<'a>> EvaluateForward<'a> for GroupByOperand<CO, CO> {
    type InputValue = CO::InputValue;
    type ReturnValue = BoxedIterator<'a, (CO::GroupKey, CO::ReturnValue)>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        indices: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        let partitions = CO::partition(medrecord, indices, &self.context.discriminator);

        let indices = partitions
            .map(|(key, partition)| Ok((key, self.operand.evaluate_forward(medrecord, partition)?)))
            .collect::<MedRecordResult<Vec<_>>>()?
            .into_iter();

        Ok(Box::new(indices))
    }
}

impl<CO: GroupableOperand, O: Attribute> Attribute for GroupByOperand<CO, O>
where
    O::ReturnOperand: DeepClone,
{
    type ReturnOperand = GroupByOperand<CO, O::ReturnOperand>;

    fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.attribute(attribute);

        Wrapper::<GroupByOperand<CO, O::ReturnOperand>>::new(self.context.deep_clone(), operand)
    }
}

impl<CO: GroupableOperand, O: Max> Max for GroupByOperand<CO, O>
where
    O::ReturnOperand: DeepClone,
{
    type ReturnOperand = GroupByOperand<CO, O::ReturnOperand>;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.max();

        Wrapper::<GroupByOperand<CO, O::ReturnOperand>>::new(self.context.deep_clone(), operand)
    }
}

impl<CO: GroupableOperand, O> GroupByOperand<CO, O> {
    pub(crate) fn new(context: Context<CO>, operand: Wrapper<O>) -> Self {
        Self { context, operand }
    }
}

impl<CO: GroupableOperand, O> Wrapper<GroupByOperand<CO, O>> {
    pub(crate) fn new(context: Context<CO>, operand: Wrapper<O>) -> Self {
        GroupByOperand::new(context, operand).into()
    }
}

#[cfg(test)]
mod tests {
    use super::EdgeOperandGroupDiscriminator;
    use crate::{
        medrecord::querying::nodes::EdgeDirection, prelude::MedRecordAttribute, MedRecord,
    };

    #[test]
    fn test_group_by() {
        let medrecord = MedRecord::from_admissions_example_dataset();

        let result = medrecord
            .query_nodes(|nodes| {
                let mut edges = nodes.edges(EdgeDirection::Outgoing);

                edges.has_attribute(MedRecordAttribute::from("duration_days"));

                let group_by = edges.group_by(EdgeOperandGroupDiscriminator::SourceNode);

                group_by.attribute("duration_days").max()
            })
            .evaluate()
            .unwrap()
            .collect::<Vec<_>>();

        println!("{:?}", result);
    }
}
