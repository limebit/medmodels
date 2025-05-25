use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            edges::EdgeOperation,
            operand_traits::{Attribute, Max},
            BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, ReadWriteOrPanic,
            ReduceInput,
        },
        EdgeOperand, MedRecordAttribute, Wrapper,
    },
    prelude::{EdgeIndex, MedRecordValue, NodeIndex},
    MedRecord,
};
use std::{collections::HashMap, fmt::Debug};

pub trait GroupableOperand: DeepClone {
    type Discriminator: Debug + Clone;

    fn group_by(&mut self, discriminator: Self::Discriminator) -> Wrapper<RootGroupOperand<Self>>
    where
        Self: Sized;
}

impl<O: GroupableOperand> Wrapper<O> {
    pub fn group_by(&self, discriminator: O::Discriminator) -> Wrapper<RootGroupOperand<O>> {
        self.0.write_or_panic().group_by(discriminator)
    }
}

pub trait PartitionGroups<'a>: GroupableOperand {
    type GroupKey;
    type Values;

    fn partition(
        medrecord: &'a MedRecord,
        values: Self::Values,
        discriminator: &Self::Discriminator,
    ) -> BoxedIterator<'a, (Self::GroupKey, Self::Values)>;

    fn merge(values: BoxedIterator<'a, (Self::GroupKey, Self::Values)>) -> Self::Values;
}

#[derive(Debug, Clone)]
pub enum EdgeOperandGroupDiscriminator {
    SourceNode,
    TargetNode,
    Attribute(MedRecordAttribute),
}

impl GroupableOperand for EdgeOperand {
    type Discriminator = EdgeOperandGroupDiscriminator;

    fn group_by(&mut self, discriminator: Self::Discriminator) -> Wrapper<RootGroupOperand<Self>> {
        let operand = Wrapper::<RootGroupOperand<Self>>::new(self.deep_clone(), discriminator);

        self.operations.push(EdgeOperation::GroupBy {
            operand: operand.clone(),
        });

        operand
    }
}

pub trait EvaluateBackwardGrouped<'a> {
    type ReturnValue;

    fn evaluate_backward_grouped(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, Self::ReturnValue>>;
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

#[derive(Debug, Clone)]
pub struct RootGroupOperand<O: GroupableOperand> {
    operand: O,
    discriminator: O::Discriminator,
}

impl<O: GroupableOperand + DeepClone> DeepClone for RootGroupOperand<O> {
    fn deep_clone(&self) -> Self {
        Self {
            operand: self.operand.deep_clone(),
            discriminator: self.discriminator.clone(),
        }
    }
}

impl<'a, O> EvaluateForward<'a> for RootGroupOperand<O>
where
    O: 'a + PartitionGroups<'a, Values = O::InputValue> + EvaluateForward<'a>,
{
    type InputValue = O::InputValue;
    type ReturnValue = BoxedIterator<'a, (O::GroupKey, <O as EvaluateForward<'a>>::ReturnValue)>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        indices: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        let partitions = O::partition(medrecord, indices, &self.discriminator);

        let indices: Vec<_> = partitions
            .map(|(key, partition)| Ok((key, self.operand.evaluate_forward(medrecord, partition)?)))
            .collect::<MedRecordResult<_>>()?;

        Ok(Box::new(indices.into_iter()))
    }
}

impl<'a, O> EvaluateBackward<'a> for RootGroupOperand<O>
where
    O: 'a + PartitionGroups<'a, Values = O::ReturnValue> + EvaluateBackward<'a>,
{
    type ReturnValue = BoxedIterator<'a, <O as EvaluateBackward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let values = self.operand.evaluate_backward(medrecord)?;

        Ok(Box::new(
            O::partition(medrecord, values, &self.discriminator).map(|(_, partition)| partition),
        ))
    }
}

impl<'a, O> EvaluateBackwardGrouped<'a> for RootGroupOperand<O>
where
    O: 'a + PartitionGroups<'a, Values = O::ReturnValue> + EvaluateBackward<'a>,
{
    type ReturnValue = <O as EvaluateBackward<'a>>::ReturnValue;

    #[inline]
    fn evaluate_backward_grouped(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, Self::ReturnValue>> {
        self.evaluate_backward(medrecord)
    }
}

impl<O: GroupableOperand + Attribute> Attribute for RootGroupOperand<O> {
    type ReturnOperand = GroupOperand<Self, O::ReturnOperand>;

    fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.attribute(attribute);

        Wrapper::<GroupOperand<Self, O::ReturnOperand>>::new(self.deep_clone(), operand)
    }
}

impl<O: GroupableOperand + Max> Max for RootGroupOperand<O> {
    type ReturnOperand = GroupOperand<Self, O::ReturnOperand>;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.max();

        Wrapper::<GroupOperand<Self, O::ReturnOperand>>::new(self.deep_clone(), operand)
    }
}

impl<O: GroupableOperand> RootGroupOperand<O> {
    pub(crate) fn new(operand: O, discriminator: O::Discriminator) -> Self {
        Self {
            operand,
            discriminator,
        }
    }
}

impl<O: GroupableOperand> Wrapper<RootGroupOperand<O>> {
    pub(crate) fn new(operand: O, discriminator: O::Discriminator) -> Self {
        RootGroupOperand::new(operand, discriminator).into()
    }
}

#[derive(Debug, Clone)]
pub struct GroupOperand<CO, O> {
    context: CO,
    operand: Wrapper<O>,
}

impl<CO: DeepClone, O: DeepClone> DeepClone for GroupOperand<CO, O> {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            operand: self.operand.deep_clone(),
        }
    }
}

impl<'a, CO, O> EvaluateBackward<'a> for GroupOperand<CO, O>
where
    CO: EvaluateBackwardGrouped<'a>,
    O: 'a
        + EvaluateForward<'a, InputValue = <O as ReduceInput<'a, CO::ReturnValue>>::ReturnValue>
        + ReduceInput<'a, CO::ReturnValue>,
{
    type ReturnValue = BoxedIterator<'a, <O as EvaluateForward<'a>>::ReturnValue>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let partitions = self.context.evaluate_backward_grouped(medrecord)?;

        let indices: Vec<_> = partitions
            .map(|partition| {
                let reduced_partition = self.operand.reduce_input(medrecord, partition)?;

                self.operand.evaluate_forward(medrecord, reduced_partition)
            })
            .collect::<MedRecordResult<_>>()?;

        Ok(Box::new(indices.into_iter()))
    }
}

impl<'a, CO, O> EvaluateBackwardGrouped<'a> for GroupOperand<CO, O>
where
    CO: EvaluateBackwardGrouped<'a>,
    O: 'a
        + EvaluateForward<'a, InputValue = <O as ReduceInput<'a, CO::ReturnValue>>::ReturnValue>
        + ReduceInput<'a, CO::ReturnValue>,
{
    type ReturnValue = <O as EvaluateForward<'a>>::ReturnValue;

    #[inline]
    fn evaluate_backward_grouped(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, Self::ReturnValue>> {
        self.evaluate_backward(medrecord)
    }
}

impl<CO, O: Attribute> Attribute for GroupOperand<CO, O>
where
    GroupOperand<CO, O>: DeepClone,
{
    type ReturnOperand = GroupOperand<Self, O::ReturnOperand>;

    fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.attribute(attribute);

        Wrapper::<GroupOperand<Self, O::ReturnOperand>>::new(self.deep_clone(), operand)
    }
}

impl<CO, O: Max> Max for GroupOperand<CO, O>
where
    GroupOperand<CO, O>: DeepClone,
{
    type ReturnOperand = GroupOperand<Self, O::ReturnOperand>;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.max();

        Wrapper::<GroupOperand<Self, O::ReturnOperand>>::new(self.deep_clone(), operand)
    }
}

impl<CO, O> GroupOperand<CO, O> {
    pub(crate) fn new(context: CO, operand: Wrapper<O>) -> Self {
        Self { context, operand }
    }
}

impl<CO, O> Wrapper<GroupOperand<CO, O>> {
    pub(crate) fn new(context: CO, operand: Wrapper<O>) -> Self {
        GroupOperand::new(context, operand).into()
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

        let result: Vec<_> = medrecord
            .query_nodes(|nodes| {
                let mut edges = nodes.edges(EdgeDirection::Outgoing);

                edges.has_attribute(MedRecordAttribute::from("duration_days"));

                let group_by = edges.group_by(EdgeOperandGroupDiscriminator::SourceNode);

                group_by.attribute("duration_days").max()
            })
            .evaluate()
            .unwrap()
            .collect();

        println!("{:?}", result);
    }
}
