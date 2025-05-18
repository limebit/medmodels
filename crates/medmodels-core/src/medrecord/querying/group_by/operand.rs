use super::operation::GroupByOperation;
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            edges::{self, EdgeOperation},
            DeepClone, EvaluateBackward, EvaluateForward, ReadWriteOrPanic,
        },
        EdgeOperand, MedRecordAttribute, Wrapper,
    },
};
use std::fmt::Debug;

pub trait GroupableOperand: DeepClone {
    type Discriminator: Clone;

    fn group_by(&mut self, discriminator: Self::Discriminator) -> Wrapper<GroupByOperand<Self>>
    where
        Self: Sized;

    fn new_group_by_context(context: Box<GroupByOperand<Self>>) -> Self
    where
        Self: Sized;
}

impl<O: GroupableOperand> Wrapper<O> {
    pub fn group_by(&self, discriminator: O::Discriminator) -> Wrapper<GroupByOperand<O>> {
        self.0.write_or_panic().group_by(discriminator)
    }
}

#[derive(Debug, Clone)]
pub enum EdgeOperandGroupDiscriminator {
    SourceNode,
    TargetNode,
    Attribute(MedRecordAttribute),
}

impl GroupableOperand for EdgeOperand {
    type Discriminator = EdgeOperandGroupDiscriminator;

    fn group_by(&mut self, discriminator: Self::Discriminator) -> Wrapper<GroupByOperand<Self>> {
        let operand = Wrapper::<GroupByOperand<Self>>::new(self.deep_clone(), discriminator);

        self.operations.push(EdgeOperation::GroupBy {
            operand: operand.clone(),
        });

        operand
    }

    fn new_group_by_context(context: Box<GroupByOperand<Self>>) -> Self {
        Self::new(Some(edges::Context::GroupBy { operand: context }))
    }
}

#[derive(Debug, Clone)]
pub struct GroupByOperand<O: GroupableOperand> {
    context: O,
    discriminator: O::Discriminator,
    operations: Vec<GroupByOperation<O>>,
}

impl<O: GroupableOperand> DeepClone for GroupByOperand<O>
where
    O: DeepClone,
{
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            discriminator: self.discriminator.clone(),
            operations: self.operations.deep_clone(),
        }
    }
}

impl<'a, O: GroupableOperand + EvaluateForward<'a>> EvaluateForward<'a> for GroupByOperand<O> {
    type Indices = O::Indices;
    type ReturnValue = O::ReturnValue;

    fn evaluate_forward(
        &self,
        _medrecord: &'a crate::MedRecord,
        _indices: Self::Indices,
    ) -> MedRecordResult<Self::ReturnValue> {
        todo!()
    }
}

impl<'a, O: GroupableOperand + EvaluateBackward<'a>> EvaluateBackward<'a> for GroupByOperand<O> {
    type ReturnValue = O::ReturnValue;

    fn evaluate_backward(
        &self,
        _medrecord: &'a crate::MedRecord,
    ) -> MedRecordResult<Self::ReturnValue> {
        todo!()
    }
}

impl<O: GroupableOperand> GroupByOperand<O> {
    pub(crate) fn new(context: O, discriminator: O::Discriminator) -> Self {
        Self {
            context,
            discriminator,
            operations: Vec::new(),
        }
    }

    fn for_each<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut O),
    {
        let mut operand = O::new_group_by_context(Box::new(self.deep_clone()));

        query(&mut operand);

        self.operations.push(GroupByOperation::ForEach {
            operand: operand.deep_clone(),
        });
    }
}

impl<O: GroupableOperand> Wrapper<GroupByOperand<O>> {
    pub(crate) fn new(context: O, discriminator: O::Discriminator) -> Self {
        GroupByOperand::new(context, discriminator).into()
    }

    pub fn for_each<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut O),
    {
        self.0.write_or_panic().for_each(query);
    }
}

#[cfg(test)]
mod tests {
    use super::EdgeOperandGroupDiscriminator;
    use crate::{medrecord::querying::nodes::EdgeDirection, MedRecord};

    #[test]
    fn test_group_by() {
        let medrecord = MedRecord::from_admissions_example_dataset();

        let _result = medrecord
            .query_nodes(|nodes| {
                let edges = nodes.edges(EdgeDirection::Outgoing);

                let _group_by = edges.group_by(EdgeOperandGroupDiscriminator::SourceNode);

                edges.index()
            })
            .evaluate();
    }
}
