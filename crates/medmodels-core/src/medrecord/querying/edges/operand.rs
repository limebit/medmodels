use super::{
    operation::{EdgeIndexOperation, EdgeIndicesOperation, EdgeOperation},
    BinaryArithmeticKind, EdgeOperandContext, MultipleComparisonKind, SingleComparisonKind,
    SingleKind,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            attributes::{AttributesTreeContext, AttributesTreeOperand},
            edges::{group_by, EdgeIndicesOperandContext, EdgeOperandGroupDiscriminator},
            group_by::{GroupKey, GroupOperand, PartitionGroups},
            nodes::{self, NodeOperand},
            operand_traits::{
                Add, Attribute, Attributes, Contains, Count, EitherOr, EndsWith, EqualTo, Exclude,
                GreaterThan, GreaterThanOrEqualTo, HasAttribute, InGroup, Index, IsIn, IsMax,
                IsMin, IsNotIn, LessThan, LessThanOrEqualTo, Max, Min, Mod, Mul, NotEqualTo, Pow,
                Random, SourceNode, StartsWith, Sub, Sum, TargetNode,
            },
            values::{self, MultipleValuesWithIndexOperand},
            wrapper::{CardinalityWrapper, Wrapper},
            BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, EvaluateForwardGrouped,
            GroupedIterator, ReadWriteOrPanic, ReduceInput, RootOperand,
        },
        EdgeIndex, Group, MedRecordAttribute,
    },
    prelude::MedRecordValue,
    MedRecord,
};
use medmodels_utils::aliases::MrHashSet;
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
};

#[derive(Debug, Clone)]
pub struct EdgeOperand {
    context: Option<EdgeOperandContext>,
    operations: Vec<EdgeOperation>,
}

impl DeepClone for EdgeOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            operations: self
                .operations
                .iter()
                .map(|operation| operation.deep_clone())
                .collect(),
        }
    }
}

impl RootOperand for EdgeOperand {
    type Index = EdgeIndex;
    type Discriminator = EdgeOperandGroupDiscriminator;

    fn _evaluate_forward<'a>(
        &self,
        medrecord: &'a MedRecord,
        edge_indices: BoxedIterator<'a, &'a Self::Index>,
    ) -> MedRecordResult<BoxedIterator<'a, &'a Self::Index>> {
        self.operations
            .iter()
            .try_fold(edge_indices, |edge_indices, operation| {
                operation.evaluate(medrecord, edge_indices)
            })
    }

    fn _evaluate_forward_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>>> {
        self.operations
            .iter()
            .try_fold(edge_indices, |edge_indices, operation| {
                operation.evaluate_grouped(medrecord, edge_indices)
            })
    }

    fn _evaluate_backward<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, &'a Self::Index>> {
        let edge_indices: BoxedIterator<_> = match &self.context {
            Some(EdgeOperandContext::Edges { operand, kind }) => {
                let node_indices = operand.evaluate_backward(medrecord)?;

                match kind {
                    nodes::EdgeDirection::Incoming => {
                        Box::new(node_indices.flat_map(|node_index| {
                            medrecord
                                .incoming_edges(node_index)
                                .expect("Node must exist.")
                        }))
                    }
                    nodes::EdgeDirection::Outgoing => {
                        Box::new(node_indices.flat_map(|node_index| {
                            medrecord
                                .outgoing_edges(node_index)
                                .expect("Node must exist.")
                        }))
                    }
                    nodes::EdgeDirection::Both => Box::new(node_indices.flat_map(|node_index| {
                        medrecord
                            .incoming_edges(node_index)
                            .expect("Node must exist")
                            .chain(
                                medrecord
                                    .outgoing_edges(node_index)
                                    .expect("Node must exist"),
                            )
                    })),
                }
            }
            Some(EdgeOperandContext::GroupBy { operand }) => {
                operand.evaluate_backward(medrecord)?
            }
            None => Box::new(medrecord.edge_indices()),
        };

        self.evaluate_forward(medrecord, edge_indices)
    }

    fn _evaluate_backward_grouped_operand<'a>(
        group_operand: &GroupOperand<Self>,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>>> {
        match &group_operand.context {
            group_by::EdgeOperandContext::Discriminator(discriminator) => {
                let edge_indices = group_operand.operand.evaluate_backward(medrecord)?;

                Ok(Self::partition(medrecord, edge_indices, discriminator))
            }
            group_by::EdgeOperandContext::Edges(operand) => {
                let partitions = operand.evaluate_backward(medrecord)?;

                let Some(EdgeOperandContext::Edges {
                    operand: _,
                    ref kind,
                }) = group_operand.operand.0.read_or_panic().context
                else {
                    unreachable!()
                };

                let indices: Vec<_> = partitions
                    .map(|(key, partition)| {
                        let reduced_partition: BoxedIterator<_> = match kind {
                            nodes::EdgeDirection::Incoming => {
                                Box::new(partition.flat_map(|node_index| {
                                    medrecord
                                        .incoming_edges(node_index)
                                        .expect("Node must exist.")
                                }))
                            }
                            nodes::EdgeDirection::Outgoing => {
                                Box::new(partition.flat_map(|node_index| {
                                    medrecord
                                        .outgoing_edges(node_index)
                                        .expect("Node must exist.")
                                }))
                            }
                            nodes::EdgeDirection::Both => {
                                Box::new(partition.flat_map(|node_index| {
                                    medrecord
                                        .incoming_edges(node_index)
                                        .expect("Node must exist")
                                        .chain(
                                            medrecord
                                                .outgoing_edges(node_index)
                                                .expect("Node must exist"),
                                        )
                                }))
                            }
                        };

                        let partition = group_operand
                            .operand
                            .evaluate_forward(medrecord, reduced_partition)?;

                        Ok((key, partition))
                    })
                    .collect::<MedRecordResult<_>>()?;

                Ok(Box::new(indices.into_iter()))
            }
        }
    }

    fn _group_by(&mut self, discriminator: Self::Discriminator) -> Wrapper<GroupOperand<Self>> {
        let edge_operand = Wrapper::<Self>::new(Some(EdgeOperandContext::GroupBy {
            operand: Box::new(self.deep_clone()),
        }));
        let operand = Wrapper::<GroupOperand<Self>>::new(discriminator.into(), edge_operand);

        self.operations.push(EdgeOperation::GroupBy {
            operand: operand.clone(),
        });

        operand
    }

    fn _partition<'a>(
        medrecord: &'a MedRecord,
        edge_indices: BoxedIterator<'a, &'a Self::Index>,
        discriminator: &Self::Discriminator,
    ) -> GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>> {
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
                        GroupKey::NodeIndex(key),
                        Box::new(group.into_iter()) as BoxedIterator<_>,
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
                        GroupKey::NodeIndex(key),
                        Box::new(group.into_iter()) as BoxedIterator<_>,
                    )
                }))
            }
            EdgeOperandGroupDiscriminator::Parallel => {
                let mut buckets: HashMap<
                    (&'a MedRecordAttribute, &'a MedRecordAttribute),
                    Vec<&'a EdgeIndex>,
                > = HashMap::new();

                for edge_index in edge_indices {
                    let endpoints = medrecord
                        .edge_endpoints(edge_index)
                        .expect("Edge must exist");

                    buckets
                        .entry((endpoints.0, endpoints.1))
                        .or_default()
                        .push(edge_index);
                }

                Box::new(buckets.into_iter().map(|(key, group)| {
                    (
                        GroupKey::TupleKey((
                            Box::new(GroupKey::NodeIndex(key.0)),
                            Box::new(GroupKey::NodeIndex(key.1)),
                        )),
                        Box::new(group.into_iter()) as BoxedIterator<_>,
                    )
                }))
            }
            EdgeOperandGroupDiscriminator::Attribute(attribute) => {
                let mut buckets: Vec<(Option<&'a MedRecordValue>, Vec<&'a EdgeIndex>)> = Vec::new();

                for edge_index in edge_indices {
                    let value = medrecord
                        .edge_attributes(edge_index)
                        .expect("Edge must exist")
                        .get(attribute);

                    if let Some((_, bucket)) = buckets.iter_mut().find(|(k, _)| *k == value) {
                        bucket.push(edge_index);
                    } else {
                        buckets.push((value, vec![edge_index]));
                    }
                }

                Box::new(buckets.into_iter().map(|(key, group)| {
                    (
                        GroupKey::OptionalValue(key),
                        Box::new(group.into_iter()) as BoxedIterator<_>,
                    )
                }))
            }
        }
    }

    fn _merge<'a>(
        edge_indices: GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>>,
    ) -> BoxedIterator<'a, &'a Self::Index> {
        Box::new(edge_indices.flat_map(|(_, edge_indices)| edge_indices))
    }
}

impl Attribute for EdgeOperand {
    type ReturnOperand = MultipleValuesWithIndexOperand<Self>;

    fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            values::MultipleValuesWithIndexContext::Operand((self.deep_clone(), attribute)),
        );

        self.operations.push(EdgeOperation::Values {
            operand: operand.clone(),
        });

        operand
    }
}

impl Attributes for EdgeOperand {
    type ReturnOperand = AttributesTreeOperand<Self>;

    fn attributes(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand =
            Wrapper::<Self::ReturnOperand>::new(AttributesTreeContext::Operand(self.deep_clone()));

        self.operations.push(EdgeOperation::Attributes {
            operand: operand.clone(),
        });

        operand
    }
}

impl Index for EdgeOperand {
    type ReturnOperand = EdgeIndicesOperand;

    fn index(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(EdgeIndicesOperandContext::EdgeOperand(
            self.deep_clone(),
        ));

        self.operations.push(EdgeOperation::Indices {
            operand: operand.clone(),
        });

        operand
    }
}

impl InGroup for EdgeOperand {
    fn in_group<G: Into<CardinalityWrapper<Group>>>(&mut self, group: G) {
        self.operations.push(EdgeOperation::InGroup {
            group: group.into(),
        });
    }
}

impl HasAttribute for EdgeOperand {
    fn has_attribute<A: Into<CardinalityWrapper<MedRecordAttribute>>>(&mut self, attribute: A) {
        self.operations.push(EdgeOperation::HasAttribute {
            attribute: attribute.into(),
        });
    }
}

impl SourceNode for EdgeOperand {
    type ReturnOperand = NodeOperand;

    fn source_node(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand =
            Wrapper::<Self::ReturnOperand>::new(Some(nodes::NodeOperandContext::SourceNode {
                operand: self.deep_clone(),
            }));

        self.operations.push(EdgeOperation::SourceNode {
            operand: operand.clone(),
        });

        operand
    }
}

impl TargetNode for EdgeOperand {
    type ReturnOperand = NodeOperand;

    fn target_node(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand =
            Wrapper::<Self::ReturnOperand>::new(Some(nodes::NodeOperandContext::TargetNode {
                operand: self.deep_clone(),
            }));

        self.operations.push(EdgeOperation::TargetNode {
            operand: operand.clone(),
        });

        operand
    }
}

impl EitherOr for EdgeOperand {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut either_operand = Wrapper::<Self>::new(self.context.deep_clone());
        let mut or_operand = Wrapper::<Self>::new(self.context.deep_clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(EdgeOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }
}

impl Exclude for EdgeOperand {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand = Wrapper::<Self>::new(self.context.deep_clone());

        query(&mut operand);

        self.operations.push(EdgeOperation::Exclude { operand });
    }
}

impl EdgeOperand {
    pub(crate) fn new(context: Option<EdgeOperandContext>) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }
}

impl Wrapper<EdgeOperand> {
    pub(crate) fn new(context: Option<EdgeOperandContext>) -> Self {
        EdgeOperand::new(context).into()
    }
}

#[derive(Debug, Clone)]
pub enum EdgeIndexComparisonOperand {
    Operand(EdgeIndexOperand),
    Index(EdgeIndex),
}

impl DeepClone for EdgeIndexComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Index(index) => Self::Index(*index),
        }
    }
}

impl From<Wrapper<EdgeIndexOperand>> for EdgeIndexComparisonOperand {
    fn from(index: Wrapper<EdgeIndexOperand>) -> Self {
        Self::Operand(index.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeIndexOperand>> for EdgeIndexComparisonOperand {
    fn from(index: &Wrapper<EdgeIndexOperand>) -> Self {
        Self::Operand(index.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<EdgeIndex>> From<V> for EdgeIndexComparisonOperand {
    fn from(index: V) -> Self {
        Self::Index(index.into())
    }
}

impl EdgeIndexComparisonOperand {
    pub(crate) fn evaluate_backward(
        &self,
        medrecord: &MedRecord,
    ) -> MedRecordResult<Option<EdgeIndex>> {
        match self {
            Self::Operand(operand) => operand.evaluate_backward(medrecord),
            Self::Index(index) => Ok(Some(*index)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EdgeIndicesComparisonOperand {
    Operand(EdgeIndicesOperand),
    Indices(MrHashSet<EdgeIndex>),
}

impl DeepClone for EdgeIndicesComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Indices(indices) => Self::Indices(indices.clone()),
        }
    }
}

impl From<Wrapper<EdgeIndicesOperand>> for EdgeIndicesComparisonOperand {
    fn from(indices: Wrapper<EdgeIndicesOperand>) -> Self {
        Self::Operand(indices.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeIndicesOperand>> for EdgeIndicesComparisonOperand {
    fn from(indices: &Wrapper<EdgeIndicesOperand>) -> Self {
        Self::Operand(indices.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<EdgeIndex>> From<Vec<V>> for EdgeIndicesComparisonOperand {
    fn from(indices: Vec<V>) -> Self {
        Self::Indices(indices.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<EdgeIndex>> From<HashSet<V>> for EdgeIndicesComparisonOperand {
    fn from(indices: HashSet<V>) -> Self {
        Self::Indices(indices.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<EdgeIndex>> From<MrHashSet<V>> for EdgeIndicesComparisonOperand {
    fn from(indices: MrHashSet<V>) -> Self {
        Self::Indices(indices.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<EdgeIndex> + Clone, const N: usize> From<[V; N]> for EdgeIndicesComparisonOperand {
    fn from(indices: [V; N]) -> Self {
        indices.to_vec().into()
    }
}

impl EdgeIndicesComparisonOperand {
    pub(crate) fn evaluate_backward(
        &self,
        medrecord: &MedRecord,
    ) -> MedRecordResult<MrHashSet<EdgeIndex>> {
        Ok(match self {
            Self::Operand(operand) => operand.evaluate_backward(medrecord)?.collect(),
            Self::Indices(indices) => indices.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct EdgeIndicesOperand {
    context: EdgeIndicesOperandContext,
    operations: Vec<EdgeIndicesOperation>,
}

impl DeepClone for EdgeIndicesOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a> EvaluateForward<'a> for EdgeIndicesOperand {
    type InputValue = BoxedIterator<'a, EdgeIndex>;
    type ReturnValue = BoxedIterator<'a, EdgeIndex>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        edge_indices: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        let edge_indices: BoxedIterator<_> = Box::new(edge_indices);

        self.operations
            .iter()
            .try_fold(edge_indices, |index_tuples, operation| {
                operation.evaluate(medrecord, index_tuples)
            })
    }
}

impl<'a> EvaluateForwardGrouped<'a> for EdgeIndicesOperand {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        self.operations
            .iter()
            .try_fold(edge_indices, |index_tuples, operation| {
                operation.evaluate_grouped(medrecord, index_tuples)
            })
    }
}

impl<'a> EvaluateBackward<'a> for EdgeIndicesOperand {
    type ReturnValue = BoxedIterator<'a, EdgeIndex>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let edge_indices = self.context.evaluate_backward(medrecord)?;

        self.evaluate_forward(medrecord, edge_indices)
    }
}

impl Max for EdgeIndicesOperand {
    type ReturnOperand = EdgeIndexOperand;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKind::Max);

        self.operations
            .push(EdgeIndicesOperation::EdgeIndexOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl Min for EdgeIndicesOperand {
    type ReturnOperand = EdgeIndexOperand;

    fn min(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKind::Min);

        self.operations
            .push(EdgeIndicesOperation::EdgeIndexOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl Count for EdgeIndicesOperand {
    type ReturnOperand = EdgeIndexOperand;

    fn count(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKind::Count);

        self.operations
            .push(EdgeIndicesOperation::EdgeIndexOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl Sum for EdgeIndicesOperand {
    type ReturnOperand = EdgeIndexOperand;

    fn sum(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKind::Sum);

        self.operations
            .push(EdgeIndicesOperation::EdgeIndexOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl Random for EdgeIndicesOperand {
    type ReturnOperand = EdgeIndexOperand;

    fn random(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKind::Random);

        self.operations
            .push(EdgeIndicesOperation::EdgeIndexOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl GreaterThan for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThan,
            });
    }
}

impl GreaterThanOrEqualTo for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThanOrEqualTo,
            });
    }
}

impl LessThan for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThan,
            });
    }
}

impl LessThanOrEqualTo for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThanOrEqualTo,
            });
    }
}

impl EqualTo for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EqualTo,
            });
    }
}

impl NotEqualTo for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::NotEqualTo,
            });
    }
}

impl StartsWith for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::StartsWith,
            });
    }
}

impl EndsWith for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EndsWith,
            });
    }
}

impl Contains for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::Contains,
            });
    }
}

impl IsIn for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndicesComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }
}

impl IsNotIn for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndicesComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsNotIn,
            });
    }
}

impl Add for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Add,
            });
    }
}

impl Sub for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Sub,
            });
    }
}

impl Mul for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mul,
            });
    }
}

impl Pow for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Pow,
            });
    }
}

impl Mod for EdgeIndicesOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndicesOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mod,
            });
    }
}

impl IsMax for EdgeIndicesOperand {
    fn is_max(&mut self) {
        self.operations.push(EdgeIndicesOperation::IsMax);
    }
}

impl IsMin for EdgeIndicesOperand {
    fn is_min(&mut self) {
        self.operations.push(EdgeIndicesOperation::IsMin);
    }
}

impl EitherOr for EdgeIndicesOperand {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut either_operand = Wrapper::<Self>::new(self.context.clone());
        let mut or_operand = Wrapper::<Self>::new(self.context.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(EdgeIndicesOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }
}

impl Exclude for EdgeIndicesOperand {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand = Wrapper::<Self>::new(self.context.clone());

        query(&mut operand);

        self.operations
            .push(EdgeIndicesOperation::Exclude { operand });
    }
}

impl EdgeIndicesOperand {
    pub(crate) fn new(context: EdgeIndicesOperandContext) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    pub(crate) fn push_merge_operation(&mut self, operand: Wrapper<EdgeIndicesOperand>) {
        self.operations.push(EdgeIndicesOperation::Merge {
            operand: operand.clone(),
        });
    }
}

impl Wrapper<EdgeIndicesOperand> {
    pub(crate) fn new(context: EdgeIndicesOperandContext) -> Self {
        EdgeIndicesOperand::new(context).into()
    }

    pub(crate) fn push_merge_operation(&self, operand: Wrapper<EdgeIndicesOperand>) {
        self.0.write_or_panic().push_merge_operation(operand);
    }
}

#[derive(Debug, Clone)]
pub struct EdgeIndexOperand {
    context: EdgeIndicesOperand,
    pub(crate) kind: SingleKind,
    operations: Vec<EdgeIndexOperation>,
}

impl DeepClone for EdgeIndexOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a> EvaluateForward<'a> for EdgeIndexOperand {
    type InputValue = Option<EdgeIndex>;
    type ReturnValue = Option<EdgeIndex>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        edge_index: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        self.operations
            .iter()
            .try_fold(edge_index, |edge_index, operation| {
                operation.evaluate(medrecord, edge_index)
            })
    }
}

impl<'a> EvaluateForwardGrouped<'a> for EdgeIndexOperand {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        edge_indices: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        self.operations
            .iter()
            .try_fold(edge_indices, |edge_indices, operation| {
                operation.evaluate_grouped(medrecord, edge_indices)
            })
    }
}

impl<'a> EvaluateBackward<'a> for EdgeIndexOperand {
    type ReturnValue = Option<EdgeIndex>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let edge_indices = self.context.evaluate_backward(medrecord)?;

        let edge_index = self.reduce_input(edge_indices)?;

        self.evaluate_forward(medrecord, edge_index)
    }
}

impl<'a> ReduceInput<'a> for EdgeIndexOperand {
    type Context = EdgeIndicesOperand;

    #[inline]
    fn reduce_input(
        &self,
        edge_indices: <Self::Context as EvaluateBackward<'a>>::ReturnValue,
    ) -> MedRecordResult<<Self as EvaluateForward<'a>>::InputValue> {
        Ok(match self.kind {
            SingleKind::Max => EdgeIndicesOperation::get_max(edge_indices),
            SingleKind::Min => EdgeIndicesOperation::get_min(edge_indices),
            SingleKind::Count => Some(EdgeIndicesOperation::get_count(edge_indices)),
            SingleKind::Sum => Some(EdgeIndicesOperation::get_sum(edge_indices)),
            SingleKind::Random => EdgeIndicesOperation::get_random(edge_indices),
        })
    }
}

impl GreaterThan for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThan,
            });
    }
}

impl GreaterThanOrEqualTo for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThanOrEqualTo,
            });
    }
}

impl LessThan for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThan,
            });
    }
}

impl LessThanOrEqualTo for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThanOrEqualTo,
            });
    }
}

impl EqualTo for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EqualTo,
            });
    }
}

impl NotEqualTo for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::NotEqualTo,
            });
    }
}

impl StartsWith for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::StartsWith,
            });
    }
}

impl EndsWith for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EndsWith,
            });
    }
}

impl Contains for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::Contains,
            });
    }
}

impl IsIn for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndicesComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }
}

impl IsNotIn for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndicesComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsNotIn,
            });
    }
}

impl Add for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Add,
            });
    }
}

impl Sub for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Sub,
            });
    }
}

impl Mul for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mul,
            });
    }
}

impl Pow for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Pow,
            });
    }
}

impl Mod for EdgeIndexOperand {
    type ComparisonOperand = EdgeIndexComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(EdgeIndexOperation::BinaryArithmeticOperation {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mod,
            });
    }
}

impl EitherOr for EdgeIndexOperand {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut either_operand = Wrapper::<Self>::new(self.context.clone(), self.kind.clone());
        let mut or_operand = Wrapper::<Self>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(EdgeIndexOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }
}

impl Exclude for EdgeIndexOperand {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand = Wrapper::<Self>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(EdgeIndexOperation::Exclude { operand });
    }
}

impl EdgeIndexOperand {
    pub(crate) fn new(context: EdgeIndicesOperand, kind: SingleKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    pub(crate) fn push_merge_operation(&mut self, operand: Wrapper<EdgeIndicesOperand>) {
        self.operations.push(EdgeIndexOperation::Merge { operand });
    }
}

impl Wrapper<EdgeIndexOperand> {
    pub(crate) fn new(context: EdgeIndicesOperand, kind: SingleKind) -> Self {
        EdgeIndexOperand::new(context, kind).into()
    }

    pub(crate) fn push_merge_operation(&self, operand: Wrapper<EdgeIndicesOperand>) {
        self.0.write_or_panic().push_merge_operation(operand);
    }
}
