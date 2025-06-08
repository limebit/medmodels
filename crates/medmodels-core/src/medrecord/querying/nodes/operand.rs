use super::{
    operation::{EdgeDirection, NodeIndexOperation, NodeIndicesOperation, NodeOperation},
    BinaryArithmeticKind, Context, MultipleComparisonKind, SingleComparisonKind, SingleKind,
    UnaryArithmeticKind,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            attributes::AttributesTreeOperand,
            edges::{self, EdgeOperand, EdgeOperandGroupDiscriminator},
            group_by::GroupOperand,
            operand_traits::{
                Abs, Add, Attribute, Attributes, Contains, Count, Edges, EitherOr, EndsWith,
                EqualTo, Exclude, GreaterThan, GreaterThanOrEqualTo, HasAttribute, InGroup, Index,
                IsIn, IsInt, IsMax, IsMin, IsNotIn, IsString, LessThan, LessThanOrEqualTo,
                Lowercase, Max, Min, Mod, Mul, Neighbors, NotEqualTo, Pow, Random, Slice,
                StartsWith, Sub, Sum, Trim, TrimEnd, TrimStart, Uppercase,
            },
            values::{self, MultipleValuesWithIndexOperand},
            wrapper::{CardinalityWrapper, Wrapper},
            BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, EvaluateForwardGrouped,
            GroupedIterator, ReadWriteOrPanic, ReduceInput, RootOperand,
        },
        Group, MedRecordAttribute, NodeIndex,
    },
    MedRecord,
};
use std::{collections::HashSet, fmt::Debug};

#[derive(Debug, Clone)]
pub struct NodeOperand {
    context: Option<Context>,
    operations: Vec<NodeOperation>,
}

impl DeepClone for NodeOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            operations: self.operations.deep_clone(),
        }
    }
}

impl RootOperand for NodeOperand {
    type Index = NodeIndex;
    type Discriminator = EdgeOperandGroupDiscriminator;

    fn _evaluate_forward<'a>(
        &self,
        medrecord: &'a MedRecord,
        node_indices: BoxedIterator<'a, &'a Self::Index>,
    ) -> MedRecordResult<BoxedIterator<'a, &'a Self::Index>> {
        self.operations
            .iter()
            .try_fold(node_indices, |node_indices, operation| {
                operation.evaluate(medrecord, node_indices)
            })
    }

    fn _evaluate_forward_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>>> {
        self.operations
            .iter()
            .try_fold(node_indices, |node_indices, operation| {
                operation.evaluate_grouped(medrecord, node_indices)
            })
    }

    fn _evaluate_backward<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, &'a Self::Index>> {
        let node_indices: BoxedIterator<&NodeIndex> = match &self.context {
            Some(Context::Neighbors { operand, direction }) => {
                let node_indices = operand.evaluate_backward(medrecord)?;

                match direction {
                    EdgeDirection::Incoming => Box::new(node_indices.flat_map(move |node_index| {
                        medrecord
                            .neighbors_incoming(node_index)
                            .expect("Node must exist")
                    })),
                    EdgeDirection::Outgoing => Box::new(node_indices.flat_map(move |node_index| {
                        medrecord
                            .neighbors_outgoing(node_index)
                            .expect("Node must exist")
                    })),
                    EdgeDirection::Both => Box::new(node_indices.flat_map(move |node_index| {
                        medrecord
                            .neighbors_undirected(node_index)
                            .expect("Node must exist")
                    })),
                }
            }
            Some(Context::SourceNode { operand }) => {
                let edge_indices = operand.evaluate_backward(medrecord)?;

                Box::new(edge_indices.map(move |edge_index| {
                    medrecord
                        .edge_endpoints(edge_index)
                        .expect("Node must exist")
                        .0
                }))
            }
            Some(Context::TargetNode { operand }) => {
                let edge_indices = operand.evaluate_backward(medrecord)?;

                Box::new(edge_indices.map(move |edge_index| {
                    medrecord
                        .edge_endpoints(edge_index)
                        .expect("Node must exist")
                        .1
                }))
            }
            None => Box::new(medrecord.node_indices()),
        };

        self.evaluate_forward(medrecord, node_indices)
    }

    fn _evaluate_backward_grouped_operand<'a>(
        _group_operand: &GroupOperand<Self>,
        _medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, BoxedIterator<'a, &'a Self::Index>>> {
        todo!()
    }

    fn _group_by(&mut self, _discriminator: Self::Discriminator) -> Wrapper<GroupOperand<Self>> {
        todo!()
    }

    fn _partition<'a>(
        _medrecord: &'a MedRecord,
        _node_indices: BoxedIterator<'a, &'a Self::Index>,
        _discriminator: &Self::Discriminator,
    ) -> GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>> {
        todo!()
    }

    fn _merge<'a>(
        _node_indices: GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>>,
    ) -> BoxedIterator<'a, &'a Self::Index> {
        todo!()
    }
}

impl Attribute for NodeOperand {
    type ReturnOperand = MultipleValuesWithIndexOperand<Self>;

    fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(
            values::MultipleValuesWithIndexContext::Operand((self.deep_clone(), attribute)),
        );

        self.operations.push(NodeOperation::Values {
            operand: operand.clone(),
        });

        operand
    }
}

impl Attributes for NodeOperand {
    type ReturnOperand = AttributesTreeOperand<Self>;

    fn attributes(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(self.deep_clone());

        self.operations.push(NodeOperation::Attributes {
            operand: operand.clone(),
        });

        operand
    }
}

impl Index for NodeOperand {
    type ReturnOperand = NodeIndicesOperand;

    fn index(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(self.deep_clone());

        self.operations.push(NodeOperation::Indices {
            operand: operand.clone(),
        });

        operand
    }
}

impl InGroup for NodeOperand {
    fn in_group<G: Into<CardinalityWrapper<Group>>>(&mut self, group: G) {
        self.operations.push(NodeOperation::InGroup {
            group: group.into(),
        });
    }
}

impl HasAttribute for NodeOperand {
    fn has_attribute<A: Into<CardinalityWrapper<MedRecordAttribute>>>(&mut self, attribute: A) {
        self.operations.push(NodeOperation::HasAttribute {
            attribute: attribute.into(),
        });
    }
}

impl Edges for NodeOperand {
    type ReturnOperand = EdgeOperand;

    fn edges(&mut self, edge_direction: EdgeDirection) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(Some(edges::EdgeOperandContext::Edges {
            operand: Box::new(self.deep_clone()),
            kind: edge_direction.clone(),
        }));

        self.operations.push(NodeOperation::Edges {
            operand: operand.clone(),
            direction: edge_direction,
        });

        operand
    }
}

impl Neighbors for NodeOperand {
    type ReturnOperand = Self;

    fn neighbors(&mut self, edge_direction: EdgeDirection) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(Some(Context::Neighbors {
            operand: Box::new(self.deep_clone()),
            direction: edge_direction.clone(),
        }));

        self.operations.push(NodeOperation::Neighbors {
            operand: operand.clone(),
            direction: edge_direction,
        });

        operand
    }
}

impl EitherOr for NodeOperand {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut either_operand = Wrapper::<Self::QueryOperand>::new(self.context.clone());
        let mut or_operand = Wrapper::<Self::QueryOperand>::new(self.context.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(NodeOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }
}

impl Exclude for NodeOperand {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand = Wrapper::<Self::QueryOperand>::new(self.context.clone());

        query(&mut operand);

        self.operations.push(NodeOperation::Exclude { operand });
    }
}

impl NodeOperand {
    pub(crate) fn new(context: Option<Context>) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }
}

impl Wrapper<NodeOperand> {
    pub(crate) fn new(context: Option<Context>) -> Self {
        NodeOperand::new(context).into()
    }
}

#[derive(Debug, Clone)]
pub enum NodeIndexComparisonOperand {
    Operand(NodeIndexOperand),
    Index(NodeIndex),
}

impl DeepClone for NodeIndexComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Index(index) => Self::Index(index.clone()),
        }
    }
}

impl From<Wrapper<NodeIndexOperand>> for NodeIndexComparisonOperand {
    fn from(index: Wrapper<NodeIndexOperand>) -> Self {
        Self::Operand(index.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeIndexOperand>> for NodeIndexComparisonOperand {
    fn from(index: &Wrapper<NodeIndexOperand>) -> Self {
        Self::Operand(index.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<NodeIndex>> From<V> for NodeIndexComparisonOperand {
    fn from(index: V) -> Self {
        Self::Index(index.into())
    }
}

impl NodeIndexComparisonOperand {
    pub(crate) fn evaluate_backward(
        &self,
        medrecord: &MedRecord,
    ) -> MedRecordResult<Option<NodeIndex>> {
        match self {
            Self::Operand(operand) => operand.evaluate_backward(medrecord),
            Self::Index(index) => Ok(Some(index.clone())),
        }
    }
}

#[derive(Debug, Clone)]
pub enum NodeIndicesComparisonOperand {
    Operand(NodeIndicesOperand),
    Indices(HashSet<NodeIndex>),
}

impl DeepClone for NodeIndicesComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Indices(indices) => Self::Indices(indices.clone()),
        }
    }
}

impl From<Wrapper<NodeIndicesOperand>> for NodeIndicesComparisonOperand {
    fn from(indices: Wrapper<NodeIndicesOperand>) -> Self {
        Self::Operand(indices.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeIndicesOperand>> for NodeIndicesComparisonOperand {
    fn from(indices: &Wrapper<NodeIndicesOperand>) -> Self {
        Self::Operand(indices.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<NodeIndex>> From<Vec<V>> for NodeIndicesComparisonOperand {
    fn from(indices: Vec<V>) -> Self {
        Self::Indices(indices.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<NodeIndex>> From<HashSet<V>> for NodeIndicesComparisonOperand {
    fn from(indices: HashSet<V>) -> Self {
        Self::Indices(indices.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<NodeIndex> + Clone, const N: usize> From<[V; N]> for NodeIndicesComparisonOperand {
    fn from(indices: [V; N]) -> Self {
        indices.to_vec().into()
    }
}

impl NodeIndicesComparisonOperand {
    pub(crate) fn evaluate_backward(
        &self,
        medrecord: &MedRecord,
    ) -> MedRecordResult<HashSet<NodeIndex>> {
        Ok(match self {
            Self::Operand(operand) => operand.evaluate_backward(medrecord)?.collect(),
            Self::Indices(indices) => indices.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct NodeIndicesOperand {
    context: NodeOperand,
    operations: Vec<NodeIndicesOperation>,
}

impl DeepClone for NodeIndicesOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a> EvaluateForward<'a> for NodeIndicesOperand {
    type InputValue = BoxedIterator<'a, NodeIndex>;
    type ReturnValue = BoxedIterator<'a, NodeIndex>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        node_indices: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        let node_indices = Box::new(node_indices) as BoxedIterator<NodeIndex>;

        self.operations
            .iter()
            .try_fold(node_indices, |index_tuples, operation| {
                operation.evaluate(medrecord, index_tuples)
            })
    }
}

impl<'a> EvaluateForwardGrouped<'a> for NodeIndicesOperand {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        self.operations
            .iter()
            .try_fold(node_indices, |index_tuples, operation| {
                operation.evaluate_grouped(medrecord, index_tuples)
            })
    }
}

impl<'a> EvaluateBackward<'a> for NodeIndicesOperand {
    type ReturnValue = BoxedIterator<'a, NodeIndex>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let node_indices = self.context.evaluate_backward(medrecord)?;

        let node_indices = self.reduce_input(node_indices)?;

        self.evaluate_forward(medrecord, Box::new(node_indices))
    }
}

impl<'a> ReduceInput<'a> for NodeIndicesOperand {
    type Context = NodeOperand;

    #[inline]
    fn reduce_input(
        &self,
        node_indices: <Self::Context as EvaluateBackward<'a>>::ReturnValue,
    ) -> MedRecordResult<<Self as EvaluateForward<'a>>::InputValue> {
        Ok(Box::new(node_indices.cloned()))
    }
}

impl Max for NodeIndicesOperand {
    type ReturnOperand = NodeIndexOperand;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKind::Max);

        self.operations
            .push(NodeIndicesOperation::NodeIndexOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl Min for NodeIndicesOperand {
    type ReturnOperand = NodeIndexOperand;

    fn min(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKind::Min);

        self.operations
            .push(NodeIndicesOperation::NodeIndexOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl Count for NodeIndicesOperand {
    type ReturnOperand = NodeIndexOperand;

    fn count(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKind::Count);

        self.operations
            .push(NodeIndicesOperation::NodeIndexOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl Sum for NodeIndicesOperand {
    type ReturnOperand = NodeIndexOperand;

    fn sum(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKind::Sum);

        self.operations
            .push(NodeIndicesOperation::NodeIndexOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl Random for NodeIndicesOperand {
    type ReturnOperand = NodeIndexOperand;

    fn random(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<Self::ReturnOperand>::new(self.deep_clone(), SingleKind::Random);

        self.operations
            .push(NodeIndicesOperation::NodeIndexOperation {
                operand: operand.clone(),
            });

        operand
    }
}

impl GreaterThan for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThan,
            });
    }
}

impl GreaterThanOrEqualTo for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThanOrEqualTo,
            });
    }
}

impl LessThan for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThan,
            });
    }
}

impl LessThanOrEqualTo for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThanOrEqualTo,
            });
    }
}

impl EqualTo for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EqualTo,
            });
    }
}

impl NotEqualTo for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::NotEqualTo,
            });
    }
}

impl StartsWith for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::StartsWith,
            });
    }
}

impl EndsWith for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EndsWith,
            });
    }
}

impl Contains for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::Contains,
            });
    }
}

impl IsIn for NodeIndicesOperand {
    type ComparisonOperand = NodeIndicesComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }
}

impl IsNotIn for NodeIndicesOperand {
    type ComparisonOperand = NodeIndicesComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsNotIn,
            });
    }
}

impl Add for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::BinaryArithmeticOpration {
                operand: value.into(),
                kind: BinaryArithmeticKind::Add,
            });
    }
}

impl Sub for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::BinaryArithmeticOpration {
                operand: value.into(),
                kind: BinaryArithmeticKind::Sub,
            });
    }
}

impl Mul for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::BinaryArithmeticOpration {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mul,
            });
    }
}

impl Pow for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::BinaryArithmeticOpration {
                operand: value.into(),
                kind: BinaryArithmeticKind::Pow,
            });
    }
}

impl Mod for NodeIndicesOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndicesOperation::BinaryArithmeticOpration {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mod,
            });
    }
}

impl Abs for NodeIndicesOperand {
    fn abs(&mut self) {
        self.operations
            .push(NodeIndicesOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Abs,
            });
    }
}

impl Trim for NodeIndicesOperand {
    fn trim(&mut self) {
        self.operations
            .push(NodeIndicesOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Trim,
            });
    }
}

impl TrimStart for NodeIndicesOperand {
    fn trim_start(&mut self) {
        self.operations
            .push(NodeIndicesOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimStart,
            });
    }
}

impl TrimEnd for NodeIndicesOperand {
    fn trim_end(&mut self) {
        self.operations
            .push(NodeIndicesOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimEnd,
            });
    }
}

impl Lowercase for NodeIndicesOperand {
    fn lowercase(&mut self) {
        self.operations
            .push(NodeIndicesOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Lowercase,
            });
    }
}

impl Uppercase for NodeIndicesOperand {
    fn uppercase(&mut self) {
        self.operations
            .push(NodeIndicesOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Uppercase,
            });
    }
}

impl Slice for NodeIndicesOperand {
    fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(NodeIndicesOperation::Slice(start..end));
    }
}

impl IsString for NodeIndicesOperand {
    fn is_string(&mut self) {
        self.operations.push(NodeIndicesOperation::IsString);
    }
}

impl IsInt for NodeIndicesOperand {
    fn is_int(&mut self) {
        self.operations.push(NodeIndicesOperation::IsInt);
    }
}

impl IsMax for NodeIndicesOperand {
    fn is_max(&mut self) {
        self.operations.push(NodeIndicesOperation::IsMax);
    }
}

impl IsMin for NodeIndicesOperand {
    fn is_min(&mut self) {
        self.operations.push(NodeIndicesOperation::IsMin);
    }
}

impl EitherOr for NodeIndicesOperand {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut either_operand = Wrapper::<Self::QueryOperand>::new(self.context.clone());
        let mut or_operand = Wrapper::<Self::QueryOperand>::new(self.context.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(NodeIndicesOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }
}

impl Exclude for NodeIndicesOperand {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand = Wrapper::<Self::QueryOperand>::new(self.context.clone());

        query(&mut operand);

        self.operations
            .push(NodeIndicesOperation::Exclude { operand });
    }
}

impl NodeIndicesOperand {
    pub(crate) fn new(context: NodeOperand) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }
}

impl Wrapper<NodeIndicesOperand> {
    pub(crate) fn new(context: NodeOperand) -> Self {
        NodeIndicesOperand::new(context).into()
    }
}

#[derive(Debug, Clone)]
pub struct NodeIndexOperand {
    context: NodeIndicesOperand,
    pub(crate) kind: SingleKind,
    operations: Vec<NodeIndexOperation>,
}

impl DeepClone for NodeIndexOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self.operations.iter().map(DeepClone::deep_clone).collect(),
        }
    }
}

impl<'a> EvaluateForward<'a> for NodeIndexOperand {
    type InputValue = Option<NodeIndex>;
    type ReturnValue = Option<NodeIndex>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        node_index: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        self.operations
            .iter()
            .try_fold(node_index, |node_index, operation| {
                operation.evaluate(medrecord, node_index)
            })
    }
}

impl<'a> EvaluateForwardGrouped<'a> for NodeIndexOperand {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        node_indices: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
        self.operations
            .iter()
            .try_fold(node_indices, |node_indices, operation| {
                operation.evaluate_grouped(medrecord, node_indices)
            })
    }
}

impl<'a> EvaluateBackward<'a> for NodeIndexOperand {
    type ReturnValue = Option<NodeIndex>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let node_indices = self.context.evaluate_backward(medrecord)?;

        let node_index = self.reduce_input(node_indices)?;

        self.evaluate_forward(medrecord, node_index)
    }
}

impl<'a> ReduceInput<'a> for NodeIndexOperand {
    type Context = NodeIndicesOperand;

    #[inline]
    fn reduce_input(
        &self,
        node_indices: <Self::Context as EvaluateBackward<'a>>::ReturnValue,
    ) -> MedRecordResult<<Self as EvaluateForward<'a>>::InputValue> {
        Ok(match self.kind {
            SingleKind::Max => NodeIndicesOperation::get_max(node_indices)?,
            SingleKind::Min => NodeIndicesOperation::get_min(node_indices)?,
            SingleKind::Count => Some(NodeIndicesOperation::get_count(node_indices)),
            SingleKind::Sum => NodeIndicesOperation::get_sum(node_indices)?,
            SingleKind::Random => NodeIndicesOperation::get_random(node_indices),
        })
    }
}

impl GreaterThan for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThan,
            });
    }
}

impl GreaterThanOrEqualTo for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::GreaterThanOrEqualTo,
            });
    }
}

impl LessThan for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThan,
            });
    }
}

impl LessThanOrEqualTo for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::LessThanOrEqualTo,
            });
    }
}

impl EqualTo for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EqualTo,
            });
    }
}

impl NotEqualTo for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::NotEqualTo,
            });
    }
}

impl StartsWith for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::StartsWith,
            });
    }
}

impl EndsWith for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::EndsWith,
            });
    }
}

impl Contains for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndexComparisonOperation {
                operand: value.into(),
                kind: SingleComparisonKind::Contains,
            });
    }
}

impl IsIn for NodeIndexOperand {
    type ComparisonOperand = NodeIndicesComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }
}

impl IsNotIn for NodeIndexOperand {
    type ComparisonOperand = NodeIndicesComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsNotIn,
            });
    }
}

impl Add for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::BinaryArithmeticOpration {
                operand: value.into(),
                kind: BinaryArithmeticKind::Add,
            });
    }
}

impl Sub for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::BinaryArithmeticOpration {
                operand: value.into(),
                kind: BinaryArithmeticKind::Sub,
            });
    }
}

impl Mul for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::BinaryArithmeticOpration {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mul,
            });
    }
}

impl Pow for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::BinaryArithmeticOpration {
                operand: value.into(),
                kind: BinaryArithmeticKind::Pow,
            });
    }
}

impl Mod for NodeIndexOperand {
    type ComparisonOperand = NodeIndexComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operations
            .push(NodeIndexOperation::BinaryArithmeticOpration {
                operand: value.into(),
                kind: BinaryArithmeticKind::Mod,
            });
    }
}

impl Abs for NodeIndexOperand {
    fn abs(&mut self) {
        self.operations
            .push(NodeIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Abs,
            });
    }
}

impl Trim for NodeIndexOperand {
    fn trim(&mut self) {
        self.operations
            .push(NodeIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Trim,
            });
    }
}

impl TrimStart for NodeIndexOperand {
    fn trim_start(&mut self) {
        self.operations
            .push(NodeIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimStart,
            });
    }
}

impl TrimEnd for NodeIndexOperand {
    fn trim_end(&mut self) {
        self.operations
            .push(NodeIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::TrimEnd,
            });
    }
}

impl Lowercase for NodeIndexOperand {
    fn lowercase(&mut self) {
        self.operations
            .push(NodeIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Lowercase,
            });
    }
}

impl Uppercase for NodeIndexOperand {
    fn uppercase(&mut self) {
        self.operations
            .push(NodeIndexOperation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::Uppercase,
            });
    }
}

impl Slice for NodeIndexOperand {
    fn slice(&mut self, start: usize, end: usize) {
        self.operations.push(NodeIndexOperation::Slice(start..end));
    }
}

impl IsString for NodeIndexOperand {
    fn is_string(&mut self) {
        self.operations.push(NodeIndexOperation::IsString);
    }
}

impl IsInt for NodeIndexOperand {
    fn is_int(&mut self) {
        self.operations.push(NodeIndexOperation::IsInt);
    }
}

impl EitherOr for NodeIndexOperand {
    type QueryOperand = Self;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut either_operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(NodeIndexOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }
}

impl Exclude for NodeIndexOperand {
    type QueryOperand = Self;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        let mut operand =
            Wrapper::<Self::QueryOperand>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(NodeIndexOperation::Exclude { operand });
    }
}

impl NodeIndexOperand {
    pub(crate) fn new(context: NodeIndicesOperand, kind: SingleKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }
}

impl Wrapper<NodeIndexOperand> {
    pub(crate) fn new(context: NodeIndicesOperand, kind: SingleKind) -> Self {
        NodeIndexOperand::new(context, kind).into()
    }
}
