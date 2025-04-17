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
            edges::{self, EdgeOperand},
            traits::{DeepClone, ReadWriteOrPanic},
            values::{self, MultipleValuesOperand},
            wrapper::{CardinalityWrapper, Wrapper},
            BoxedIterator,
        },
        Group, MedRecordAttribute, NodeIndex,
    },
    MedRecord,
};
use std::{collections::HashSet, fmt::Debug};

#[derive(Debug, Clone)]
pub struct NodeOperand {
    pub(crate) context: Option<Context>,
    operations: Vec<NodeOperation>,
}

impl DeepClone for NodeOperand {
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

impl NodeOperand {
    pub(crate) fn new(context: Option<Context>) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate_forward<'a>(
        &self,
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, &'a NodeIndex>> {
        let node_indices = Box::new(node_indices) as BoxedIterator<'a, &'a NodeIndex>;

        self.operations
            .iter()
            .try_fold(node_indices, |node_indices, operation| {
                operation.evaluate(medrecord, node_indices)
            })
    }

    pub(crate) fn evaluate_backward<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, &'a NodeIndex>> {
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

    pub fn attribute(
        &mut self,
        attribute: MedRecordAttribute,
    ) -> Wrapper<MultipleValuesOperand<Self>> {
        let operand = Wrapper::<MultipleValuesOperand<Self>>::new(values::Context::Operand((
            self.deep_clone(),
            attribute,
        )));

        self.operations.push(NodeOperation::Values {
            operand: operand.clone(),
        });

        operand
    }

    pub fn attributes(&mut self) -> Wrapper<AttributesTreeOperand<Self>> {
        let operand = Wrapper::<AttributesTreeOperand<Self>>::new(self.deep_clone());

        self.operations.push(NodeOperation::Attributes {
            operand: operand.clone(),
        });

        operand
    }

    pub fn index(&mut self) -> Wrapper<NodeIndicesOperand> {
        let operand = Wrapper::<NodeIndicesOperand>::new(self.deep_clone());

        self.operations.push(NodeOperation::Indices {
            operand: operand.clone(),
        });

        operand
    }

    pub fn in_group<G>(&mut self, group: G)
    where
        G: Into<CardinalityWrapper<Group>>,
    {
        self.operations.push(NodeOperation::InGroup {
            group: group.into(),
        });
    }

    pub fn has_attribute<A>(&mut self, attribute: A)
    where
        A: Into<CardinalityWrapper<MedRecordAttribute>>,
    {
        self.operations.push(NodeOperation::HasAttribute {
            attribute: attribute.into(),
        });
    }

    pub fn edges(&mut self, direction: EdgeDirection) -> Wrapper<EdgeOperand> {
        let operand = Wrapper::<EdgeOperand>::new(Some(edges::Context::Edges {
            operand: Box::new(self.clone()),
            kind: direction.clone(),
        }));

        self.operations.push(NodeOperation::Edges {
            operand: operand.clone(),
            direction,
        });

        operand
    }

    pub fn neighbors(&mut self, direction: EdgeDirection) -> Wrapper<NodeOperand> {
        let operand = Wrapper::<NodeOperand>::new(Some(Context::Neighbors {
            operand: Box::new(self.clone()),
            direction: direction.clone(),
        }));

        self.operations.push(NodeOperation::Neighbors {
            operand: operand.clone(),
            direction,
        });

        operand
    }

    pub fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<NodeOperand>),
        OQ: FnOnce(&mut Wrapper<NodeOperand>),
    {
        let mut either_operand = Wrapper::<NodeOperand>::new(self.context.clone());
        let mut or_operand = Wrapper::<NodeOperand>::new(self.context.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(NodeOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<NodeOperand>),
    {
        let mut operand = Wrapper::<NodeOperand>::new(self.context.clone());

        query(&mut operand);

        self.operations.push(NodeOperation::Exclude { operand });
    }
}

impl Wrapper<NodeOperand> {
    pub(crate) fn new(context: Option<Context>) -> Self {
        NodeOperand::new(context).into()
    }

    pub(crate) fn evaluate_forward<'a>(
        &self,
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
    ) -> MedRecordResult<BoxedIterator<'a, &'a NodeIndex>> {
        self.0
            .read_or_panic()
            .evaluate_forward(medrecord, node_indices)
    }

    pub fn attribute<A>(&mut self, attribute: A) -> Wrapper<MultipleValuesOperand<NodeOperand>>
    where
        A: Into<MedRecordAttribute>,
    {
        self.0.write_or_panic().attribute(attribute.into())
    }

    pub fn attributes(&mut self) -> Wrapper<AttributesTreeOperand<NodeOperand>> {
        self.0.write_or_panic().attributes()
    }

    pub fn index(&mut self) -> Wrapper<NodeIndicesOperand> {
        self.0.write_or_panic().index()
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

    pub fn edges(&mut self, direction: EdgeDirection) -> Wrapper<EdgeOperand> {
        self.0.write_or_panic().edges(direction)
    }

    pub fn neighbors(&mut self, direction: EdgeDirection) -> Wrapper<NodeOperand> {
        self.0.write_or_panic().neighbors(direction)
    }

    pub fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<NodeOperand>),
        OQ: FnOnce(&mut Wrapper<NodeOperand>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<NodeOperand>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

macro_rules! implement_index_operation {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self) -> Wrapper<NodeIndexOperand> {
            let operand = Wrapper::<NodeIndexOperand>::new(self.deep_clone(), SingleKind::$variant);

            self.operations
                .push(NodeIndicesOperation::NodeIndexOperation {
                    operand: operand.clone(),
                });

            operand
        }
    };
}

macro_rules! implement_single_index_comparison_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name<V: Into<NodeIndexComparisonOperand>>(&mut self, index: V) {
            self.operations
                .push($operation::NodeIndexComparisonOperation {
                    operand: index.into(),
                    kind: SingleComparisonKind::$kind,
                });
        }
    };
}

macro_rules! implement_binary_arithmetic_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name<V: Into<NodeIndexComparisonOperand>>(&mut self, index: V) {
            self.operations.push($operation::BinaryArithmeticOpration {
                operand: index.into(),
                kind: BinaryArithmeticKind::$kind,
            });
        }
    };
}

macro_rules! implement_unary_arithmetic_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name(&mut self) {
            self.operations.push($operation::UnaryArithmeticOperation {
                kind: UnaryArithmeticKind::$kind,
            });
        }
    };
}

macro_rules! implement_assertion_operation {
    ($name:ident, $operation:expr) => {
        pub fn $name(&mut self) {
            self.operations.push($operation);
        }
    };
}

macro_rules! implement_wrapper_operand {
    ($name:ident) => {
        pub fn $name(&self) {
            self.0.write_or_panic().$name()
        }
    };
}

macro_rules! implement_wrapper_operand_with_return {
    ($name:ident, $return_operand:ident) => {
        pub fn $name(&self) -> Wrapper<$return_operand> {
            self.0.write_or_panic().$name()
        }
    };
}

macro_rules! implement_wrapper_operand_with_argument {
    ($name:ident, $index_type:ty) => {
        pub fn $name(&self, index: $index_type) {
            self.0.write_or_panic().$name(index)
        }
    };
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
    pub(crate) context: NodeOperand,
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

impl NodeIndicesOperand {
    pub(crate) fn new(context: NodeOperand) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate_forward<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = NodeIndex> + 'a,
    ) -> MedRecordResult<impl Iterator<Item = NodeIndex> + 'a> {
        let indices = Box::new(indices) as BoxedIterator<NodeIndex>;

        self.operations
            .iter()
            .try_fold(indices, |index_tuples, operation| {
                operation.evaluate(medrecord, index_tuples)
            })
    }

    pub(crate) fn evaluate_backward<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = NodeIndex> + 'a> {
        let node_indices = self.context.evaluate_backward(medrecord)?.cloned();

        self.evaluate_forward(medrecord, node_indices)
    }

    implement_index_operation!(max, Max);
    implement_index_operation!(min, Min);
    implement_index_operation!(count, Count);
    implement_index_operation!(sum, Sum);
    implement_index_operation!(first, First);
    implement_index_operation!(last, Last);

    implement_single_index_comparison_operation!(greater_than, NodeIndicesOperation, GreaterThan);
    implement_single_index_comparison_operation!(
        greater_than_or_equal_to,
        NodeIndicesOperation,
        GreaterThanOrEqualTo
    );
    implement_single_index_comparison_operation!(less_than, NodeIndicesOperation, LessThan);
    implement_single_index_comparison_operation!(
        less_than_or_equal_to,
        NodeIndicesOperation,
        LessThanOrEqualTo
    );
    implement_single_index_comparison_operation!(equal_to, NodeIndicesOperation, EqualTo);
    implement_single_index_comparison_operation!(not_equal_to, NodeIndicesOperation, NotEqualTo);
    implement_single_index_comparison_operation!(starts_with, NodeIndicesOperation, StartsWith);
    implement_single_index_comparison_operation!(ends_with, NodeIndicesOperation, EndsWith);
    implement_single_index_comparison_operation!(contains, NodeIndicesOperation, Contains);

    pub fn is_in<V: Into<NodeIndicesComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }

    pub fn is_not_in<V: Into<NodeIndicesComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsNotIn,
            });
    }

    implement_binary_arithmetic_operation!(add, NodeIndicesOperation, Add);
    implement_binary_arithmetic_operation!(sub, NodeIndicesOperation, Sub);
    implement_binary_arithmetic_operation!(mul, NodeIndicesOperation, Mul);
    implement_binary_arithmetic_operation!(pow, NodeIndicesOperation, Pow);
    implement_binary_arithmetic_operation!(r#mod, NodeIndicesOperation, Mod);

    implement_unary_arithmetic_operation!(abs, NodeIndicesOperation, Abs);
    implement_unary_arithmetic_operation!(trim, NodeIndicesOperation, Trim);
    implement_unary_arithmetic_operation!(trim_start, NodeIndicesOperation, TrimStart);
    implement_unary_arithmetic_operation!(trim_end, NodeIndicesOperation, TrimEnd);
    implement_unary_arithmetic_operation!(lowercase, NodeIndicesOperation, Lowercase);
    implement_unary_arithmetic_operation!(uppercase, NodeIndicesOperation, Uppercase);

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations
            .push(NodeIndicesOperation::Slice(start..end));
    }

    implement_assertion_operation!(is_string, NodeIndicesOperation::IsString);
    implement_assertion_operation!(is_int, NodeIndicesOperation::IsInt);
    implement_assertion_operation!(is_max, NodeIndicesOperation::IsMax);
    implement_assertion_operation!(is_min, NodeIndicesOperation::IsMin);

    pub fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<NodeIndicesOperand>),
        OQ: FnOnce(&mut Wrapper<NodeIndicesOperand>),
    {
        let mut either_operand = Wrapper::<NodeIndicesOperand>::new(self.context.clone());
        let mut or_operand = Wrapper::<NodeIndicesOperand>::new(self.context.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(NodeIndicesOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<NodeIndicesOperand>),
    {
        let mut operand = Wrapper::<NodeIndicesOperand>::new(self.context.clone());

        query(&mut operand);

        self.operations
            .push(NodeIndicesOperation::Exclude { operand });
    }
}

impl Wrapper<NodeIndicesOperand> {
    pub(crate) fn new(context: NodeOperand) -> Self {
        NodeIndicesOperand::new(context).into()
    }

    pub(crate) fn evaluate_forward<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = NodeIndex> + 'a,
    ) -> MedRecordResult<impl Iterator<Item = NodeIndex> + 'a> {
        self.0.read_or_panic().evaluate_forward(medrecord, indices)
    }

    pub(crate) fn evaluate_backward<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = NodeIndex> + 'a> {
        self.0.read_or_panic().evaluate_backward(medrecord)
    }

    implement_wrapper_operand_with_return!(max, NodeIndexOperand);
    implement_wrapper_operand_with_return!(min, NodeIndexOperand);
    implement_wrapper_operand_with_return!(count, NodeIndexOperand);
    implement_wrapper_operand_with_return!(sum, NodeIndexOperand);
    implement_wrapper_operand_with_return!(first, NodeIndexOperand);
    implement_wrapper_operand_with_return!(last, NodeIndexOperand);

    implement_wrapper_operand_with_argument!(greater_than, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<NodeIndexComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(less_than, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<NodeIndexComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(equal_to, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(not_equal_to, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(starts_with, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(ends_with, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(contains, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_in, impl Into<NodeIndicesComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_not_in, impl Into<NodeIndicesComparisonOperand>);
    implement_wrapper_operand_with_argument!(add, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(sub, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(mul, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(pow, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<NodeIndexComparisonOperand>);

    implement_wrapper_operand!(abs);
    implement_wrapper_operand!(trim);
    implement_wrapper_operand!(trim_start);
    implement_wrapper_operand!(trim_end);
    implement_wrapper_operand!(lowercase);
    implement_wrapper_operand!(uppercase);

    pub fn slice(&self, start: usize, end: usize) {
        self.0.write_or_panic().slice(start, end)
    }

    implement_wrapper_operand!(is_string);
    implement_wrapper_operand!(is_int);
    implement_wrapper_operand!(is_max);
    implement_wrapper_operand!(is_min);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<NodeIndicesOperand>),
        OQ: FnOnce(&mut Wrapper<NodeIndicesOperand>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<NodeIndicesOperand>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

#[derive(Debug, Clone)]
pub struct NodeIndexOperand {
    pub(crate) context: NodeIndicesOperand,
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

impl NodeIndexOperand {
    pub(crate) fn new(context: NodeIndicesOperand, kind: SingleKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate_forward(
        &self,
        medrecord: &MedRecord,
        index: NodeIndex,
    ) -> MedRecordResult<Option<NodeIndex>> {
        self.operations
            .iter()
            .try_fold(Some(index), |index, operation| {
                if let Some(index) = index {
                    operation.evaluate(medrecord, index)
                } else {
                    Ok(None)
                }
            })
    }

    pub(crate) fn evaluate_backward(
        &self,
        medrecord: &MedRecord,
    ) -> MedRecordResult<Option<NodeIndex>> {
        let node_indices = self.context.evaluate_backward(medrecord)?;

        let node_index = match self.kind {
            SingleKind::Max => NodeIndicesOperation::get_max(node_indices)?.clone(),
            SingleKind::Min => NodeIndicesOperation::get_min(node_indices)?.clone(),
            SingleKind::Count => NodeIndicesOperation::get_count(node_indices),
            SingleKind::Sum => NodeIndicesOperation::get_sum(node_indices)?,
            SingleKind::First => NodeIndicesOperation::get_first(node_indices)?,
            SingleKind::Last => NodeIndicesOperation::get_last(node_indices)?,
        };

        self.evaluate_forward(medrecord, node_index)
    }

    implement_single_index_comparison_operation!(greater_than, NodeIndexOperation, GreaterThan);
    implement_single_index_comparison_operation!(
        greater_than_or_equal_to,
        NodeIndexOperation,
        GreaterThanOrEqualTo
    );
    implement_single_index_comparison_operation!(less_than, NodeIndexOperation, LessThan);
    implement_single_index_comparison_operation!(
        less_than_or_equal_to,
        NodeIndexOperation,
        LessThanOrEqualTo
    );
    implement_single_index_comparison_operation!(equal_to, NodeIndexOperation, EqualTo);
    implement_single_index_comparison_operation!(not_equal_to, NodeIndexOperation, NotEqualTo);
    implement_single_index_comparison_operation!(starts_with, NodeIndexOperation, StartsWith);
    implement_single_index_comparison_operation!(ends_with, NodeIndexOperation, EndsWith);
    implement_single_index_comparison_operation!(contains, NodeIndexOperation, Contains);

    pub fn is_in<V: Into<NodeIndicesComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }

    pub fn is_not_in<V: Into<NodeIndicesComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsNotIn,
            });
    }

    implement_binary_arithmetic_operation!(add, NodeIndexOperation, Add);
    implement_binary_arithmetic_operation!(sub, NodeIndexOperation, Sub);
    implement_binary_arithmetic_operation!(mul, NodeIndexOperation, Mul);
    implement_binary_arithmetic_operation!(pow, NodeIndexOperation, Pow);
    implement_binary_arithmetic_operation!(r#mod, NodeIndexOperation, Mod);

    implement_unary_arithmetic_operation!(abs, NodeIndexOperation, Abs);
    implement_unary_arithmetic_operation!(trim, NodeIndexOperation, Trim);
    implement_unary_arithmetic_operation!(trim_start, NodeIndexOperation, TrimStart);
    implement_unary_arithmetic_operation!(trim_end, NodeIndexOperation, TrimEnd);
    implement_unary_arithmetic_operation!(lowercase, NodeIndexOperation, Lowercase);
    implement_unary_arithmetic_operation!(uppercase, NodeIndexOperation, Uppercase);

    pub fn slice(&mut self, start: usize, end: usize) {
        self.operations.push(NodeIndexOperation::Slice(start..end));
    }

    implement_assertion_operation!(is_string, NodeIndexOperation::IsString);
    implement_assertion_operation!(is_int, NodeIndexOperation::IsInt);
    pub fn eiter_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<NodeIndexOperand>),
        OQ: FnOnce(&mut Wrapper<NodeIndexOperand>),
    {
        let mut either_operand =
            Wrapper::<NodeIndexOperand>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<NodeIndexOperand>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(NodeIndexOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<NodeIndexOperand>),
    {
        let mut operand = Wrapper::<NodeIndexOperand>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(NodeIndexOperation::Exclude { operand });
    }
}

impl Wrapper<NodeIndexOperand> {
    pub(crate) fn new(context: NodeIndicesOperand, kind: SingleKind) -> Self {
        NodeIndexOperand::new(context, kind).into()
    }

    pub(crate) fn evaluate_forward(
        &self,
        medrecord: &MedRecord,
        index: NodeIndex,
    ) -> MedRecordResult<Option<NodeIndex>> {
        self.0.read_or_panic().evaluate_forward(medrecord, index)
    }

    pub(crate) fn evaluate_backward(
        &self,
        medrecord: &MedRecord,
    ) -> MedRecordResult<Option<NodeIndex>> {
        self.0.read_or_panic().evaluate_backward(medrecord)
    }

    implement_wrapper_operand_with_argument!(greater_than, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<NodeIndexComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(less_than, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<NodeIndexComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(equal_to, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(not_equal_to, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(starts_with, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(ends_with, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(contains, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_in, impl Into<NodeIndicesComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_not_in, impl Into<NodeIndicesComparisonOperand>);
    implement_wrapper_operand_with_argument!(add, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(sub, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(mul, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(pow, impl Into<NodeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<NodeIndexComparisonOperand>);

    implement_wrapper_operand!(abs);
    implement_wrapper_operand!(trim);
    implement_wrapper_operand!(trim_start);
    implement_wrapper_operand!(trim_end);
    implement_wrapper_operand!(lowercase);
    implement_wrapper_operand!(uppercase);

    pub fn slice(&self, start: usize, end: usize) {
        self.0.write_or_panic().slice(start, end)
    }

    implement_wrapper_operand!(is_string);
    implement_wrapper_operand!(is_int);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<NodeIndexOperand>),
        OQ: FnOnce(&mut Wrapper<NodeIndexOperand>),
    {
        self.0.write_or_panic().eiter_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<NodeIndexOperand>),
    {
        self.0.write_or_panic().exclude(query);
    }
}
