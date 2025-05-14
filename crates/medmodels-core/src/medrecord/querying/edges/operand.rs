use super::{
    operation::{EdgeIndexOperation, EdgeIndicesOperation, EdgeOperation},
    BinaryArithmeticKind, Context, MultipleComparisonKind, SingleComparisonKind, SingleKind,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            attributes::AttributesTreeOperand,
            nodes::{self, NodeOperand},
            operand_traits::Attribute,
            values::{self, MultipleValuesOperand},
            wrapper::{CardinalityWrapper, Wrapper},
            BoxedIterator, DeepClone, EvaluateBackward, EvaluateForward, ReadWriteOrPanic,
        },
        EdgeIndex, Group, MedRecordAttribute,
    },
    MedRecord,
};
use medmodels_utils::aliases::MrHashSet;
use std::{collections::HashSet, fmt::Debug};

#[derive(Debug, Clone)]
pub struct EdgeOperand {
    context: Option<Context>,
    pub(crate) operations: Vec<EdgeOperation>,
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

impl<'a> EvaluateForward<'a> for EdgeOperand {
    type InputValue = BoxedIterator<'a, &'a EdgeIndex>;
    type ReturnValue = BoxedIterator<'a, &'a EdgeIndex>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        edge_indices: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        self.operations
            .iter()
            .try_fold(edge_indices, |edge_indices, operation| {
                operation.evaluate(medrecord, edge_indices)
            })
    }
}

impl<'a> EvaluateBackward<'a> for EdgeOperand {
    type ReturnValue = BoxedIterator<'a, &'a EdgeIndex>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let edge_indices: BoxedIterator<&EdgeIndex> = match &self.context {
            Some(Context::Edges { operand, kind }) => {
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
            None => Box::new(medrecord.edge_indices()),
        };

        self.evaluate_forward(medrecord, edge_indices)
    }
}

impl Attribute for EdgeOperand {
    type ReturnOperand = MultipleValuesOperand<Self>;

    fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<Self::ReturnOperand> {
        let operand = Wrapper::<MultipleValuesOperand<Self>>::new(values::Context::Operand((
            self.deep_clone(),
            attribute,
        )));

        self.operations.push(EdgeOperation::Values {
            operand: operand.clone(),
        });

        operand
    }
}

impl EdgeOperand {
    pub(crate) fn new(context: Option<Context>) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    pub fn attributes(&mut self) -> Wrapper<AttributesTreeOperand<Self>> {
        let operand = Wrapper::<AttributesTreeOperand<Self>>::new(self.deep_clone());

        self.operations.push(EdgeOperation::Attributes {
            operand: operand.clone(),
        });

        operand
    }

    pub fn index(&mut self) -> Wrapper<EdgeIndicesOperand> {
        let operand = Wrapper::<EdgeIndicesOperand>::new(self.deep_clone());

        self.operations.push(EdgeOperation::Indices {
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
        let operand = Wrapper::<NodeOperand>::new(Some(nodes::Context::SourceNode {
            operand: self.deep_clone(),
        }));

        self.operations.push(EdgeOperation::SourceNode {
            operand: operand.clone(),
        });

        operand
    }

    pub fn target_node(&mut self) -> Wrapper<NodeOperand> {
        let operand = Wrapper::<NodeOperand>::new(Some(nodes::Context::TargetNode {
            operand: self.deep_clone(),
        }));

        self.operations.push(EdgeOperation::TargetNode {
            operand: operand.clone(),
        });

        operand
    }

    pub fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<EdgeOperand>),
        OQ: FnOnce(&mut Wrapper<EdgeOperand>),
    {
        let mut either_operand = Wrapper::<EdgeOperand>::new(self.context.deep_clone());
        let mut or_operand = Wrapper::<EdgeOperand>::new(self.context.deep_clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(EdgeOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<EdgeOperand>),
    {
        let mut operand = Wrapper::<EdgeOperand>::new(self.context.deep_clone());

        query(&mut operand);

        self.operations.push(EdgeOperation::Exclude { operand });
    }
}

impl Wrapper<EdgeOperand> {
    pub(crate) fn new(context: Option<Context>) -> Self {
        EdgeOperand::new(context).into()
    }

    pub fn attributes(&self) -> Wrapper<AttributesTreeOperand<EdgeOperand>> {
        self.0.write_or_panic().attributes()
    }

    pub fn index(&self) -> Wrapper<EdgeIndicesOperand> {
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

    pub fn source_node(&self) -> Wrapper<NodeOperand> {
        self.0.write_or_panic().source_node()
    }

    pub fn target_node(&self) -> Wrapper<NodeOperand> {
        self.0.write_or_panic().target_node()
    }

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<EdgeOperand>),
        OQ: FnOnce(&mut Wrapper<EdgeOperand>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<EdgeOperand>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

macro_rules! implement_index_operation {
    ($name:ident, $variant:ident) => {
        pub fn $name(&mut self) -> Wrapper<EdgeIndexOperand> {
            let operand = Wrapper::<EdgeIndexOperand>::new(self.deep_clone(), SingleKind::$variant);

            self.operations
                .push(EdgeIndicesOperation::EdgeIndexOperation {
                    operand: operand.clone(),
                });

            operand
        }
    };
}

macro_rules! implement_single_index_comparison_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name<V: Into<EdgeIndexComparisonOperand>>(&mut self, index: V) {
            self.operations
                .push($operation::EdgeIndexComparisonOperation {
                    operand: index.into(),
                    kind: SingleComparisonKind::$kind,
                });
        }
    };
}

macro_rules! implement_binary_arithmetic_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name<V: Into<EdgeIndexComparisonOperand>>(&mut self, index: V) {
            self.operations.push($operation::BinaryArithmeticOpration {
                operand: index.into(),
                kind: BinaryArithmeticKind::$kind,
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
    context: EdgeOperand,
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
        let edge_indices = Box::new(edge_indices) as BoxedIterator<EdgeIndex>;

        self.operations
            .iter()
            .try_fold(edge_indices, |index_tuples, operation| {
                operation.evaluate(medrecord, index_tuples)
            })
    }
}

impl<'a> EvaluateBackward<'a> for EdgeIndicesOperand {
    type ReturnValue = BoxedIterator<'a, EdgeIndex>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let edge_indices = self.context.evaluate_backward(medrecord)?.cloned();

        self.evaluate_forward(medrecord, Box::new(edge_indices))
    }
}

impl EdgeIndicesOperand {
    pub(crate) fn new(context: EdgeOperand) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    implement_index_operation!(max, Max);
    implement_index_operation!(min, Min);
    implement_index_operation!(count, Count);
    implement_index_operation!(sum, Sum);
    implement_index_operation!(random, Random);

    implement_single_index_comparison_operation!(greater_than, EdgeIndicesOperation, GreaterThan);
    implement_single_index_comparison_operation!(
        greater_than_or_equal_to,
        EdgeIndicesOperation,
        GreaterThanOrEqualTo
    );
    implement_single_index_comparison_operation!(less_than, EdgeIndicesOperation, LessThan);
    implement_single_index_comparison_operation!(
        less_than_or_equal_to,
        EdgeIndicesOperation,
        LessThanOrEqualTo
    );
    implement_single_index_comparison_operation!(equal_to, EdgeIndicesOperation, EqualTo);
    implement_single_index_comparison_operation!(not_equal_to, EdgeIndicesOperation, NotEqualTo);
    implement_single_index_comparison_operation!(starts_with, EdgeIndicesOperation, StartsWith);
    implement_single_index_comparison_operation!(ends_with, EdgeIndicesOperation, EndsWith);
    implement_single_index_comparison_operation!(contains, EdgeIndicesOperation, Contains);

    pub fn is_in<V: Into<EdgeIndicesComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }

    pub fn is_not_in<V: Into<EdgeIndicesComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsNotIn,
            });
    }

    implement_binary_arithmetic_operation!(add, EdgeIndicesOperation, Add);
    implement_binary_arithmetic_operation!(sub, EdgeIndicesOperation, Sub);
    implement_binary_arithmetic_operation!(mul, EdgeIndicesOperation, Mul);
    implement_binary_arithmetic_operation!(pow, EdgeIndicesOperation, Pow);
    implement_binary_arithmetic_operation!(r#mod, EdgeIndicesOperation, Mod);

    implement_assertion_operation!(is_max, EdgeIndicesOperation::IsMax);
    implement_assertion_operation!(is_min, EdgeIndicesOperation::IsMin);

    pub fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<EdgeIndicesOperand>),
        OQ: FnOnce(&mut Wrapper<EdgeIndicesOperand>),
    {
        let mut either_operand = Wrapper::<EdgeIndicesOperand>::new(self.context.clone());
        let mut or_operand = Wrapper::<EdgeIndicesOperand>::new(self.context.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(EdgeIndicesOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<EdgeIndicesOperand>),
    {
        let mut operand = Wrapper::<EdgeIndicesOperand>::new(self.context.clone());

        query(&mut operand);

        self.operations
            .push(EdgeIndicesOperation::Exclude { operand });
    }
}

impl Wrapper<EdgeIndicesOperand> {
    pub(crate) fn new(context: EdgeOperand) -> Self {
        EdgeIndicesOperand::new(context).into()
    }

    implement_wrapper_operand_with_return!(max, EdgeIndexOperand);
    implement_wrapper_operand_with_return!(min, EdgeIndexOperand);
    implement_wrapper_operand_with_return!(count, EdgeIndexOperand);
    implement_wrapper_operand_with_return!(sum, EdgeIndexOperand);
    implement_wrapper_operand_with_return!(random, EdgeIndexOperand);

    implement_wrapper_operand_with_argument!(greater_than, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<EdgeIndexComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(less_than, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<EdgeIndexComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(equal_to, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(not_equal_to, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(starts_with, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(ends_with, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(contains, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_in, impl Into<EdgeIndicesComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_not_in, impl Into<EdgeIndicesComparisonOperand>);
    implement_wrapper_operand_with_argument!(add, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(sub, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(mul, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(pow, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<EdgeIndexComparisonOperand>);

    implement_wrapper_operand!(is_max);
    implement_wrapper_operand!(is_min);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<EdgeIndicesOperand>),
        OQ: FnOnce(&mut Wrapper<EdgeIndicesOperand>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<EdgeIndicesOperand>),
    {
        self.0.write_or_panic().exclude(query);
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
    type InputValue = EdgeIndex;
    type ReturnValue = Option<EdgeIndex>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        edge_index: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        self.operations
            .iter()
            .try_fold(Some(edge_index), |index, operation| {
                if let Some(index) = index {
                    operation.evaluate(medrecord, index)
                } else {
                    Ok(None)
                }
            })
    }
}

impl<'a> EvaluateBackward<'a> for EdgeIndexOperand {
    type ReturnValue = Option<EdgeIndex>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        let edge_indices = self.context.evaluate_backward(medrecord)?;

        let edge_index = match self.kind {
            SingleKind::Max => EdgeIndicesOperation::get_max(edge_indices)?,
            SingleKind::Min => EdgeIndicesOperation::get_min(edge_indices)?,
            SingleKind::Count => EdgeIndicesOperation::get_count(edge_indices),
            SingleKind::Sum => EdgeIndicesOperation::get_sum(edge_indices),
            SingleKind::Random => EdgeIndicesOperation::get_random(edge_indices)?,
        };

        self.evaluate_forward(medrecord, edge_index)
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

    implement_single_index_comparison_operation!(greater_than, EdgeIndexOperation, GreaterThan);
    implement_single_index_comparison_operation!(
        greater_than_or_equal_to,
        EdgeIndexOperation,
        GreaterThanOrEqualTo
    );
    implement_single_index_comparison_operation!(less_than, EdgeIndexOperation, LessThan);
    implement_single_index_comparison_operation!(
        less_than_or_equal_to,
        EdgeIndexOperation,
        LessThanOrEqualTo
    );
    implement_single_index_comparison_operation!(equal_to, EdgeIndexOperation, EqualTo);
    implement_single_index_comparison_operation!(not_equal_to, EdgeIndexOperation, NotEqualTo);
    implement_single_index_comparison_operation!(starts_with, EdgeIndexOperation, StartsWith);
    implement_single_index_comparison_operation!(ends_with, EdgeIndexOperation, EndsWith);
    implement_single_index_comparison_operation!(contains, EdgeIndexOperation, Contains);

    pub fn is_in<V: Into<EdgeIndicesComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }

    pub fn is_not_in<V: Into<EdgeIndicesComparisonOperand>>(&mut self, indices: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndicesComparisonOperation {
                operand: indices.into(),
                kind: MultipleComparisonKind::IsNotIn,
            });
    }

    implement_binary_arithmetic_operation!(add, EdgeIndexOperation, Add);
    implement_binary_arithmetic_operation!(sub, EdgeIndexOperation, Sub);
    implement_binary_arithmetic_operation!(mul, EdgeIndexOperation, Mul);
    implement_binary_arithmetic_operation!(pow, EdgeIndexOperation, Pow);
    implement_binary_arithmetic_operation!(r#mod, EdgeIndexOperation, Mod);

    pub fn eiter_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<EdgeIndexOperand>),
        OQ: FnOnce(&mut Wrapper<EdgeIndexOperand>),
    {
        let mut either_operand =
            Wrapper::<EdgeIndexOperand>::new(self.context.clone(), self.kind.clone());
        let mut or_operand =
            Wrapper::<EdgeIndexOperand>::new(self.context.clone(), self.kind.clone());

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(EdgeIndexOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }

    pub fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<EdgeIndexOperand>),
    {
        let mut operand = Wrapper::<EdgeIndexOperand>::new(self.context.clone(), self.kind.clone());

        query(&mut operand);

        self.operations
            .push(EdgeIndexOperation::Exclude { operand });
    }
}

impl Wrapper<EdgeIndexOperand> {
    pub(crate) fn new(context: EdgeIndicesOperand, kind: SingleKind) -> Self {
        EdgeIndexOperand::new(context, kind).into()
    }

    implement_wrapper_operand_with_argument!(greater_than, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        greater_than_or_equal_to,
        impl Into<EdgeIndexComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(less_than, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(
        less_than_or_equal_to,
        impl Into<EdgeIndexComparisonOperand>
    );
    implement_wrapper_operand_with_argument!(equal_to, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(not_equal_to, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(starts_with, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(ends_with, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(contains, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_in, impl Into<EdgeIndicesComparisonOperand>);
    implement_wrapper_operand_with_argument!(is_not_in, impl Into<EdgeIndicesComparisonOperand>);
    implement_wrapper_operand_with_argument!(add, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(sub, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(mul, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(pow, impl Into<EdgeIndexComparisonOperand>);
    implement_wrapper_operand_with_argument!(r#mod, impl Into<EdgeIndexComparisonOperand>);

    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<EdgeIndexOperand>),
        OQ: FnOnce(&mut Wrapper<EdgeIndexOperand>),
    {
        self.0.write_or_panic().eiter_or(either_query, or_query);
    }

    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<EdgeIndexOperand>),
    {
        self.0.write_or_panic().exclude(query);
    }
}
