use super::{
    operation::{EdgeDirection, NodeIndexOperation, NodeIndicesOperation, NodeOperation},
    BinaryArithmeticKind, MultipleComparisonKind, SingleComparisonKind, SingleKind,
    UnaryArithmeticKind,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            attributes::{self, AttributesTreeOperand},
            edges::EdgeOperand,
            traits::{DeepClone, ReadWriteOrPanic},
            values::{self, MultipleValuesOperand},
            wrapper::{CardinalityWrapper, Wrapper},
            BoxedIterator,
        },
        Group, MedRecordAttribute, NodeIndex,
    },
    MedRecord,
};
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct NodeOperand {
    operations: Vec<NodeOperation>,
}

impl DeepClone for NodeOperand {
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

impl NodeOperand {
    pub(crate) fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, &'a NodeIndex>> {
        let node_indices = Box::new(medrecord.node_indices()) as BoxedIterator<'a, &'a NodeIndex>;

        self.operations
            .iter()
            .try_fold(node_indices, |node_indices, operation| {
                operation.evaluate(medrecord, node_indices)
            })
    }

    pub fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<MultipleValuesOperand> {
        let operand = Wrapper::<MultipleValuesOperand>::new(
            values::Context::NodeOperand(self.deep_clone()),
            attribute,
        );

        self.operations.push(NodeOperation::Values {
            operand: operand.clone(),
        });

        operand
    }

    pub fn attributes(&mut self) -> Wrapper<AttributesTreeOperand> {
        let operand = Wrapper::<AttributesTreeOperand>::new(attributes::Context::NodeOperand(
            self.deep_clone(),
        ));

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

    pub fn outgoing_edges(&mut self) -> Wrapper<EdgeOperand> {
        let operand = Wrapper::<EdgeOperand>::new();

        self.operations.push(NodeOperation::OutgoingEdges {
            operand: operand.clone(),
        });

        operand
    }

    pub fn incoming_edges(&mut self) -> Wrapper<EdgeOperand> {
        let operand = Wrapper::<EdgeOperand>::new();

        self.operations.push(NodeOperation::IncomingEdges {
            operand: operand.clone(),
        });

        operand
    }

    pub fn neighbors(&mut self, direction: EdgeDirection) -> Wrapper<NodeOperand> {
        let operand = Wrapper::<NodeOperand>::new();

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
        let mut either_operand = Wrapper::<NodeOperand>::new();
        let mut or_operand = Wrapper::<NodeOperand>::new();

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(NodeOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }
}

impl Wrapper<NodeOperand> {
    pub(crate) fn new() -> Self {
        NodeOperand::new().into()
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, &'a NodeIndex>> {
        self.0.read_or_panic().evaluate(medrecord)
    }

    pub fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<MultipleValuesOperand> {
        self.0.write_or_panic().attribute(attribute)
    }

    pub fn attributes(&mut self) -> Wrapper<AttributesTreeOperand> {
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

    pub fn outgoing_edges(&mut self) -> Wrapper<EdgeOperand> {
        self.0.write_or_panic().outgoing_edges()
    }

    pub fn incoming_edges(&mut self) -> Wrapper<EdgeOperand> {
        self.0.write_or_panic().incoming_edges()
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
}

macro_rules! implement_value_operation {
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

macro_rules! implement_single_value_comparison_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name<V: Into<NodeIndexComparisonOperand>>(&mut self, value: V) {
            self.operations
                .push($operation::NodeIndexComparisonOperation {
                    operand: value.into(),
                    kind: SingleComparisonKind::$kind,
                });
        }
    };
}

macro_rules! implement_binary_arithmetic_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name<V: Into<NodeIndexComparisonOperand>>(&mut self, value: V) {
            self.operations.push($operation::BinaryArithmeticOpration {
                operand: value.into(),
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
    ($name:ident, $value_type:ty) => {
        pub fn $name(&self, value: $value_type) {
            self.0.write_or_panic().$name(value)
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
            Self::Index(value) => Self::Index(value.clone()),
        }
    }
}

impl From<Wrapper<NodeIndexOperand>> for NodeIndexComparisonOperand {
    fn from(value: Wrapper<NodeIndexOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeIndexOperand>> for NodeIndexComparisonOperand {
    fn from(value: &Wrapper<NodeIndexOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<NodeIndex>> From<V> for NodeIndexComparisonOperand {
    fn from(value: V) -> Self {
        Self::Index(value.into())
    }
}

#[derive(Debug, Clone)]
pub enum NodeIndicesComparisonOperand {
    Operand(NodeIndicesOperand),
    Indices(Vec<NodeIndex>),
}

impl DeepClone for NodeIndicesComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Indices(value) => Self::Indices(value.clone()),
        }
    }
}

impl From<Wrapper<NodeIndicesOperand>> for NodeIndicesComparisonOperand {
    fn from(value: Wrapper<NodeIndicesOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<NodeIndicesOperand>> for NodeIndicesComparisonOperand {
    fn from(value: &Wrapper<NodeIndicesOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<NodeIndex>> From<Vec<V>> for NodeIndicesComparisonOperand {
    fn from(value: Vec<V>) -> Self {
        Self::Indices(value.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<NodeIndex> + Clone, const N: usize> From<[V; N]> for NodeIndicesComparisonOperand {
    fn from(value: [V; N]) -> Self {
        value.to_vec().into()
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

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = NodeIndex> + 'a,
    ) -> MedRecordResult<impl Iterator<Item = NodeIndex> + 'a> {
        let values = Box::new(values) as BoxedIterator<NodeIndex>;

        self.operations
            .iter()
            .try_fold(values, |value_tuples, operation| {
                operation.evaluate(medrecord, value_tuples)
            })
    }

    implement_value_operation!(max, Max);
    implement_value_operation!(min, Min);
    implement_value_operation!(count, Count);
    implement_value_operation!(sum, Sum);
    implement_value_operation!(first, First);
    implement_value_operation!(last, Last);

    implement_single_value_comparison_operation!(greater_than, NodeIndicesOperation, GreaterThan);
    implement_single_value_comparison_operation!(
        greater_than_or_equal_to,
        NodeIndicesOperation,
        GreaterThanOrEqualTo
    );
    implement_single_value_comparison_operation!(less_than, NodeIndicesOperation, LessThan);
    implement_single_value_comparison_operation!(
        less_than_or_equal_to,
        NodeIndicesOperation,
        LessThanOrEqualTo
    );
    implement_single_value_comparison_operation!(equal_to, NodeIndicesOperation, EqualTo);
    implement_single_value_comparison_operation!(not_equal_to, NodeIndicesOperation, NotEqualTo);
    implement_single_value_comparison_operation!(starts_with, NodeIndicesOperation, StartsWith);
    implement_single_value_comparison_operation!(ends_with, NodeIndicesOperation, EndsWith);
    implement_single_value_comparison_operation!(contains, NodeIndicesOperation, Contains);

    pub fn is_in<V: Into<NodeIndicesComparisonOperand>>(&mut self, values: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndicesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }

    pub fn is_not_in<V: Into<NodeIndicesComparisonOperand>>(&mut self, values: V) {
        self.operations
            .push(NodeIndicesOperation::NodeIndicesComparisonOperation {
                operand: values.into(),
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
}

impl Wrapper<NodeIndicesOperand> {
    pub(crate) fn new(context: NodeOperand) -> Self {
        NodeIndicesOperand::new(context).into()
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = NodeIndex> + 'a,
    ) -> MedRecordResult<impl Iterator<Item = NodeIndex> + 'a> {
        self.0.read_or_panic().evaluate(medrecord, values)
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

    pub(crate) fn evaluate(
        &self,
        medrecord: &MedRecord,
        value: NodeIndex,
    ) -> MedRecordResult<Option<NodeIndex>> {
        self.operations
            .iter()
            .try_fold(Some(value), |value, operation| {
                if let Some(value) = value {
                    operation.evaluate(medrecord, value)
                } else {
                    Ok(None)
                }
            })
    }

    implement_single_value_comparison_operation!(greater_than, NodeIndexOperation, GreaterThan);
    implement_single_value_comparison_operation!(
        greater_than_or_equal_to,
        NodeIndexOperation,
        GreaterThanOrEqualTo
    );
    implement_single_value_comparison_operation!(less_than, NodeIndexOperation, LessThan);
    implement_single_value_comparison_operation!(
        less_than_or_equal_to,
        NodeIndexOperation,
        LessThanOrEqualTo
    );
    implement_single_value_comparison_operation!(equal_to, NodeIndexOperation, EqualTo);
    implement_single_value_comparison_operation!(not_equal_to, NodeIndexOperation, NotEqualTo);
    implement_single_value_comparison_operation!(starts_with, NodeIndexOperation, StartsWith);
    implement_single_value_comparison_operation!(ends_with, NodeIndexOperation, EndsWith);
    implement_single_value_comparison_operation!(contains, NodeIndexOperation, Contains);

    pub fn is_in<V: Into<NodeIndicesComparisonOperand>>(&mut self, values: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndicesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }

    pub fn is_not_in<V: Into<NodeIndicesComparisonOperand>>(&mut self, values: V) {
        self.operations
            .push(NodeIndexOperation::NodeIndicesComparisonOperation {
                operand: values.into(),
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
}

impl Wrapper<NodeIndexOperand> {
    pub(crate) fn new(context: NodeIndicesOperand, kind: SingleKind) -> Self {
        NodeIndexOperand::new(context, kind).into()
    }

    pub(crate) fn evaluate(
        &self,
        medrecord: &MedRecord,
        value: NodeIndex,
    ) -> MedRecordResult<Option<NodeIndex>> {
        self.0.read_or_panic().evaluate(medrecord, value)
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
}
