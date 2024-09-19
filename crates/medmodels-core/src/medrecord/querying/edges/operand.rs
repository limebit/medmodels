use super::{
    operation::{EdgeIndexOperation, EdgeIndicesOperation, EdgeOperation},
    BinaryArithmeticKind, MultipleComparisonKind, SingleComparisonKind, SingleKind,
};
use crate::{
    errors::MedRecordResult,
    medrecord::{
        querying::{
            attributes::{self, AttributesTreeOperand},
            nodes::NodeOperand,
            traits::{DeepClone, ReadWriteOrPanic},
            values::{self, MultipleValuesOperand},
            wrapper::Wrapper,
            BoxedIterator,
        },
        CardinalityWrapper, EdgeIndex, Group, MedRecordAttribute,
    },
    MedRecord,
};
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct EdgeOperand {
    pub(crate) operations: Vec<EdgeOperation>,
}

impl DeepClone for EdgeOperand {
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

impl EdgeOperand {
    pub(crate) fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        let edge_indices = Box::new(medrecord.edge_indices()) as BoxedIterator<&'a EdgeIndex>;

        self.operations
            .iter()
            .try_fold(edge_indices, |edge_indices, operation| {
                operation.evaluate(medrecord, edge_indices)
            })
    }

    pub fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<MultipleValuesOperand> {
        let operand = Wrapper::<MultipleValuesOperand>::new(
            values::Context::EdgeOperand(self.deep_clone()),
            attribute,
        );

        self.operations.push(EdgeOperation::Values {
            operand: operand.clone(),
        });

        operand
    }

    pub fn attributes(&mut self) -> Wrapper<AttributesTreeOperand> {
        let operand = Wrapper::<AttributesTreeOperand>::new(attributes::Context::EdgeOperand(
            self.deep_clone(),
        ));

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
        let operand = Wrapper::<NodeOperand>::new();

        self.operations.push(EdgeOperation::SourceNode {
            operand: operand.clone(),
        });

        operand
    }

    pub fn target_node(&mut self) -> Wrapper<NodeOperand> {
        let operand = Wrapper::<NodeOperand>::new();

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
        let mut either_operand = Wrapper::<EdgeOperand>::new();
        let mut or_operand = Wrapper::<EdgeOperand>::new();

        either_query(&mut either_operand);
        or_query(&mut or_operand);

        self.operations.push(EdgeOperation::EitherOr {
            either: either_operand,
            or: or_operand,
        });
    }
}

impl Wrapper<EdgeOperand> {
    pub(crate) fn new() -> Self {
        EdgeOperand::new().into()
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<impl Iterator<Item = &'a EdgeIndex>> {
        self.0.read_or_panic().evaluate(medrecord)
    }

    pub fn attribute<A>(&self, attribute: A) -> Wrapper<MultipleValuesOperand>
    where
        A: Into<MedRecordAttribute>,
    {
        self.0.write_or_panic().attribute(attribute.into())
    }

    pub fn attributes(&self) -> Wrapper<AttributesTreeOperand> {
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
}

macro_rules! implement_value_operation {
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

macro_rules! implement_single_value_comparison_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name<V: Into<EdgeIndexComparisonOperand>>(&mut self, value: V) {
            self.operations
                .push($operation::EdgeIndexComparisonOperation {
                    operand: value.into(),
                    kind: SingleComparisonKind::$kind,
                });
        }
    };
}

macro_rules! implement_binary_arithmetic_operation {
    ($name:ident, $operation:ident, $kind:ident) => {
        pub fn $name<V: Into<EdgeIndexComparisonOperand>>(&mut self, value: V) {
            self.operations.push($operation::BinaryArithmeticOpration {
                operand: value.into(),
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
    ($name:ident, $value_type:ty) => {
        pub fn $name(&self, value: $value_type) {
            self.0.write_or_panic().$name(value)
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
            Self::Index(value) => Self::Index(*value),
        }
    }
}

impl From<Wrapper<EdgeIndexOperand>> for EdgeIndexComparisonOperand {
    fn from(value: Wrapper<EdgeIndexOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeIndexOperand>> for EdgeIndexComparisonOperand {
    fn from(value: &Wrapper<EdgeIndexOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<EdgeIndex>> From<V> for EdgeIndexComparisonOperand {
    fn from(value: V) -> Self {
        Self::Index(value.into())
    }
}

#[derive(Debug, Clone)]
pub enum EdgeIndicesComparisonOperand {
    Operand(EdgeIndicesOperand),
    Indices(Vec<EdgeIndex>),
}

impl DeepClone for EdgeIndicesComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Operand(operand) => Self::Operand(operand.deep_clone()),
            Self::Indices(value) => Self::Indices(value.clone()),
        }
    }
}

impl From<Wrapper<EdgeIndicesOperand>> for EdgeIndicesComparisonOperand {
    fn from(value: Wrapper<EdgeIndicesOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<EdgeIndicesOperand>> for EdgeIndicesComparisonOperand {
    fn from(value: &Wrapper<EdgeIndicesOperand>) -> Self {
        Self::Operand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<EdgeIndex>> From<Vec<V>> for EdgeIndicesComparisonOperand {
    fn from(value: Vec<V>) -> Self {
        Self::Indices(value.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<EdgeIndex> + Clone, const N: usize> From<[V; N]> for EdgeIndicesComparisonOperand {
    fn from(value: [V; N]) -> Self {
        value.to_vec().into()
    }
}

#[derive(Debug, Clone)]
pub struct EdgeIndicesOperand {
    pub(crate) context: EdgeOperand,
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

impl EdgeIndicesOperand {
    pub(crate) fn new(context: EdgeOperand) -> Self {
        Self {
            context,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = EdgeIndex> + 'a,
    ) -> MedRecordResult<impl Iterator<Item = EdgeIndex> + 'a> {
        let values = Box::new(values) as BoxedIterator<EdgeIndex>;

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

    implement_single_value_comparison_operation!(greater_than, EdgeIndicesOperation, GreaterThan);
    implement_single_value_comparison_operation!(
        greater_than_or_equal_to,
        EdgeIndicesOperation,
        GreaterThanOrEqualTo
    );
    implement_single_value_comparison_operation!(less_than, EdgeIndicesOperation, LessThan);
    implement_single_value_comparison_operation!(
        less_than_or_equal_to,
        EdgeIndicesOperation,
        LessThanOrEqualTo
    );
    implement_single_value_comparison_operation!(equal_to, EdgeIndicesOperation, EqualTo);
    implement_single_value_comparison_operation!(not_equal_to, EdgeIndicesOperation, NotEqualTo);
    implement_single_value_comparison_operation!(starts_with, EdgeIndicesOperation, StartsWith);
    implement_single_value_comparison_operation!(ends_with, EdgeIndicesOperation, EndsWith);
    implement_single_value_comparison_operation!(contains, EdgeIndicesOperation, Contains);

    pub fn is_in<V: Into<EdgeIndicesComparisonOperand>>(&mut self, values: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndicesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }

    pub fn is_not_in<V: Into<EdgeIndicesComparisonOperand>>(&mut self, values: V) {
        self.operations
            .push(EdgeIndicesOperation::EdgeIndicesComparisonOperation {
                operand: values.into(),
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
}

impl Wrapper<EdgeIndicesOperand> {
    pub(crate) fn new(context: EdgeOperand) -> Self {
        EdgeIndicesOperand::new(context).into()
    }

    pub(crate) fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = EdgeIndex> + 'a,
    ) -> MedRecordResult<impl Iterator<Item = EdgeIndex> + 'a> {
        self.0.read_or_panic().evaluate(medrecord, values)
    }

    implement_wrapper_operand_with_return!(max, EdgeIndexOperand);
    implement_wrapper_operand_with_return!(min, EdgeIndexOperand);
    implement_wrapper_operand_with_return!(count, EdgeIndexOperand);
    implement_wrapper_operand_with_return!(sum, EdgeIndexOperand);
    implement_wrapper_operand_with_return!(first, EdgeIndexOperand);
    implement_wrapper_operand_with_return!(last, EdgeIndexOperand);

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
}

#[derive(Debug, Clone)]
pub struct EdgeIndexOperand {
    pub(crate) context: EdgeIndicesOperand,
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

impl EdgeIndexOperand {
    pub(crate) fn new(context: EdgeIndicesOperand, kind: SingleKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate(
        &self,
        medrecord: &MedRecord,
        value: EdgeIndex,
    ) -> MedRecordResult<Option<EdgeIndex>> {
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

    implement_single_value_comparison_operation!(greater_than, EdgeIndexOperation, GreaterThan);
    implement_single_value_comparison_operation!(
        greater_than_or_equal_to,
        EdgeIndexOperation,
        GreaterThanOrEqualTo
    );
    implement_single_value_comparison_operation!(less_than, EdgeIndexOperation, LessThan);
    implement_single_value_comparison_operation!(
        less_than_or_equal_to,
        EdgeIndexOperation,
        LessThanOrEqualTo
    );
    implement_single_value_comparison_operation!(equal_to, EdgeIndexOperation, EqualTo);
    implement_single_value_comparison_operation!(not_equal_to, EdgeIndexOperation, NotEqualTo);
    implement_single_value_comparison_operation!(starts_with, EdgeIndexOperation, StartsWith);
    implement_single_value_comparison_operation!(ends_with, EdgeIndexOperation, EndsWith);
    implement_single_value_comparison_operation!(contains, EdgeIndexOperation, Contains);

    pub fn is_in<V: Into<EdgeIndicesComparisonOperand>>(&mut self, values: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndicesComparisonOperation {
                operand: values.into(),
                kind: MultipleComparisonKind::IsIn,
            });
    }

    pub fn is_not_in<V: Into<EdgeIndicesComparisonOperand>>(&mut self, values: V) {
        self.operations
            .push(EdgeIndexOperation::EdgeIndicesComparisonOperation {
                operand: values.into(),
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
}

impl Wrapper<EdgeIndexOperand> {
    pub(crate) fn new(context: EdgeIndicesOperand, kind: SingleKind) -> Self {
        EdgeIndexOperand::new(context, kind).into()
    }

    pub(crate) fn evaluate(
        &self,
        medrecord: &MedRecord,
        value: EdgeIndex,
    ) -> MedRecordResult<Option<EdgeIndex>> {
        self.0.read_or_panic().evaluate(medrecord, value)
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
}
