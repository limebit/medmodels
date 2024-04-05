mod edge_operation;
mod node_operation;
mod operand;

pub use self::{
    edge_operation::EdgeOperation,
    node_operation::NodeOperation,
    operand::{
        edge, node, ArithmeticOperation, EdgeAttributeOperand, EdgeIndexOperand, EdgeOperand,
        NodeAttributeOperand, NodeIndexOperand, NodeOperand, TransformationOperation, ValueOperand,
    },
};
use crate::{
    errors::MedRecordError,
    medrecord::{
        datatypes::{
            Ceil, Contains, EndsWith, Floor, Lowercase, PartialNeq, Round, Slice, StartsWith, Trim,
            TrimEnd, TrimStart, Uppercase,
        },
        Attributes, MedRecord, MedRecordAttribute, MedRecordValue,
    },
};

macro_rules! implement_attribute_evaluate {
    ($name: ident, $evaluate: ident) => {
        fn $name<'a, P>(
            node_indices: impl Iterator<Item = &'a Self::IndexType>,
            attribute_operand: MedRecordAttribute,
            value_operand: ValueOperand,
            attributes_for_index_fn: P,
        ) -> impl Iterator<Item = &'a Self::IndexType>
        where
            P: Fn(&Self::IndexType) -> Result<&'a Attributes, MedRecordError>,
            Self::IndexType: 'a,
        {
            node_indices.filter(move |index| {
                let Ok(attributes) = attributes_for_index_fn(index) else {
                    return false;
                };

                let Some(value) = attributes.get(&attribute_operand) else {
                    return false;
                };

                match &value_operand {
                    ValueOperand::Value(value_operand) => value.$evaluate(value_operand),
                    ValueOperand::Evaluate(value_attribute) => {
                        let Some(other) = attributes.get(&value_attribute) else {
                            return false;
                        };

                        value.$evaluate(other)
                    }
                    ValueOperand::ArithmeticOperation(
                        operation,
                        value_attribute,
                        value_operand,
                    ) => {
                        let Some(other) = attributes.get(&value_attribute) else {
                            return false;
                        };

                        let operation = match operation {
                            ArithmeticOperation::Addition => other.clone() + value_operand.clone(),
                            ArithmeticOperation::Subtraction => {
                                other.clone() - value_operand.clone()
                            }
                            ArithmeticOperation::Multiplication => {
                                other.clone() * value_operand.clone()
                            }
                            ArithmeticOperation::Division => other.clone() / value_operand.clone(),
                        };

                        match operation {
                            Ok(operation) => value.$evaluate(&operation),
                            Err(_) => false,
                        }
                    }
                    ValueOperand::TransformationOperation(operation, value_attribute) => {
                        let Some(other) = attributes.get(&value_attribute) else {
                            return false;
                        };

                        let operation = match operation {
                            TransformationOperation::Round => other.clone().round(),
                            TransformationOperation::Ceil => other.clone().ceil(),
                            TransformationOperation::Floor => other.clone().floor(),
                            TransformationOperation::Trim => other.clone().trim(),
                            TransformationOperation::TrimStart => other.clone().trim_start(),
                            TransformationOperation::TrimEnd => other.clone().trim_end(),
                            TransformationOperation::Lowercase => other.clone().lowercase(),
                            TransformationOperation::Uppercase => other.clone().uppercase(),
                        };

                        value.$evaluate(&operation)
                    }
                    ValueOperand::Slice(value_attribute, range) => {
                        let Some(other) = attributes.get(&value_attribute) else {
                            return false;
                        };

                        value.$evaluate(&other.clone().slice(range.clone()))
                    }
                }
            })
        }
    };
}

macro_rules! implement_index_evaluate {
    ($name: ident, $evaluate: ident) => {
        fn $name<'a>(
            indices: impl Iterator<Item = &'a Self::IndexType>,
            operand: Self::IndexType,
        ) -> impl Iterator<Item = &'a Self::IndexType>
        where
            Self::IndexType: 'a,
        {
            indices.filter(move |index| (*index).$evaluate(&operand))
        }
    };
}

pub(super) trait Operation: Sized {
    type IndexType: PartialEq + PartialNeq + PartialOrd;

    fn evaluate_and<'a>(
        medrecord: &'a MedRecord,
        indices: Vec<&'a Self::IndexType>,
        operation1: Self,
        operation2: Self,
    ) -> impl Iterator<Item = &'a Self::IndexType> {
        let operation1_indices = operation1
            .evaluate(medrecord, indices.clone().into_iter())
            .collect::<Vec<_>>();
        let operation2_indices = operation2
            .evaluate(medrecord, indices.clone().into_iter())
            .collect::<Vec<_>>();

        indices.into_iter().filter(move |index| {
            operation1_indices.contains(index) && operation2_indices.contains(index)
        })
    }

    fn evaluate_or<'a>(
        medrecord: &'a MedRecord,
        indices: Vec<&'a Self::IndexType>,
        operation1: Self,
        operation2: Self,
    ) -> impl Iterator<Item = &'a Self::IndexType> {
        let operation1_indices = operation1
            .evaluate(medrecord, indices.clone().into_iter())
            .collect::<Vec<_>>();
        let operation2_indices = operation2
            .evaluate(medrecord, indices.clone().into_iter())
            .collect::<Vec<_>>();

        indices.into_iter().filter(move |index| {
            operation1_indices.contains(index) || operation2_indices.contains(index)
        })
    }

    fn evaluate_not<'a>(
        medrecord: &'a MedRecord,
        indices: Vec<&'a Self::IndexType>,
        operation: Self,
    ) -> impl Iterator<Item = &'a Self::IndexType> {
        let operation_indices = operation
            .evaluate(medrecord, indices.clone().into_iter())
            .collect::<Vec<_>>();

        indices
            .into_iter()
            .filter(move |index| !operation_indices.contains(index))
    }

    fn evaluate_attribute_in<'a, P>(
        node_indices: impl Iterator<Item = &'a Self::IndexType>,
        attribute_operand: MedRecordAttribute,
        value_operands: Vec<MedRecordValue>,
        attributes_for_index_fn: P,
    ) -> impl Iterator<Item = &'a Self::IndexType>
    where
        P: Fn(&Self::IndexType) -> Result<&'a Attributes, MedRecordError>,
        Self::IndexType: 'a,
    {
        node_indices.filter(move |index| {
            let Ok(attributes) = attributes_for_index_fn(index) else {
                return false;
            };

            let Some(value) = attributes.get(&attribute_operand) else {
                return false;
            };

            value_operands.contains(value)
        })
    }

    fn evaluate_attribute_not_in<'a, P>(
        node_indices: impl Iterator<Item = &'a Self::IndexType>,
        attribute_operand: MedRecordAttribute,
        value_operands: Vec<MedRecordValue>,
        attributes_for_index_fn: P,
    ) -> impl Iterator<Item = &'a Self::IndexType>
    where
        P: Fn(&Self::IndexType) -> Result<&'a Attributes, MedRecordError>,
        Self::IndexType: 'a,
    {
        node_indices.filter(move |index| {
            let Ok(attributes) = attributes_for_index_fn(index) else {
                return false;
            };

            let Some(value) = attributes.get(&attribute_operand) else {
                return false;
            };

            !value_operands.contains(value)
        })
    }

    implement_attribute_evaluate!(evaluate_attribute_gt, gt);
    implement_attribute_evaluate!(evaluate_attribute_lt, lt);
    implement_attribute_evaluate!(evaluate_attribute_gte, ge);
    implement_attribute_evaluate!(evaluate_attribute_lte, le);
    implement_attribute_evaluate!(evaluate_attribute_eq, eq);
    implement_attribute_evaluate!(evaluate_attribute_neq, neq);
    implement_attribute_evaluate!(evaluate_attribute_starts_with, starts_with);
    implement_attribute_evaluate!(evaluate_attribute_ends_with, ends_with);
    implement_attribute_evaluate!(evaluate_attribute_contains, contains);

    fn evaluate_has_attribute<'a, P>(
        node_indices: impl Iterator<Item = &'a Self::IndexType>,
        attribute_operand: MedRecordAttribute,
        attributes_for_index_fn: P,
    ) -> impl Iterator<Item = &'a Self::IndexType>
    where
        P: Fn(&Self::IndexType) -> Result<&'a Attributes, MedRecordError>,
        Self::IndexType: 'a,
    {
        node_indices.filter(move |index| {
            let Ok(attributes) = attributes_for_index_fn(index) else {
                return false;
            };

            attributes.contains_key(&attribute_operand)
        })
    }

    fn evaluate_attribute<'a, P>(
        indices: impl Iterator<Item = &'a Self::IndexType> + 'a,
        operation: AttributeOperation,
        attributes_for_index_fn: P,
    ) -> Box<dyn Iterator<Item = &'a Self::IndexType> + 'a>
    where
        P: Fn(&Self::IndexType) -> Result<&'a Attributes, MedRecordError> + 'a,
        Self: 'a,
    {
        match operation {
            AttributeOperation::Gt(attribute_operand, value_operand) => {
                Box::new(Self::evaluate_attribute_gt(
                    indices,
                    attribute_operand,
                    value_operand,
                    attributes_for_index_fn,
                ))
            }
            AttributeOperation::Lt(attribute_operand, value_operand) => {
                Box::new(Self::evaluate_attribute_lt(
                    indices,
                    attribute_operand,
                    value_operand,
                    attributes_for_index_fn,
                ))
            }
            AttributeOperation::Gte(attribute_operand, value_operand) => {
                Box::new(Self::evaluate_attribute_gte(
                    indices,
                    attribute_operand,
                    value_operand,
                    attributes_for_index_fn,
                ))
            }
            AttributeOperation::Lte(attribute_operand, value_operand) => {
                Box::new(Self::evaluate_attribute_lte(
                    indices,
                    attribute_operand,
                    value_operand,
                    attributes_for_index_fn,
                ))
            }
            AttributeOperation::Eq(attribute_operand, value_operand) => {
                Box::new(Self::evaluate_attribute_eq(
                    indices,
                    attribute_operand,
                    value_operand,
                    attributes_for_index_fn,
                ))
            }
            AttributeOperation::Neq(attribute_operand, value_operand) => {
                Box::new(Self::evaluate_attribute_neq(
                    indices,
                    attribute_operand,
                    value_operand,
                    attributes_for_index_fn,
                ))
            }
            AttributeOperation::In(attribute_operand, value_operands) => {
                Box::new(Self::evaluate_attribute_in(
                    indices,
                    attribute_operand,
                    value_operands,
                    attributes_for_index_fn,
                ))
            }
            AttributeOperation::NotIn(attribute_operand, value_operands) => {
                Box::new(Self::evaluate_attribute_not_in(
                    indices,
                    attribute_operand,
                    value_operands,
                    attributes_for_index_fn,
                ))
            }
            AttributeOperation::StartsWith(attribute_operand, value_operand) => {
                Box::new(Self::evaluate_attribute_starts_with(
                    indices,
                    attribute_operand,
                    value_operand,
                    attributes_for_index_fn,
                ))
            }
            AttributeOperation::EndsWith(attribute_operand, value_operand) => {
                Box::new(Self::evaluate_attribute_ends_with(
                    indices,
                    attribute_operand,
                    value_operand,
                    attributes_for_index_fn,
                ))
            }
            AttributeOperation::Contains(attribute_operand, value_operand) => {
                Box::new(Self::evaluate_attribute_contains(
                    indices,
                    attribute_operand,
                    value_operand,
                    attributes_for_index_fn,
                ))
            }
        }
    }

    implement_index_evaluate!(evaluate_index_gt, gt);
    implement_index_evaluate!(evaluate_index_lt, lt);
    implement_index_evaluate!(evaluate_index_gte, ge);
    implement_index_evaluate!(evaluate_index_lte, le);
    implement_index_evaluate!(evaluate_index_eq, eq);

    fn evaluate_index_in<'a>(
        indices: impl Iterator<Item = &'a Self::IndexType>,
        operands: Vec<Self::IndexType>,
    ) -> impl Iterator<Item = &'a Self::IndexType>
    where
        Self::IndexType: 'a,
    {
        indices.filter(move |index| operands.contains(index))
    }

    fn evaluate<'a>(
        self,
        medrecord: &'a MedRecord,
        indices: impl Iterator<Item = &'a Self::IndexType> + 'a,
    ) -> Box<dyn Iterator<Item = &'a Self::IndexType> + 'a>;
}

#[derive(Debug, Clone)]
pub enum AttributeOperation {
    Gt(MedRecordAttribute, ValueOperand),
    Lt(MedRecordAttribute, ValueOperand),
    Gte(MedRecordAttribute, ValueOperand),
    Lte(MedRecordAttribute, ValueOperand),
    Eq(MedRecordAttribute, ValueOperand),
    Neq(MedRecordAttribute, ValueOperand),
    In(MedRecordAttribute, Vec<MedRecordValue>),
    NotIn(MedRecordAttribute, Vec<MedRecordValue>),
    StartsWith(MedRecordAttribute, ValueOperand),
    EndsWith(MedRecordAttribute, ValueOperand),
    Contains(MedRecordAttribute, ValueOperand),
}
