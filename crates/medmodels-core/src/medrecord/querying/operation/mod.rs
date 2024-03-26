mod edge_operation;
mod node_operation;
mod operand;

pub use self::{
    edge_operation::EdgeOperation,
    node_operation::NodeOperation,
    operand::{edge, node},
};
use crate::{
    errors::MedRecordError,
    medrecord::{Attributes, MedRecord, MedRecordAttribute},
};
use operand::ValueOperand;

macro_rules! implement_attribute_comparison {
    ($name: ident, $operation: tt) => {
        fn $name<'a, P>(
            node_indices: impl Iterator<Item = &'a Self::IndexType>,
            attribute_operand: Self::AttributeOperand,
            value_operand: ValueOperand,
            attributes_for_index_fn: P,
        ) -> impl Iterator<Item = &'a Self::IndexType> where
        P: Fn(&Self::IndexType) -> Result<&'a Attributes, MedRecordError>,
        Self::IndexType: 'a,  {
            let attribute = attribute_operand.into();

            node_indices.filter(move |index| {
                let Ok(attributes) = attributes_for_index_fn(index) else {
                    return false;
                };

                let Some(value) = attributes.get(&attribute) else {
                    return false;
                };

                *value $operation value_operand
            })
        }
    };
}

macro_rules! implement_index_comparison {
    ($name: ident, $operation: tt) => {
        fn $name<'a>(
            indices: impl Iterator<Item = &'a Self::IndexType>,
            operand: Self::IndexType,
        ) -> impl Iterator<Item = &'a Self::IndexType>
        where Self::IndexType: 'a {
            indices.filter(move |index| {
                *index $operation &operand
            })
        }
    };
}

pub(super) trait Operation: Sized {
    type IndexType: PartialEq + PartialOrd;
    type AttributeOperand: Into<MedRecordAttribute>;

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
        attribute_operand: Self::AttributeOperand,
        value_operands: Vec<ValueOperand>,
        attributes_for_index_fn: P,
    ) -> impl Iterator<Item = &'a Self::IndexType>
    where
        P: Fn(&Self::IndexType) -> Result<&'a Attributes, MedRecordError>,
        Self::IndexType: 'a,
    {
        let attribute = attribute_operand.into();

        node_indices.filter(move |index| {
            let Ok(attributes) = attributes_for_index_fn(index) else {
                return false;
            };

            let Some(value) = attributes.get(&attribute) else {
                return false;
            };

            value_operands.contains(value)
        })
    }

    implement_attribute_comparison!(evaluate_attribute_gt, >);
    implement_attribute_comparison!(evaluate_attribute_gte, >=);
    implement_attribute_comparison!(evaluate_attribute_eq, ==);

    fn evaluate_attribute_starts_with<'a, P>(
        node_indices: impl Iterator<Item = &'a Self::IndexType>,
        _attribute_operand: Self::AttributeOperand,
        _value_operand: ValueOperand,
        _attributes_for_index_fn: P,
    ) -> impl Iterator<Item = &'a Self::IndexType>
    where
        P: Fn(&Self::IndexType) -> Result<&'a Attributes, MedRecordError>,
        Self::IndexType: 'a,
    {
        // let attribute = attribute_operand.into();

        // node_indices.filter(move |index| {
        //     let Ok(attributes) = medrecord.node_attributes(index) else {
        //         return false;
        //     };

        //     let Some(value) = attributes.get(&attribute) else {
        //         return false;
        //     };

        //     match value {
        //         crate::medrecord::MedRecordValue::String(value) => value.starts_with(pat),
        //         crate::medrecord::MedRecordValue::Int(_) => todo!(),
        //         crate::medrecord::MedRecordValue::Float(_) => todo!(),
        //         crate::medrecord::MedRecordValue::Bool(_) => return false,
        //     }
        // })
        node_indices
    }

    fn evaluate_attribute_ends_with<'a, P>(
        node_indices: impl Iterator<Item = &'a Self::IndexType>,
        _attribute_operand: Self::AttributeOperand,
        _value_operand: ValueOperand,
        _attributes_for_index_fn: P,
    ) -> impl Iterator<Item = &'a Self::IndexType>
    where
        P: Fn(&Self::IndexType) -> Result<&'a Attributes, MedRecordError>,
        Self::IndexType: 'a,
    {
        node_indices
    }

    fn evaluate_attribute_contains<'a, P>(
        node_indices: impl Iterator<Item = &'a Self::IndexType>,
        _attribute_operand: Self::AttributeOperand,
        _value_operand: ValueOperand,
        _attributes_for_index_fn: P,
    ) -> impl Iterator<Item = &'a Self::IndexType>
    where
        P: Fn(&Self::IndexType) -> Result<&'a Attributes, MedRecordError>,
        Self::IndexType: 'a,
    {
        node_indices
    }

    fn evaluate_has_attribute<'a, P>(
        node_indices: impl Iterator<Item = &'a Self::IndexType>,
        attribute_operand: Self::AttributeOperand,
        attributes_for_index_fn: P,
    ) -> impl Iterator<Item = &'a Self::IndexType>
    where
        P: Fn(&Self::IndexType) -> Result<&'a Attributes, MedRecordError>,
        Self::IndexType: 'a,
    {
        let attribute = attribute_operand.into();

        node_indices.filter(move |index| {
            let Ok(attributes) = attributes_for_index_fn(index) else {
                return false;
            };

            attributes.contains_key(&attribute)
        })
    }

    fn evaluate_attribute<'a, P>(
        indices: impl Iterator<Item = &'a Self::IndexType> + 'a,
        operation: AttributeOperation<Self::AttributeOperand>,
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
            AttributeOperation::Gte(attribute_operand, value_operand) => {
                Box::new(Self::evaluate_attribute_gte(
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
            AttributeOperation::In(attribute_operand, value_operands) => {
                Box::new(Self::evaluate_attribute_in(
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

    implement_index_comparison!(evaluate_index_gt, >);
    implement_index_comparison!(evaluate_index_gte, >=);
    implement_index_comparison!(evaluate_index_eq, ==);

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
pub enum AttributeOperation<T> {
    Gt(T, ValueOperand),
    Gte(T, ValueOperand),
    Eq(T, ValueOperand),
    In(T, Vec<ValueOperand>),
    StartsWith(T, ValueOperand),
    EndsWith(T, ValueOperand),
    Contains(T, ValueOperand),
}
