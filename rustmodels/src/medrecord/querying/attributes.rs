use std::ops::Deref;

use crate::medrecord::{
    attribute::PyMedRecordAttribute,
    errors::PyMedRecordError,
    querying::values::{
        PyEdgeMultipleValuesWithIndexGroupOperand, PyEdgeMultipleValuesWithIndexOperand,
        PyNodeMultipleValuesWithIndexGroupOperand, PyNodeMultipleValuesWithIndexOperand,
    },
};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{
        querying::{
            attributes::{
                AttributesTreeOperand, MultipleAttributesComparisonOperand,
                MultipleAttributesWithIndexOperand, MultipleAttributesWithoutIndexOperand,
                SingleAttributeComparisonOperand, SingleAttributeWithIndexOperand,
                SingleAttributeWithoutIndexOperand,
            },
            edges::EdgeOperand,
            group_by::GroupOperand,
            nodes::NodeOperand,
            wrapper::Wrapper,
            DeepClone,
        },
        MedRecordAttribute,
    },
};
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyFunction},
    Bound, FromPyObject, PyAny, PyResult,
};

#[repr(transparent)]
pub struct PySingleAttributeComparisonOperand(SingleAttributeComparisonOperand);

impl From<SingleAttributeComparisonOperand> for PySingleAttributeComparisonOperand {
    fn from(operand: SingleAttributeComparisonOperand) -> Self {
        Self(operand)
    }
}

impl From<PySingleAttributeComparisonOperand> for SingleAttributeComparisonOperand {
    fn from(operand: PySingleAttributeComparisonOperand) -> Self {
        operand.0
    }
}

impl Deref for PySingleAttributeComparisonOperand {
    type Target = SingleAttributeComparisonOperand;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl FromPyObject<'_> for PySingleAttributeComparisonOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(attribute) = ob.extract::<PyMedRecordAttribute>() {
            Ok(SingleAttributeComparisonOperand::Attribute(attribute.into()).into())
        } else if let Ok(operand) = ob.extract::<PyNodeSingleAttributeWithIndexOperand>() {
            Ok(PySingleAttributeComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyNodeSingleAttributeWithoutIndexOperand>() {
            Ok(PySingleAttributeComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyEdgeSingleAttributeWithIndexOperand>() {
            Ok(PySingleAttributeComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyEdgeSingleAttributeWithoutIndexOperand>() {
            Ok(PySingleAttributeComparisonOperand(operand.0.into()))
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {ob} into MedRecordValue or SingleValueOperand",
                )))
                .into(),
            )
        }
    }
}

#[repr(transparent)]
pub struct PyMultipleAttributesComparisonOperand(MultipleAttributesComparisonOperand);

impl From<MultipleAttributesComparisonOperand> for PyMultipleAttributesComparisonOperand {
    fn from(operand: MultipleAttributesComparisonOperand) -> Self {
        Self(operand)
    }
}

impl From<PyMultipleAttributesComparisonOperand> for MultipleAttributesComparisonOperand {
    fn from(operand: PyMultipleAttributesComparisonOperand) -> Self {
        operand.0
    }
}

impl Deref for PyMultipleAttributesComparisonOperand {
    type Target = MultipleAttributesComparisonOperand;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl FromPyObject<'_> for PyMultipleAttributesComparisonOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(values) = ob.extract::<Vec<PyMedRecordAttribute>>() {
            Ok(MultipleAttributesComparisonOperand::Attributes(
                values.into_iter().map(MedRecordAttribute::from).collect(),
            )
            .into())
        } else if let Ok(operand) = ob.extract::<PyNodeMultipleAttributesWithIndexOperand>() {
            Ok(PyMultipleAttributesComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyNodeMultipleAttributesWithoutIndexOperand>() {
            Ok(PyMultipleAttributesComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyEdgeMultipleAttributesWithIndexOperand>() {
            Ok(PyMultipleAttributesComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyEdgeMultipleAttributesWithoutIndexOperand>() {
            Ok(PyMultipleAttributesComparisonOperand(operand.0.into()))
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {ob} into List[MedRecordAttribute] or MultipleAttributesOperand",
                )))
                .into(),
            )
        }
    }
}

macro_rules! implement_attributes_tree_operand {
    ($name:ident, $generic:ty, $multiple_attributes_operand:ty) => {
        #[pyclass]
        #[repr(transparent)]
        #[derive(Clone)]
        pub struct $name(Wrapper<AttributesTreeOperand<$generic>>);

        impl From<Wrapper<AttributesTreeOperand<$generic>>> for $name {
            fn from(operand: Wrapper<AttributesTreeOperand<$generic>>) -> Self {
                Self(operand)
            }
        }

        impl From<$name> for Wrapper<AttributesTreeOperand<$generic>> {
            fn from(operand: $name) -> Self {
                operand.0
            }
        }

        impl Deref for $name {
            type Target = Wrapper<AttributesTreeOperand<$generic>>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        #[pymethods]
        impl $name {
            pub fn max(&self) -> $multiple_attributes_operand {
                self.0.max().into()
            }

            pub fn min(&self) -> $multiple_attributes_operand {
                self.0.min().into()
            }

            pub fn count(&self) -> $multiple_attributes_operand {
                self.0.count().into()
            }

            pub fn sum(&self) -> $multiple_attributes_operand {
                self.0.sum().into()
            }

            pub fn random(&self) -> $multiple_attributes_operand {
                self.0.random().into()
            }

            pub fn greater_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than(attribute);
            }

            pub fn greater_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than_or_equal_to(attribute);
            }

            pub fn less_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than(attribute);
            }

            pub fn less_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than_or_equal_to(attribute);
            }

            pub fn equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.equal_to(attribute);
            }

            pub fn not_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.not_equal_to(attribute);
            }

            pub fn starts_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.starts_with(attribute);
            }

            pub fn ends_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.ends_with(attribute);
            }

            pub fn contains(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.contains(attribute);
            }

            pub fn is_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_in(attributes);
            }

            pub fn is_not_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_not_in(attributes);
            }

            pub fn add(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.add(attribute);
            }

            pub fn sub(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.sub(attribute);
            }

            pub fn mul(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.mul(attribute);
            }

            pub fn pow(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.pow(attribute);
            }

            pub fn r#mod(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.r#mod(attribute);
            }

            pub fn abs(&self) {
                self.0.abs();
            }

            pub fn trim(&self) {
                self.0.trim();
            }

            pub fn trim_start(&self) {
                self.0.trim_start();
            }

            pub fn trim_end(&self) {
                self.0.trim_end();
            }

            pub fn lowercase(&self) {
                self.0.lowercase();
            }

            pub fn uppercase(&self) {
                self.0.uppercase();
            }

            pub fn slice(&self, start: usize, end: usize) {
                self.0.slice(start, end);
            }

            pub fn is_string(&self) {
                self.0.is_string();
            }

            pub fn is_int(&self) {
                self.0.is_int();
            }

            pub fn is_max(&self) {
                self.0.is_max();
            }

            pub fn is_min(&self) {
                self.0.is_min();
            }

            pub fn either_or(
                &mut self,
                either: &Bound<'_, PyFunction>,
                or: &Bound<'_, PyFunction>,
            ) {
                self.0.either_or(
                    |operand| {
                        either
                            .call1(($name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                    |operand| {
                        or.call1(($name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                );
            }

            pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
                self.0.exclude(|operand| {
                    query
                        .call1(($name::from(operand.clone()),))
                        .expect("Call must succeed");
                });
            }

            pub fn deep_clone(&self) -> $name {
                self.0.deep_clone().into()
            }
        }
    };
}

implement_attributes_tree_operand!(
    PyNodeAttributesTreeOperand,
    NodeOperand,
    PyNodeMultipleAttributesWithIndexOperand
);
implement_attributes_tree_operand!(
    PyEdgeAttributesTreeOperand,
    EdgeOperand,
    PyEdgeMultipleAttributesWithIndexOperand
);

macro_rules! implement_attributes_tree_group_operand {
    ($name:ident, $ungrouped_name:ident, $generic:ty, $multiple_attributes_operand:ty) => {
        #[pyclass]
        #[repr(transparent)]
        #[derive(Clone)]
        pub struct $name(Wrapper<GroupOperand<AttributesTreeOperand<$generic>>>);

        impl From<Wrapper<GroupOperand<AttributesTreeOperand<$generic>>>> for $name {
            fn from(operand: Wrapper<GroupOperand<AttributesTreeOperand<$generic>>>) -> Self {
                Self(operand)
            }
        }

        impl From<$name> for Wrapper<GroupOperand<AttributesTreeOperand<$generic>>> {
            fn from(operand: $name) -> Self {
                operand.0
            }
        }

        impl Deref for $name {
            type Target = Wrapper<GroupOperand<AttributesTreeOperand<$generic>>>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        #[pymethods]
        impl $name {
            pub fn max(&self) -> $multiple_attributes_operand {
                self.0.max().into()
            }

            pub fn min(&self) -> $multiple_attributes_operand {
                self.0.min().into()
            }

            pub fn count(&self) -> $multiple_attributes_operand {
                self.0.count().into()
            }

            pub fn sum(&self) -> $multiple_attributes_operand {
                self.0.sum().into()
            }

            pub fn random(&self) -> $multiple_attributes_operand {
                self.0.random().into()
            }

            pub fn greater_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than(attribute);
            }

            pub fn greater_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than_or_equal_to(attribute);
            }

            pub fn less_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than(attribute);
            }

            pub fn less_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than_or_equal_to(attribute);
            }

            pub fn equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.equal_to(attribute);
            }

            pub fn not_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.not_equal_to(attribute);
            }

            pub fn starts_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.starts_with(attribute);
            }

            pub fn ends_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.ends_with(attribute);
            }

            pub fn contains(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.contains(attribute);
            }

            pub fn is_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_in(attributes);
            }

            pub fn is_not_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_not_in(attributes);
            }

            pub fn add(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.add(attribute);
            }

            pub fn sub(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.sub(attribute);
            }

            pub fn mul(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.mul(attribute);
            }

            pub fn pow(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.pow(attribute);
            }

            pub fn r#mod(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.r#mod(attribute);
            }

            pub fn abs(&self) {
                self.0.abs();
            }

            pub fn trim(&self) {
                self.0.trim();
            }

            pub fn trim_start(&self) {
                self.0.trim_start();
            }

            pub fn trim_end(&self) {
                self.0.trim_end();
            }

            pub fn lowercase(&self) {
                self.0.lowercase();
            }

            pub fn uppercase(&self) {
                self.0.uppercase();
            }

            pub fn slice(&self, start: usize, end: usize) {
                self.0.slice(start, end);
            }

            pub fn is_string(&self) {
                self.0.is_string();
            }

            pub fn is_int(&self) {
                self.0.is_int();
            }

            pub fn is_max(&self) {
                self.0.is_max();
            }

            pub fn is_min(&self) {
                self.0.is_min();
            }

            pub fn either_or(
                &mut self,
                either: &Bound<'_, PyFunction>,
                or: &Bound<'_, PyFunction>,
            ) {
                self.0.either_or(
                    |operand| {
                        either
                            .call1(($ungrouped_name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                    |operand| {
                        or.call1(($ungrouped_name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                );
            }

            pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
                self.0.exclude(|operand| {
                    query
                        .call1(($ungrouped_name::from(operand.clone()),))
                        .expect("Call must succeed");
                });
            }

            pub fn ungroup(&mut self) -> $ungrouped_name {
                self.0.ungroup().into()
            }

            pub fn deep_clone(&self) -> $name {
                self.0.deep_clone().into()
            }
        }
    };
}

implement_attributes_tree_group_operand!(
    PyNodeAttributesTreeGroupOperand,
    PyNodeAttributesTreeOperand,
    NodeOperand,
    PyNodeMultipleAttributesWithIndexGroupOperand
);
implement_attributes_tree_group_operand!(
    PyEdgeAttributesTreeGroupOperand,
    PyEdgeAttributesTreeOperand,
    EdgeOperand,
    PyEdgeMultipleAttributesWithIndexGroupOperand
);

macro_rules! implement_multiple_attributes_operand {
    ($name:ident, $kind:ident, $generic:ty, $py_single_attribute_with_index_operand:ty, $py_single_attribute_without_index_operand:ty) => {
        #[pyclass]
        #[repr(transparent)]
        #[derive(Clone)]
        pub struct $name(Wrapper<$kind<$generic>>);

        impl From<Wrapper<$kind<$generic>>> for $name {
            fn from(operand: Wrapper<$kind<$generic>>) -> Self {
                Self(operand)
            }
        }

        impl From<$name> for Wrapper<$kind<$generic>> {
            fn from(operand: $name) -> Self {
                operand.0
            }
        }

        impl Deref for $name {
            type Target = Wrapper<$kind<$generic>>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        #[pymethods]
        impl $name {
            pub fn max(&self) -> $py_single_attribute_with_index_operand {
                self.0.max().into()
            }

            pub fn min(&self) -> $py_single_attribute_with_index_operand {
                self.0.min().into()
            }

            pub fn count(&self) -> $py_single_attribute_without_index_operand {
                self.0.count().into()
            }

            pub fn sum(&self) -> $py_single_attribute_without_index_operand {
                self.0.sum().into()
            }

            pub fn random(&self) -> $py_single_attribute_with_index_operand {
                self.0.random().into()
            }

            pub fn greater_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than(attribute);
            }

            pub fn greater_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than_or_equal_to(attribute);
            }

            pub fn less_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than(attribute);
            }

            pub fn less_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than_or_equal_to(attribute);
            }

            pub fn equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.equal_to(attribute);
            }

            pub fn not_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.not_equal_to(attribute);
            }

            pub fn starts_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.starts_with(attribute);
            }

            pub fn ends_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.ends_with(attribute);
            }

            pub fn contains(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.contains(attribute);
            }

            pub fn is_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_in(attributes);
            }

            pub fn is_not_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_not_in(attributes);
            }

            pub fn add(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.add(attribute);
            }

            pub fn sub(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.sub(attribute);
            }

            pub fn mul(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.mul(attribute);
            }

            pub fn pow(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.pow(attribute);
            }

            pub fn r#mod(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.r#mod(attribute);
            }

            pub fn abs(&self) {
                self.0.abs();
            }

            pub fn trim(&self) {
                self.0.trim();
            }

            pub fn trim_start(&self) {
                self.0.trim_start();
            }

            pub fn trim_end(&self) {
                self.0.trim_end();
            }

            pub fn lowercase(&self) {
                self.0.lowercase();
            }

            pub fn uppercase(&self) {
                self.0.uppercase();
            }

            pub fn slice(&self, start: usize, end: usize) {
                self.0.slice(start, end);
            }

            pub fn is_string(&self) {
                self.0.is_string();
            }

            pub fn is_int(&self) {
                self.0.is_int();
            }

            pub fn is_max(&self) {
                self.0.is_max();
            }

            pub fn is_min(&self) {
                self.0.is_min();
            }

            pub fn either_or(
                &mut self,
                either: &Bound<'_, PyFunction>,
                or: &Bound<'_, PyFunction>,
            ) {
                self.0.either_or(
                    |operand| {
                        either
                            .call1(($name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                    |operand| {
                        or.call1(($name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                );
            }

            pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
                self.0.exclude(|operand| {
                    query
                        .call1(($name::from(operand.clone()),))
                        .expect("Call must succeed");
                });
            }

            pub fn deep_clone(&self) -> $name {
                self.0.deep_clone().into()
            }
        }
    };
    ($name:ident, $kind:ident, $generic:ty, $py_single_attribute_with_index_operand:ty, $py_single_attribute_without_index_operand:ty, $py_multiple_values_operand:ty) => {
        #[pyclass]
        #[repr(transparent)]
        #[derive(Clone)]
        pub struct $name(Wrapper<$kind<$generic>>);

        impl From<Wrapper<$kind<$generic>>> for $name {
            fn from(operand: Wrapper<$kind<$generic>>) -> Self {
                Self(operand)
            }
        }

        impl From<$name> for Wrapper<$kind<$generic>> {
            fn from(operand: $name) -> Self {
                operand.0
            }
        }

        impl Deref for $name {
            type Target = Wrapper<$kind<$generic>>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        #[pymethods]
        impl $name {
            pub fn max(&self) -> $py_single_attribute_with_index_operand {
                self.0.max().into()
            }

            pub fn min(&self) -> $py_single_attribute_with_index_operand {
                self.0.min().into()
            }

            pub fn count(&self) -> $py_single_attribute_without_index_operand {
                self.0.count().into()
            }

            pub fn sum(&self) -> $py_single_attribute_without_index_operand {
                self.0.sum().into()
            }

            pub fn random(&self) -> $py_single_attribute_with_index_operand {
                self.0.random().into()
            }

            pub fn greater_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than(attribute);
            }

            pub fn greater_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than_or_equal_to(attribute);
            }

            pub fn less_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than(attribute);
            }

            pub fn less_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than_or_equal_to(attribute);
            }

            pub fn equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.equal_to(attribute);
            }

            pub fn not_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.not_equal_to(attribute);
            }

            pub fn starts_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.starts_with(attribute);
            }

            pub fn ends_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.ends_with(attribute);
            }

            pub fn contains(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.contains(attribute);
            }

            pub fn is_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_in(attributes);
            }

            pub fn is_not_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_not_in(attributes);
            }

            pub fn add(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.add(attribute);
            }

            pub fn sub(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.sub(attribute);
            }

            pub fn mul(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.mul(attribute);
            }

            pub fn pow(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.pow(attribute);
            }

            pub fn r#mod(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.r#mod(attribute);
            }

            pub fn abs(&self) {
                self.0.abs();
            }

            pub fn trim(&self) {
                self.0.trim();
            }

            pub fn trim_start(&self) {
                self.0.trim_start();
            }

            pub fn trim_end(&self) {
                self.0.trim_end();
            }

            pub fn lowercase(&self) {
                self.0.lowercase();
            }

            pub fn uppercase(&self) {
                self.0.uppercase();
            }

            pub fn to_values(&self) -> $py_multiple_values_operand {
                self.0.to_values().into()
            }

            pub fn slice(&self, start: usize, end: usize) {
                self.0.slice(start, end);
            }

            pub fn is_string(&self) {
                self.0.is_string();
            }

            pub fn is_int(&self) {
                self.0.is_int();
            }

            pub fn is_max(&self) {
                self.0.is_max();
            }

            pub fn is_min(&self) {
                self.0.is_min();
            }

            pub fn either_or(
                &mut self,
                either: &Bound<'_, PyFunction>,
                or: &Bound<'_, PyFunction>,
            ) {
                self.0.either_or(
                    |operand| {
                        either
                            .call1(($name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                    |operand| {
                        or.call1(($name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                );
            }

            pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
                self.0.exclude(|operand| {
                    query
                        .call1(($name::from(operand.clone()),))
                        .expect("Call must succeed");
                });
            }

            pub fn deep_clone(&self) -> $name {
                self.0.deep_clone().into()
            }
        }
    };
}

implement_multiple_attributes_operand!(
    PyNodeMultipleAttributesWithIndexOperand,
    MultipleAttributesWithIndexOperand,
    NodeOperand,
    PyNodeSingleAttributeWithIndexOperand,
    PyNodeSingleAttributeWithoutIndexOperand,
    PyNodeMultipleValuesWithIndexOperand
);
implement_multiple_attributes_operand!(
    PyNodeMultipleAttributesWithoutIndexOperand,
    MultipleAttributesWithoutIndexOperand,
    NodeOperand,
    PyNodeSingleAttributeWithoutIndexOperand,
    PyNodeSingleAttributeWithoutIndexOperand
);
implement_multiple_attributes_operand!(
    PyEdgeMultipleAttributesWithIndexOperand,
    MultipleAttributesWithIndexOperand,
    EdgeOperand,
    PyEdgeSingleAttributeWithIndexOperand,
    PyEdgeSingleAttributeWithoutIndexOperand,
    PyEdgeMultipleValuesWithIndexOperand
);
implement_multiple_attributes_operand!(
    PyEdgeMultipleAttributesWithoutIndexOperand,
    MultipleAttributesWithoutIndexOperand,
    EdgeOperand,
    PyEdgeSingleAttributeWithoutIndexOperand,
    PyEdgeSingleAttributeWithoutIndexOperand
);

macro_rules! implement_multiple_attributes_grouped_operand {
    ($name:ident, $ungrouped_name:ident, $kind:ident, $generic:ty, $py_single_attribute_with_index_operand:ty, $py_single_attribute_without_index_operand:ty, $py_multiple_values_operand:ty) => {
        #[pyclass]
        #[repr(transparent)]
        #[derive(Clone)]
        pub struct $name(Wrapper<GroupOperand<$kind<$generic>>>);

        impl From<Wrapper<GroupOperand<$kind<$generic>>>> for $name {
            fn from(operand: Wrapper<GroupOperand<$kind<$generic>>>) -> Self {
                Self(operand)
            }
        }

        impl From<$name> for Wrapper<GroupOperand<$kind<$generic>>> {
            fn from(operand: $name) -> Self {
                operand.0
            }
        }

        impl Deref for $name {
            type Target = Wrapper<GroupOperand<$kind<$generic>>>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        #[pymethods]
        impl $name {
            pub fn max(&self) -> $py_single_attribute_with_index_operand {
                self.0.max().into()
            }

            pub fn min(&self) -> $py_single_attribute_with_index_operand {
                self.0.min().into()
            }

            pub fn count(&self) -> $py_single_attribute_without_index_operand {
                self.0.count().into()
            }

            pub fn sum(&self) -> $py_single_attribute_without_index_operand {
                self.0.sum().into()
            }

            pub fn random(&self) -> $py_single_attribute_with_index_operand {
                self.0.random().into()
            }

            pub fn greater_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than(attribute);
            }

            pub fn greater_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than_or_equal_to(attribute);
            }

            pub fn less_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than(attribute);
            }

            pub fn less_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than_or_equal_to(attribute);
            }

            pub fn equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.equal_to(attribute);
            }

            pub fn not_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.not_equal_to(attribute);
            }

            pub fn starts_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.starts_with(attribute);
            }

            pub fn ends_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.ends_with(attribute);
            }

            pub fn contains(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.contains(attribute);
            }

            pub fn is_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_in(attributes);
            }

            pub fn is_not_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_not_in(attributes);
            }

            pub fn add(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.add(attribute);
            }

            pub fn sub(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.sub(attribute);
            }

            pub fn mul(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.mul(attribute);
            }

            pub fn pow(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.pow(attribute);
            }

            pub fn r#mod(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.r#mod(attribute);
            }

            pub fn abs(&self) {
                self.0.abs();
            }

            pub fn trim(&self) {
                self.0.trim();
            }

            pub fn trim_start(&self) {
                self.0.trim_start();
            }

            pub fn trim_end(&self) {
                self.0.trim_end();
            }

            pub fn lowercase(&self) {
                self.0.lowercase();
            }

            pub fn uppercase(&self) {
                self.0.uppercase();
            }

            pub fn to_values(&self) -> $py_multiple_values_operand {
                self.0.to_values().into()
            }

            pub fn slice(&self, start: usize, end: usize) {
                self.0.slice(start, end);
            }

            pub fn is_string(&self) {
                self.0.is_string();
            }

            pub fn is_int(&self) {
                self.0.is_int();
            }

            pub fn is_max(&self) {
                self.0.is_max();
            }

            pub fn is_min(&self) {
                self.0.is_min();
            }

            pub fn either_or(
                &mut self,
                either: &Bound<'_, PyFunction>,
                or: &Bound<'_, PyFunction>,
            ) {
                self.0.either_or(
                    |operand| {
                        either
                            .call1(($ungrouped_name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                    |operand| {
                        or.call1(($ungrouped_name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                );
            }

            pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
                self.0.exclude(|operand| {
                    query
                        .call1(($ungrouped_name::from(operand.clone()),))
                        .expect("Call must succeed");
                });
            }

            pub fn ungroup(&mut self) -> $ungrouped_name {
                self.0.ungroup().into()
            }

            pub fn deep_clone(&self) -> $name {
                self.0.deep_clone().into()
            }
        }
    };
}

implement_multiple_attributes_grouped_operand!(
    PyNodeMultipleAttributesWithIndexGroupOperand,
    PyNodeMultipleAttributesWithIndexOperand,
    MultipleAttributesWithIndexOperand,
    NodeOperand,
    PyNodeSingleAttributeWithIndexGroupOperand,
    PyNodeSingleAttributeWithoutIndexGroupOperand,
    PyNodeMultipleValuesWithIndexGroupOperand
);
implement_multiple_attributes_grouped_operand!(
    PyEdgeMultipleAttributesWithIndexGroupOperand,
    PyEdgeMultipleAttributesWithIndexOperand,
    MultipleAttributesWithIndexOperand,
    EdgeOperand,
    PyEdgeSingleAttributeWithIndexGroupOperand,
    PyEdgeSingleAttributeWithoutIndexGroupOperand,
    PyEdgeMultipleValuesWithIndexGroupOperand
);

macro_rules! implement_single_attribute_operand {
    ($name:ident, $kind:ident, $generic:ty) => {
        #[pyclass]
        #[repr(transparent)]
        #[derive(Clone)]
        pub struct $name(Wrapper<$kind<$generic>>);

        impl From<Wrapper<$kind<$generic>>> for $name {
            fn from(operand: Wrapper<$kind<$generic>>) -> Self {
                Self(operand)
            }
        }

        impl From<$name> for Wrapper<$kind<$generic>> {
            fn from(operand: $name) -> Self {
                operand.0
            }
        }

        impl Deref for $name {
            type Target = Wrapper<$kind<$generic>>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        #[pymethods]
        impl $name {
            pub fn greater_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than(attribute);
            }

            pub fn greater_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than_or_equal_to(attribute);
            }

            pub fn less_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than(attribute);
            }

            pub fn less_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than_or_equal_to(attribute);
            }

            pub fn equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.equal_to(attribute);
            }

            pub fn not_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.not_equal_to(attribute);
            }

            pub fn starts_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.starts_with(attribute);
            }

            pub fn ends_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.ends_with(attribute);
            }

            pub fn contains(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.contains(attribute);
            }

            pub fn is_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_in(attributes);
            }

            pub fn is_not_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_not_in(attributes);
            }

            pub fn add(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.add(attribute);
            }

            pub fn sub(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.sub(attribute);
            }

            pub fn mul(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.mul(attribute);
            }

            pub fn pow(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.pow(attribute);
            }

            pub fn r#mod(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.r#mod(attribute);
            }

            pub fn abs(&self) {
                self.0.abs();
            }

            pub fn trim(&self) {
                self.0.trim();
            }

            pub fn trim_start(&self) {
                self.0.trim_start();
            }

            pub fn trim_end(&self) {
                self.0.trim_end();
            }

            pub fn lowercase(&self) {
                self.0.lowercase();
            }

            pub fn uppercase(&self) {
                self.0.uppercase();
            }

            pub fn slice(&self, start: usize, end: usize) {
                self.0.slice(start, end);
            }

            pub fn is_string(&self) {
                self.0.is_string();
            }

            pub fn is_int(&self) {
                self.0.is_int();
            }

            pub fn either_or(
                &mut self,
                either: &Bound<'_, PyFunction>,
                or: &Bound<'_, PyFunction>,
            ) {
                self.0.either_or(
                    |operand| {
                        either
                            .call1(($name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                    |operand| {
                        or.call1(($name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                );
            }

            pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
                self.0.exclude(|operand| {
                    query
                        .call1(($name::from(operand.clone()),))
                        .expect("Call must succeed");
                });
            }

            pub fn deep_clone(&self) -> $name {
                self.0.deep_clone().into()
            }
        }
    };
}

implement_single_attribute_operand!(
    PyNodeSingleAttributeWithIndexOperand,
    SingleAttributeWithIndexOperand,
    NodeOperand
);
implement_single_attribute_operand!(
    PyNodeSingleAttributeWithoutIndexOperand,
    SingleAttributeWithoutIndexOperand,
    NodeOperand
);
implement_single_attribute_operand!(
    PyEdgeSingleAttributeWithIndexOperand,
    SingleAttributeWithIndexOperand,
    EdgeOperand
);
implement_single_attribute_operand!(
    PyEdgeSingleAttributeWithoutIndexOperand,
    SingleAttributeWithoutIndexOperand,
    EdgeOperand
);

macro_rules! implement_single_attribute_grouped_operand {
    ($name:ident, $ungrouped_name:ident, $ungrouped_operand_name:ident, $kind:ident, $generic:ty) => {
        #[pyclass]
        #[repr(transparent)]
        #[derive(Clone)]
        pub struct $name(Wrapper<GroupOperand<$kind<$generic>>>);

        impl From<Wrapper<GroupOperand<$kind<$generic>>>> for $name {
            fn from(operand: Wrapper<GroupOperand<$kind<$generic>>>) -> Self {
                Self(operand)
            }
        }

        impl From<$name> for Wrapper<GroupOperand<$kind<$generic>>> {
            fn from(operand: $name) -> Self {
                operand.0
            }
        }

        impl Deref for $name {
            type Target = Wrapper<GroupOperand<$kind<$generic>>>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        #[pymethods]
        impl $name {
            pub fn greater_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than(attribute);
            }

            pub fn greater_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.greater_than_or_equal_to(attribute);
            }

            pub fn less_than(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than(attribute);
            }

            pub fn less_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.less_than_or_equal_to(attribute);
            }

            pub fn equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.equal_to(attribute);
            }

            pub fn not_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.not_equal_to(attribute);
            }

            pub fn starts_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.starts_with(attribute);
            }

            pub fn ends_with(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.ends_with(attribute);
            }

            pub fn contains(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.contains(attribute);
            }

            pub fn is_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_in(attributes);
            }

            pub fn is_not_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
                self.0.is_not_in(attributes);
            }

            pub fn add(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.add(attribute);
            }

            pub fn sub(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.sub(attribute);
            }

            pub fn mul(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.mul(attribute);
            }

            pub fn pow(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.pow(attribute);
            }

            pub fn r#mod(&self, attribute: PySingleAttributeComparisonOperand) {
                self.0.r#mod(attribute);
            }

            pub fn abs(&self) {
                self.0.abs();
            }

            pub fn trim(&self) {
                self.0.trim();
            }

            pub fn trim_start(&self) {
                self.0.trim_start();
            }

            pub fn trim_end(&self) {
                self.0.trim_end();
            }

            pub fn lowercase(&self) {
                self.0.lowercase();
            }

            pub fn uppercase(&self) {
                self.0.uppercase();
            }

            pub fn slice(&self, start: usize, end: usize) {
                self.0.slice(start, end);
            }

            pub fn is_string(&self) {
                self.0.is_string();
            }

            pub fn is_int(&self) {
                self.0.is_int();
            }

            pub fn either_or(
                &mut self,
                either: &Bound<'_, PyFunction>,
                or: &Bound<'_, PyFunction>,
            ) {
                self.0.either_or(
                    |operand| {
                        either
                            .call1(($ungrouped_name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                    |operand| {
                        or.call1(($ungrouped_name::from(operand.clone()),))
                            .expect("Call must succeed");
                    },
                );
            }

            pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
                self.0.exclude(|operand| {
                    query
                        .call1(($ungrouped_name::from(operand.clone()),))
                        .expect("Call must succeed");
                });
            }

            pub fn ungroup(&mut self) -> $ungrouped_operand_name {
                self.0.ungroup().into()
            }

            pub fn deep_clone(&self) -> $name {
                self.0.deep_clone().into()
            }
        }
    };
}

implement_single_attribute_grouped_operand!(
    PyNodeSingleAttributeWithIndexGroupOperand,
    PyNodeSingleAttributeWithIndexOperand,
    PyNodeMultipleAttributesWithIndexOperand,
    SingleAttributeWithIndexOperand,
    NodeOperand
);
implement_single_attribute_grouped_operand!(
    PyNodeSingleAttributeWithoutIndexGroupOperand,
    PyNodeSingleAttributeWithoutIndexOperand,
    PyNodeMultipleAttributesWithoutIndexOperand,
    SingleAttributeWithoutIndexOperand,
    NodeOperand
);
implement_single_attribute_grouped_operand!(
    PyEdgeSingleAttributeWithIndexGroupOperand,
    PyEdgeSingleAttributeWithIndexOperand,
    PyEdgeMultipleAttributesWithIndexOperand,
    SingleAttributeWithIndexOperand,
    EdgeOperand
);
implement_single_attribute_grouped_operand!(
    PyEdgeSingleAttributeWithoutIndexGroupOperand,
    PyEdgeSingleAttributeWithoutIndexOperand,
    PyEdgeMultipleAttributesWithoutIndexOperand,
    SingleAttributeWithoutIndexOperand,
    EdgeOperand
);
