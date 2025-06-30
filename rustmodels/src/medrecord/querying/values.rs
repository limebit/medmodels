use crate::medrecord::{errors::PyMedRecordError, value::PyMedRecordValue};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{
        querying::{
            edges::EdgeOperand,
            group_by::GroupOperand,
            nodes::NodeOperand,
            values::{
                MultipleValuesComparisonOperand, MultipleValuesWithIndexOperand,
                MultipleValuesWithoutIndexOperand, SingleValueComparisonOperand,
                SingleValueWithIndexOperand, SingleValueWithoutIndexOperand,
            },
            wrapper::Wrapper,
            DeepClone,
        },
        MedRecordValue,
    },
};
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyFunction},
    Bound, FromPyObject, PyAny, PyResult,
};
use std::ops::Deref;

#[repr(transparent)]
pub struct PySingleValueComparisonOperand(SingleValueComparisonOperand);

impl From<SingleValueComparisonOperand> for PySingleValueComparisonOperand {
    fn from(operand: SingleValueComparisonOperand) -> Self {
        Self(operand)
    }
}

impl From<PySingleValueComparisonOperand> for SingleValueComparisonOperand {
    fn from(operand: PySingleValueComparisonOperand) -> Self {
        operand.0
    }
}

impl FromPyObject<'_> for PySingleValueComparisonOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(value) = ob.extract::<PyMedRecordValue>() {
            Ok(SingleValueComparisonOperand::Value(value.into()).into())
        } else if let Ok(operand) = ob.extract::<PyNodeSingleValueWithIndexOperand>() {
            Ok(PySingleValueComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyNodeSingleValueWithoutIndexOperand>() {
            Ok(PySingleValueComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyEdgeSingleValueWithIndexOperand>() {
            Ok(PySingleValueComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyEdgeSingleValueWithoutIndexOperand>() {
            Ok(PySingleValueComparisonOperand(operand.0.into()))
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
pub struct PyMultipleValuesComparisonOperand(MultipleValuesComparisonOperand);

impl From<MultipleValuesComparisonOperand> for PyMultipleValuesComparisonOperand {
    fn from(operand: MultipleValuesComparisonOperand) -> Self {
        Self(operand)
    }
}

impl From<PyMultipleValuesComparisonOperand> for MultipleValuesComparisonOperand {
    fn from(operand: PyMultipleValuesComparisonOperand) -> Self {
        operand.0
    }
}

impl FromPyObject<'_> for PyMultipleValuesComparisonOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(values) = ob.extract::<Vec<PyMedRecordValue>>() {
            Ok(MultipleValuesComparisonOperand::Values(
                values.into_iter().map(MedRecordValue::from).collect(),
            )
            .into())
        } else if let Ok(operand) = ob.extract::<PyNodeMultipleValuesWithIndexOperand>() {
            Ok(PyMultipleValuesComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyNodeMultipleValuesWithoutIndexOperand>() {
            Ok(PyMultipleValuesComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyEdgeMultipleValuesWithIndexOperand>() {
            Ok(PyMultipleValuesComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyEdgeMultipleValuesWithoutIndexOperand>() {
            Ok(PyMultipleValuesComparisonOperand(operand.0.into()))
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {ob} into List[MedRecordValue] or MultipleValuesOperand",
                )))
                .into(),
            )
        }
    }
}

macro_rules! implement_multiple_values_operand {
    ($name:ident, $kind:ident, $generic:ty, $py_single_value_with_index_operand:ty, $py_single_value_without_index_operand:ty) => {
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
            pub fn max(&self) -> $py_single_value_with_index_operand {
                self.0.max().into()
            }

            pub fn min(&self) -> $py_single_value_with_index_operand {
                self.0.min().into()
            }

            pub fn mean(&self) -> $py_single_value_without_index_operand {
                self.0.mean().into()
            }

            pub fn median(&self) -> $py_single_value_without_index_operand {
                self.0.median().into()
            }

            pub fn mode(&self) -> $py_single_value_without_index_operand {
                self.0.mode().into()
            }

            pub fn std(&self) -> $py_single_value_without_index_operand {
                self.0.std().into()
            }

            pub fn var(&self) -> $py_single_value_without_index_operand {
                self.0.var().into()
            }

            pub fn count(&self) -> $py_single_value_without_index_operand {
                self.0.count().into()
            }

            pub fn sum(&self) -> $py_single_value_without_index_operand {
                self.0.sum().into()
            }

            pub fn random(&self) -> $py_single_value_with_index_operand {
                self.0.random().into()
            }

            pub fn greater_than(&self, value: PySingleValueComparisonOperand) {
                self.0.greater_than(value);
            }

            pub fn greater_than_or_equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.greater_than_or_equal_to(value);
            }

            pub fn less_than(&self, value: PySingleValueComparisonOperand) {
                self.0.less_than(value);
            }

            pub fn less_than_or_equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.less_than_or_equal_to(value);
            }

            pub fn equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.equal_to(value);
            }

            pub fn not_equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.not_equal_to(value);
            }

            pub fn starts_with(&self, value: PySingleValueComparisonOperand) {
                self.0.starts_with(value);
            }

            pub fn ends_with(&self, value: PySingleValueComparisonOperand) {
                self.0.ends_with(value);
            }

            pub fn contains(&self, value: PySingleValueComparisonOperand) {
                self.0.contains(value);
            }

            pub fn is_in(&self, values: PyMultipleValuesComparisonOperand) {
                self.0.is_in(values);
            }

            pub fn is_not_in(&self, values: PyMultipleValuesComparisonOperand) {
                self.0.is_not_in(values);
            }

            pub fn add(&self, value: PySingleValueComparisonOperand) {
                self.0.add(value);
            }

            pub fn sub(&self, value: PySingleValueComparisonOperand) {
                self.0.sub(value);
            }

            pub fn mul(&self, value: PySingleValueComparisonOperand) {
                self.0.mul(value);
            }

            pub fn div(&self, value: PySingleValueComparisonOperand) {
                self.0.div(value);
            }

            pub fn pow(&self, value: PySingleValueComparisonOperand) {
                self.0.pow(value);
            }

            pub fn r#mod(&self, value: PySingleValueComparisonOperand) {
                self.0.r#mod(value);
            }

            pub fn round(&self) {
                self.0.round();
            }

            pub fn ceil(&self) {
                self.0.ceil();
            }

            pub fn floor(&self) {
                self.0.floor();
            }

            pub fn abs(&self) {
                self.0.abs();
            }

            pub fn sqrt(&self) {
                self.0.sqrt();
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

            pub fn is_float(&self) {
                self.0.is_float();
            }

            pub fn is_bool(&self) {
                self.0.is_bool();
            }

            pub fn is_datetime(&self) {
                self.0.is_datetime();
            }

            pub fn is_duration(&self) {
                self.0.is_duration();
            }

            pub fn is_null(&self) {
                self.0.is_null();
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

implement_multiple_values_operand!(
    PyNodeMultipleValuesWithIndexOperand,
    MultipleValuesWithIndexOperand,
    NodeOperand,
    PyNodeSingleValueWithIndexOperand,
    PyNodeSingleValueWithoutIndexOperand
);
implement_multiple_values_operand!(
    PyNodeMultipleValuesWithoutIndexOperand,
    MultipleValuesWithoutIndexOperand,
    NodeOperand,
    PyNodeSingleValueWithoutIndexOperand,
    PyNodeSingleValueWithoutIndexOperand
);
implement_multiple_values_operand!(
    PyEdgeMultipleValuesWithIndexOperand,
    MultipleValuesWithIndexOperand,
    EdgeOperand,
    PyEdgeSingleValueWithIndexOperand,
    PyEdgeSingleValueWithoutIndexOperand
);
implement_multiple_values_operand!(
    PyEdgeMultipleValuesWithoutIndexOperand,
    MultipleValuesWithoutIndexOperand,
    EdgeOperand,
    PyEdgeSingleValueWithoutIndexOperand,
    PyEdgeSingleValueWithoutIndexOperand
);

macro_rules! implement_multiple_values_grouped_operand {
    ($name:ident, $ungrouped_name:ident, $kind:ident, $generic:ty, $py_single_value_with_index_operand:ty, $py_single_value_without_index_operand:ty) => {
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
            pub fn max(&self) -> $py_single_value_with_index_operand {
                self.0.max().into()
            }

            pub fn min(&self) -> $py_single_value_with_index_operand {
                self.0.min().into()
            }

            pub fn mean(&self) -> $py_single_value_without_index_operand {
                self.0.mean().into()
            }

            pub fn median(&self) -> $py_single_value_without_index_operand {
                self.0.median().into()
            }

            pub fn mode(&self) -> $py_single_value_without_index_operand {
                self.0.mode().into()
            }

            pub fn std(&self) -> $py_single_value_without_index_operand {
                self.0.std().into()
            }

            pub fn var(&self) -> $py_single_value_without_index_operand {
                self.0.var().into()
            }

            pub fn count(&self) -> $py_single_value_without_index_operand {
                self.0.count().into()
            }

            pub fn sum(&self) -> $py_single_value_without_index_operand {
                self.0.sum().into()
            }

            pub fn random(&self) -> $py_single_value_with_index_operand {
                self.0.random().into()
            }

            pub fn greater_than(&self, value: PySingleValueComparisonOperand) {
                self.0.greater_than(value);
            }

            pub fn greater_than_or_equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.greater_than_or_equal_to(value);
            }

            pub fn less_than(&self, value: PySingleValueComparisonOperand) {
                self.0.less_than(value);
            }

            pub fn less_than_or_equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.less_than_or_equal_to(value);
            }

            pub fn equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.equal_to(value);
            }

            pub fn not_equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.not_equal_to(value);
            }

            pub fn starts_with(&self, value: PySingleValueComparisonOperand) {
                self.0.starts_with(value);
            }

            pub fn ends_with(&self, value: PySingleValueComparisonOperand) {
                self.0.ends_with(value);
            }

            pub fn contains(&self, value: PySingleValueComparisonOperand) {
                self.0.contains(value);
            }

            pub fn is_in(&self, values: PyMultipleValuesComparisonOperand) {
                self.0.is_in(values);
            }

            pub fn is_not_in(&self, values: PyMultipleValuesComparisonOperand) {
                self.0.is_not_in(values);
            }

            pub fn add(&self, value: PySingleValueComparisonOperand) {
                self.0.add(value);
            }

            pub fn sub(&self, value: PySingleValueComparisonOperand) {
                self.0.sub(value);
            }

            pub fn mul(&self, value: PySingleValueComparisonOperand) {
                self.0.mul(value);
            }

            pub fn div(&self, value: PySingleValueComparisonOperand) {
                self.0.div(value);
            }

            pub fn pow(&self, value: PySingleValueComparisonOperand) {
                self.0.pow(value);
            }

            pub fn r#mod(&self, value: PySingleValueComparisonOperand) {
                self.0.r#mod(value);
            }

            pub fn round(&self) {
                self.0.round();
            }

            pub fn ceil(&self) {
                self.0.ceil();
            }

            pub fn floor(&self) {
                self.0.floor();
            }

            pub fn abs(&self) {
                self.0.abs();
            }

            pub fn sqrt(&self) {
                self.0.sqrt();
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

            pub fn is_float(&self) {
                self.0.is_float();
            }

            pub fn is_bool(&self) {
                self.0.is_bool();
            }

            pub fn is_datetime(&self) {
                self.0.is_datetime();
            }

            pub fn is_duration(&self) {
                self.0.is_duration();
            }

            pub fn is_null(&self) {
                self.0.is_null();
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

implement_multiple_values_grouped_operand!(
    PyNodeMultipleValuesWithIndexGroupOperand,
    PyNodeMultipleValuesWithIndexOperand,
    MultipleValuesWithIndexOperand,
    NodeOperand,
    PyNodeSingleValueWithIndexGroupOperand,
    PyNodeSingleValueWithoutIndexGroupOperand
);
implement_multiple_values_grouped_operand!(
    PyEdgeMultipleValuesWithIndexGroupOperand,
    PyEdgeMultipleValuesWithIndexOperand,
    MultipleValuesWithIndexOperand,
    EdgeOperand,
    PyEdgeSingleValueWithIndexGroupOperand,
    PyEdgeSingleValueWithoutIndexGroupOperand
);

macro_rules! implement_single_value_operand {
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
            pub fn greater_than(&self, value: PySingleValueComparisonOperand) {
                self.0.greater_than(value);
            }

            pub fn greater_than_or_equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.greater_than_or_equal_to(value);
            }

            pub fn less_than(&self, value: PySingleValueComparisonOperand) {
                self.0.less_than(value);
            }

            pub fn less_than_or_equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.less_than_or_equal_to(value);
            }

            pub fn equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.equal_to(value);
            }

            pub fn not_equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.not_equal_to(value);
            }

            pub fn starts_with(&self, value: PySingleValueComparisonOperand) {
                self.0.starts_with(value);
            }

            pub fn ends_with(&self, value: PySingleValueComparisonOperand) {
                self.0.ends_with(value);
            }

            pub fn contains(&self, value: PySingleValueComparisonOperand) {
                self.0.contains(value);
            }

            pub fn is_in(&self, values: PyMultipleValuesComparisonOperand) {
                self.0.is_in(values);
            }

            pub fn is_not_in(&self, values: PyMultipleValuesComparisonOperand) {
                self.0.is_not_in(values);
            }

            pub fn add(&self, value: PySingleValueComparisonOperand) {
                self.0.add(value);
            }

            pub fn sub(&self, value: PySingleValueComparisonOperand) {
                self.0.sub(value);
            }

            pub fn mul(&self, value: PySingleValueComparisonOperand) {
                self.0.mul(value);
            }

            pub fn div(&self, value: PySingleValueComparisonOperand) {
                self.0.div(value);
            }

            pub fn pow(&self, value: PySingleValueComparisonOperand) {
                self.0.pow(value);
            }

            pub fn r#mod(&self, value: PySingleValueComparisonOperand) {
                self.0.r#mod(value);
            }

            pub fn round(&self) {
                self.0.round();
            }

            pub fn ceil(&self) {
                self.0.ceil();
            }

            pub fn floor(&self) {
                self.0.floor();
            }

            pub fn abs(&self) {
                self.0.abs();
            }

            pub fn sqrt(&self) {
                self.0.sqrt();
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

            pub fn is_float(&self) {
                self.0.is_float();
            }

            pub fn is_bool(&self) {
                self.0.is_bool();
            }

            pub fn is_datetime(&self) {
                self.0.is_datetime();
            }

            pub fn is_duration(&self) {
                self.0.is_duration();
            }

            pub fn is_null(&self) {
                self.0.is_null();
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

implement_single_value_operand!(
    PyNodeSingleValueWithIndexOperand,
    SingleValueWithIndexOperand,
    NodeOperand
);
implement_single_value_operand!(
    PyEdgeSingleValueWithIndexOperand,
    SingleValueWithIndexOperand,
    EdgeOperand
);
implement_single_value_operand!(
    PyNodeSingleValueWithoutIndexOperand,
    SingleValueWithoutIndexOperand,
    NodeOperand
);
implement_single_value_operand!(
    PyEdgeSingleValueWithoutIndexOperand,
    SingleValueWithoutIndexOperand,
    EdgeOperand
);

macro_rules! implement_single_value_group_operand {
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
            pub fn greater_than(&self, value: PySingleValueComparisonOperand) {
                self.0.greater_than(value);
            }

            pub fn greater_than_or_equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.greater_than_or_equal_to(value);
            }

            pub fn less_than(&self, value: PySingleValueComparisonOperand) {
                self.0.less_than(value);
            }

            pub fn less_than_or_equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.less_than_or_equal_to(value);
            }

            pub fn equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.equal_to(value);
            }

            pub fn not_equal_to(&self, value: PySingleValueComparisonOperand) {
                self.0.not_equal_to(value);
            }

            pub fn starts_with(&self, value: PySingleValueComparisonOperand) {
                self.0.starts_with(value);
            }

            pub fn ends_with(&self, value: PySingleValueComparisonOperand) {
                self.0.ends_with(value);
            }

            pub fn contains(&self, value: PySingleValueComparisonOperand) {
                self.0.contains(value);
            }

            pub fn is_in(&self, values: PyMultipleValuesComparisonOperand) {
                self.0.is_in(values);
            }

            pub fn is_not_in(&self, values: PyMultipleValuesComparisonOperand) {
                self.0.is_not_in(values);
            }

            pub fn add(&self, value: PySingleValueComparisonOperand) {
                self.0.add(value);
            }

            pub fn sub(&self, value: PySingleValueComparisonOperand) {
                self.0.sub(value);
            }

            pub fn mul(&self, value: PySingleValueComparisonOperand) {
                self.0.mul(value);
            }

            pub fn div(&self, value: PySingleValueComparisonOperand) {
                self.0.div(value);
            }

            pub fn pow(&self, value: PySingleValueComparisonOperand) {
                self.0.pow(value);
            }

            pub fn r#mod(&self, value: PySingleValueComparisonOperand) {
                self.0.r#mod(value);
            }

            pub fn round(&self) {
                self.0.round();
            }

            pub fn ceil(&self) {
                self.0.ceil();
            }

            pub fn floor(&self) {
                self.0.floor();
            }

            pub fn abs(&self) {
                self.0.abs();
            }

            pub fn sqrt(&self) {
                self.0.sqrt();
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

            pub fn is_float(&self) {
                self.0.is_float();
            }

            pub fn is_bool(&self) {
                self.0.is_bool();
            }

            pub fn is_datetime(&self) {
                self.0.is_datetime();
            }

            pub fn is_duration(&self) {
                self.0.is_duration();
            }

            pub fn is_null(&self) {
                self.0.is_null();
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

implement_single_value_group_operand!(
    PyNodeSingleValueWithIndexGroupOperand,
    PyNodeSingleValueWithIndexOperand,
    PyNodeMultipleValuesWithIndexOperand,
    SingleValueWithIndexOperand,
    NodeOperand
);
implement_single_value_group_operand!(
    PyEdgeSingleValueWithIndexGroupOperand,
    PyEdgeSingleValueWithIndexOperand,
    PyEdgeMultipleValuesWithIndexOperand,
    SingleValueWithIndexOperand,
    EdgeOperand
);
implement_single_value_group_operand!(
    PyNodeSingleValueWithoutIndexGroupOperand,
    PyNodeSingleValueWithoutIndexOperand,
    PyNodeMultipleValuesWithoutIndexOperand,
    SingleValueWithoutIndexOperand,
    NodeOperand
);
implement_single_value_group_operand!(
    PyEdgeSingleValueWithoutIndexGroupOperand,
    PyEdgeSingleValueWithoutIndexOperand,
    PyEdgeMultipleValuesWithoutIndexOperand,
    SingleValueWithoutIndexOperand,
    EdgeOperand
);
