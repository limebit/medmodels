use crate::medrecord::{errors::PyMedRecordError, value::PyMedRecordValue};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{
        DeepClone, EdgeOperand, MedRecordValue, MultipleValuesComparisonOperand,
        MultipleValuesOperand, NodeOperand, SingleValueComparisonOperand, SingleValueOperand,
        Wrapper,
    },
};
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyFunction},
    Bound, FromPyObject, PyAny, PyResult,
};

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
        } else if let Ok(operand) = ob.extract::<PyNodeSingleValueOperand>() {
            Ok(PySingleValueComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyEdgeSingleValueOperand>() {
            Ok(PySingleValueComparisonOperand(operand.0.into()))
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {} into MedRecordValue or SingleValueOperand",
                    ob,
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
        } else if let Ok(operand) = ob.extract::<PyNodeMultipleValuesOperand>() {
            Ok(PyMultipleValuesComparisonOperand(operand.0.into()))
        } else if let Ok(operand) = ob.extract::<PyEdgeMultipleValuesOperand>() {
            Ok(PyMultipleValuesComparisonOperand(operand.0.into()))
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {} into List[MedRecordValue] or MultipleValuesOperand",
                    ob,
                )))
                .into(),
            )
        }
    }
}

macro_rules! implement_multiple_values_operand {
    ($name:ident, $generic:ty, $py_single_value_operand:ty) => {
        #[pyclass]
        #[repr(transparent)]
        #[derive(Clone)]
        pub struct $name(Wrapper<MultipleValuesOperand<$generic>>);

        impl From<Wrapper<MultipleValuesOperand<$generic>>> for $name {
            fn from(operand: Wrapper<MultipleValuesOperand<$generic>>) -> Self {
                Self(operand)
            }
        }

        impl From<$name> for Wrapper<MultipleValuesOperand<$generic>> {
            fn from(operand: $name) -> Self {
                operand.0
            }
        }

        #[pymethods]
        impl $name {
            pub fn max(&self) -> $py_single_value_operand {
                self.0.max().into()
            }

            pub fn min(&self) -> $py_single_value_operand {
                self.0.min().into()
            }

            pub fn mean(&self) -> $py_single_value_operand {
                self.0.mean().into()
            }

            pub fn median(&self) -> $py_single_value_operand {
                self.0.median().into()
            }

            pub fn mode(&self) -> $py_single_value_operand {
                self.0.mode().into()
            }

            pub fn std(&self) -> $py_single_value_operand {
                self.0.std().into()
            }

            pub fn var(&self) -> $py_single_value_operand {
                self.0.var().into()
            }

            pub fn count(&self) -> $py_single_value_operand {
                self.0.count().into()
            }

            pub fn sum(&self) -> $py_single_value_operand {
                self.0.sum().into()
            }

            pub fn first(&self) -> $py_single_value_operand {
                self.0.first().into()
            }

            pub fn last(&self) -> $py_single_value_operand {
                self.0.last().into()
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
    PyNodeMultipleValuesOperand,
    NodeOperand,
    PyNodeSingleValueOperand
);
implement_multiple_values_operand!(
    PyEdgeMultipleValuesOperand,
    EdgeOperand,
    PyEdgeSingleValueOperand
);

macro_rules! implement_single_value_operand {
    ($name:ident, $generic:ty) => {
        #[pyclass]
        #[repr(transparent)]
        #[derive(Clone)]
        pub struct $name(Wrapper<SingleValueOperand<$generic>>);

        impl From<Wrapper<SingleValueOperand<$generic>>> for $name {
            fn from(operand: Wrapper<SingleValueOperand<$generic>>) -> Self {
                Self(operand)
            }
        }

        impl From<$name> for Wrapper<SingleValueOperand<$generic>> {
            fn from(operand: $name) -> Self {
                operand.0
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

implement_single_value_operand!(PyNodeSingleValueOperand, NodeOperand);
implement_single_value_operand!(PyEdgeSingleValueOperand, EdgeOperand);
