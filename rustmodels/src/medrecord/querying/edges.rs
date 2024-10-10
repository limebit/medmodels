use super::{
    attributes::PyAttributesTreeOperand, nodes::PyNodeOperand, values::PyMultipleValuesOperand,
    PyGroupCardinalityWrapper, PyMedRecordAttributeCardinalityWrapper,
};
use crate::medrecord::{attribute::PyMedRecordAttribute, errors::PyMedRecordError};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{
        DeepClone, EdgeIndex, EdgeIndexComparisonOperand, EdgeIndexOperand,
        EdgeIndicesComparisonOperand, EdgeIndicesOperand, EdgeOperand, Wrapper,
    },
};
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyFunction},
    Bound, FromPyObject, PyAny, PyResult,
};

#[pyclass]
#[repr(transparent)]
pub struct PyEdgeOperand(Wrapper<EdgeOperand>);

impl From<Wrapper<EdgeOperand>> for PyEdgeOperand {
    fn from(operand: Wrapper<EdgeOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyEdgeOperand> for Wrapper<EdgeOperand> {
    fn from(operand: PyEdgeOperand) -> Self {
        operand.0
    }
}

#[pymethods]
impl PyEdgeOperand {
    pub fn attribute(&mut self, attribute: PyMedRecordAttribute) -> PyMultipleValuesOperand {
        self.0.attribute(attribute).into()
    }

    pub fn attributes(&mut self) -> PyAttributesTreeOperand {
        self.0.attributes().into()
    }

    pub fn index(&mut self) -> PyEdgeIndicesOperand {
        self.0.index().into()
    }

    pub fn in_group(&mut self, group: PyGroupCardinalityWrapper) {
        self.0.in_group(group);
    }

    pub fn has_attribute(&mut self, attribute: PyMedRecordAttributeCardinalityWrapper) {
        self.0.has_attribute(attribute);
    }

    pub fn source_node(&mut self) -> PyNodeOperand {
        self.0.source_node().into()
    }

    pub fn target_node(&mut self) -> PyNodeOperand {
        self.0.target_node().into()
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyEdgeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyEdgeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn deep_clone(&self) -> PyEdgeOperand {
        self.0.deep_clone().into()
    }
}

#[repr(transparent)]
pub struct PyEdgeIndexComparisonOperand(EdgeIndexComparisonOperand);

impl From<EdgeIndexComparisonOperand> for PyEdgeIndexComparisonOperand {
    fn from(operand: EdgeIndexComparisonOperand) -> Self {
        Self(operand)
    }
}

impl From<PyEdgeIndexComparisonOperand> for EdgeIndexComparisonOperand {
    fn from(operand: PyEdgeIndexComparisonOperand) -> Self {
        operand.0
    }
}

impl<'a> FromPyObject<'a> for PyEdgeIndexComparisonOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(index) = ob.extract::<EdgeIndex>() {
            Ok(EdgeIndexComparisonOperand::Index(index).into())
        } else if let Ok(operand) = ob.extract::<PyEdgeIndexOperand>() {
            Ok(PyEdgeIndexComparisonOperand(operand.0.into()))
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {} into EdgeIndex or EdgeIndexOperand",
                    ob,
                )))
                .into(),
            )
        }
    }
}

#[repr(transparent)]
pub struct PyEdgeIndicesComparisonOperand(EdgeIndicesComparisonOperand);

impl From<EdgeIndicesComparisonOperand> for PyEdgeIndicesComparisonOperand {
    fn from(operand: EdgeIndicesComparisonOperand) -> Self {
        Self(operand)
    }
}

impl From<PyEdgeIndicesComparisonOperand> for EdgeIndicesComparisonOperand {
    fn from(operand: PyEdgeIndicesComparisonOperand) -> Self {
        operand.0
    }
}

impl<'a> FromPyObject<'a> for PyEdgeIndicesComparisonOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(indices) = ob.extract::<Vec<EdgeIndex>>() {
            Ok(EdgeIndicesComparisonOperand::Indices(indices).into())
        } else if let Ok(operand) = ob.extract::<PyEdgeIndicesOperand>() {
            Ok(PyEdgeIndicesComparisonOperand(operand.0.into()))
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {} into List[EdgeIndex] or EdgeIndicesOperand",
                    ob,
                )))
                .into(),
            )
        }
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyEdgeIndicesOperand(Wrapper<EdgeIndicesOperand>);

impl From<Wrapper<EdgeIndicesOperand>> for PyEdgeIndicesOperand {
    fn from(operand: Wrapper<EdgeIndicesOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyEdgeIndicesOperand> for Wrapper<EdgeIndicesOperand> {
    fn from(operand: PyEdgeIndicesOperand) -> Self {
        operand.0
    }
}

#[pymethods]
impl PyEdgeIndicesOperand {
    pub fn max(&mut self) -> PyEdgeIndexOperand {
        self.0.max().into()
    }

    pub fn min(&mut self) -> PyEdgeIndexOperand {
        self.0.min().into()
    }

    pub fn count(&mut self) -> PyEdgeIndexOperand {
        self.0.count().into()
    }

    pub fn sum(&mut self) -> PyEdgeIndexOperand {
        self.0.sum().into()
    }

    pub fn first(&mut self) -> PyEdgeIndexOperand {
        self.0.first().into()
    }

    pub fn last(&mut self) -> PyEdgeIndexOperand {
        self.0.last().into()
    }

    pub fn greater_than(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.greater_than(index);
    }

    pub fn greater_than_or_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.greater_than_or_equal_to(index);
    }

    pub fn less_than(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.less_than(index);
    }

    pub fn less_than_or_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.less_than_or_equal_to(index);
    }

    pub fn equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.equal_to(index);
    }

    pub fn not_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.not_equal_to(index);
    }

    pub fn starts_with(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.starts_with(index);
    }

    pub fn ends_with(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.ends_with(index);
    }

    pub fn contains(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.contains(index);
    }

    pub fn is_in(&mut self, indices: PyEdgeIndicesComparisonOperand) {
        self.0.is_in(indices);
    }

    pub fn is_not_in(&mut self, indices: PyEdgeIndicesComparisonOperand) {
        self.0.is_not_in(indices);
    }

    pub fn add(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.add(index);
    }

    pub fn sub(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.sub(index);
    }

    pub fn mul(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.mul(index);
    }

    pub fn pow(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.pow(index);
    }

    pub fn r#mod(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.r#mod(index);
    }

    pub fn is_max(&mut self) {
        self.0.is_max()
    }

    pub fn is_min(&mut self) {
        self.0.is_min()
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyEdgeIndicesOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyEdgeIndicesOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn deep_clone(&self) -> PyEdgeIndicesOperand {
        self.0.deep_clone().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyEdgeIndexOperand(Wrapper<EdgeIndexOperand>);

impl From<Wrapper<EdgeIndexOperand>> for PyEdgeIndexOperand {
    fn from(operand: Wrapper<EdgeIndexOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyEdgeIndexOperand> for Wrapper<EdgeIndexOperand> {
    fn from(operand: PyEdgeIndexOperand) -> Self {
        operand.0
    }
}

#[pymethods]
impl PyEdgeIndexOperand {
    pub fn greater_than(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.greater_than(index);
    }

    pub fn greater_than_or_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.greater_than_or_equal_to(index);
    }

    pub fn less_than(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.less_than(index);
    }

    pub fn less_than_or_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.less_than_or_equal_to(index);
    }

    pub fn equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.equal_to(index);
    }

    pub fn not_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.not_equal_to(index);
    }

    pub fn starts_with(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.starts_with(index);
    }

    pub fn ends_with(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.ends_with(index);
    }

    pub fn contains(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.contains(index);
    }

    pub fn is_in(&mut self, indices: PyEdgeIndicesComparisonOperand) {
        self.0.is_in(indices);
    }

    pub fn is_not_in(&mut self, indices: PyEdgeIndicesComparisonOperand) {
        self.0.is_not_in(indices);
    }

    pub fn add(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.add(index);
    }

    pub fn sub(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.sub(index);
    }

    pub fn mul(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.mul(index);
    }

    pub fn pow(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.pow(index);
    }

    pub fn r#mod(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.r#mod(index);
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyEdgeIndexOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyEdgeIndexOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn deep_clone(&self) -> PyEdgeIndexOperand {
        self.0.deep_clone().into()
    }
}
