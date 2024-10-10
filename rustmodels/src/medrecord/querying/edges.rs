use super::{
    attributes::PyAttributesTreeOperand, nodes::PyNodeOperand, values::PyMultipleValuesOperand,
    PyGroupCardinalityWrapper, PyMedRecordAttributeCardinalityWrapper,
};
use crate::medrecord::attribute::PyMedRecordAttribute;
use medmodels_core::medrecord::{EdgeIndexOperand, EdgeIndicesOperand, EdgeOperand, Wrapper};
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyFunction},
    Bound,
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
            |edge| {
                either
                    .call1((PyEdgeOperand::from(edge.clone()),))
                    .expect("Call must succeed");
            },
            |edge| {
                or.call1((PyEdgeOperand::from(edge.clone()),))
                    .expect("Call must succeed");
            },
        );
    }
}

#[pyclass]
#[repr(transparent)]
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

#[pyclass]
#[repr(transparent)]
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
