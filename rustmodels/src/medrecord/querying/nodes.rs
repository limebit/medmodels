use super::{
    attributes::PyAttributesTreeOperand, edges::PyEdgeOperand, values::PyMultipleValuesOperand,
    PyGroupCardinalityWrapper, PyMedRecordAttributeCardinalityWrapper,
};
use crate::medrecord::attribute::PyMedRecordAttribute;
use medmodels_core::medrecord::{EdgeDirection, NodeIndicesOperand, NodeOperand, Wrapper};
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyFunction},
    Bound,
};

#[pyclass]
#[derive(Clone)]
pub enum PyEdgeDirection {
    Incoming = 0,
    Outgoing = 1,
    Both = 2,
}

impl From<EdgeDirection> for PyEdgeDirection {
    fn from(value: EdgeDirection) -> Self {
        match value {
            EdgeDirection::Incoming => Self::Incoming,
            EdgeDirection::Outgoing => Self::Outgoing,
            EdgeDirection::Both => Self::Both,
        }
    }
}

impl From<PyEdgeDirection> for EdgeDirection {
    fn from(value: PyEdgeDirection) -> Self {
        match value {
            PyEdgeDirection::Incoming => Self::Incoming,
            PyEdgeDirection::Outgoing => Self::Outgoing,
            PyEdgeDirection::Both => Self::Both,
        }
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyNodeOperand(Wrapper<NodeOperand>);

impl From<Wrapper<NodeOperand>> for PyNodeOperand {
    fn from(operand: Wrapper<NodeOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyNodeOperand> for Wrapper<NodeOperand> {
    fn from(operand: PyNodeOperand) -> Self {
        operand.0
    }
}

#[pymethods]
impl PyNodeOperand {
    pub fn attribute(&mut self, attribute: PyMedRecordAttribute) -> PyMultipleValuesOperand {
        self.0.attribute(attribute).into()
    }

    pub fn attributes(&mut self) -> PyAttributesTreeOperand {
        self.0.attributes().into()
    }

    pub fn index(&mut self) -> PyNodeIndicesOperand {
        self.0.index().into()
    }

    pub fn in_group(&mut self, group: PyGroupCardinalityWrapper) {
        self.0.in_group(group);
    }

    pub fn has_attribute(&mut self, attribute: PyMedRecordAttributeCardinalityWrapper) {
        self.0.has_attribute(attribute);
    }

    pub fn outgoing_edges(&mut self) -> PyEdgeOperand {
        self.0.outgoing_edges().into()
    }

    pub fn incoming_edges(&mut self) -> PyEdgeOperand {
        self.0.incoming_edges().into()
    }

    pub fn neighbors(&mut self, direction: PyEdgeDirection) -> PyNodeOperand {
        self.0.neighbors(direction.into()).into()
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |node| {
                either
                    .call1((PyNodeOperand::from(node.clone()),))
                    .expect("Call must succeed");
            },
            |node| {
                or.call1((PyNodeOperand::from(node.clone()),))
                    .expect("Call must succeed");
            },
        );
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyNodeIndicesOperand(Wrapper<NodeIndicesOperand>);

impl From<Wrapper<NodeIndicesOperand>> for PyNodeIndicesOperand {
    fn from(operand: Wrapper<NodeIndicesOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyNodeIndicesOperand> for Wrapper<NodeIndicesOperand> {
    fn from(operand: PyNodeIndicesOperand) -> Self {
        operand.0
    }
}
