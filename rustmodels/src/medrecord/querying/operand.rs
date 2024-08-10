use super::values::{
    PyComparisonOperand, PyGroupCardinalityWrapper, PyMedRecordAttributeCardinalityWrapper,
};
use crate::medrecord::attribute::PyMedRecordAttribute;
use medmodels_core::medrecord::{
    EdgeOperand, EdgeValueOperand, EdgeValuesOperand, NodeOperand, NodeValueOperand,
    NodeValuesOperand, Wrapper,
};
use pyo3::{pyclass, pymethods};

#[pyclass]
#[repr(transparent)]
pub struct PyNodeOperand(Wrapper<NodeOperand>);

impl From<Wrapper<NodeOperand>> for PyNodeOperand {
    fn from(node_operand: Wrapper<NodeOperand>) -> Self {
        Self(node_operand)
    }
}

impl From<PyNodeOperand> for Wrapper<NodeOperand> {
    fn from(py_node: PyNodeOperand) -> Self {
        py_node.0
    }
}

#[pymethods]
impl PyNodeOperand {
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
}

#[pyclass]
#[repr(transparent)]
pub struct PyEdgeOperand(Wrapper<EdgeOperand>);

impl From<Wrapper<EdgeOperand>> for PyEdgeOperand {
    fn from(edge_operand: Wrapper<EdgeOperand>) -> Self {
        Self(edge_operand)
    }
}

impl From<PyEdgeOperand> for Wrapper<EdgeOperand> {
    fn from(py_edge: PyEdgeOperand) -> Self {
        py_edge.0
    }
}

#[pymethods]
impl PyEdgeOperand {
    fn attribute(&mut self, attribute: PyMedRecordAttribute) -> PyEdgeValuesOperand {
        self.0.attribute(attribute).into()
    }

    fn in_group(&mut self, group: PyGroupCardinalityWrapper) {
        self.0.in_group(group);
    }

    fn has_attribute(&mut self, attribute: PyMedRecordAttributeCardinalityWrapper) {
        self.0.has_attribute(attribute);
    }

    fn source_node(&mut self) -> PyNodeOperand {
        self.0.target_node().into()
    }

    fn target_node(&mut self) -> PyNodeOperand {
        self.0.target_node().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyNodeValuesOperand(Wrapper<NodeValuesOperand>);

impl From<Wrapper<NodeValuesOperand>> for PyNodeValuesOperand {
    fn from(node_values_operand: Wrapper<NodeValuesOperand>) -> Self {
        Self(node_values_operand)
    }
}

impl From<PyNodeValuesOperand> for Wrapper<NodeValuesOperand> {
    fn from(py_node_values: PyNodeValuesOperand) -> Self {
        py_node_values.0
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyEdgeValuesOperand(Wrapper<EdgeValuesOperand>);

impl From<Wrapper<EdgeValuesOperand>> for PyEdgeValuesOperand {
    fn from(edge_values_operand: Wrapper<EdgeValuesOperand>) -> Self {
        Self(edge_values_operand)
    }
}

impl From<PyEdgeValuesOperand> for Wrapper<EdgeValuesOperand> {
    fn from(py_edge_values: PyEdgeValuesOperand) -> Self {
        py_edge_values.0
    }
}

#[pymethods]
impl PyEdgeValuesOperand {
    pub fn max(&mut self) -> PyEdgeValueOperand {
        self.0.max().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyNodeValueOperand(Wrapper<NodeValueOperand>);

impl From<Wrapper<NodeValueOperand>> for PyNodeValueOperand {
    fn from(node_value_operand: Wrapper<NodeValueOperand>) -> Self {
        Self(node_value_operand)
    }
}

impl From<PyNodeValueOperand> for Wrapper<NodeValueOperand> {
    fn from(py_node_value: PyNodeValueOperand) -> Self {
        py_node_value.0
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyEdgeValueOperand(Wrapper<EdgeValueOperand>);

impl From<Wrapper<EdgeValueOperand>> for PyEdgeValueOperand {
    fn from(edge_value_operand: Wrapper<EdgeValueOperand>) -> Self {
        Self(edge_value_operand)
    }
}

impl From<PyEdgeValueOperand> for Wrapper<EdgeValueOperand> {
    fn from(py_edge_value: PyEdgeValueOperand) -> Self {
        py_edge_value.0
    }
}

#[pymethods]
impl PyEdgeValueOperand {
    pub fn less_than(&mut self, value: PyComparisonOperand) {
        self.0.less_than(value);
    }
}
