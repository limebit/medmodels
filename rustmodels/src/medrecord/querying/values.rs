use medmodels_core::medrecord::{MultipleValuesOperand, Wrapper};
use pyo3::pyclass;

#[pyclass]
#[repr(transparent)]
pub struct PyMultipleValuesOperand(Wrapper<MultipleValuesOperand>);

impl From<Wrapper<MultipleValuesOperand>> for PyMultipleValuesOperand {
    fn from(operand: Wrapper<MultipleValuesOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyMultipleValuesOperand> for Wrapper<MultipleValuesOperand> {
    fn from(operand: PyMultipleValuesOperand) -> Self {
        operand.0
    }
}
