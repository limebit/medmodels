use medmodels_core::medrecord::{AttributesTreeOperand, Wrapper};
use pyo3::pyclass;

#[pyclass]
#[repr(transparent)]
pub struct PyAttributesTreeOperand(Wrapper<AttributesTreeOperand>);

impl From<Wrapper<AttributesTreeOperand>> for PyAttributesTreeOperand {
    fn from(operand: Wrapper<AttributesTreeOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyAttributesTreeOperand> for Wrapper<AttributesTreeOperand> {
    fn from(operand: PyAttributesTreeOperand) -> Self {
        operand.0
    }
}
