mod gil_hash_map;
mod medrecord;

use medrecord::{
    querying::{
        PyEdgeAttributeOperand, PyEdgeIndexOperand, PyEdgeOperand, PyEdgeOperation,
        PyNodeAttributeOperand, PyNodeIndexOperand, PyNodeOperand, PyNodeOperation,
        PyValueArithmeticOperation, PyValueTransformationOperation,
    },
    PyMedRecord,
};
use pyo3::prelude::*;

#[pymodule]
fn _medmodels(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMedRecord>()?;
    m.add_class::<PyValueArithmeticOperation>()?;
    m.add_class::<PyValueTransformationOperation>()?;
    m.add_class::<PyNodeOperation>()?;
    m.add_class::<PyEdgeOperation>()?;
    m.add_class::<PyNodeAttributeOperand>()?;
    m.add_class::<PyEdgeAttributeOperand>()?;
    m.add_class::<PyNodeIndexOperand>()?;
    m.add_class::<PyEdgeIndexOperand>()?;
    m.add_class::<PyNodeOperand>()?;
    m.add_class::<PyEdgeOperand>()?;
    Ok(())
}
