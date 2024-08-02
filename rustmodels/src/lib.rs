mod gil_hash_map;
mod medrecord;

use medrecord::{
    datatype::{PyAny, PyBool, PyDateTime, PyFloat, PyInt, PyNull, PyOption, PyString, PyUnion},
    querying::{
        PyEdgeAttributeOperand, PyEdgeIndexOperand, PyEdgeOperand, PyEdgeOperation,
        PyNodeAttributeOperand, PyNodeIndexOperand, PyNodeOperand, PyNodeOperation,
        PyValueArithmeticOperation, PyValueTransformationOperation,
    },
    schema::{PyGroupSchema, PySchema},
    PyMedRecord,
};
use pyo3::prelude::*;

#[pymodule]
fn _medmodels(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMedRecord>()?;

    m.add_class::<PyString>()?;
    m.add_class::<PyInt>()?;
    m.add_class::<PyFloat>()?;
    m.add_class::<PyBool>()?;
    m.add_class::<PyDateTime>()?;
    m.add_class::<PyNull>()?;
    m.add_class::<PyAny>()?;
    m.add_class::<PyUnion>()?;
    m.add_class::<PyOption>()?;

    m.add_class::<PyGroupSchema>()?;
    m.add_class::<PySchema>()?;

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
