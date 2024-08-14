mod gil_hash_map;
mod medrecord;

use medrecord::{
    datatype::{PyAny, PyBool, PyDateTime, PyFloat, PyInt, PyNull, PyOption, PyString, PyUnion},
    querying::{
        PyEdgeOperand, PyEdgeValueOperand, PyEdgeValuesOperand, PyNodeOperand, PyNodeValueOperand,
        PyNodeValuesOperand,
    },
    schema::{PyAttributeDataType, PyAttributeType, PyGroupSchema, PySchema},
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

    m.add_class::<PyAttributeDataType>()?;
    m.add_class::<PyAttributeType>()?;
    m.add_class::<PyGroupSchema>()?;
    m.add_class::<PySchema>()?;

    m.add_class::<PyNodeOperand>()?;
    m.add_class::<PyEdgeOperand>()?;
    m.add_class::<PyNodeValuesOperand>()?;
    m.add_class::<PyEdgeValuesOperand>()?;
    m.add_class::<PyNodeValueOperand>()?;
    m.add_class::<PyEdgeValueOperand>()?;

    Ok(())
}
