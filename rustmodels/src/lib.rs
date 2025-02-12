mod gil_hash_map;
mod medrecord;

use medrecord::{
    datatype::{
        PyAny, PyBool, PyDateTime, PyDuration, PyFloat, PyInt, PyNull, PyOption, PyString, PyUnion,
    },
    querying::{
        attributes::{
            PyAttributesTreeOperand, PyMultipleAttributesOperand, PySingleAttributeOperand,
        },
        edges::{PyEdgeIndexOperand, PyEdgeIndicesOperand, PyEdgeOperand},
        nodes::{PyEdgeDirection, PyNodeIndexOperand, PyNodeIndicesOperand, PyNodeOperand},
        values::{PyMultipleValuesOperand, PySingleValueOperand},
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
    m.add_class::<PyDuration>()?;
    m.add_class::<PyNull>()?;
    m.add_class::<PyAny>()?;
    m.add_class::<PyUnion>()?;
    m.add_class::<PyOption>()?;

    m.add_class::<PyAttributeDataType>()?;
    m.add_class::<PyAttributeType>()?;
    m.add_class::<PyGroupSchema>()?;
    m.add_class::<PySchema>()?;

    m.add_class::<PyEdgeDirection>()?;

    m.add_class::<PyNodeOperand>()?;
    m.add_class::<PyNodeIndicesOperand>()?;
    m.add_class::<PyNodeIndexOperand>()?;

    m.add_class::<PyEdgeOperand>()?;
    m.add_class::<PyEdgeIndicesOperand>()?;
    m.add_class::<PyEdgeIndexOperand>()?;

    m.add_class::<PyMultipleValuesOperand>()?;
    m.add_class::<PySingleValueOperand>()?;

    m.add_class::<PyAttributesTreeOperand>()?;
    m.add_class::<PyMultipleAttributesOperand>()?;
    m.add_class::<PySingleAttributeOperand>()?;

    Ok(())
}
