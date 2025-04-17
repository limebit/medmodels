mod gil_hash_map;
mod medrecord;

use medrecord::{
    datatype::{
        PyAny, PyBool, PyDateTime, PyDuration, PyFloat, PyInt, PyNull, PyOption, PyString, PyUnion,
    },
    querying::{
        attributes::{
            PyEdgeAttributesTreeOperand, PyEdgeMultipleAttributesOperand,
            PyEdgeSingleAttributeOperand, PyNodeAttributesTreeOperand,
            PyNodeMultipleAttributesOperand, PyNodeSingleAttributeOperand,
        },
        edges::{PyEdgeIndexOperand, PyEdgeIndicesOperand, PyEdgeOperand},
        nodes::{PyEdgeDirection, PyNodeIndexOperand, PyNodeIndicesOperand, PyNodeOperand},
        values::{
            PyEdgeMultipleValuesOperand, PyEdgeSingleValueOperand, PyNodeMultipleValuesOperand,
            PyNodeSingleValueOperand,
        },
    },
    schema::{PyAttributeDataType, PyAttributeType, PyGroupSchema, PySchema, PySchemaType},
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
    m.add_class::<PySchemaType>()?;
    m.add_class::<PySchema>()?;

    m.add_class::<PyEdgeDirection>()?;

    m.add_class::<PyNodeOperand>()?;
    m.add_class::<PyNodeIndicesOperand>()?;
    m.add_class::<PyNodeIndexOperand>()?;

    m.add_class::<PyEdgeOperand>()?;
    m.add_class::<PyEdgeIndicesOperand>()?;
    m.add_class::<PyEdgeIndexOperand>()?;

    m.add_class::<PyNodeMultipleValuesOperand>()?;
    m.add_class::<PyEdgeMultipleValuesOperand>()?;
    m.add_class::<PyNodeSingleValueOperand>()?;
    m.add_class::<PyEdgeSingleValueOperand>()?;

    m.add_class::<PyNodeAttributesTreeOperand>()?;
    m.add_class::<PyEdgeAttributesTreeOperand>()?;
    m.add_class::<PyNodeMultipleAttributesOperand>()?;
    m.add_class::<PyEdgeMultipleAttributesOperand>()?;
    m.add_class::<PyNodeSingleAttributeOperand>()?;
    m.add_class::<PyEdgeSingleAttributeOperand>()?;

    Ok(())
}
