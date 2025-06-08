#![recursion_limit = "256"]

mod gil_hash_map;
mod medrecord;

use medrecord::{
    datatype::{
        PyAny, PyBool, PyDateTime, PyDuration, PyFloat, PyInt, PyNull, PyOption, PyString, PyUnion,
    },
    querying::{
        attributes::{
            PyEdgeAttributesTreeOperand, PyEdgeMultipleAttributesOperandWithIndex,
            PyEdgeMultipleAttributesOperandWithoutIndex, PyEdgeSingleAttributeOperandWithIndex,
            PyEdgeSingleAttributeOperandWithoutIndex, PyNodeAttributesTreeOperand,
            PyNodeMultipleAttributesOperandWithIndex, PyNodeMultipleAttributesOperandWithoutIndex,
            PyNodeSingleAttributeOperandWithIndex, PyNodeSingleAttributeOperandWithoutIndex,
        },
        edges::{PyEdgeIndexOperand, PyEdgeIndicesOperand, PyEdgeOperand},
        nodes::{PyEdgeDirection, PyNodeIndexOperand, PyNodeIndicesOperand, PyNodeOperand},
        values::{
            PyEdgeMultipleValuesOperandWithIndex, PyEdgeMultipleValuesOperandWithoutIndex,
            PyEdgeSingleValueOperandWithIndex, PyEdgeSingleValueOperandWithoutIndex,
            PyNodeMultipleValuesOperandWithIndex, PyNodeMultipleValuesOperandWithoutIndex,
            PyNodeSingleValueOperandWithIndex, PyNodeSingleValueOperandWithoutIndex,
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

    m.add_class::<PyNodeMultipleValuesOperandWithIndex>()?;
    m.add_class::<PyNodeMultipleValuesOperandWithoutIndex>()?;
    m.add_class::<PyEdgeMultipleValuesOperandWithIndex>()?;
    m.add_class::<PyEdgeMultipleValuesOperandWithoutIndex>()?;
    m.add_class::<PyNodeSingleValueOperandWithIndex>()?;
    m.add_class::<PyNodeSingleValueOperandWithoutIndex>()?;
    m.add_class::<PyEdgeSingleValueOperandWithIndex>()?;
    m.add_class::<PyEdgeSingleValueOperandWithoutIndex>()?;

    m.add_class::<PyNodeAttributesTreeOperand>()?;
    m.add_class::<PyEdgeAttributesTreeOperand>()?;
    m.add_class::<PyNodeMultipleAttributesOperandWithIndex>()?;
    m.add_class::<PyNodeMultipleAttributesOperandWithoutIndex>()?;
    m.add_class::<PyEdgeMultipleAttributesOperandWithIndex>()?;
    m.add_class::<PyEdgeMultipleAttributesOperandWithoutIndex>()?;
    m.add_class::<PyNodeSingleAttributeOperandWithIndex>()?;
    m.add_class::<PyNodeSingleAttributeOperandWithoutIndex>()?;
    m.add_class::<PyEdgeSingleAttributeOperandWithIndex>()?;
    m.add_class::<PyEdgeSingleAttributeOperandWithoutIndex>()?;

    Ok(())
}
