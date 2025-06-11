#![recursion_limit = "256"]

mod gil_hash_map;
mod medrecord;

use medrecord::{
    datatype::{
        PyAny, PyBool, PyDateTime, PyDuration, PyFloat, PyInt, PyNull, PyOption, PyString, PyUnion,
    },
    querying::{
        attributes::{
            PyEdgeAttributesTreeGroupOperand, PyEdgeAttributesTreeOperand,
            PyEdgeMultipleAttributesWithIndexGroupOperand,
            PyEdgeMultipleAttributesWithIndexOperand, PyEdgeMultipleAttributesWithoutIndexOperand,
            PyEdgeSingleAttributeWithIndexGroupOperand, PyEdgeSingleAttributeWithIndexOperand,
            PyEdgeSingleAttributeWithoutIndexGroupOperand,
            PyEdgeSingleAttributeWithoutIndexOperand, PyNodeAttributesTreeGroupOperand,
            PyNodeAttributesTreeOperand, PyNodeMultipleAttributesWithIndexGroupOperand,
            PyNodeMultipleAttributesWithIndexOperand, PyNodeMultipleAttributesWithoutIndexOperand,
            PyNodeSingleAttributeWithIndexGroupOperand, PyNodeSingleAttributeWithIndexOperand,
            PyNodeSingleAttributeWithoutIndexGroupOperand,
            PyNodeSingleAttributeWithoutIndexOperand,
        },
        edges::{
            EdgeOperandGroupDiscriminator, PyEdgeGroupOperand, PyEdgeIndexGroupOperand,
            PyEdgeIndexOperand, PyEdgeIndicesGroupOperand, PyEdgeIndicesOperand, PyEdgeOperand,
        },
        nodes::{
            NodeOperandGroupDiscriminator, PyEdgeDirection, PyNodeGroupOperand,
            PyNodeIndexGroupOperand, PyNodeIndexOperand, PyNodeIndicesGroupOperand,
            PyNodeIndicesOperand, PyNodeOperand,
        },
        values::{
            PyEdgeMultipleValuesWithIndexGroupOperand, PyEdgeMultipleValuesWithIndexOperand,
            PyEdgeMultipleValuesWithoutIndexOperand, PyEdgeSingleValueWithIndexGroupOperand,
            PyEdgeSingleValueWithIndexOperand, PyEdgeSingleValueWithoutIndexGroupOperand,
            PyEdgeSingleValueWithoutIndexOperand, PyNodeMultipleValuesWithIndexGroupOperand,
            PyNodeMultipleValuesWithIndexOperand, PyNodeMultipleValuesWithoutIndexOperand,
            PyNodeSingleValueWithIndexGroupOperand, PyNodeSingleValueWithIndexOperand,
            PyNodeSingleValueWithoutIndexGroupOperand, PyNodeSingleValueWithoutIndexOperand,
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

    m.add_class::<NodeOperandGroupDiscriminator>()?;
    m.add_class::<PyNodeOperand>()?;
    m.add_class::<PyNodeGroupOperand>()?;
    m.add_class::<PyNodeIndicesOperand>()?;
    m.add_class::<PyNodeIndicesGroupOperand>()?;
    m.add_class::<PyNodeIndexOperand>()?;
    m.add_class::<PyNodeIndexGroupOperand>()?;

    m.add_class::<EdgeOperandGroupDiscriminator>()?;
    m.add_class::<PyEdgeOperand>()?;
    m.add_class::<PyEdgeGroupOperand>()?;
    m.add_class::<PyEdgeIndicesOperand>()?;
    m.add_class::<PyEdgeIndicesGroupOperand>()?;
    m.add_class::<PyEdgeIndexOperand>()?;
    m.add_class::<PyEdgeIndexGroupOperand>()?;

    m.add_class::<PyNodeMultipleValuesWithIndexOperand>()?;
    m.add_class::<PyNodeMultipleValuesWithIndexGroupOperand>()?;
    m.add_class::<PyNodeMultipleValuesWithoutIndexOperand>()?;
    m.add_class::<PyEdgeMultipleValuesWithIndexOperand>()?;
    m.add_class::<PyEdgeMultipleValuesWithIndexGroupOperand>()?;
    m.add_class::<PyEdgeMultipleValuesWithoutIndexOperand>()?;
    m.add_class::<PyNodeSingleValueWithIndexOperand>()?;
    m.add_class::<PyNodeSingleValueWithIndexGroupOperand>()?;
    m.add_class::<PyNodeSingleValueWithoutIndexOperand>()?;
    m.add_class::<PyNodeSingleValueWithoutIndexGroupOperand>()?;
    m.add_class::<PyEdgeSingleValueWithIndexOperand>()?;
    m.add_class::<PyEdgeSingleValueWithIndexGroupOperand>()?;
    m.add_class::<PyEdgeSingleValueWithoutIndexOperand>()?;
    m.add_class::<PyEdgeSingleValueWithoutIndexGroupOperand>()?;

    m.add_class::<PyNodeAttributesTreeOperand>()?;
    m.add_class::<PyNodeAttributesTreeGroupOperand>()?;
    m.add_class::<PyEdgeAttributesTreeOperand>()?;
    m.add_class::<PyEdgeAttributesTreeGroupOperand>()?;
    m.add_class::<PyNodeMultipleAttributesWithIndexOperand>()?;
    m.add_class::<PyNodeMultipleAttributesWithIndexGroupOperand>()?;
    m.add_class::<PyNodeMultipleAttributesWithoutIndexOperand>()?;
    m.add_class::<PyEdgeMultipleAttributesWithIndexOperand>()?;
    m.add_class::<PyEdgeMultipleAttributesWithIndexGroupOperand>()?;
    m.add_class::<PyEdgeMultipleAttributesWithoutIndexOperand>()?;
    m.add_class::<PyNodeSingleAttributeWithIndexOperand>()?;
    m.add_class::<PyNodeSingleAttributeWithIndexGroupOperand>()?;
    m.add_class::<PyNodeSingleAttributeWithoutIndexOperand>()?;
    m.add_class::<PyNodeSingleAttributeWithoutIndexGroupOperand>()?;
    m.add_class::<PyEdgeSingleAttributeWithIndexOperand>()?;
    m.add_class::<PyEdgeSingleAttributeWithIndexGroupOperand>()?;
    m.add_class::<PyEdgeSingleAttributeWithoutIndexOperand>()?;
    m.add_class::<PyEdgeSingleAttributeWithoutIndexGroupOperand>()?;

    Ok(())
}
