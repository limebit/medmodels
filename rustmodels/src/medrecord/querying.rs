use super::{attribute::PyMedRecordAttribute, value::PyMedRecordValue, Lut};
use crate::{
    gil_hash_map::GILHashMap,
    medrecord::{
        errors::PyMedRecordError, value::convert_pyobject_to_medrecordvalue, PyGroup, PyNodeIndex,
    },
};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{
        ArithmeticOperation, EdgeAttributeOperand, EdgeIndex, EdgeIndexOperand, EdgeOperand,
        EdgeOperation, MedRecordAttribute, MedRecordValue, NodeAttributeOperand, NodeIndexOperand,
        NodeOperand, NodeOperation, TransformationOperation, ValueOperand,
    },
};
use pyo3::{
    pyclass, pymethods, types::PyAnyMethods, Bound, FromPyObject, IntoPy, PyAny, PyObject,
    PyResult, Python,
};
use std::ops::Range;

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyValueArithmeticOperation(ArithmeticOperation, MedRecordAttribute, MedRecordValue);

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyValueTransformationOperation(TransformationOperation, MedRecordAttribute);

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyValueSliceOperation(MedRecordAttribute, Range<usize>);

#[repr(transparent)]
#[derive(Clone, Debug)]
pub(crate) struct PyValueOperand(ValueOperand);

impl From<ValueOperand> for PyValueOperand {
    fn from(value: ValueOperand) -> Self {
        PyValueOperand(value)
    }
}

impl From<PyValueOperand> for ValueOperand {
    fn from(value: PyValueOperand) -> Self {
        value.0
    }
}

static PYVALUEOPERAND_CONVERSION_LUT: Lut<ValueOperand> = GILHashMap::new();

fn convert_pyobject_to_valueoperand(ob: &Bound<'_, PyAny>) -> PyResult<ValueOperand> {
    if let Ok(value) = convert_pyobject_to_medrecordvalue(ob) {
        return Ok(ValueOperand::Value(value));
    };

    fn convert_node_attribute_operand(ob: &Bound<'_, PyAny>) -> PyResult<ValueOperand> {
        Ok(ValueOperand::Evaluate(MedRecordAttribute::from(
            ob.extract::<PyNodeAttributeOperand>()?.0,
        )))
    }

    fn convert_edge_attribute_operand(ob: &Bound<'_, PyAny>) -> PyResult<ValueOperand> {
        Ok(ValueOperand::Evaluate(MedRecordAttribute::from(
            ob.extract::<PyEdgeAttributeOperand>()?.0,
        )))
    }

    fn convert_arithmetic_operation(ob: &Bound<'_, PyAny>) -> PyResult<ValueOperand> {
        let operation = ob.extract::<PyValueArithmeticOperation>()?;

        Ok(ValueOperand::ArithmeticOperation(
            operation.0,
            operation.1,
            operation.2,
        ))
    }

    fn convert_transformation_operation(ob: &Bound<'_, PyAny>) -> PyResult<ValueOperand> {
        let operation = ob.extract::<PyValueTransformationOperation>()?;

        Ok(ValueOperand::TransformationOperation(
            operation.0,
            operation.1,
        ))
    }

    fn convert_slice_operation(ob: &Bound<'_, PyAny>) -> PyResult<ValueOperand> {
        let operation = ob.extract::<PyValueSliceOperation>()?;

        Ok(ValueOperand::Slice(operation.0, operation.1))
    }

    fn throw_error(ob: &Bound<'_, PyAny>) -> PyResult<ValueOperand> {
        Err(
            PyMedRecordError::from(MedRecordError::ConversionError(format!(
                "Failed to convert {} into ValueOperand",
                ob,
            )))
            .into(),
        )
    }

    let type_pointer = ob.get_type_ptr() as usize;

    Python::with_gil(|py| {
        PYVALUEOPERAND_CONVERSION_LUT.map(py, |lut| {
            let conversion_function = lut.entry(type_pointer).or_insert_with(|| {
                if ob.is_instance_of::<PyNodeAttributeOperand>() {
                    convert_node_attribute_operand
                } else if ob.is_instance_of::<PyEdgeAttributeOperand>() {
                    convert_edge_attribute_operand
                } else if ob.is_instance_of::<PyValueArithmeticOperation>() {
                    convert_arithmetic_operation
                } else if ob.is_instance_of::<PyValueTransformationOperation>() {
                    convert_transformation_operation
                } else if ob.is_instance_of::<PyValueSliceOperation>() {
                    convert_slice_operation
                } else {
                    throw_error
                }
            });

            conversion_function(ob)
        })
    })
}

impl<'a> FromPyObject<'a> for PyValueOperand {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        convert_pyobject_to_valueoperand(ob).map(PyValueOperand::from)
    }
}

impl IntoPy<PyObject> for PyValueOperand {
    fn into_py(self, py: pyo3::prelude::Python<'_>) -> PyObject {
        match self.0 {
            ValueOperand::Value(value) => PyMedRecordValue::from(value).into_py(py),
            ValueOperand::Evaluate(attribute) => PyMedRecordAttribute::from(attribute).into_py(py),
            ValueOperand::ArithmeticOperation(operation, attribute, value) => {
                PyValueArithmeticOperation(operation, attribute, value).into_py(py)
            }
            ValueOperand::TransformationOperation(operation, attribute) => {
                PyValueTransformationOperation(operation, attribute).into_py(py)
            }
            ValueOperand::Slice(attribute, range) => {
                PyValueSliceOperation(attribute, range).into_py(py)
            }
        }
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct PyNodeOperation(NodeOperation);

impl From<NodeOperation> for PyNodeOperation {
    fn from(value: NodeOperation) -> Self {
        PyNodeOperation(value)
    }
}

impl From<PyNodeOperation> for NodeOperation {
    fn from(value: PyNodeOperation) -> Self {
        value.0
    }
}

#[pymethods]
impl PyNodeOperation {
    fn logical_and(&self, operation: PyNodeOperation) -> PyNodeOperation {
        self.clone().0.and(operation.into()).into()
    }

    fn logical_or(&self, operation: PyNodeOperation) -> PyNodeOperation {
        self.clone().0.or(operation.into()).into()
    }

    fn logical_xor(&self, operation: PyNodeOperation) -> PyNodeOperation {
        self.clone().0.xor(operation.into()).into()
    }

    fn logical_not(&self) -> PyNodeOperation {
        self.clone().0.not().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct PyEdgeOperation(EdgeOperation);

impl From<EdgeOperation> for PyEdgeOperation {
    fn from(value: EdgeOperation) -> Self {
        PyEdgeOperation(value)
    }
}

impl From<PyEdgeOperation> for EdgeOperation {
    fn from(value: PyEdgeOperation) -> Self {
        value.0
    }
}

#[pymethods]
impl PyEdgeOperation {
    fn logical_and(&self, operation: PyEdgeOperation) -> PyEdgeOperation {
        self.clone().0.and(operation.into()).into()
    }

    fn logical_or(&self, operation: PyEdgeOperation) -> PyEdgeOperation {
        self.clone().0.or(operation.into()).into()
    }

    fn logical_xor(&self, operation: PyEdgeOperation) -> PyEdgeOperation {
        self.clone().0.xor(operation.into()).into()
    }

    fn logical_not(&self) -> PyEdgeOperation {
        self.clone().0.not().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct PyNodeAttributeOperand(pub NodeAttributeOperand);

impl From<NodeAttributeOperand> for PyNodeAttributeOperand {
    fn from(value: NodeAttributeOperand) -> Self {
        PyNodeAttributeOperand(value)
    }
}

impl From<PyNodeAttributeOperand> for NodeAttributeOperand {
    fn from(value: PyNodeAttributeOperand) -> Self {
        value.0
    }
}

#[pymethods]
impl PyNodeAttributeOperand {
    fn greater(&self, operand: PyValueOperand) -> PyNodeOperation {
        self.clone().0.greater(ValueOperand::from(operand)).into()
    }
    fn less(&self, operand: PyValueOperand) -> PyNodeOperation {
        self.clone().0.less(ValueOperand::from(operand)).into()
    }
    fn greater_or_equal(&self, operand: PyValueOperand) -> PyNodeOperation {
        self.clone()
            .0
            .greater_or_equal(ValueOperand::from(operand))
            .into()
    }
    fn less_or_equal(&self, operand: PyValueOperand) -> PyNodeOperation {
        self.clone()
            .0
            .less_or_equal(ValueOperand::from(operand))
            .into()
    }

    fn equal(&self, operand: PyValueOperand) -> PyNodeOperation {
        self.clone().0.equal(ValueOperand::from(operand)).into()
    }
    fn not_equal(&self, operand: PyValueOperand) -> PyNodeOperation {
        self.clone().0.not_equal(ValueOperand::from(operand)).into()
    }

    fn is_in(&self, operands: Vec<PyMedRecordValue>) -> PyNodeOperation {
        self.clone().0.r#in(operands).into()
    }
    fn not_in(&self, operands: Vec<PyMedRecordValue>) -> PyNodeOperation {
        self.clone().0.not_in(operands).into()
    }

    fn starts_with(&self, operand: PyValueOperand) -> PyNodeOperation {
        self.clone()
            .0
            .starts_with(ValueOperand::from(operand))
            .into()
    }

    fn ends_with(&self, operand: PyValueOperand) -> PyNodeOperation {
        self.clone().0.ends_with(ValueOperand::from(operand)).into()
    }

    fn contains(&self, operand: PyValueOperand) -> PyNodeOperation {
        self.clone().0.contains(ValueOperand::from(operand)).into()
    }

    fn add(&self, value: PyMedRecordValue) -> PyValueOperand {
        self.clone().0.add(value).into()
    }

    fn sub(&self, value: PyMedRecordValue) -> PyValueOperand {
        self.clone().0.sub(value).into()
    }

    fn mul(&self, value: PyMedRecordValue) -> PyValueOperand {
        self.clone().0.mul(value).into()
    }

    fn div(&self, value: PyMedRecordValue) -> PyValueOperand {
        self.clone().0.div(value).into()
    }

    fn pow(&self, value: PyMedRecordValue) -> PyValueOperand {
        self.clone().0.pow(value).into()
    }

    fn r#mod(&self, value: PyMedRecordValue) -> PyValueOperand {
        self.clone().0.r#mod(value).into()
    }

    fn round(&self) -> PyValueOperand {
        self.clone().0.round().into()
    }

    fn ceil(&self) -> PyValueOperand {
        self.clone().0.ceil().into()
    }

    fn floor(&self) -> PyValueOperand {
        self.clone().0.floor().into()
    }

    fn abs(&self) -> PyValueOperand {
        self.clone().0.abs().into()
    }

    fn sqrt(&self) -> PyValueOperand {
        self.clone().0.sqrt().into()
    }

    fn trim(&self) -> PyValueOperand {
        self.clone().0.trim().into()
    }

    fn trim_start(&self) -> PyValueOperand {
        self.clone().0.trim_start().into()
    }

    fn trim_end(&self) -> PyValueOperand {
        self.clone().0.trim_end().into()
    }

    fn lowercase(&self) -> PyValueOperand {
        self.clone().0.lowercase().into()
    }

    fn uppercase(&self) -> PyValueOperand {
        self.clone().0.uppercase().into()
    }

    fn slice(&self, start: usize, end: usize) -> PyResult<PyValueOperand> {
        Ok(self.clone().0.slice(Range { start, end }).into())
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct PyEdgeAttributeOperand(EdgeAttributeOperand);

impl From<EdgeAttributeOperand> for PyEdgeAttributeOperand {
    fn from(value: EdgeAttributeOperand) -> Self {
        PyEdgeAttributeOperand(value)
    }
}

impl From<PyEdgeAttributeOperand> for EdgeAttributeOperand {
    fn from(value: PyEdgeAttributeOperand) -> Self {
        value.0
    }
}

#[pymethods]
impl PyEdgeAttributeOperand {
    fn greater(&self, operand: PyValueOperand) -> PyEdgeOperation {
        self.clone().0.greater(ValueOperand::from(operand)).into()
    }
    fn less(&self, operand: PyValueOperand) -> PyEdgeOperation {
        self.clone().0.less(ValueOperand::from(operand)).into()
    }
    fn greater_or_equal(&self, operand: PyValueOperand) -> PyEdgeOperation {
        self.clone()
            .0
            .greater_or_equal(ValueOperand::from(operand))
            .into()
    }
    fn less_or_equal(&self, operand: PyValueOperand) -> PyEdgeOperation {
        self.clone()
            .0
            .less_or_equal(ValueOperand::from(operand))
            .into()
    }

    fn equal(&self, operand: PyValueOperand) -> PyEdgeOperation {
        self.clone().0.equal(ValueOperand::from(operand)).into()
    }
    fn not_equal(&self, operand: PyValueOperand) -> PyEdgeOperation {
        self.clone().0.not_equal(ValueOperand::from(operand)).into()
    }

    fn is_in(&self, operand: Vec<PyMedRecordValue>) -> PyEdgeOperation {
        self.clone().0.r#in(operand).into()
    }
    fn not_in(&self, operand: Vec<PyMedRecordValue>) -> PyEdgeOperation {
        self.clone().0.not_in(operand).into()
    }

    fn starts_with(&self, operand: PyValueOperand) -> PyEdgeOperation {
        self.clone()
            .0
            .starts_with(ValueOperand::from(operand))
            .into()
    }

    fn ends_with(&self, operand: PyValueOperand) -> PyEdgeOperation {
        self.clone().0.ends_with(ValueOperand::from(operand)).into()
    }

    fn contains(&self, operand: PyValueOperand) -> PyEdgeOperation {
        self.clone().0.contains(ValueOperand::from(operand)).into()
    }

    fn add(&self, value: PyMedRecordValue) -> PyValueOperand {
        self.clone().0.add(value).into()
    }

    fn sub(&self, value: PyMedRecordValue) -> PyValueOperand {
        self.clone().0.sub(value).into()
    }

    fn mul(&self, value: PyMedRecordValue) -> PyValueOperand {
        self.clone().0.mul(value).into()
    }

    fn div(&self, value: PyMedRecordValue) -> PyValueOperand {
        self.clone().0.div(value).into()
    }

    fn pow(&self, value: PyMedRecordValue) -> PyValueOperand {
        self.clone().0.pow(value).into()
    }

    fn r#mod(&self, value: PyMedRecordValue) -> PyValueOperand {
        self.clone().0.r#mod(value).into()
    }

    fn round(&self) -> PyValueOperand {
        self.clone().0.round().into()
    }

    fn ceil(&self) -> PyValueOperand {
        self.clone().0.ceil().into()
    }

    fn floor(&self) -> PyValueOperand {
        self.clone().0.floor().into()
    }

    fn abs(&self) -> PyValueOperand {
        self.clone().0.abs().into()
    }

    fn sqrt(&self) -> PyValueOperand {
        self.clone().0.sqrt().into()
    }

    fn trim(&self) -> PyValueOperand {
        self.clone().0.trim().into()
    }

    fn trim_start(&self) -> PyValueOperand {
        self.clone().0.trim_start().into()
    }

    fn trim_end(&self) -> PyValueOperand {
        self.clone().0.trim_end().into()
    }

    fn lowercase(&self) -> PyValueOperand {
        self.clone().0.lowercase().into()
    }

    fn uppercase(&self) -> PyValueOperand {
        self.clone().0.uppercase().into()
    }

    fn slice(&self, start: usize, end: usize) -> PyResult<PyValueOperand> {
        Ok(self.clone().0.slice(Range { start, end }).into())
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct PyNodeIndexOperand(NodeIndexOperand);

impl From<NodeIndexOperand> for PyNodeIndexOperand {
    fn from(value: NodeIndexOperand) -> Self {
        PyNodeIndexOperand(value)
    }
}

impl From<PyNodeIndexOperand> for NodeIndexOperand {
    fn from(value: PyNodeIndexOperand) -> Self {
        value.0
    }
}

#[pymethods]
impl PyNodeIndexOperand {
    fn greater(&self, operand: PyNodeIndex) -> PyNodeOperation {
        self.clone().0.greater(operand).into()
    }
    fn less(&self, operand: PyNodeIndex) -> PyNodeOperation {
        self.clone().0.less(operand).into()
    }
    fn greater_or_equal(&self, operand: PyNodeIndex) -> PyNodeOperation {
        self.clone().0.greater_or_equal(operand).into()
    }
    fn less_or_equal(&self, operand: PyNodeIndex) -> PyNodeOperation {
        self.clone().0.less_or_equal(operand).into()
    }

    fn equal(&self, operand: PyNodeIndex) -> PyNodeOperation {
        self.clone().0.equal(operand).into()
    }
    fn not_equal(&self, operand: PyNodeIndex) -> PyNodeOperation {
        self.clone().0.not_equal(operand).into()
    }

    fn is_in(&self, operand: Vec<PyNodeIndex>) -> PyNodeOperation {
        self.clone().0.r#in(operand).into()
    }
    fn not_in(&self, operand: Vec<PyNodeIndex>) -> PyNodeOperation {
        self.clone().0.not_in(operand).into()
    }

    fn starts_with(&self, operand: PyNodeIndex) -> PyNodeOperation {
        self.clone().0.starts_with(operand).into()
    }

    fn ends_with(&self, operand: PyNodeIndex) -> PyNodeOperation {
        self.clone().0.ends_with(operand).into()
    }

    fn contains(&self, operand: PyNodeIndex) -> PyNodeOperation {
        self.clone().0.contains(operand).into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct PyEdgeIndexOperand(EdgeIndexOperand);

impl From<EdgeIndexOperand> for PyEdgeIndexOperand {
    fn from(value: EdgeIndexOperand) -> Self {
        PyEdgeIndexOperand(value)
    }
}

impl From<PyEdgeIndexOperand> for EdgeIndexOperand {
    fn from(value: PyEdgeIndexOperand) -> Self {
        value.0
    }
}

#[pymethods]
impl PyEdgeIndexOperand {
    fn greater(&self, operand: EdgeIndex) -> PyEdgeOperation {
        self.clone().0.greater(operand).into()
    }
    fn less(&self, operand: EdgeIndex) -> PyEdgeOperation {
        self.clone().0.less(operand).into()
    }
    fn greater_or_equal(&self, operand: EdgeIndex) -> PyEdgeOperation {
        self.clone().0.greater_or_equal(operand).into()
    }
    fn less_or_equal(&self, operand: EdgeIndex) -> PyEdgeOperation {
        self.clone().0.less_or_equal(operand).into()
    }

    fn equal(&self, operand: EdgeIndex) -> PyEdgeOperation {
        self.clone().0.equal(operand).into()
    }
    fn not_equal(&self, operand: EdgeIndex) -> PyEdgeOperation {
        self.clone().0.not_equal(operand).into()
    }

    fn is_in(&self, operand: Vec<EdgeIndex>) -> PyEdgeOperation {
        self.clone().0.r#in(operand).into()
    }
    fn not_in(&self, operand: Vec<EdgeIndex>) -> PyEdgeOperation {
        self.clone().0.not_in(operand).into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct PyNodeOperand(NodeOperand);

#[pymethods]
impl PyNodeOperand {
    #[new]
    fn new() -> Self {
        Self(NodeOperand)
    }

    fn in_group(&self, operand: PyGroup) -> PyNodeOperation {
        self.clone().0.in_group(operand).into()
    }

    fn has_attribute(&self, operand: PyMedRecordAttribute) -> PyNodeOperation {
        self.clone().0.has_attribute(operand).into()
    }

    fn has_outgoing_edge_with(&self, operation: PyEdgeOperation) -> PyNodeOperation {
        self.clone()
            .0
            .has_outgoing_edge_with(operation.into())
            .into()
    }
    fn has_incoming_edge_with(&self, operation: PyEdgeOperation) -> PyNodeOperation {
        self.clone()
            .0
            .has_incoming_edge_with(operation.into())
            .into()
    }
    fn has_edge_with(&self, operation: PyEdgeOperation) -> PyNodeOperation {
        self.clone().0.has_edge_with(operation.into()).into()
    }

    fn has_neighbor_with(&self, operation: PyNodeOperation) -> PyNodeOperation {
        self.clone().0.has_neighbor_with(operation.into()).into()
    }

    fn attribute(&self, attribute: PyMedRecordAttribute) -> PyNodeAttributeOperand {
        self.clone().0.attribute(attribute).into()
    }

    fn index(&self) -> PyNodeIndexOperand {
        self.clone().0.index().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone, Debug)]
pub struct PyEdgeOperand(EdgeOperand);

#[pymethods]
impl PyEdgeOperand {
    #[new]
    fn new() -> Self {
        Self(EdgeOperand)
    }

    fn connected_target(&self, operand: PyNodeIndex) -> PyEdgeOperation {
        self.clone().0.connected_target(operand).into()
    }

    fn connected_source(&self, operand: PyNodeIndex) -> PyEdgeOperation {
        self.clone().0.connected_source(operand).into()
    }

    fn connected(&self, operand: PyNodeIndex) -> PyEdgeOperation {
        self.clone().0.connected(operand).into()
    }

    fn in_group(&self, operand: PyGroup) -> PyEdgeOperation {
        self.clone().0.in_group(operand).into()
    }

    fn has_attribute(&self, operand: PyMedRecordAttribute) -> PyEdgeOperation {
        self.clone().0.has_attribute(operand).into()
    }

    fn connected_source_with(&self, operation: PyNodeOperation) -> PyEdgeOperation {
        self.clone()
            .0
            .connected_source_with(operation.into())
            .into()
    }

    fn connected_target_with(&self, operation: PyNodeOperation) -> PyEdgeOperation {
        self.clone()
            .0
            .connected_target_with(operation.into())
            .into()
    }

    fn connected_with(&self, operation: PyNodeOperation) -> PyEdgeOperation {
        self.clone().0.connected_with(operation.into()).into()
    }

    fn has_parallel_edges_with(&self, operation: PyEdgeOperation) -> PyEdgeOperation {
        self.clone()
            .0
            .has_parallel_edges_with(operation.into())
            .into()
    }

    fn has_parallel_edges_with_self_comparison(
        &self,
        operation: PyEdgeOperation,
    ) -> PyEdgeOperation {
        self.clone()
            .0
            .has_parallel_edges_with_self_comparison(operation.into())
            .into()
    }

    fn attribute(&self, attribute: PyMedRecordAttribute) -> PyEdgeAttributeOperand {
        self.clone().0.attribute(attribute).into()
    }

    fn index(&self) -> PyEdgeIndexOperand {
        self.clone().0.index().into()
    }
}
