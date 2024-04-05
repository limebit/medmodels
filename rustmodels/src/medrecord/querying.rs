use crate::{
    gil_hash_map::GILHashMap,
    medrecord::{
        conversion::{convert_pyobject_to_medrecordvalue, PyMedRecordAttribute, PyMedRecordValue},
        errors::PyMedRecordError,
        PyGroup, PyNodeIndex,
    },
};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{
        ArithmeticOperation, EdgeAttributeOperand, EdgeIndex, EdgeIndexOperand, EdgeOperand,
        EdgeOperation, MedRecordAttribute, NodeAttributeOperand, NodeIndexOperand, NodeOperand,
        NodeOperation, TransformationOperation, ValueOperand,
    },
};
use pyo3::{
    pyclass, pymethods, types::PyType, FromPyObject, IntoPy, PyAny, PyObject, PyResult, Python,
};

#[pyclass]
#[derive(Clone, Debug)]
pub enum PyArithmeticOperation {
    Addition,
    Subtraction,
    Multiplication,
    Division,
}

impl From<ArithmeticOperation> for PyArithmeticOperation {
    fn from(value: ArithmeticOperation) -> Self {
        match value {
            ArithmeticOperation::Addition => PyArithmeticOperation::Addition,
            ArithmeticOperation::Subtraction => PyArithmeticOperation::Subtraction,
            ArithmeticOperation::Multiplication => PyArithmeticOperation::Multiplication,
            ArithmeticOperation::Division => PyArithmeticOperation::Division,
        }
    }
}

impl From<PyArithmeticOperation> for ArithmeticOperation {
    fn from(value: PyArithmeticOperation) -> Self {
        match value {
            PyArithmeticOperation::Addition => ArithmeticOperation::Addition,
            PyArithmeticOperation::Subtraction => ArithmeticOperation::Subtraction,
            PyArithmeticOperation::Multiplication => ArithmeticOperation::Multiplication,
            PyArithmeticOperation::Division => ArithmeticOperation::Division,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub enum PyTransformationOperation {
    Round,
    Ceil,
    Floor,

    Trim,
    TrimStart,
    TrimEnd,

    Lowercase,
    Uppercase,
}

impl From<TransformationOperation> for PyTransformationOperation {
    fn from(value: TransformationOperation) -> Self {
        match value {
            TransformationOperation::Round => PyTransformationOperation::Round,
            TransformationOperation::Ceil => PyTransformationOperation::Ceil,
            TransformationOperation::Floor => PyTransformationOperation::Floor,
            TransformationOperation::Trim => PyTransformationOperation::Trim,
            TransformationOperation::TrimStart => PyTransformationOperation::TrimStart,
            TransformationOperation::TrimEnd => PyTransformationOperation::TrimEnd,
            TransformationOperation::Lowercase => PyTransformationOperation::Lowercase,
            TransformationOperation::Uppercase => PyTransformationOperation::Uppercase,
        }
    }
}

impl From<PyTransformationOperation> for TransformationOperation {
    fn from(value: PyTransformationOperation) -> Self {
        match value {
            PyTransformationOperation::Round => TransformationOperation::Round,
            PyTransformationOperation::Ceil => TransformationOperation::Ceil,
            PyTransformationOperation::Floor => TransformationOperation::Floor,
            PyTransformationOperation::Trim => TransformationOperation::Trim,
            PyTransformationOperation::TrimStart => TransformationOperation::TrimStart,
            PyTransformationOperation::TrimEnd => TransformationOperation::TrimEnd,
            PyTransformationOperation::Lowercase => TransformationOperation::Lowercase,
            PyTransformationOperation::Uppercase => TransformationOperation::Uppercase,
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyValueArithmeticOperation(
    PyArithmeticOperation,
    PyMedRecordAttribute,
    PyMedRecordValue,
);

#[pyclass]
#[derive(Clone, Debug)]
pub struct PyValueTransformationOperation(PyTransformationOperation, PyMedRecordAttribute);

#[derive(Debug)]
pub(crate) enum PyValueOperand {
    Value(PyMedRecordValue),
    Evaluate(PyMedRecordAttribute),
    ArithmeticOperation(PyValueArithmeticOperation),
    TransformationOperation(PyValueTransformationOperation),
}

static PYVALUEOPERAND_CONVERSION_LUT: GILHashMap<usize, fn(&PyAny) -> PyResult<PyValueOperand>> =
    GILHashMap::new();

impl<'a> FromPyObject<'a> for PyValueOperand {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        if let Ok(value) = convert_pyobject_to_medrecordvalue(ob) {
            return Ok(PyValueOperand::Value(value.into()));
        };

        fn convert_node_attribute_operand(ob: &PyAny) -> PyResult<PyValueOperand> {
            Ok(PyValueOperand::Evaluate(
                MedRecordAttribute::from(ob.extract::<PyNodeAttributeOperand>()?.0).into(),
            ))
        }

        fn convert_edge_attribute_operand(ob: &PyAny) -> PyResult<PyValueOperand> {
            Ok(PyValueOperand::Evaluate(
                MedRecordAttribute::from(ob.extract::<PyEdgeAttributeOperand>()?.0).into(),
            ))
        }

        fn convert_arithmetic_operation(ob: &PyAny) -> PyResult<PyValueOperand> {
            Ok(PyValueOperand::ArithmeticOperation(
                ob.extract::<PyValueArithmeticOperation>()?,
            ))
        }

        fn convert_transformation_operation(ob: &PyAny) -> PyResult<PyValueOperand> {
            Ok(PyValueOperand::TransformationOperation(
                ob.extract::<PyValueTransformationOperation>()?,
            ))
        }

        fn throw_error(ob: &PyAny) -> PyResult<PyValueOperand> {
            Err(PyMedRecordError(MedRecordError::ConversionError(format!(
                "Failed to convert {} into ValueOperand",
                ob,
            )))
            .into())
        }

        let type_pointer = PyType::as_type_ptr(ob.get_type()) as usize;

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
                    } else {
                        throw_error
                    }
                });

                conversion_function(ob)
            })
        })
    }
}

impl IntoPy<PyObject> for PyValueOperand {
    fn into_py(self, py: pyo3::prelude::Python<'_>) -> PyObject {
        match self {
            PyValueOperand::Value(value) => value.into_py(py),
            PyValueOperand::Evaluate(attribute) => attribute.into_py(py),
            PyValueOperand::ArithmeticOperation(operation) => operation.into_py(py),
            PyValueOperand::TransformationOperation(operation) => operation.into_py(py),
        }
    }
}

impl From<ValueOperand> for PyValueOperand {
    fn from(value: ValueOperand) -> Self {
        match value {
            ValueOperand::Value(value) => PyValueOperand::Value(value.into()),
            ValueOperand::Evaluate(value) => PyValueOperand::Evaluate(value.into()),
            ValueOperand::ArithmeticOperation(operation, attribute, value) => {
                PyValueOperand::ArithmeticOperation(PyValueArithmeticOperation(
                    operation.into(),
                    attribute.into(),
                    value.into(),
                ))
            }
            ValueOperand::TransformationOperation(operation, attribute) => {
                PyValueOperand::TransformationOperation(PyValueTransformationOperation(
                    operation.into(),
                    attribute.into(),
                ))
            }
            ValueOperand::Slice(_attribute, _range) => todo!(),
        }
    }
}

impl From<PyValueOperand> for ValueOperand {
    fn from(value: PyValueOperand) -> Self {
        match value {
            PyValueOperand::Value(value) => ValueOperand::Value(value.into()),
            PyValueOperand::Evaluate(attribute) => ValueOperand::Evaluate(attribute.into()),
            PyValueOperand::ArithmeticOperation(operation) => ValueOperand::ArithmeticOperation(
                operation.0.into(),
                operation.1.into(),
                operation.2.into(),
            ),
            PyValueOperand::TransformationOperation(operation) => {
                ValueOperand::TransformationOperation(operation.0.into(), operation.1.into())
            }
        }
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
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
#[derive(Clone)]
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
#[derive(Clone)]
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

    fn round(&self) -> PyValueOperand {
        self.clone().0.round().into()
    }

    fn ceil(&self) -> PyValueOperand {
        self.clone().0.ceil().into()
    }

    fn floor(&self) -> PyValueOperand {
        self.clone().0.floor().into()
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
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
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

    fn round(&self) -> PyValueOperand {
        self.clone().0.round().into()
    }

    fn ceil(&self) -> PyValueOperand {
        self.clone().0.ceil().into()
    }

    fn floor(&self) -> PyValueOperand {
        self.clone().0.floor().into()
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
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
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
#[derive(Clone)]
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
#[derive(Clone)]
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
#[derive(Clone)]
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
