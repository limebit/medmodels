pub mod attributes;
pub mod edges;
pub mod nodes;
pub mod values;

use crate::gil_hash_map::GILHashMap;

use super::{
    attribute::PyMedRecordAttribute, errors::PyMedRecordError, traits::DeepFrom,
    value::PyMedRecordValue, Lut, PyNodeIndex,
};
use attributes::{
    PyEdgeAttributesTreeOperand, PyEdgeMultipleAttributesOperand, PyEdgeSingleAttributeOperand,
    PyNodeAttributesTreeOperand, PyNodeMultipleAttributesOperand, PyNodeSingleAttributeOperand,
};
use edges::{PyEdgeIndexOperand, PyEdgeIndicesOperand};
use medmodels_core::{
    errors::{MedRecordError, MedRecordResult},
    medrecord::{
        CardinalityWrapper, EdgeAttributesTreeOperand, EdgeIndexOperand, EdgeIndicesOperand,
        EdgeMultipleAttributesOperand, EdgeMultipleValuesOperand, EdgeSingleAttributeOperand,
        EdgeSingleValueOperand, MedRecordAttribute, NodeAttributesTreeOperand, NodeIndexOperand,
        NodeIndicesOperand, NodeMultipleAttributesOperand, NodeMultipleValuesOperand,
        NodeSingleAttributeOperand, NodeSingleValueOperand, OptionalIndexWrapper, ReturnOperand,
        Wrapper,
    },
    MedRecord,
};
use nodes::{PyNodeIndexOperand, PyNodeIndicesOperand};
use pyo3::{
    types::{PyAnyMethods, PyList, PyNone},
    Bound, FromPyObject, IntoPyObject, IntoPyObjectExt, PyAny, PyErr, PyResult, Python,
};
use std::collections::HashMap;
use values::{
    PyEdgeMultipleValuesOperand, PyEdgeSingleValueOperand, PyNodeMultipleValuesOperand,
    PyNodeSingleValueOperand,
};

pub enum PyReturnOperand {
    NodeAttributesTree(PyNodeAttributesTreeOperand),
    EdgeAttributesTree(PyEdgeAttributesTreeOperand),
    NodeMultipleAttributes(PyNodeMultipleAttributesOperand),
    EdgeMultipleAttributes(PyEdgeMultipleAttributesOperand),
    NodeSingleAttribute(PyNodeSingleAttributeOperand),
    EdgeSingleAttribute(PyEdgeSingleAttributeOperand),
    EdgeIndices(PyEdgeIndicesOperand),
    EdgeIndex(PyEdgeIndexOperand),
    NodeIndices(PyNodeIndicesOperand),
    NodeIndex(PyNodeIndexOperand),
    NodeMultipleValues(PyNodeMultipleValuesOperand),
    EdgeMultipleValues(PyEdgeMultipleValuesOperand),
    NodeSingleValue(PyNodeSingleValueOperand),
    EdgeSingleValue(PyEdgeSingleValueOperand),
    Vector(Vec<Self>),
}

impl<'a> ReturnOperand<'a> for PyReturnOperand {
    type ReturnValue = PyReturnValue<'a>;

    fn evaluate(self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        match self {
            PyReturnOperand::NodeAttributesTree(operand) => {
                Wrapper::<NodeAttributesTreeOperand>::from(operand)
                    .evaluate(medrecord)
                    .map(PyReturnValue::NodeAttributesTree)
            }
            PyReturnOperand::EdgeAttributesTree(operand) => {
                Wrapper::<EdgeAttributesTreeOperand>::from(operand)
                    .evaluate(medrecord)
                    .map(PyReturnValue::EdgeAttributesTree)
            }
            PyReturnOperand::NodeMultipleAttributes(operand) => {
                Wrapper::<NodeMultipleAttributesOperand>::from(operand)
                    .evaluate(medrecord)
                    .map(PyReturnValue::NodeMultipleAttributes)
            }
            PyReturnOperand::EdgeMultipleAttributes(operand) => {
                Wrapper::<EdgeMultipleAttributesOperand>::from(operand)
                    .evaluate(medrecord)
                    .map(PyReturnValue::EdgeMultipleAttributes)
            }
            PyReturnOperand::NodeSingleAttribute(operand) => {
                Wrapper::<NodeSingleAttributeOperand>::from(operand)
                    .evaluate(medrecord)
                    .map(PyReturnValue::NodeSingleAttribute)
            }
            PyReturnOperand::EdgeSingleAttribute(operand) => {
                Wrapper::<EdgeSingleAttributeOperand>::from(operand)
                    .evaluate(medrecord)
                    .map(PyReturnValue::EdgeSingleAttribute)
            }
            PyReturnOperand::EdgeIndices(operand) => Wrapper::<EdgeIndicesOperand>::from(operand)
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeIndices),
            PyReturnOperand::EdgeIndex(operand) => Wrapper::<EdgeIndexOperand>::from(operand)
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeIndex),
            PyReturnOperand::NodeIndices(operand) => Wrapper::<NodeIndicesOperand>::from(operand)
                .evaluate(medrecord)
                .map(PyReturnValue::NodeIndices),
            PyReturnOperand::NodeIndex(operand) => Wrapper::<NodeIndexOperand>::from(operand)
                .evaluate(medrecord)
                .map(PyReturnValue::NodeIndex),
            PyReturnOperand::NodeMultipleValues(operand) => {
                Wrapper::<NodeMultipleValuesOperand>::from(operand)
                    .evaluate(medrecord)
                    .map(PyReturnValue::NodeMultipleValues)
            }
            PyReturnOperand::EdgeMultipleValues(operand) => {
                Wrapper::<EdgeMultipleValuesOperand>::from(operand)
                    .evaluate(medrecord)
                    .map(PyReturnValue::EdgeMultipleValues)
            }
            PyReturnOperand::NodeSingleValue(operand) => {
                Wrapper::<NodeSingleValueOperand>::from(operand)
                    .evaluate(medrecord)
                    .map(PyReturnValue::NodeSingleValue)
            }
            PyReturnOperand::EdgeSingleValue(operand) => {
                Wrapper::<EdgeSingleValueOperand>::from(operand)
                    .evaluate(medrecord)
                    .map(PyReturnValue::EdgeSingleValue)
            }
            PyReturnOperand::Vector(operand) => operand
                .into_iter()
                .map(|item| item.evaluate(medrecord))
                .collect::<MedRecordResult<Vec<_>>>()
                .map(PyReturnValue::Vector),
        }
    }
}

static RETURNOPERAND_CONVERSION_LUT: Lut<PyReturnOperand> = GILHashMap::new();

pub(crate) fn convert_pyobject_to_pyreturnoperand(
    ob: &Bound<'_, PyAny>,
) -> PyResult<PyReturnOperand> {
    fn convert_py_node_attributes_tree_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeAttributesTree(
            ob.extract::<PyNodeAttributesTreeOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_attributes_tree_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeAttributesTree(
            ob.extract::<PyEdgeAttributesTreeOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_multiple_attributes_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeMultipleAttributes(
            ob.extract::<PyNodeMultipleAttributesOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_multiple_attributes_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeMultipleAttributes(
            ob.extract::<PyEdgeMultipleAttributesOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_attribute_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleAttribute(
            ob.extract::<PyNodeSingleAttributeOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_single_attribute_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleAttribute(
            ob.extract::<PyEdgeSingleAttributeOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_indices_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeIndices(
            ob.extract::<PyEdgeIndicesOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_index_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeIndex(
            ob.extract::<PyEdgeIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_indices_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeIndices(
            ob.extract::<PyNodeIndicesOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_index_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeIndex(
            ob.extract::<PyNodeIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_multiple_values_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeMultipleValues(
            ob.extract::<PyNodeMultipleValuesOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_multiple_values_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeMultipleValues(
            ob.extract::<PyEdgeMultipleValuesOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_value_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleValue(
            ob.extract::<PyNodeSingleValueOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_single_value_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleValue(
            ob.extract::<PyEdgeSingleValueOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_list(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::Vector(
            ob.extract::<Vec<PyReturnOperand>>()?,
        ))
    }

    fn throw_error(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Err(
            PyMedRecordError::from(MedRecordError::ConversionError(format!(
                "Failed to convert {} into query ReturnOperand",
                ob,
            )))
            .into(),
        )
    }

    let type_pointer = ob.get_type_ptr() as usize;

    Python::with_gil(|py| {
        RETURNOPERAND_CONVERSION_LUT.map(py, |lut| {
            let conversion_function = lut.entry(type_pointer).or_insert_with(|| {
                if ob.is_instance_of::<PyNodeAttributesTreeOperand>() {
                    convert_py_node_attributes_tree_operand
                } else if ob.is_instance_of::<PyEdgeAttributesTreeOperand>() {
                    convert_py_edge_attributes_tree_operand
                } else if ob.is_instance_of::<PyNodeMultipleAttributesOperand>() {
                    convert_py_node_multiple_attributes_operand
                } else if ob.is_instance_of::<PyEdgeMultipleAttributesOperand>() {
                    convert_py_edge_multiple_attributes_operand
                } else if ob.is_instance_of::<PyNodeSingleAttributeOperand>() {
                    convert_py_node_single_attribute_operand
                } else if ob.is_instance_of::<PyEdgeSingleAttributeOperand>() {
                    convert_py_edge_single_attribute_operand
                } else if ob.is_instance_of::<PyEdgeIndicesOperand>() {
                    convert_py_edge_indices_operand
                } else if ob.is_instance_of::<PyEdgeIndexOperand>() {
                    convert_py_edge_index_operand
                } else if ob.is_instance_of::<PyNodeIndicesOperand>() {
                    convert_py_node_indices_operand
                } else if ob.is_instance_of::<PyNodeIndexOperand>() {
                    convert_py_node_index_operand
                } else if ob.is_instance_of::<PyNodeMultipleValuesOperand>() {
                    convert_py_node_multiple_values_operand
                } else if ob.is_instance_of::<PyEdgeMultipleValuesOperand>() {
                    convert_py_edge_multiple_values_operand
                } else if ob.is_instance_of::<PyNodeSingleValueOperand>() {
                    convert_py_node_single_value_operand
                } else if ob.is_instance_of::<PyEdgeSingleValueOperand>() {
                    convert_py_edge_single_value_operand
                } else if ob.is_instance_of::<PyList>() {
                    convert_py_list
                } else {
                    throw_error
                }
            });

            conversion_function(ob)
        })
    })
}

impl FromPyObject<'_> for PyReturnOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        convert_pyobject_to_pyreturnoperand(ob)
    }
}

pub enum PyReturnValue<'a> {
    NodeAttributesTree(<Wrapper<NodeAttributesTreeOperand> as ReturnOperand<'a>>::ReturnValue),
    EdgeAttributesTree(<Wrapper<EdgeAttributesTreeOperand> as ReturnOperand<'a>>::ReturnValue),
    NodeMultipleAttributes(
        <Wrapper<NodeMultipleAttributesOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeMultipleAttributes(
        <Wrapper<EdgeMultipleAttributesOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeSingleAttribute(<Wrapper<NodeSingleAttributeOperand> as ReturnOperand<'a>>::ReturnValue),
    EdgeSingleAttribute(<Wrapper<EdgeSingleAttributeOperand> as ReturnOperand<'a>>::ReturnValue),
    EdgeIndices(<Wrapper<EdgeIndicesOperand> as ReturnOperand<'a>>::ReturnValue),
    EdgeIndex(<Wrapper<EdgeIndexOperand> as ReturnOperand<'a>>::ReturnValue),
    NodeIndices(<Wrapper<NodeIndicesOperand> as ReturnOperand<'a>>::ReturnValue),
    NodeIndex(<Wrapper<NodeIndexOperand> as ReturnOperand<'a>>::ReturnValue),
    NodeMultipleValues(<Wrapper<NodeMultipleValuesOperand> as ReturnOperand<'a>>::ReturnValue),
    EdgeMultipleValues(<Wrapper<EdgeMultipleValuesOperand> as ReturnOperand<'a>>::ReturnValue),
    NodeSingleValue(<Wrapper<NodeSingleValueOperand> as ReturnOperand<'a>>::ReturnValue),
    EdgeSingleValue(<Wrapper<EdgeSingleValueOperand> as ReturnOperand<'a>>::ReturnValue),
    Vector(Vec<Self>),
}

impl<'py> IntoPyObject<'py> for PyReturnValue<'_> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            PyReturnValue::NodeAttributesTree(iterator) => iterator
                .map(|item| {
                    (
                        PyNodeIndex::from(item.0.clone()),
                        Vec::<PyMedRecordAttribute>::deep_from(item.1),
                    )
                })
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeAttributesTree(iterator) => iterator
                .map(|item| (item.0, Vec::<PyMedRecordAttribute>::deep_from(item.1)))
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeMultipleAttributes(iterator) => iterator
                .map(|item| {
                    (
                        PyNodeIndex::from(item.0.clone()),
                        PyMedRecordAttribute::from(item.1),
                    )
                })
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeMultipleAttributes(iterator) => iterator
                .map(|item| (item.0, PyMedRecordAttribute::from(item.1)))
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeSingleAttribute(attribute) => match attribute {
                Some(attribute) => match attribute {
                    OptionalIndexWrapper::WithIndex((index, attribute)) => (
                        PyNodeIndex::from(index.clone()),
                        PyMedRecordAttribute::from(attribute),
                    )
                        .into_bound_py_any(py),
                    OptionalIndexWrapper::WithoutIndex(attribute) => {
                        PyMedRecordAttribute::from(attribute).into_bound_py_any(py)
                    }
                },
                None => PyNone::get(py).into_bound_py_any(py),
            },
            PyReturnValue::EdgeSingleAttribute(attribute) => match attribute {
                Some(attribute) => match attribute {
                    OptionalIndexWrapper::WithIndex((index, attribute)) => {
                        (index, PyMedRecordAttribute::from(attribute)).into_bound_py_any(py)
                    }
                    OptionalIndexWrapper::WithoutIndex(attribute) => {
                        PyMedRecordAttribute::from(attribute).into_bound_py_any(py)
                    }
                },
                None => PyNone::get(py).into_bound_py_any(py),
            },
            PyReturnValue::EdgeIndices(iterator) => {
                iterator.collect::<Vec<_>>().into_bound_py_any(py)
            }
            PyReturnValue::EdgeIndex(index) => index.into_bound_py_any(py),
            PyReturnValue::NodeIndices(iterator) => iterator
                .map(PyMedRecordAttribute::from)
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeIndex(index) => {
                Option::<PyNodeIndex>::deep_from(index).into_bound_py_any(py)
            }
            PyReturnValue::NodeMultipleValues(iterator) => iterator
                .map(|item| {
                    (
                        PyNodeIndex::from(item.0.clone()),
                        PyMedRecordValue::from(item.1),
                    )
                })
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeMultipleValues(iterator) => iterator
                .map(|item| (item.0, PyMedRecordValue::from(item.1)))
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeSingleValue(value) => match value {
                Some(value) => match value {
                    OptionalIndexWrapper::WithIndex((index, value)) => (
                        PyNodeIndex::from(index.clone()),
                        PyMedRecordValue::from(value),
                    )
                        .into_bound_py_any(py),
                    OptionalIndexWrapper::WithoutIndex(value) => {
                        PyMedRecordValue::from(value).into_bound_py_any(py)
                    }
                },
                None => PyNone::get(py).into_bound_py_any(py),
            },
            PyReturnValue::EdgeSingleValue(value) => match value {
                Some(value) => match value {
                    OptionalIndexWrapper::WithIndex((index, value)) => {
                        (index, PyMedRecordValue::from(value)).into_bound_py_any(py)
                    }
                    OptionalIndexWrapper::WithoutIndex(value) => {
                        PyMedRecordValue::from(value).into_bound_py_any(py)
                    }
                },
                None => PyNone::get(py).into_bound_py_any(py),
            },
            PyReturnValue::Vector(vector) => vector.into_bound_py_any(py),
        }
    }
}

#[repr(transparent)]
pub struct PyMedRecordAttributeCardinalityWrapper(CardinalityWrapper<MedRecordAttribute>);

impl From<CardinalityWrapper<MedRecordAttribute>> for PyMedRecordAttributeCardinalityWrapper {
    fn from(attribute: CardinalityWrapper<MedRecordAttribute>) -> Self {
        Self(attribute)
    }
}

impl From<PyMedRecordAttributeCardinalityWrapper> for CardinalityWrapper<MedRecordAttribute> {
    fn from(attribute: PyMedRecordAttributeCardinalityWrapper) -> Self {
        attribute.0
    }
}

impl FromPyObject<'_> for PyMedRecordAttributeCardinalityWrapper {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(attribute) = ob.extract::<PyMedRecordAttribute>() {
            Ok(CardinalityWrapper::Single(MedRecordAttribute::from(attribute)).into())
        } else if let Ok(attributes) = ob.extract::<Vec<PyMedRecordAttribute>>() {
            Ok(CardinalityWrapper::Multiple(
                attributes
                    .into_iter()
                    .map(MedRecordAttribute::from)
                    .collect(),
            )
            .into())
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {} into MedRecordAttribute or List[MedREcordAttribute]",
                    ob,
                )))
                .into(),
            )
        }
    }
}

type PyGroupCardinalityWrapper = PyMedRecordAttributeCardinalityWrapper;
