pub mod attributes;
pub mod edges;
pub mod nodes;
pub mod values;

use crate::{
    gil_hash_map::GILHashMap,
    medrecord::querying::values::{
        PyEdgeMultipleValuesWithIndexOperand, PyEdgeMultipleValuesWithoutIndexOperand,
        PyNodeMultipleValuesWithIndexOperand, PyNodeMultipleValuesWithoutIndexOperand,
    },
};

use super::{
    attribute::PyMedRecordAttribute, errors::PyMedRecordError, traits::DeepFrom,
    value::PyMedRecordValue, Lut, PyNodeIndex,
};
use attributes::{
    PyEdgeAttributesTreeOperand, PyEdgeMultipleAttributesWithIndexOperand,
    PyEdgeMultipleAttributesWithoutIndexOperand, PyEdgeSingleAttributeWithIndexOperand,
    PyEdgeSingleAttributeWithoutIndexOperand, PyNodeAttributesTreeOperand,
    PyNodeMultipleAttributesWithIndexOperand, PyNodeMultipleAttributesWithoutIndexOperand,
    PyNodeSingleAttributeWithIndexOperand, PyNodeSingleAttributeWithoutIndexOperand,
};
use edges::{PyEdgeIndexOperand, PyEdgeIndicesOperand};
use medmodels_core::{
    errors::{MedRecordError, MedRecordResult},
    medrecord::{
        querying::{
            attributes::{
                EdgeAttributesTreeOperand, EdgeMultipleAttributesWithIndexOperand,
                EdgeMultipleAttributesWithoutIndexOperand, EdgeSingleAttributeWithIndexOperand,
                EdgeSingleAttributeWithoutIndexOperand, NodeAttributesTreeOperand,
                NodeMultipleAttributesWithIndexOperand, NodeMultipleAttributesWithoutIndexOperand,
                NodeSingleAttributeWithIndexOperand, NodeSingleAttributeWithoutIndexOperand,
            },
            edges::{EdgeIndexOperand, EdgeIndicesOperand},
            nodes::{NodeIndexOperand, NodeIndicesOperand},
            values::{
                EdgeMultipleValuesWithIndexOperand, EdgeMultipleValuesWithoutIndexOperand,
                EdgeSingleValueWithIndexOperand, EdgeSingleValueWithoutIndexOperand,
                NodeMultipleValuesWithIndexOperand, NodeMultipleValuesWithoutIndexOperand,
                NodeSingleValueWithIndexOperand, NodeSingleValueWithoutIndexOperand,
            },
            wrapper::{CardinalityWrapper, Wrapper},
            ReturnOperand,
        },
        MedRecordAttribute,
    },
    MedRecord,
};
use nodes::{PyNodeIndexOperand, PyNodeIndicesOperand};
use pyo3::{
    types::{PyAnyMethods, PyList},
    Bound, FromPyObject, IntoPyObject, IntoPyObjectExt, PyAny, PyErr, PyResult, Python,
};
use std::collections::HashMap;
use values::{
    PyEdgeSingleValueWithIndexOperand, PyEdgeSingleValueWithoutIndexOperand,
    PyNodeSingleValueWithIndexOperand, PyNodeSingleValueWithoutIndexOperand,
};

pub enum PyReturnOperand {
    NodeAttributesTree(PyNodeAttributesTreeOperand),
    EdgeAttributesTree(PyEdgeAttributesTreeOperand),
    NodeMultipleAttributesWithIndex(PyNodeMultipleAttributesWithIndexOperand),
    NodeMultipleAttributesWithoutIndex(PyNodeMultipleAttributesWithoutIndexOperand),
    EdgeMultipleAttributesWithIndex(PyEdgeMultipleAttributesWithIndexOperand),
    EdgeMultipleAttributesWithoutIndex(PyEdgeMultipleAttributesWithoutIndexOperand),
    NodeSingleAttributeWithIndex(PyNodeSingleAttributeWithIndexOperand),
    NodeSingleAttributeWithoutIndex(PyNodeSingleAttributeWithoutIndexOperand),
    EdgeSingleAttributeWithIndex(PyEdgeSingleAttributeWithIndexOperand),
    EdgeSingleAttributeWithoutIndex(PyEdgeSingleAttributeWithoutIndexOperand),
    EdgeIndices(PyEdgeIndicesOperand),
    EdgeIndex(PyEdgeIndexOperand),
    NodeIndices(PyNodeIndicesOperand),
    NodeIndex(PyNodeIndexOperand),
    NodeMultipleValuesWithIndex(PyNodeMultipleValuesWithIndexOperand),
    NodeMultipleValuesWithoutIndex(PyNodeMultipleValuesWithoutIndexOperand),
    EdgeMultipleValuesWithIndex(PyEdgeMultipleValuesWithIndexOperand),
    EdgeMultipleValuesWithoutIndex(PyEdgeMultipleValuesWithoutIndexOperand),
    NodeSingleValueWithIndex(PyNodeSingleValueWithIndexOperand),
    NodeSingleValueWithoutIndex(PyNodeSingleValueWithoutIndexOperand),
    EdgeSingleValueWithIndex(PyEdgeSingleValueWithIndexOperand),
    EdgeSingleValueWithoutIndex(PyEdgeSingleValueWithoutIndexOperand),
    Vector(Vec<Self>),
}

impl<'a> ReturnOperand<'a> for PyReturnOperand {
    type ReturnValue = PyReturnValue<'a>;

    fn evaluate(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        match self {
            PyReturnOperand::NodeAttributesTree(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeAttributesTree),
            PyReturnOperand::EdgeAttributesTree(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeAttributesTree),
            PyReturnOperand::NodeMultipleAttributesWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeMultipleAttributesWithIndex),
            PyReturnOperand::NodeMultipleAttributesWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeMultipleAttributesWithoutIndex),
            PyReturnOperand::EdgeMultipleAttributesWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeMultipleAttributesWithIndex),
            PyReturnOperand::EdgeMultipleAttributesWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeMultipleAttributesWithoutIndex),
            PyReturnOperand::NodeSingleAttributeWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeSingleAttributeWithIndex),
            PyReturnOperand::NodeSingleAttributeWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeSingleAttributeWithoutIndex),
            PyReturnOperand::EdgeSingleAttributeWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeSingleAttributeWithIndex),
            PyReturnOperand::EdgeSingleAttributeWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeSingleAttributeWithoutIndex),
            PyReturnOperand::EdgeIndices(operand) => {
                operand.evaluate(medrecord).map(PyReturnValue::EdgeIndices)
            }
            PyReturnOperand::EdgeIndex(operand) => {
                operand.evaluate(medrecord).map(PyReturnValue::EdgeIndex)
            }
            PyReturnOperand::NodeIndices(operand) => {
                operand.evaluate(medrecord).map(PyReturnValue::NodeIndices)
            }
            PyReturnOperand::NodeIndex(operand) => {
                operand.evaluate(medrecord).map(PyReturnValue::NodeIndex)
            }
            PyReturnOperand::NodeMultipleValuesWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeMultipleValuesWithIndex),
            PyReturnOperand::NodeMultipleValuesWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeMultipleValuesWithoutIndex),
            PyReturnOperand::EdgeMultipleValuesWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeMultipleValuesWithIndex),
            PyReturnOperand::EdgeMultipleValuesWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeMultipleValuesWithoutIndex),
            PyReturnOperand::NodeSingleValueWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeSingleValueWithIndex),
            PyReturnOperand::NodeSingleValueWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeSingleValueWithoutIndex),
            PyReturnOperand::EdgeSingleValueWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeSingleValueWithIndex),
            PyReturnOperand::EdgeSingleValueWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeSingleValueWithoutIndex),
            PyReturnOperand::Vector(operand) => operand
                .iter()
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

    fn convert_py_node_multiple_attributes_operand_with_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeMultipleAttributesWithIndex(
            ob.extract::<PyNodeMultipleAttributesWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }
    fn convert_py_node_multiple_attributes_operand_without_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeMultipleAttributesWithoutIndex(
            ob.extract::<PyNodeMultipleAttributesWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_multiple_attributes_operand_with_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeMultipleAttributesWithIndex(
            ob.extract::<PyEdgeMultipleAttributesWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_multiple_attributes_operand_without_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeMultipleAttributesWithoutIndex(
            ob.extract::<PyEdgeMultipleAttributesWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_attribute_operand_with_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleAttributeWithIndex(
            ob.extract::<PyNodeSingleAttributeWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_attribute_operand_without_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleAttributeWithoutIndex(
            ob.extract::<PyNodeSingleAttributeWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_single_attribute_operand_with_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleAttributeWithIndex(
            ob.extract::<PyEdgeSingleAttributeWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_single_attribute_operand_without_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleAttributeWithoutIndex(
            ob.extract::<PyEdgeSingleAttributeWithoutIndexOperand>()
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

    fn convert_py_node_multiple_values_operand_with_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeMultipleValuesWithIndex(
            ob.extract::<PyNodeMultipleValuesWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_multiple_values_operand_without_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeMultipleValuesWithoutIndex(
            ob.extract::<PyNodeMultipleValuesWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_multiple_values_operand_with_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeMultipleValuesWithIndex(
            ob.extract::<PyEdgeMultipleValuesWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_multiple_values_operand_without_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeMultipleValuesWithoutIndex(
            ob.extract::<PyEdgeMultipleValuesWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_value_operand_with_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleValueWithIndex(
            ob.extract::<PyNodeSingleValueWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_value_operand_without_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleValueWithoutIndex(
            ob.extract::<PyNodeSingleValueWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_single_value_operand_with_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleValueWithIndex(
            ob.extract::<PyEdgeSingleValueWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }
    fn convert_py_edge_single_value_operand_without_index(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleValueWithoutIndex(
            ob.extract::<PyEdgeSingleValueWithoutIndexOperand>()
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
                } else if ob.is_instance_of::<PyNodeMultipleAttributesWithIndexOperand>() {
                    convert_py_node_multiple_attributes_operand_with_index
                } else if ob.is_instance_of::<PyNodeMultipleAttributesWithoutIndexOperand>() {
                    convert_py_node_multiple_attributes_operand_without_index
                } else if ob.is_instance_of::<PyEdgeMultipleAttributesWithIndexOperand>() {
                    convert_py_edge_multiple_attributes_operand_with_index
                } else if ob.is_instance_of::<PyEdgeMultipleAttributesWithoutIndexOperand>() {
                    convert_py_edge_multiple_attributes_operand_without_index
                } else if ob.is_instance_of::<PyNodeSingleAttributeWithIndexOperand>() {
                    convert_py_node_single_attribute_operand_with_index
                } else if ob.is_instance_of::<PyNodeSingleAttributeWithoutIndexOperand>() {
                    convert_py_node_single_attribute_operand_without_index
                } else if ob.is_instance_of::<PyEdgeSingleAttributeWithIndexOperand>() {
                    convert_py_edge_single_attribute_operand_with_index
                } else if ob.is_instance_of::<PyEdgeSingleAttributeWithoutIndexOperand>() {
                    convert_py_edge_single_attribute_operand_without_index
                } else if ob.is_instance_of::<PyEdgeIndicesOperand>() {
                    convert_py_edge_indices_operand
                } else if ob.is_instance_of::<PyEdgeIndexOperand>() {
                    convert_py_edge_index_operand
                } else if ob.is_instance_of::<PyNodeIndicesOperand>() {
                    convert_py_node_indices_operand
                } else if ob.is_instance_of::<PyNodeIndexOperand>() {
                    convert_py_node_index_operand
                } else if ob.is_instance_of::<PyNodeMultipleValuesWithIndexOperand>() {
                    convert_py_node_multiple_values_operand_with_index
                } else if ob.is_instance_of::<PyNodeMultipleValuesWithoutIndexOperand>() {
                    convert_py_node_multiple_values_operand_without_index
                } else if ob.is_instance_of::<PyEdgeMultipleValuesWithIndexOperand>() {
                    convert_py_edge_multiple_values_operand_with_index
                } else if ob.is_instance_of::<PyEdgeMultipleValuesWithoutIndexOperand>() {
                    convert_py_edge_multiple_values_operand_without_index
                } else if ob.is_instance_of::<PyNodeSingleValueWithIndexOperand>() {
                    convert_py_node_single_value_operand_with_index
                } else if ob.is_instance_of::<PyNodeSingleValueWithoutIndexOperand>() {
                    convert_py_node_single_value_operand_without_index
                } else if ob.is_instance_of::<PyEdgeSingleValueWithIndexOperand>() {
                    convert_py_edge_single_value_operand_with_index
                } else if ob.is_instance_of::<PyEdgeSingleValueWithoutIndexOperand>() {
                    convert_py_edge_single_value_operand_without_index
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
    NodeMultipleAttributesWithIndex(
        <Wrapper<NodeMultipleAttributesWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeMultipleAttributesWithoutIndex(
        <Wrapper<NodeMultipleAttributesWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeMultipleAttributesWithIndex(
        <Wrapper<EdgeMultipleAttributesWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeMultipleAttributesWithoutIndex(
        <Wrapper<EdgeMultipleAttributesWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeSingleAttributeWithIndex(
        <Wrapper<NodeSingleAttributeWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeSingleAttributeWithoutIndex(
        <Wrapper<NodeSingleAttributeWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeSingleAttributeWithIndex(
        <Wrapper<EdgeSingleAttributeWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeSingleAttributeWithoutIndex(
        <Wrapper<EdgeSingleAttributeWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeIndices(<Wrapper<EdgeIndicesOperand> as ReturnOperand<'a>>::ReturnValue),
    EdgeIndex(<Wrapper<EdgeIndexOperand> as ReturnOperand<'a>>::ReturnValue),
    NodeIndices(<Wrapper<NodeIndicesOperand> as ReturnOperand<'a>>::ReturnValue),
    NodeIndex(<Wrapper<NodeIndexOperand> as ReturnOperand<'a>>::ReturnValue),
    NodeMultipleValuesWithIndex(
        <Wrapper<NodeMultipleValuesWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeMultipleValuesWithoutIndex(
        <Wrapper<NodeMultipleValuesWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeMultipleValuesWithIndex(
        <Wrapper<EdgeMultipleValuesWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeMultipleValuesWithoutIndex(
        <Wrapper<EdgeMultipleValuesWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeSingleValueWithIndex(
        <Wrapper<NodeSingleValueWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeSingleValueWithoutIndex(
        <Wrapper<NodeSingleValueWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeSingleValueWithIndex(
        <Wrapper<EdgeSingleValueWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeSingleValueWithoutIndex(
        <Wrapper<EdgeSingleValueWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
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
            PyReturnValue::NodeMultipleAttributesWithIndex(iterator) => iterator
                .map(|item| {
                    (
                        PyNodeIndex::from(item.0.clone()),
                        PyMedRecordAttribute::from(item.1),
                    )
                })
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeMultipleAttributesWithoutIndex(iterator) => iterator
                .map(PyMedRecordAttribute::from)
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeMultipleAttributesWithIndex(iterator) => iterator
                .map(|item| (item.0, PyMedRecordAttribute::from(item.1)))
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeMultipleAttributesWithoutIndex(iterator) => iterator
                .map(PyMedRecordAttribute::from)
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeSingleAttributeWithIndex(attribute) => attribute
                .map(|(index, attribute)| {
                    (
                        PyMedRecordAttribute::from(index.clone()),
                        PyMedRecordAttribute::from(attribute),
                    )
                })
                .into_bound_py_any(py),
            PyReturnValue::NodeSingleAttributeWithoutIndex(attribute) => attribute
                .map(PyMedRecordAttribute::from)
                .into_bound_py_any(py),
            PyReturnValue::EdgeSingleAttributeWithIndex(attribute) => attribute
                .map(|(index, attribute)| (index, PyMedRecordAttribute::from(attribute)))
                .into_bound_py_any(py),
            PyReturnValue::EdgeSingleAttributeWithoutIndex(attribute) => attribute
                .map(PyMedRecordAttribute::from)
                .into_bound_py_any(py),
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
            PyReturnValue::NodeMultipleValuesWithIndex(iterator) => iterator
                .map(|item| {
                    (
                        PyNodeIndex::from(item.0.clone()),
                        PyMedRecordValue::from(item.1),
                    )
                })
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeMultipleValuesWithoutIndex(iterator) => iterator
                .map(PyMedRecordValue::from)
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeMultipleValuesWithIndex(iterator) => iterator
                .map(|item| (item.0, PyMedRecordValue::from(item.1)))
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeMultipleValuesWithoutIndex(iterator) => iterator
                .map(PyMedRecordValue::from)
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeSingleValueWithIndex(value) => value
                .map(|(index, value)| {
                    (
                        PyMedRecordAttribute::from(index.clone()),
                        PyMedRecordValue::from(value),
                    )
                })
                .into_bound_py_any(py),
            PyReturnValue::NodeSingleValueWithoutIndex(value) => {
                value.map(PyMedRecordValue::from).into_bound_py_any(py)
            }
            PyReturnValue::EdgeSingleValueWithIndex(value) => value
                .map(|(index, value)| (index, PyMedRecordValue::from(value)))
                .into_bound_py_any(py),
            PyReturnValue::EdgeSingleValueWithoutIndex(value) => {
                value.map(PyMedRecordValue::from).into_bound_py_any(py)
            }
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
