pub mod attributes;
pub mod edges;
pub mod nodes;
pub mod values;

use super::{
    attribute::PyMedRecordAttribute, errors::PyMedRecordError, traits::DeepFrom,
    value::PyMedRecordValue, Lut, PyNodeIndex,
};
use crate::{
    gil_hash_map::GILHashMap,
    medrecord::querying::{
        attributes::{
            PyEdgeSingleAttributeWithoutIndexGroupOperand,
            PyNodeSingleAttributeWithoutIndexGroupOperand,
        },
        edges::{PyEdgeIndexGroupOperand, PyEdgeIndicesGroupOperand},
        nodes::{PyNodeIndexGroupOperand, PyNodeIndicesGroupOperand},
        values::{
            PyEdgeSingleValueWithoutIndexGroupOperand, PyNodeSingleValueWithoutIndexGroupOperand,
        },
    },
};
use attributes::{
    PyEdgeAttributesTreeGroupOperand, PyEdgeAttributesTreeOperand,
    PyEdgeMultipleAttributesWithIndexGroupOperand, PyEdgeMultipleAttributesWithIndexOperand,
    PyEdgeMultipleAttributesWithoutIndexOperand, PyEdgeSingleAttributeWithIndexGroupOperand,
    PyEdgeSingleAttributeWithIndexOperand, PyEdgeSingleAttributeWithoutIndexOperand,
    PyNodeAttributesTreeGroupOperand, PyNodeAttributesTreeOperand,
    PyNodeMultipleAttributesWithIndexGroupOperand, PyNodeMultipleAttributesWithIndexOperand,
    PyNodeMultipleAttributesWithoutIndexOperand, PyNodeSingleAttributeWithIndexGroupOperand,
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
            group_by::{GroupKey, GroupOperand},
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
    PyEdgeMultipleValuesWithIndexGroupOperand, PyEdgeMultipleValuesWithIndexOperand,
    PyEdgeMultipleValuesWithoutIndexOperand, PyEdgeSingleValueWithIndexGroupOperand,
    PyEdgeSingleValueWithIndexOperand, PyEdgeSingleValueWithoutIndexOperand,
    PyNodeMultipleValuesWithIndexGroupOperand, PyNodeMultipleValuesWithIndexOperand,
    PyNodeMultipleValuesWithoutIndexOperand, PyNodeSingleValueWithIndexGroupOperand,
    PyNodeSingleValueWithIndexOperand, PyNodeSingleValueWithoutIndexOperand,
};

#[derive(Debug, Clone)]
pub enum PyGroupKey {
    NodeIndex(PyNodeIndex),
    Value(PyMedRecordValue),
    OptionalValue(Option<PyMedRecordValue>),
    TupleKey((Box<PyGroupKey>, Box<PyGroupKey>)),
}

impl From<GroupKey<'_>> for PyGroupKey {
    fn from(key: GroupKey<'_>) -> Self {
        match key {
            GroupKey::NodeIndex(index) => PyGroupKey::NodeIndex(PyNodeIndex::from(index.clone())),
            GroupKey::Value(value) => PyGroupKey::Value(PyMedRecordValue::from(value.clone())),
            GroupKey::OptionalValue(value) => {
                PyGroupKey::OptionalValue(value.cloned().map(PyMedRecordValue::from))
            }
            GroupKey::TupleKey((left, right)) => PyGroupKey::TupleKey((
                Box::new(PyGroupKey::from(*left)),
                Box::new(PyGroupKey::from(*right)),
            )),
        }
    }
}

impl<'py> IntoPyObject<'py> for PyGroupKey {
    type Target = pyo3::PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            PyGroupKey::NodeIndex(index) => index.into_pyobject(py),
            PyGroupKey::Value(value) => value.into_pyobject(py),
            PyGroupKey::OptionalValue(value) => value.into_pyobject(py),
            PyGroupKey::TupleKey((left, right)) => {
                let left = left.into_pyobject(py)?;
                let right = right.into_pyobject(py)?;
                (left, right).into_bound_py_any(py)
            }
        }
    }
}

pub enum PyReturnOperand {
    NodeAttributesTree(PyNodeAttributesTreeOperand),
    NodeAttributesTreeGroup(PyNodeAttributesTreeGroupOperand),
    EdgeAttributesTree(PyEdgeAttributesTreeOperand),
    EdgeAttributesTreeGroup(PyEdgeAttributesTreeGroupOperand),
    NodeMultipleAttributesWithIndex(PyNodeMultipleAttributesWithIndexOperand),
    NodeMultipleAttributesWithIndexGroup(PyNodeMultipleAttributesWithIndexGroupOperand),
    NodeMultipleAttributesWithoutIndex(PyNodeMultipleAttributesWithoutIndexOperand),
    EdgeMultipleAttributesWithIndex(PyEdgeMultipleAttributesWithIndexOperand),
    EdgeMultipleAttributesWithIndexGroup(PyEdgeMultipleAttributesWithIndexGroupOperand),
    EdgeMultipleAttributesWithoutIndex(PyEdgeMultipleAttributesWithoutIndexOperand),
    NodeSingleAttributeWithIndex(PyNodeSingleAttributeWithIndexOperand),
    NodeSingleAttributeWithIndexGroup(PyNodeSingleAttributeWithIndexGroupOperand),
    NodeSingleAttributeWithoutIndex(PyNodeSingleAttributeWithoutIndexOperand),
    NodeSingleAttributeWithoutIndexGroup(PyNodeSingleAttributeWithoutIndexGroupOperand),
    EdgeSingleAttributeWithIndex(PyEdgeSingleAttributeWithIndexOperand),
    EdgeSingleAttributeWithIndexGroup(PyEdgeSingleAttributeWithIndexGroupOperand),
    EdgeSingleAttributeWithoutIndex(PyEdgeSingleAttributeWithoutIndexOperand),
    EdgeSingleAttributeWithoutIndexGroup(PyEdgeSingleAttributeWithoutIndexGroupOperand),
    EdgeIndices(PyEdgeIndicesOperand),
    EdgeIndicesGroup(PyEdgeIndicesGroupOperand),
    EdgeIndex(PyEdgeIndexOperand),
    EdgeIndexGroup(PyEdgeIndexGroupOperand),
    NodeIndices(PyNodeIndicesOperand),
    NodeIndicesGroup(PyNodeIndicesGroupOperand),
    NodeIndex(PyNodeIndexOperand),
    NodeIndexGroup(PyNodeIndexGroupOperand),
    NodeMultipleValuesWithIndex(PyNodeMultipleValuesWithIndexOperand),
    NodeMultipleValuesWithIndexGroup(PyNodeMultipleValuesWithIndexGroupOperand),
    NodeMultipleValuesWithoutIndex(PyNodeMultipleValuesWithoutIndexOperand),
    EdgeMultipleValuesWithIndex(PyEdgeMultipleValuesWithIndexOperand),
    EdgeMultipleValuesWithIndexGroup(PyEdgeMultipleValuesWithIndexGroupOperand),
    EdgeMultipleValuesWithoutIndex(PyEdgeMultipleValuesWithoutIndexOperand),
    NodeSingleValueWithIndex(PyNodeSingleValueWithIndexOperand),
    NodeSingleValueWithIndexGroup(PyNodeSingleValueWithIndexGroupOperand),
    NodeSingleValueWithoutIndex(PyNodeSingleValueWithoutIndexOperand),
    NodeSingleValueWithoutIndexGroup(PyNodeSingleValueWithoutIndexGroupOperand),
    EdgeSingleValueWithIndex(PyEdgeSingleValueWithIndexOperand),
    EdgeSingleValueWithIndexGroup(PyEdgeSingleValueWithIndexGroupOperand),
    EdgeSingleValueWithoutIndex(PyEdgeSingleValueWithoutIndexOperand),
    EdgeSingleValueWithoutIndexGroup(PyEdgeSingleValueWithoutIndexGroupOperand),
    Vector(Vec<Self>),
}

impl<'a> ReturnOperand<'a> for PyReturnOperand {
    type ReturnValue = PyReturnValue<'a>;

    fn evaluate(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        match self {
            PyReturnOperand::NodeAttributesTree(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeAttributesTree),
            PyReturnOperand::NodeAttributesTreeGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeAttributesTreeGroup),
            PyReturnOperand::EdgeAttributesTree(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeAttributesTree),
            PyReturnOperand::EdgeAttributesTreeGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeAttributesTreeGroup),
            PyReturnOperand::NodeMultipleAttributesWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeMultipleAttributesWithIndex),
            PyReturnOperand::NodeMultipleAttributesWithIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeMultipleAttributesWithIndexGroup),
            PyReturnOperand::NodeMultipleAttributesWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeMultipleAttributesWithoutIndex),
            PyReturnOperand::EdgeMultipleAttributesWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeMultipleAttributesWithIndex),
            PyReturnOperand::EdgeMultipleAttributesWithIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeMultipleAttributesWithIndexGroup),
            PyReturnOperand::EdgeMultipleAttributesWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeMultipleAttributesWithoutIndex),
            PyReturnOperand::NodeSingleAttributeWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeSingleAttributeWithIndex),
            PyReturnOperand::NodeSingleAttributeWithIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeSingleAttributeWithIndexGroup),
            PyReturnOperand::NodeSingleAttributeWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeSingleAttributeWithoutIndex),
            PyReturnOperand::NodeSingleAttributeWithoutIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeSingleAttributeWithoutIndexGroup),
            PyReturnOperand::EdgeSingleAttributeWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeSingleAttributeWithIndex),
            PyReturnOperand::EdgeSingleAttributeWithIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeSingleAttributeWithIndexGroup),
            PyReturnOperand::EdgeSingleAttributeWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeSingleAttributeWithoutIndex),
            PyReturnOperand::EdgeSingleAttributeWithoutIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeSingleAttributeWithoutIndexGroup),
            PyReturnOperand::EdgeIndices(operand) => {
                operand.evaluate(medrecord).map(PyReturnValue::EdgeIndices)
            }
            PyReturnOperand::EdgeIndicesGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeIndicesGroup),
            PyReturnOperand::EdgeIndex(operand) => {
                operand.evaluate(medrecord).map(PyReturnValue::EdgeIndex)
            }
            PyReturnOperand::EdgeIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeIndexGroup),
            PyReturnOperand::NodeIndices(operand) => {
                operand.evaluate(medrecord).map(PyReturnValue::NodeIndices)
            }
            PyReturnOperand::NodeIndicesGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeIndicesGroup),
            PyReturnOperand::NodeIndex(operand) => {
                operand.evaluate(medrecord).map(PyReturnValue::NodeIndex)
            }
            PyReturnOperand::NodeIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeIndexGroup),
            PyReturnOperand::NodeMultipleValuesWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeMultipleValuesWithIndex),
            PyReturnOperand::NodeMultipleValuesWithIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeMultipleValuesWithIndexGroup),
            PyReturnOperand::NodeMultipleValuesWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeMultipleValuesWithoutIndex),
            PyReturnOperand::EdgeMultipleValuesWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeMultipleValuesWithIndex),
            PyReturnOperand::EdgeMultipleValuesWithIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeMultipleValuesWithIndexGroup),
            PyReturnOperand::EdgeMultipleValuesWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeMultipleValuesWithoutIndex),
            PyReturnOperand::NodeSingleValueWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeSingleValueWithIndex),
            PyReturnOperand::NodeSingleValueWithIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeSingleValueWithIndexGroup),
            PyReturnOperand::NodeSingleValueWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeSingleValueWithoutIndex),
            PyReturnOperand::NodeSingleValueWithoutIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::NodeSingleValueWithoutIndexGroup),
            PyReturnOperand::EdgeSingleValueWithIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeSingleValueWithIndex),
            PyReturnOperand::EdgeSingleValueWithIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeSingleValueWithIndexGroup),
            PyReturnOperand::EdgeSingleValueWithoutIndex(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeSingleValueWithoutIndex),
            PyReturnOperand::EdgeSingleValueWithoutIndexGroup(operand) => operand
                .evaluate(medrecord)
                .map(PyReturnValue::EdgeSingleValueWithoutIndexGroup),
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
    fn convert_py_node_attributes_tree_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeAttributesTreeGroup(
            ob.extract::<PyNodeAttributesTreeGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_attributes_tree_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeAttributesTree(
            ob.extract::<PyEdgeAttributesTreeOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_attributes_tree_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeAttributesTreeGroup(
            ob.extract::<PyEdgeAttributesTreeGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_multiple_attributes_with_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeMultipleAttributesWithIndex(
            ob.extract::<PyNodeMultipleAttributesWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_multiple_attributes_with_index_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeMultipleAttributesWithIndexGroup(
            ob.extract::<PyNodeMultipleAttributesWithIndexGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_multiple_attributes_without_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeMultipleAttributesWithoutIndex(
            ob.extract::<PyNodeMultipleAttributesWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_multiple_attributes_with_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeMultipleAttributesWithIndex(
            ob.extract::<PyEdgeMultipleAttributesWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_multiple_attributes_with_index_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeMultipleAttributesWithIndexGroup(
            ob.extract::<PyEdgeMultipleAttributesWithIndexGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_multiple_attributes_without_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeMultipleAttributesWithoutIndex(
            ob.extract::<PyEdgeMultipleAttributesWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_attribute_with_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleAttributeWithIndex(
            ob.extract::<PyNodeSingleAttributeWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_attribute_with_index_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleAttributeWithIndexGroup(
            ob.extract::<PyNodeSingleAttributeWithIndexGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_attribute_without_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleAttributeWithoutIndex(
            ob.extract::<PyNodeSingleAttributeWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_attribute_without_index_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleAttributeWithoutIndexGroup(
            ob.extract::<PyNodeSingleAttributeWithoutIndexGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_single_attribute_with_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleAttributeWithIndex(
            ob.extract::<PyEdgeSingleAttributeWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_single_attribute_with_index_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleAttributeWithIndexGroup(
            ob.extract::<PyEdgeSingleAttributeWithIndexGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_single_attribute_without_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleAttributeWithoutIndex(
            ob.extract::<PyEdgeSingleAttributeWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_single_attribute_without_index_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleAttributeWithoutIndexGroup(
            ob.extract::<PyEdgeSingleAttributeWithoutIndexGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_indices_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeIndices(
            ob.extract::<PyEdgeIndicesOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_indices_group_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeIndicesGroup(
            ob.extract::<PyEdgeIndicesGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_index_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeIndex(
            ob.extract::<PyEdgeIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_index_group_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeIndexGroup(
            ob.extract::<PyEdgeIndexGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_indices_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeIndices(
            ob.extract::<PyNodeIndicesOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_indices_group_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeIndicesGroup(
            ob.extract::<PyNodeIndicesGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_index_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeIndex(
            ob.extract::<PyNodeIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_index_group_operand(ob: &Bound<'_, PyAny>) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeIndexGroup(
            ob.extract::<PyNodeIndexGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_multiple_values_with_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeMultipleValuesWithIndex(
            ob.extract::<PyNodeMultipleValuesWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_multiple_values_with_index_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeMultipleValuesWithIndexGroup(
            ob.extract::<PyNodeMultipleValuesWithIndexGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_multiple_values_without_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeMultipleValuesWithoutIndex(
            ob.extract::<PyNodeMultipleValuesWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_multiple_values_with_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeMultipleValuesWithIndex(
            ob.extract::<PyEdgeMultipleValuesWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_multiple_values_with_index_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeMultipleValuesWithIndexGroup(
            ob.extract::<PyEdgeMultipleValuesWithIndexGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_multiple_values_without_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeMultipleValuesWithoutIndex(
            ob.extract::<PyEdgeMultipleValuesWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_value_with_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleValueWithIndex(
            ob.extract::<PyNodeSingleValueWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_value_with_index_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleValueWithIndexGroup(
            ob.extract::<PyNodeSingleValueWithIndexGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_value_without_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleValueWithoutIndex(
            ob.extract::<PyNodeSingleValueWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_node_single_value_without_index_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::NodeSingleValueWithoutIndexGroup(
            ob.extract::<PyNodeSingleValueWithoutIndexGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_single_value_with_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleValueWithIndex(
            ob.extract::<PyEdgeSingleValueWithIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_single_value_with_index_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleValueWithIndexGroup(
            ob.extract::<PyEdgeSingleValueWithIndexGroupOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_single_value_without_index_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleValueWithoutIndex(
            ob.extract::<PyEdgeSingleValueWithoutIndexOperand>()
                .expect("Extraction must succeed"),
        ))
    }

    fn convert_py_edge_single_value_without_index_group_operand(
        ob: &Bound<'_, PyAny>,
    ) -> PyResult<PyReturnOperand> {
        Ok(PyReturnOperand::EdgeSingleValueWithoutIndexGroup(
            ob.extract::<PyEdgeSingleValueWithoutIndexGroupOperand>()
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
                "Failed to convert {ob} into query ReturnOperand",
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
                } else if ob.is_instance_of::<PyNodeAttributesTreeGroupOperand>() {
                    convert_py_node_attributes_tree_group_operand
                } else if ob.is_instance_of::<PyEdgeAttributesTreeOperand>() {
                    convert_py_edge_attributes_tree_operand
                } else if ob.is_instance_of::<PyEdgeAttributesTreeGroupOperand>() {
                    convert_py_edge_attributes_tree_group_operand
                } else if ob.is_instance_of::<PyNodeMultipleAttributesWithIndexOperand>() {
                    convert_py_node_multiple_attributes_with_index_operand
                } else if ob.is_instance_of::<PyNodeMultipleAttributesWithIndexGroupOperand>() {
                    convert_py_node_multiple_attributes_with_index_group_operand
                } else if ob.is_instance_of::<PyNodeMultipleAttributesWithoutIndexOperand>() {
                    convert_py_node_multiple_attributes_without_index_operand
                } else if ob.is_instance_of::<PyEdgeMultipleAttributesWithIndexOperand>() {
                    convert_py_edge_multiple_attributes_with_index_operand
                } else if ob.is_instance_of::<PyEdgeMultipleAttributesWithIndexGroupOperand>() {
                    convert_py_edge_multiple_attributes_with_index_group_operand
                } else if ob.is_instance_of::<PyEdgeMultipleAttributesWithoutIndexOperand>() {
                    convert_py_edge_multiple_attributes_without_index_operand
                } else if ob.is_instance_of::<PyNodeSingleAttributeWithIndexOperand>() {
                    convert_py_node_single_attribute_with_index_operand
                } else if ob.is_instance_of::<PyNodeSingleAttributeWithIndexGroupOperand>() {
                    convert_py_node_single_attribute_with_index_group_operand
                } else if ob.is_instance_of::<PyNodeSingleAttributeWithoutIndexOperand>() {
                    convert_py_node_single_attribute_without_index_operand
                } else if ob.is_instance_of::<PyNodeSingleAttributeWithoutIndexGroupOperand>() {
                    convert_py_node_single_attribute_without_index_group_operand
                } else if ob.is_instance_of::<PyEdgeSingleAttributeWithIndexOperand>() {
                    convert_py_edge_single_attribute_with_index_operand
                } else if ob.is_instance_of::<PyEdgeSingleAttributeWithIndexGroupOperand>() {
                    convert_py_edge_single_attribute_with_index_group_operand
                } else if ob.is_instance_of::<PyEdgeSingleAttributeWithoutIndexOperand>() {
                    convert_py_edge_single_attribute_without_index_operand
                } else if ob.is_instance_of::<PyEdgeSingleAttributeWithoutIndexGroupOperand>() {
                    convert_py_edge_single_attribute_without_index_group_operand
                } else if ob.is_instance_of::<PyEdgeIndicesOperand>() {
                    convert_py_edge_indices_operand
                } else if ob.is_instance_of::<PyEdgeIndicesGroupOperand>() {
                    convert_py_edge_indices_group_operand
                } else if ob.is_instance_of::<PyEdgeIndexOperand>() {
                    convert_py_edge_index_operand
                } else if ob.is_instance_of::<PyEdgeIndexGroupOperand>() {
                    convert_py_edge_index_group_operand
                } else if ob.is_instance_of::<PyNodeIndicesOperand>() {
                    convert_py_node_indices_operand
                } else if ob.is_instance_of::<PyNodeIndicesGroupOperand>() {
                    convert_py_node_indices_group_operand
                } else if ob.is_instance_of::<PyNodeIndexOperand>() {
                    convert_py_node_index_operand
                } else if ob.is_instance_of::<PyNodeIndexGroupOperand>() {
                    convert_py_node_index_group_operand
                } else if ob.is_instance_of::<PyNodeMultipleValuesWithIndexOperand>() {
                    convert_py_node_multiple_values_with_index_operand
                } else if ob.is_instance_of::<PyNodeMultipleValuesWithIndexGroupOperand>() {
                    convert_py_node_multiple_values_with_index_group_operand
                } else if ob.is_instance_of::<PyNodeMultipleValuesWithoutIndexOperand>() {
                    convert_py_node_multiple_values_without_index_operand
                } else if ob.is_instance_of::<PyEdgeMultipleValuesWithIndexOperand>() {
                    convert_py_edge_multiple_values_with_index_operand
                } else if ob.is_instance_of::<PyEdgeMultipleValuesWithIndexGroupOperand>() {
                    convert_py_edge_multiple_values_with_index_group_operand
                } else if ob.is_instance_of::<PyEdgeMultipleValuesWithoutIndexOperand>() {
                    convert_py_edge_multiple_values_without_index_operand
                } else if ob.is_instance_of::<PyNodeSingleValueWithIndexOperand>() {
                    convert_py_node_single_value_with_index_operand
                } else if ob.is_instance_of::<PyNodeSingleValueWithIndexGroupOperand>() {
                    convert_py_node_single_value_with_index_group_operand
                } else if ob.is_instance_of::<PyNodeSingleValueWithoutIndexOperand>() {
                    convert_py_node_single_value_without_index_operand
                } else if ob.is_instance_of::<PyNodeSingleValueWithoutIndexGroupOperand>() {
                    convert_py_node_single_value_without_index_group_operand
                } else if ob.is_instance_of::<PyEdgeSingleValueWithIndexOperand>() {
                    convert_py_edge_single_value_with_index_operand
                } else if ob.is_instance_of::<PyEdgeSingleValueWithIndexGroupOperand>() {
                    convert_py_edge_single_value_with_index_group_operand
                } else if ob.is_instance_of::<PyEdgeSingleValueWithoutIndexOperand>() {
                    convert_py_edge_single_value_without_index_operand
                } else if ob.is_instance_of::<PyEdgeSingleValueWithoutIndexGroupOperand>() {
                    convert_py_edge_single_value_without_index_group_operand
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
    NodeAttributesTreeGroup(<Wrapper<GroupOperand<NodeAttributesTreeOperand>> as ReturnOperand<'a>>::ReturnValue),
    EdgeAttributesTree(<Wrapper<EdgeAttributesTreeOperand> as ReturnOperand<'a>>::ReturnValue),
    EdgeAttributesTreeGroup(<Wrapper<GroupOperand<EdgeAttributesTreeOperand>> as ReturnOperand<'a>>::ReturnValue),
    NodeMultipleAttributesWithIndex(
        <Wrapper<NodeMultipleAttributesWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeMultipleAttributesWithIndexGroup(
        <Wrapper<GroupOperand<NodeMultipleAttributesWithIndexOperand>> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeMultipleAttributesWithoutIndex(
        <Wrapper<NodeMultipleAttributesWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeMultipleAttributesWithIndex(
        <Wrapper<EdgeMultipleAttributesWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeMultipleAttributesWithIndexGroup(
        <Wrapper<GroupOperand<EdgeMultipleAttributesWithIndexOperand>> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeMultipleAttributesWithoutIndex(
        <Wrapper<EdgeMultipleAttributesWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeSingleAttributeWithIndex(
        <Wrapper<NodeSingleAttributeWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeSingleAttributeWithIndexGroup(
        <Wrapper<GroupOperand<NodeSingleAttributeWithIndexOperand>> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeSingleAttributeWithoutIndex(
        <Wrapper<NodeSingleAttributeWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeSingleAttributeWithoutIndexGroup(
        <Wrapper<GroupOperand<NodeSingleAttributeWithoutIndexOperand>> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeSingleAttributeWithIndex(
        <Wrapper<EdgeSingleAttributeWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeSingleAttributeWithIndexGroup(
        <Wrapper<GroupOperand<EdgeSingleAttributeWithIndexOperand>> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeSingleAttributeWithoutIndex(
        <Wrapper<EdgeSingleAttributeWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeSingleAttributeWithoutIndexGroup(
        <Wrapper<GroupOperand<EdgeSingleAttributeWithoutIndexOperand>> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeIndices(<Wrapper<EdgeIndicesOperand> as ReturnOperand<'a>>::ReturnValue),
    EdgeIndicesGroup(<Wrapper<GroupOperand<EdgeIndicesOperand>> as ReturnOperand<'a>>::ReturnValue),
    EdgeIndex(<Wrapper<EdgeIndexOperand> as ReturnOperand<'a>>::ReturnValue),
    EdgeIndexGroup(<Wrapper<GroupOperand<EdgeIndexOperand>> as ReturnOperand<'a>>::ReturnValue),
    NodeIndices(<Wrapper<NodeIndicesOperand> as ReturnOperand<'a>>::ReturnValue),
    NodeIndicesGroup(<Wrapper<GroupOperand<NodeIndicesOperand>> as ReturnOperand<'a>>::ReturnValue),
    NodeIndex(<Wrapper<NodeIndexOperand> as ReturnOperand<'a>>::ReturnValue),
    NodeIndexGroup(<Wrapper<GroupOperand<NodeIndexOperand>> as ReturnOperand<'a>>::ReturnValue),
    NodeMultipleValuesWithIndex(
        <Wrapper<NodeMultipleValuesWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeMultipleValuesWithIndexGroup(
        <Wrapper<GroupOperand<NodeMultipleValuesWithIndexOperand>> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeMultipleValuesWithoutIndex(
        <Wrapper<NodeMultipleValuesWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeMultipleValuesWithIndex(
        <Wrapper<EdgeMultipleValuesWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeMultipleValuesWithIndexGroup(
        <Wrapper<GroupOperand<EdgeMultipleValuesWithIndexOperand>> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeMultipleValuesWithoutIndex(
        <Wrapper<EdgeMultipleValuesWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeSingleValueWithIndex(
        <Wrapper<NodeSingleValueWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeSingleValueWithIndexGroup(
        <Wrapper<GroupOperand<NodeSingleValueWithIndexOperand>> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeSingleValueWithoutIndex(
        <Wrapper<NodeSingleValueWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    NodeSingleValueWithoutIndexGroup(
        <Wrapper<GroupOperand<NodeSingleValueWithoutIndexOperand>> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeSingleValueWithIndex(
        <Wrapper<EdgeSingleValueWithIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeSingleValueWithIndexGroup(
        <Wrapper<GroupOperand<EdgeSingleValueWithIndexOperand>> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeSingleValueWithoutIndex(
        <Wrapper<EdgeSingleValueWithoutIndexOperand> as ReturnOperand<'a>>::ReturnValue,
    ),
    EdgeSingleValueWithoutIndexGroup(
        <Wrapper<GroupOperand<EdgeSingleValueWithoutIndexOperand>> as ReturnOperand<'a>>::ReturnValue,
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
            PyReturnValue::NodeAttributesTreeGroup(iterator) => iterator
                .map(|(key, items)| {
                    (
                        PyGroupKey::from(key),
                        items
                            .map(|item| {
                                (
                                    PyNodeIndex::from(item.0.clone()),
                                    Vec::<PyMedRecordAttribute>::deep_from(item.1),
                                )
                            })
                            .collect::<HashMap<_, _>>(),
                    )
                })
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeAttributesTree(iterator) => iterator
                .map(|item| (item.0, Vec::<PyMedRecordAttribute>::deep_from(item.1)))
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeAttributesTreeGroup(iterator) => iterator
                .map(|(key, items)| {
                    (
                        PyGroupKey::from(key),
                        items
                            .map(|item| (item.0, Vec::<PyMedRecordAttribute>::deep_from(item.1)))
                            .collect::<HashMap<_, _>>(),
                    )
                })
                .collect::<Vec<_>>()
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
            PyReturnValue::NodeMultipleAttributesWithIndexGroup(iterator) => iterator
                .map(|(key, items)| {
                    (
                        PyGroupKey::from(key),
                        items
                            .map(|item| {
                                (
                                    PyNodeIndex::from(item.0.clone()),
                                    PyMedRecordAttribute::from(item.1),
                                )
                            })
                            .collect::<HashMap<_, _>>(),
                    )
                })
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeMultipleAttributesWithoutIndex(iterator) => iterator
                .map(PyMedRecordAttribute::from)
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeMultipleAttributesWithIndex(iterator) => iterator
                .map(|item| (item.0, PyMedRecordAttribute::from(item.1)))
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeMultipleAttributesWithIndexGroup(iterator) => iterator
                .map(|(key, items)| {
                    (
                        PyGroupKey::from(key),
                        items
                            .map(|item| (item.0, PyMedRecordAttribute::from(item.1)))
                            .collect::<HashMap<_, _>>(),
                    )
                })
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeMultipleAttributesWithoutIndex(iterator) => iterator
                .map(PyMedRecordAttribute::from)
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeSingleAttributeWithIndex(attribute) => attribute
                .map(|item| {
                    (
                        PyMedRecordAttribute::from(item.0.clone()),
                        PyMedRecordAttribute::from(item.1),
                    )
                })
                .into_bound_py_any(py),
            PyReturnValue::NodeSingleAttributeWithIndexGroup(attribute) => attribute
                .map(|(key, items)| {
                    (
                        PyGroupKey::from(key),
                        items.map(|item| {
                            (
                                PyMedRecordAttribute::from(item.0.clone()),
                                PyMedRecordAttribute::from(item.1),
                            )
                        }),
                    )
                })
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeSingleAttributeWithoutIndex(attribute) => attribute
                .map(PyMedRecordAttribute::from)
                .into_bound_py_any(py),
            PyReturnValue::NodeSingleAttributeWithoutIndexGroup(attribute) => attribute
                .map(|(key, item)| (PyGroupKey::from(key), item.map(PyMedRecordAttribute::from)))
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeSingleAttributeWithIndex(attribute) => attribute
                .map(|item| (item.0, PyMedRecordAttribute::from(item.1)))
                .into_bound_py_any(py),
            PyReturnValue::EdgeSingleAttributeWithIndexGroup(attribute) => attribute
                .map(|(key, items)| {
                    (
                        PyGroupKey::from(key),
                        items.map(|item| (item.0, PyMedRecordAttribute::from(item.1))),
                    )
                })
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeSingleAttributeWithoutIndex(attribute) => attribute
                .map(PyMedRecordAttribute::from)
                .into_bound_py_any(py),
            PyReturnValue::EdgeSingleAttributeWithoutIndexGroup(attribute) => attribute
                .map(|(key, item)| (PyGroupKey::from(key), item.map(PyMedRecordAttribute::from)))
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeIndices(iterator) => {
                iterator.collect::<Vec<_>>().into_bound_py_any(py)
            }
            PyReturnValue::EdgeIndicesGroup(iterator) => iterator
                .map(|(key, iterator)| (PyGroupKey::from(key), iterator.collect::<Vec<_>>()))
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeIndex(index) => index.into_bound_py_any(py),
            PyReturnValue::EdgeIndexGroup(index) => index
                .map(|(key, index)| (PyGroupKey::from(key), index))
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeIndices(iterator) => iterator
                .map(PyMedRecordAttribute::from)
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeIndicesGroup(iterator) => iterator
                .map(|(key, iterator)| {
                    (
                        PyGroupKey::from(key),
                        iterator.map(PyMedRecordAttribute::from).collect::<Vec<_>>(),
                    )
                })
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeIndex(index) => {
                Option::<PyNodeIndex>::deep_from(index).into_bound_py_any(py)
            }
            PyReturnValue::NodeIndexGroup(index) => index
                .map(|(key, index)| {
                    (
                        PyGroupKey::from(key),
                        Option::<PyNodeIndex>::deep_from(index),
                    )
                })
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeMultipleValuesWithIndex(iterator) => iterator
                .map(|item| {
                    (
                        PyNodeIndex::from(item.0.clone()),
                        PyMedRecordValue::from(item.1),
                    )
                })
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeMultipleValuesWithIndexGroup(iterator) => iterator
                .map(|(key, items)| {
                    (
                        PyGroupKey::from(key),
                        items
                            .map(|item| {
                                (
                                    PyNodeIndex::from(item.0.clone()),
                                    PyMedRecordValue::from(item.1),
                                )
                            })
                            .collect::<HashMap<_, _>>(),
                    )
                })
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeMultipleValuesWithoutIndex(iterator) => iterator
                .map(PyMedRecordValue::from)
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeMultipleValuesWithIndex(iterator) => iterator
                .map(|item| (item.0, PyMedRecordValue::from(item.1)))
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeMultipleValuesWithIndexGroup(iterator) => iterator
                .map(|(key, items)| {
                    (
                        PyGroupKey::from(key),
                        items
                            .map(|item| (item.0, PyMedRecordValue::from(item.1)))
                            .collect::<HashMap<_, _>>(),
                    )
                })
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeMultipleValuesWithoutIndex(iterator) => iterator
                .map(PyMedRecordValue::from)
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeSingleValueWithIndex(value) => value
                .map(|item| {
                    (
                        PyMedRecordAttribute::from(item.0.clone()),
                        PyMedRecordValue::from(item.1),
                    )
                })
                .into_bound_py_any(py),
            PyReturnValue::NodeSingleValueWithIndexGroup(value) => value
                .map(|(key, items)| {
                    (
                        PyGroupKey::from(key),
                        items.map(|item| {
                            (
                                PyMedRecordAttribute::from(item.0.clone()),
                                PyMedRecordValue::from(item.1),
                            )
                        }),
                    )
                })
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::NodeSingleValueWithoutIndex(value) => {
                value.map(PyMedRecordValue::from).into_bound_py_any(py)
            }
            PyReturnValue::NodeSingleValueWithoutIndexGroup(value) => value
                .map(|(key, item)| (PyGroupKey::from(key), item.map(PyMedRecordValue::from)))
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeSingleValueWithIndex(value) => value
                .map(|item| (item.0, PyMedRecordValue::from(item.1)))
                .into_bound_py_any(py),
            PyReturnValue::EdgeSingleValueWithIndexGroup(value) => value
                .map(|(key, items)| {
                    (
                        PyGroupKey::from(key),
                        items.map(|item| (item.0, PyMedRecordValue::from(item.1))),
                    )
                })
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            PyReturnValue::EdgeSingleValueWithoutIndex(value) => {
                value.map(PyMedRecordValue::from).into_bound_py_any(py)
            }
            PyReturnValue::EdgeSingleValueWithoutIndexGroup(value) => value
                .map(|(key, item)| (PyGroupKey::from(key), item.map(PyMedRecordValue::from)))
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
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
                    "Failed to convert {ob} into MedRecordAttribute or List[MedRecordAttribute]",
                )))
                .into(),
            )
        }
    }
}

type PyGroupCardinalityWrapper = PyMedRecordAttributeCardinalityWrapper;
