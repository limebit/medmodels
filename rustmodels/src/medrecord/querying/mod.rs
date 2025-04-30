pub mod attributes;
pub mod edges;
pub mod nodes;
pub mod values;

use super::{
    attribute::PyMedRecordAttribute, errors::PyMedRecordError, traits::DeepFrom,
    value::PyMedRecordValue, PyNodeIndex,
};
use attributes::{
    PyEdgeAttributesTreeOperand, PyEdgeMultipleAttributesOperand, PyEdgeSingleAttributeOperand,
    PyNodeAttributesTreeOperand, PyNodeMultipleAttributesOperand, PyNodeSingleAttributeOperand,
};
use edges::{PyEdgeIndexOperand, PyEdgeIndicesOperand};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{
        CardinalityWrapper, MedRecordAttribute, OptionalIndexWrapper, ReturnOperand, ReturnValue,
    },
};
use nodes::{PyNodeIndexOperand, PyNodeIndicesOperand};
use pyo3::{
    types::{PyAnyMethods, PyNone},
    Bound, FromPyObject, IntoPyObject, IntoPyObjectExt, PyAny, PyErr, PyResult, Python,
};
use std::collections::HashMap;
use values::{
    PyEdgeMultipleValuesOperand, PyEdgeSingleValueOperand, PyNodeMultipleValuesOperand,
    PyNodeSingleValueOperand,
};

pub fn parse_query_result(result: Bound<'_, PyAny>) -> ReturnOperand {
    if result.is_instance_of::<PyNodeAttributesTreeOperand>() {
        ReturnOperand::NodeAttributesTree(
            result
                .extract::<PyNodeAttributesTreeOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyEdgeAttributesTreeOperand>() {
        ReturnOperand::EdgeAttributesTree(
            result
                .extract::<PyEdgeAttributesTreeOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyNodeMultipleAttributesOperand>() {
        ReturnOperand::NodeMultipleAttributes(
            result
                .extract::<PyNodeMultipleAttributesOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyEdgeMultipleAttributesOperand>() {
        ReturnOperand::EdgeMultipleAttributes(
            result
                .extract::<PyEdgeMultipleAttributesOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyNodeSingleAttributeOperand>() {
        ReturnOperand::NodeSingleAttribute(
            result
                .extract::<PyNodeSingleAttributeOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyEdgeSingleAttributeOperand>() {
        ReturnOperand::EdgeSingleAttribute(
            result
                .extract::<PyEdgeSingleAttributeOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyEdgeIndicesOperand>() {
        ReturnOperand::EdgeIndices(
            result
                .extract::<PyEdgeIndicesOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyEdgeIndexOperand>() {
        ReturnOperand::EdgeIndex(
            result
                .extract::<PyEdgeIndexOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyNodeIndicesOperand>() {
        ReturnOperand::NodeIndices(
            result
                .extract::<PyNodeIndicesOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyNodeIndexOperand>() {
        ReturnOperand::NodeIndex(
            result
                .extract::<PyNodeIndexOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyNodeMultipleValuesOperand>() {
        ReturnOperand::NodeMultipleValues(
            result
                .extract::<PyNodeMultipleValuesOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyEdgeMultipleValuesOperand>() {
        ReturnOperand::EdgeMultipleValues(
            result
                .extract::<PyEdgeMultipleValuesOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyNodeSingleValueOperand>() {
        ReturnOperand::NodeSingleValue(
            result
                .extract::<PyNodeSingleValueOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyEdgeSingleValueOperand>() {
        ReturnOperand::EdgeSingleValue(
            result
                .extract::<PyEdgeSingleValueOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else {
        panic!("Query function is not a valid NodeQueryFunction")
    }
}

#[repr(transparent)]
pub struct PyReturnValue<'a>(ReturnValue<'a>);

impl<'a> From<ReturnValue<'a>> for PyReturnValue<'a> {
    fn from(value: ReturnValue<'a>) -> Self {
        Self(value)
    }
}

impl<'a> From<PyReturnValue<'a>> for ReturnValue<'a> {
    fn from(value: PyReturnValue<'a>) -> Self {
        value.0
    }
}

impl<'py> IntoPyObject<'py> for PyReturnValue<'_> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self.0 {
            ReturnValue::NodeAttributesTree(iterator) => iterator
                .map(|item| {
                    (
                        PyNodeIndex::from(item.0.clone()),
                        Vec::<PyMedRecordAttribute>::deep_from(item.1),
                    )
                })
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            ReturnValue::EdgeAttributesTree(iterator) => iterator
                .map(|item| (item.0, Vec::<PyMedRecordAttribute>::deep_from(item.1)))
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            ReturnValue::NodeMultipleAttributes(iterator) => iterator
                .map(|item| {
                    (
                        PyNodeIndex::from(item.0.clone()),
                        PyMedRecordAttribute::from(item.1),
                    )
                })
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            ReturnValue::EdgeMultipleAttributes(iterator) => iterator
                .map(|item| (item.0, PyMedRecordAttribute::from(item.1)))
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            ReturnValue::NodeSingleAttribute(attribute) => match attribute {
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
            ReturnValue::EdgeSingleAttribute(attribute) => match attribute {
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
            ReturnValue::EdgeIndices(iterator) => {
                iterator.collect::<Vec<_>>().into_bound_py_any(py)
            }
            ReturnValue::EdgeIndex(index) => index.into_bound_py_any(py),
            ReturnValue::NodeIndices(iterator) => iterator
                .map(PyMedRecordAttribute::from)
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
            ReturnValue::NodeIndex(index) => {
                Option::<PyNodeIndex>::deep_from(index).into_bound_py_any(py)
            }
            ReturnValue::NodeMultipleValues(iterator) => iterator
                .map(|item| {
                    (
                        PyNodeIndex::from(item.0.clone()),
                        PyMedRecordValue::from(item.1),
                    )
                })
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            ReturnValue::EdgeMultipleValues(iterator) => iterator
                .map(|item| (item.0, PyMedRecordValue::from(item.1)))
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            ReturnValue::NodeSingleValue(value) => match value {
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
            ReturnValue::EdgeSingleValue(value) => match value {
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
