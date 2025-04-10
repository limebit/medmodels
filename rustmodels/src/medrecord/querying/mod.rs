pub mod attributes;
pub mod edges;
pub mod nodes;
pub mod values;

use std::collections::HashMap;

use super::{
    attribute::PyMedRecordAttribute, errors::PyMedRecordError, traits::DeepFrom,
    value::PyMedRecordValue, PyNodeIndex,
};
use attributes::{PyAttributesTreeOperand, PyMultipleAttributesOperand, PySingleAttributeOperand};
use edges::{PyEdgeIndexOperand, PyEdgeIndicesOperand};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{CardinalityWrapper, Index, MedRecordAttribute, ReturnOperand, ReturnValue},
};
use nodes::{PyNodeIndexOperand, PyNodeIndicesOperand};
use pyo3::{
    types::PyAnyMethods, Bound, FromPyObject, IntoPyObject, IntoPyObjectExt, PyAny, PyErr,
    PyResult, Python,
};
use values::{PyMultipleValuesOperand, PySingleValueOperand};

pub fn parse_query_result(result: Bound<'_, PyAny>) -> ReturnOperand {
    if result.is_instance_of::<PyAttributesTreeOperand>() {
        ReturnOperand::AttributesTree(
            result
                .extract::<PyAttributesTreeOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PyMultipleAttributesOperand>() {
        ReturnOperand::MultipleAttributes(
            result
                .extract::<PyMultipleAttributesOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PySingleAttributeOperand>() {
        ReturnOperand::SingleAttribute(
            result
                .extract::<PySingleAttributeOperand>()
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
    } else if result.is_instance_of::<PyMultipleValuesOperand>() {
        ReturnOperand::MultipleValues(
            result
                .extract::<PyMultipleValuesOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else if result.is_instance_of::<PySingleValueOperand>() {
        ReturnOperand::SingleValue(
            result
                .extract::<PySingleValueOperand>()
                .expect("Extract must succeed")
                .into(),
        )
    } else {
        panic!("Query function is not a valid NodeQueryFunction")
    }
}

#[repr(transparent)]
#[derive(PartialEq, Eq, Hash)]
pub struct PyIndex<'a>(Index<'a>);

impl<'a> From<Index<'a>> for PyIndex<'a> {
    fn from(value: Index<'a>) -> Self {
        Self(value)
    }
}

impl<'a> From<PyIndex<'a>> for Index<'a> {
    fn from(value: PyIndex<'a>) -> Self {
        value.0
    }
}

impl<'py> IntoPyObject<'py> for PyIndex<'_> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self.0 {
            Index::NodeIndex(value) => PyMedRecordAttribute::from(value.clone()).into_pyobject(py),
            Index::EdgeIndex(value) => (*value).into_bound_py_any(py),
        }
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
            ReturnValue::AttributesTree(iterator) => iterator
                .map(|item| {
                    (
                        PyIndex::from(item.0),
                        Vec::<PyMedRecordAttribute>::deep_from(item.1),
                    )
                })
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            ReturnValue::MultipleAttributes(iterator) => iterator
                .map(|item| (PyIndex::from(item.0), PyMedRecordAttribute::from(item.1)))
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            ReturnValue::SingleAttributeWithIndex(attribute) => attribute
                .map(|(index, attribute)| {
                    (PyIndex::from(index), PyMedRecordAttribute::from(attribute))
                })
                .into_bound_py_any(py),
            ReturnValue::SingleAttribute(attribute) => {
                Option::<PyMedRecordAttribute>::deep_from(attribute).into_bound_py_any(py)
            }
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
            ReturnValue::MultipleValues(iterator) => iterator
                .map(|item| (PyIndex::from(item.0), PyMedRecordValue::from(item.1)))
                .collect::<HashMap<_, _>>()
                .into_bound_py_any(py),
            ReturnValue::SingleValueWithIndex(value) => value
                .map(|(index, value)| (PyIndex::from(index), PyMedRecordValue::from(value)))
                .into_bound_py_any(py),
            ReturnValue::SingleValue(value) => {
                Option::<PyMedRecordValue>::deep_from(value).into_bound_py_any(py)
            }
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
