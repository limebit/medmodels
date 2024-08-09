use super::operand::{
    PyEdgeValueOperand, PyEdgeValuesOperand, PyNodeValueOperand, PyNodeValuesOperand,
};
use crate::medrecord::{errors::PyMedRecordError, value::PyMedRecordValue, PyGroup};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{
        CardinalityWrapper, ComparisonOperand, Group, MedRecordValue, ValueOperand, ValuesOperand,
    },
};
use pyo3::{types::PyAnyMethods, Bound, FromPyObject, PyAny, PyResult};

#[repr(transparent)]
pub struct PyGroupCardinalityWrapper(CardinalityWrapper<Group>);

impl From<CardinalityWrapper<Group>> for PyGroupCardinalityWrapper {
    fn from(group: CardinalityWrapper<Group>) -> Self {
        Self(group)
    }
}

impl From<PyGroupCardinalityWrapper> for CardinalityWrapper<Group> {
    fn from(py_group: PyGroupCardinalityWrapper) -> Self {
        py_group.0
    }
}

impl<'a> FromPyObject<'a> for PyGroupCardinalityWrapper {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(group) = ob.extract::<PyGroup>() {
            Ok(CardinalityWrapper::Single(Group::from(group)).into())
        } else if let Ok(groups) = ob.extract::<Vec<PyGroup>>() {
            Ok(CardinalityWrapper::Multiple(groups.into_iter().map(Group::from).collect()).into())
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

#[repr(transparent)]
pub struct PyValuesOperand(ValuesOperand);

impl From<ValuesOperand> for PyValuesOperand {
    fn from(values: ValuesOperand) -> Self {
        Self(values)
    }
}

impl From<PyValuesOperand> for ValuesOperand {
    fn from(py_values: PyValuesOperand) -> Self {
        py_values.0
    }
}

impl<'a> FromPyObject<'a> for PyValuesOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(nodes) = ob.extract::<PyNodeValuesOperand>() {
            Ok(ValuesOperand::Nodes(nodes.into()).into())
        } else if let Ok(edges) = ob.extract::<PyEdgeValuesOperand>() {
            Ok(ValuesOperand::Edges(edges.into()).into())
        } else if let Ok(value) = ob.extract::<Vec<PyMedRecordValue>>() {
            Ok(
                ValuesOperand::Explicit(value.into_iter().map(MedRecordValue::from).collect())
                    .into(),
            )
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {} into ValuesOperand",
                    ob,
                )))
                .into(),
            )
        }
    }
}

#[repr(transparent)]
pub struct PyValueOperand(ValueOperand);

impl From<ValueOperand> for PyValueOperand {
    fn from(value: ValueOperand) -> Self {
        Self(value)
    }
}

impl From<PyValueOperand> for ValueOperand {
    fn from(py_value: PyValueOperand) -> Self {
        py_value.0
    }
}

impl<'a> FromPyObject<'a> for PyValueOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(nodes) = ob.extract::<PyNodeValueOperand>() {
            Ok(ValueOperand::Nodes(nodes.into()).into())
        } else if let Ok(edges) = ob.extract::<PyEdgeValueOperand>() {
            Ok(ValueOperand::Edges(edges.into()).into())
        } else if let Ok(value) = ob.extract::<PyMedRecordValue>() {
            Ok(ValueOperand::Explicit(value.into()).into())
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {} into ValueOperand",
                    ob,
                )))
                .into(),
            )
        }
    }
}

#[repr(transparent)]
pub struct PyComparisonOperand(ComparisonOperand);

impl From<ComparisonOperand> for PyComparisonOperand {
    fn from(comparison: ComparisonOperand) -> Self {
        Self(comparison)
    }
}

impl From<PyComparisonOperand> for ComparisonOperand {
    fn from(py_comparison: PyComparisonOperand) -> Self {
        py_comparison.0
    }
}

impl<'a> FromPyObject<'a> for PyComparisonOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(single) = ob.extract::<PyValueOperand>() {
            Ok(ComparisonOperand::Single(single.into()).into())
        } else if let Ok(multiple) = ob.extract::<PyValuesOperand>() {
            Ok(ComparisonOperand::Multiple(multiple.into()).into())
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {} into ComparisonOperand",
                    ob,
                )))
                .into(),
            )
        }
    }
}
