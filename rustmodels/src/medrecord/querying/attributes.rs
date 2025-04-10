use super::values::PyMultipleValuesOperand;
use crate::medrecord::{attribute::PyMedRecordAttribute, errors::PyMedRecordError};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{
        AttributesTreeOperand, DeepClone, MedRecordAttribute, MultipleAttributesComparisonOperand,
        MultipleAttributesOperand, SingleAttributeComparisonOperand, SingleAttributeOperand,
        Wrapper,
    },
};
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyFunction},
    Bound, FromPyObject, PyAny, PyResult,
};

#[repr(transparent)]
pub struct PySingleAttributeComparisonOperand(SingleAttributeComparisonOperand);

impl From<SingleAttributeComparisonOperand> for PySingleAttributeComparisonOperand {
    fn from(operand: SingleAttributeComparisonOperand) -> Self {
        Self(operand)
    }
}

impl From<PySingleAttributeComparisonOperand> for SingleAttributeComparisonOperand {
    fn from(operand: PySingleAttributeComparisonOperand) -> Self {
        operand.0
    }
}

impl FromPyObject<'_> for PySingleAttributeComparisonOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(attribute) = ob.extract::<PyMedRecordAttribute>() {
            Ok(SingleAttributeComparisonOperand::Attribute(attribute.into()).into())
        } else if let Ok(operand) = ob.extract::<PySingleAttributeOperand>() {
            Ok(PySingleAttributeComparisonOperand(operand.0.into()))
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {} into MedRecordValue or SingleValueOperand",
                    ob,
                )))
                .into(),
            )
        }
    }
}

#[repr(transparent)]
pub struct PyMultipleAttributesComparisonOperand(MultipleAttributesComparisonOperand);

impl From<MultipleAttributesComparisonOperand> for PyMultipleAttributesComparisonOperand {
    fn from(operand: MultipleAttributesComparisonOperand) -> Self {
        Self(operand)
    }
}

impl From<PyMultipleAttributesComparisonOperand> for MultipleAttributesComparisonOperand {
    fn from(operand: PyMultipleAttributesComparisonOperand) -> Self {
        operand.0
    }
}

impl FromPyObject<'_> for PyMultipleAttributesComparisonOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(values) = ob.extract::<Vec<PyMedRecordAttribute>>() {
            Ok(MultipleAttributesComparisonOperand::Attributes(
                values.into_iter().map(MedRecordAttribute::from).collect(),
            )
            .into())
        } else if let Ok(operand) = ob.extract::<PyMultipleAttributesOperand>() {
            Ok(PyMultipleAttributesComparisonOperand(operand.0.into()))
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {} into List[MedRecordAttribute] or MultipleAttributesOperand",
                    ob,
                )))
                .into(),
            )
        }
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyAttributesTreeOperand(Wrapper<AttributesTreeOperand>);

impl From<Wrapper<AttributesTreeOperand>> for PyAttributesTreeOperand {
    fn from(operand: Wrapper<AttributesTreeOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyAttributesTreeOperand> for Wrapper<AttributesTreeOperand> {
    fn from(operand: PyAttributesTreeOperand) -> Self {
        operand.0
    }
}

#[pymethods]
impl PyAttributesTreeOperand {
    pub fn max(&self) -> PyMultipleAttributesOperand {
        self.0.max().into()
    }

    pub fn min(&self) -> PyMultipleAttributesOperand {
        self.0.min().into()
    }

    pub fn count(&self) -> PyMultipleAttributesOperand {
        self.0.count().into()
    }

    pub fn sum(&self) -> PyMultipleAttributesOperand {
        self.0.sum().into()
    }

    pub fn first(&self) -> PyMultipleAttributesOperand {
        self.0.first().into()
    }

    pub fn last(&self) -> PyMultipleAttributesOperand {
        self.0.last().into()
    }

    pub fn greater_than(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.greater_than(attribute);
    }

    pub fn greater_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.greater_than_or_equal_to(attribute);
    }

    pub fn less_than(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.less_than(attribute);
    }

    pub fn less_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.less_than_or_equal_to(attribute);
    }

    pub fn equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.equal_to(attribute);
    }

    pub fn not_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.not_equal_to(attribute);
    }

    pub fn starts_with(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.starts_with(attribute);
    }

    pub fn ends_with(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.ends_with(attribute);
    }

    pub fn contains(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.contains(attribute);
    }

    pub fn is_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
        self.0.is_in(attributes);
    }

    pub fn is_not_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
        self.0.is_not_in(attributes);
    }

    pub fn add(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.add(attribute);
    }

    pub fn sub(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.sub(attribute);
    }

    pub fn mul(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.mul(attribute);
    }

    pub fn pow(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.pow(attribute);
    }

    pub fn r#mod(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.r#mod(attribute);
    }

    pub fn abs(&self) {
        self.0.abs();
    }

    pub fn trim(&self) {
        self.0.trim();
    }

    pub fn trim_start(&self) {
        self.0.trim_start();
    }

    pub fn trim_end(&self) {
        self.0.trim_end();
    }

    pub fn lowercase(&self) {
        self.0.lowercase();
    }

    pub fn uppercase(&self) {
        self.0.uppercase();
    }

    pub fn slice(&self, start: usize, end: usize) {
        self.0.slice(start, end);
    }

    pub fn is_string(&self) {
        self.0.is_string();
    }

    pub fn is_int(&self) {
        self.0.is_int();
    }

    pub fn is_max(&self) {
        self.0.is_max();
    }

    pub fn is_min(&self) {
        self.0.is_min();
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyAttributesTreeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyAttributesTreeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyAttributesTreeOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn deep_clone(&self) -> PyAttributesTreeOperand {
        self.0.deep_clone().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyMultipleAttributesOperand(Wrapper<MultipleAttributesOperand>);

impl From<Wrapper<MultipleAttributesOperand>> for PyMultipleAttributesOperand {
    fn from(operand: Wrapper<MultipleAttributesOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyMultipleAttributesOperand> for Wrapper<MultipleAttributesOperand> {
    fn from(operand: PyMultipleAttributesOperand) -> Self {
        operand.0
    }
}

#[pymethods]
impl PyMultipleAttributesOperand {
    pub fn max(&self) -> PySingleAttributeOperand {
        self.0.max().into()
    }

    pub fn min(&self) -> PySingleAttributeOperand {
        self.0.min().into()
    }

    pub fn count(&self) -> PySingleAttributeOperand {
        self.0.count().into()
    }

    pub fn sum(&self) -> PySingleAttributeOperand {
        self.0.sum().into()
    }

    pub fn first(&self) -> PySingleAttributeOperand {
        self.0.first().into()
    }

    pub fn last(&self) -> PySingleAttributeOperand {
        self.0.last().into()
    }

    pub fn greater_than(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.greater_than(attribute);
    }

    pub fn greater_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.greater_than_or_equal_to(attribute);
    }

    pub fn less_than(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.less_than(attribute);
    }

    pub fn less_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.less_than_or_equal_to(attribute);
    }

    pub fn equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.equal_to(attribute);
    }

    pub fn not_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.not_equal_to(attribute);
    }

    pub fn starts_with(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.starts_with(attribute);
    }

    pub fn ends_with(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.ends_with(attribute);
    }

    pub fn contains(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.contains(attribute);
    }

    pub fn is_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
        self.0.is_in(attributes);
    }

    pub fn is_not_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
        self.0.is_not_in(attributes);
    }

    pub fn add(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.add(attribute);
    }

    pub fn sub(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.sub(attribute);
    }

    pub fn mul(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.mul(attribute);
    }

    pub fn pow(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.pow(attribute);
    }

    pub fn r#mod(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.r#mod(attribute);
    }

    pub fn abs(&self) {
        self.0.abs();
    }

    pub fn trim(&self) {
        self.0.trim();
    }

    pub fn trim_start(&self) {
        self.0.trim_start();
    }

    pub fn trim_end(&self) {
        self.0.trim_end();
    }

    pub fn lowercase(&self) {
        self.0.lowercase();
    }

    pub fn uppercase(&self) {
        self.0.uppercase();
    }

    pub fn to_values(&self) -> PyMultipleValuesOperand {
        self.0.to_values().into()
    }

    pub fn slice(&self, start: usize, end: usize) {
        self.0.slice(start, end);
    }

    pub fn is_string(&self) {
        self.0.is_string();
    }

    pub fn is_int(&self) {
        self.0.is_int();
    }

    pub fn is_max(&self) {
        self.0.is_max();
    }

    pub fn is_min(&self) {
        self.0.is_min();
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyMultipleAttributesOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyMultipleAttributesOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyMultipleAttributesOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn deep_clone(&self) -> PyMultipleAttributesOperand {
        self.0.deep_clone().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PySingleAttributeOperand(Wrapper<SingleAttributeOperand>);

impl From<Wrapper<SingleAttributeOperand>> for PySingleAttributeOperand {
    fn from(operand: Wrapper<SingleAttributeOperand>) -> Self {
        Self(operand)
    }
}

impl From<PySingleAttributeOperand> for Wrapper<SingleAttributeOperand> {
    fn from(operand: PySingleAttributeOperand) -> Self {
        operand.0
    }
}

#[pymethods]
impl PySingleAttributeOperand {
    pub fn greater_than(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.greater_than(attribute);
    }

    pub fn greater_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.greater_than_or_equal_to(attribute);
    }

    pub fn less_than(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.less_than(attribute);
    }

    pub fn less_than_or_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.less_than_or_equal_to(attribute);
    }

    pub fn equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.equal_to(attribute);
    }

    pub fn not_equal_to(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.not_equal_to(attribute);
    }

    pub fn starts_with(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.starts_with(attribute);
    }

    pub fn ends_with(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.ends_with(attribute);
    }

    pub fn contains(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.contains(attribute);
    }

    pub fn is_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
        self.0.is_in(attributes);
    }

    pub fn is_not_in(&self, attributes: PyMultipleAttributesComparisonOperand) {
        self.0.is_not_in(attributes);
    }

    pub fn add(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.add(attribute);
    }

    pub fn sub(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.sub(attribute);
    }

    pub fn mul(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.mul(attribute);
    }

    pub fn pow(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.pow(attribute);
    }

    pub fn r#mod(&self, attribute: PySingleAttributeComparisonOperand) {
        self.0.r#mod(attribute);
    }

    pub fn abs(&self) {
        self.0.abs();
    }

    pub fn trim(&self) {
        self.0.trim();
    }

    pub fn trim_start(&self) {
        self.0.trim_start();
    }

    pub fn trim_end(&self) {
        self.0.trim_end();
    }

    pub fn lowercase(&self) {
        self.0.lowercase();
    }

    pub fn uppercase(&self) {
        self.0.uppercase();
    }

    pub fn slice(&self, start: usize, end: usize) {
        self.0.slice(start, end);
    }

    pub fn is_string(&self) {
        self.0.is_string();
    }

    pub fn is_int(&self) {
        self.0.is_int();
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PySingleAttributeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PySingleAttributeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PySingleAttributeOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn deep_clone(&self) -> PySingleAttributeOperand {
        self.0.deep_clone().into()
    }
}
