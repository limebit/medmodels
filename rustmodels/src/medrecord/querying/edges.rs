use super::{
    attributes::PyEdgeAttributesTreeOperand, nodes::PyNodeOperand, PyGroupCardinalityWrapper,
    PyMedRecordAttributeCardinalityWrapper,
};
use crate::medrecord::{
    attribute::PyMedRecordAttribute,
    errors::PyMedRecordError,
    querying::{
        attributes::PyEdgeAttributesTreeGroupOperand,
        nodes::PyNodeGroupOperand,
        values::{PyEdgeMultipleValuesWithIndexGroupOperand, PyEdgeMultipleValuesWithIndexOperand},
    },
};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{
        querying::{
            edges::{
                self, EdgeIndexComparisonOperand, EdgeIndexOperand, EdgeIndicesComparisonOperand,
                EdgeIndicesOperand, EdgeOperand,
            },
            group_by::GroupOperand,
            wrapper::Wrapper,
            DeepClone,
        },
        EdgeIndex,
    },
};
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyFunction},
    Bound, FromPyObject, PyAny, PyResult,
};
use std::ops::Deref;

#[pyclass]
#[derive(Clone)]
pub enum EdgeOperandGroupDiscriminator {
    SourceNode(),
    TargetNode(),
    Parallel(),
    Attribute(PyMedRecordAttribute),
}

impl From<EdgeOperandGroupDiscriminator> for edges::EdgeOperandGroupDiscriminator {
    fn from(discriminator: EdgeOperandGroupDiscriminator) -> Self {
        match discriminator {
            EdgeOperandGroupDiscriminator::SourceNode() => Self::SourceNode,
            EdgeOperandGroupDiscriminator::TargetNode() => Self::TargetNode,
            EdgeOperandGroupDiscriminator::Parallel() => Self::Parallel,
            EdgeOperandGroupDiscriminator::Attribute(attribute) => {
                Self::Attribute(attribute.into())
            }
        }
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyEdgeOperand(Wrapper<EdgeOperand>);

impl From<Wrapper<EdgeOperand>> for PyEdgeOperand {
    fn from(operand: Wrapper<EdgeOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyEdgeOperand> for Wrapper<EdgeOperand> {
    fn from(operand: PyEdgeOperand) -> Self {
        operand.0
    }
}

#[pymethods]
impl PyEdgeOperand {
    pub fn attribute(
        &mut self,
        attribute: PyMedRecordAttribute,
    ) -> PyEdgeMultipleValuesWithIndexOperand {
        self.0.attribute(attribute).into()
    }

    pub fn attributes(&mut self) -> PyEdgeAttributesTreeOperand {
        self.0.attributes().into()
    }

    pub fn index(&mut self) -> PyEdgeIndicesOperand {
        self.0.index().into()
    }

    pub fn in_group(&mut self, group: PyGroupCardinalityWrapper) {
        self.0.in_group(group);
    }

    pub fn has_attribute(&mut self, attribute: PyMedRecordAttributeCardinalityWrapper) {
        self.0.has_attribute(attribute);
    }

    pub fn source_node(&mut self) -> PyNodeOperand {
        self.0.source_node().into()
    }

    pub fn target_node(&mut self) -> PyNodeOperand {
        self.0.target_node().into()
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyEdgeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyEdgeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyEdgeOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn group_by(&mut self, discriminator: EdgeOperandGroupDiscriminator) -> PyEdgeGroupOperand {
        self.0.group_by(discriminator.into()).into()
    }

    pub fn deep_clone(&self) -> PyEdgeOperand {
        self.0.deep_clone().into()
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyEdgeGroupOperand(Wrapper<GroupOperand<EdgeOperand>>);

impl From<Wrapper<GroupOperand<EdgeOperand>>> for PyEdgeGroupOperand {
    fn from(operand: Wrapper<GroupOperand<EdgeOperand>>) -> Self {
        Self(operand)
    }
}

impl From<PyEdgeGroupOperand> for Wrapper<GroupOperand<EdgeOperand>> {
    fn from(operand: PyEdgeGroupOperand) -> Self {
        operand.0
    }
}

#[pymethods]
impl PyEdgeGroupOperand {
    pub fn attribute(
        &mut self,
        attribute: PyMedRecordAttribute,
    ) -> PyEdgeMultipleValuesWithIndexGroupOperand {
        self.0.attribute(attribute).into()
    }

    pub fn attributes(&mut self) -> PyEdgeAttributesTreeGroupOperand {
        self.0.attributes().into()
    }

    pub fn index(&mut self) -> PyEdgeIndicesGroupOperand {
        self.0.index().into()
    }

    pub fn in_group(&mut self, group: PyGroupCardinalityWrapper) {
        self.0.in_group(group);
    }

    pub fn has_attribute(&mut self, attribute: PyMedRecordAttributeCardinalityWrapper) {
        self.0.has_attribute(attribute);
    }

    pub fn source_node(&mut self) -> PyNodeGroupOperand {
        self.0.source_node().into()
    }

    pub fn target_node(&mut self) -> PyNodeGroupOperand {
        self.0.target_node().into()
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyEdgeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyEdgeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyEdgeOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn deep_clone(&self) -> Self {
        self.0.deep_clone().into()
    }
}

#[repr(transparent)]
pub struct PyEdgeIndexComparisonOperand(EdgeIndexComparisonOperand);

impl From<EdgeIndexComparisonOperand> for PyEdgeIndexComparisonOperand {
    fn from(operand: EdgeIndexComparisonOperand) -> Self {
        Self(operand)
    }
}

impl From<PyEdgeIndexComparisonOperand> for EdgeIndexComparisonOperand {
    fn from(operand: PyEdgeIndexComparisonOperand) -> Self {
        operand.0
    }
}

impl FromPyObject<'_> for PyEdgeIndexComparisonOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(index) = ob.extract::<EdgeIndex>() {
            Ok(EdgeIndexComparisonOperand::Index(index).into())
        } else if let Ok(operand) = ob.extract::<PyEdgeIndexOperand>() {
            Ok(PyEdgeIndexComparisonOperand(operand.0.into()))
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {ob} into EdgeIndex or EdgeIndexOperand",
                )))
                .into(),
            )
        }
    }
}

#[repr(transparent)]
pub struct PyEdgeIndicesComparisonOperand(EdgeIndicesComparisonOperand);

impl From<EdgeIndicesComparisonOperand> for PyEdgeIndicesComparisonOperand {
    fn from(operand: EdgeIndicesComparisonOperand) -> Self {
        Self(operand)
    }
}

impl From<PyEdgeIndicesComparisonOperand> for EdgeIndicesComparisonOperand {
    fn from(operand: PyEdgeIndicesComparisonOperand) -> Self {
        operand.0
    }
}

impl FromPyObject<'_> for PyEdgeIndicesComparisonOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(indices) = ob.extract::<Vec<EdgeIndex>>() {
            Ok(EdgeIndicesComparisonOperand::from(indices).into())
        } else if let Ok(operand) = ob.extract::<PyEdgeIndicesOperand>() {
            Ok(PyEdgeIndicesComparisonOperand(operand.0.into()))
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {ob} into List[EdgeIndex] or EdgeIndicesOperand",
                )))
                .into(),
            )
        }
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyEdgeIndicesOperand(Wrapper<EdgeIndicesOperand>);

impl From<Wrapper<EdgeIndicesOperand>> for PyEdgeIndicesOperand {
    fn from(operand: Wrapper<EdgeIndicesOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyEdgeIndicesOperand> for Wrapper<EdgeIndicesOperand> {
    fn from(operand: PyEdgeIndicesOperand) -> Self {
        operand.0
    }
}

impl Deref for PyEdgeIndicesOperand {
    type Target = Wrapper<EdgeIndicesOperand>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyEdgeIndicesOperand {
    pub fn max(&mut self) -> PyEdgeIndexOperand {
        self.0.max().into()
    }

    pub fn min(&mut self) -> PyEdgeIndexOperand {
        self.0.min().into()
    }

    pub fn count(&mut self) -> PyEdgeIndexOperand {
        self.0.count().into()
    }

    pub fn sum(&mut self) -> PyEdgeIndexOperand {
        self.0.sum().into()
    }

    pub fn random(&mut self) -> PyEdgeIndexOperand {
        self.0.random().into()
    }

    pub fn greater_than(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.greater_than(index);
    }

    pub fn greater_than_or_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.greater_than_or_equal_to(index);
    }

    pub fn less_than(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.less_than(index);
    }

    pub fn less_than_or_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.less_than_or_equal_to(index);
    }

    pub fn equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.equal_to(index);
    }

    pub fn not_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.not_equal_to(index);
    }

    pub fn starts_with(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.starts_with(index);
    }

    pub fn ends_with(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.ends_with(index);
    }

    pub fn contains(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.contains(index);
    }

    pub fn is_in(&mut self, indices: PyEdgeIndicesComparisonOperand) {
        self.0.is_in(indices);
    }

    pub fn is_not_in(&mut self, indices: PyEdgeIndicesComparisonOperand) {
        self.0.is_not_in(indices);
    }

    pub fn add(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.add(index);
    }

    pub fn sub(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.sub(index);
    }

    pub fn mul(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.mul(index);
    }

    pub fn pow(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.pow(index);
    }

    pub fn r#mod(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.r#mod(index);
    }

    pub fn is_max(&mut self) {
        self.0.is_max()
    }

    pub fn is_min(&mut self) {
        self.0.is_min()
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyEdgeIndicesOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyEdgeIndicesOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyEdgeIndicesOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn deep_clone(&self) -> PyEdgeIndicesOperand {
        self.0.deep_clone().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyEdgeIndicesGroupOperand(Wrapper<GroupOperand<EdgeIndicesOperand>>);

impl From<Wrapper<GroupOperand<EdgeIndicesOperand>>> for PyEdgeIndicesGroupOperand {
    fn from(operand: Wrapper<GroupOperand<EdgeIndicesOperand>>) -> Self {
        Self(operand)
    }
}

impl From<PyEdgeIndicesGroupOperand> for Wrapper<GroupOperand<EdgeIndicesOperand>> {
    fn from(operand: PyEdgeIndicesGroupOperand) -> Self {
        operand.0
    }
}

impl Deref for PyEdgeIndicesGroupOperand {
    type Target = Wrapper<GroupOperand<EdgeIndicesOperand>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyEdgeIndicesGroupOperand {
    pub fn max(&mut self) -> PyEdgeIndexGroupOperand {
        self.0.max().into()
    }

    pub fn min(&mut self) -> PyEdgeIndexGroupOperand {
        self.0.min().into()
    }

    pub fn count(&mut self) -> PyEdgeIndexGroupOperand {
        self.0.count().into()
    }

    pub fn sum(&mut self) -> PyEdgeIndexGroupOperand {
        self.0.sum().into()
    }

    pub fn random(&mut self) -> PyEdgeIndexGroupOperand {
        self.0.random().into()
    }

    pub fn greater_than(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.greater_than(index);
    }

    pub fn greater_than_or_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.greater_than_or_equal_to(index);
    }

    pub fn less_than(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.less_than(index);
    }

    pub fn less_than_or_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.less_than_or_equal_to(index);
    }

    pub fn equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.equal_to(index);
    }

    pub fn not_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.not_equal_to(index);
    }

    pub fn starts_with(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.starts_with(index);
    }

    pub fn ends_with(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.ends_with(index);
    }

    pub fn contains(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.contains(index);
    }

    pub fn is_in(&mut self, indices: PyEdgeIndicesComparisonOperand) {
        self.0.is_in(indices);
    }

    pub fn is_not_in(&mut self, indices: PyEdgeIndicesComparisonOperand) {
        self.0.is_not_in(indices);
    }

    pub fn add(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.add(index);
    }

    pub fn sub(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.sub(index);
    }

    pub fn mul(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.mul(index);
    }

    pub fn pow(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.pow(index);
    }

    pub fn r#mod(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.r#mod(index);
    }

    pub fn is_max(&mut self) {
        self.0.is_max()
    }

    pub fn is_min(&mut self) {
        self.0.is_min()
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyEdgeIndicesOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyEdgeIndicesOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyEdgeIndicesOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn ungroup(&mut self) -> PyEdgeIndicesOperand {
        self.0.ungroup().into()
    }

    pub fn deep_clone(&self) -> Self {
        self.0.deep_clone().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyEdgeIndexOperand(Wrapper<EdgeIndexOperand>);

impl From<Wrapper<EdgeIndexOperand>> for PyEdgeIndexOperand {
    fn from(operand: Wrapper<EdgeIndexOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyEdgeIndexOperand> for Wrapper<EdgeIndexOperand> {
    fn from(operand: PyEdgeIndexOperand) -> Self {
        operand.0
    }
}

impl Deref for PyEdgeIndexOperand {
    type Target = Wrapper<EdgeIndexOperand>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyEdgeIndexOperand {
    pub fn greater_than(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.greater_than(index);
    }

    pub fn greater_than_or_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.greater_than_or_equal_to(index);
    }

    pub fn less_than(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.less_than(index);
    }

    pub fn less_than_or_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.less_than_or_equal_to(index);
    }

    pub fn equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.equal_to(index);
    }

    pub fn not_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.not_equal_to(index);
    }

    pub fn starts_with(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.starts_with(index);
    }

    pub fn ends_with(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.ends_with(index);
    }

    pub fn contains(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.contains(index);
    }

    pub fn is_in(&mut self, indices: PyEdgeIndicesComparisonOperand) {
        self.0.is_in(indices);
    }

    pub fn is_not_in(&mut self, indices: PyEdgeIndicesComparisonOperand) {
        self.0.is_not_in(indices);
    }

    pub fn add(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.add(index);
    }

    pub fn sub(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.sub(index);
    }

    pub fn mul(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.mul(index);
    }

    pub fn pow(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.pow(index);
    }

    pub fn r#mod(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.r#mod(index);
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyEdgeIndexOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyEdgeIndexOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyEdgeIndexOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn deep_clone(&self) -> PyEdgeIndexOperand {
        self.0.deep_clone().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyEdgeIndexGroupOperand(Wrapper<GroupOperand<EdgeIndexOperand>>);

impl From<Wrapper<GroupOperand<EdgeIndexOperand>>> for PyEdgeIndexGroupOperand {
    fn from(operand: Wrapper<GroupOperand<EdgeIndexOperand>>) -> Self {
        Self(operand)
    }
}

impl From<PyEdgeIndexGroupOperand> for Wrapper<GroupOperand<EdgeIndexOperand>> {
    fn from(operand: PyEdgeIndexGroupOperand) -> Self {
        operand.0
    }
}

impl Deref for PyEdgeIndexGroupOperand {
    type Target = Wrapper<GroupOperand<EdgeIndexOperand>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyEdgeIndexGroupOperand {
    pub fn greater_than(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.greater_than(index);
    }

    pub fn greater_than_or_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.greater_than_or_equal_to(index);
    }

    pub fn less_than(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.less_than(index);
    }

    pub fn less_than_or_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.less_than_or_equal_to(index);
    }

    pub fn equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.equal_to(index);
    }

    pub fn not_equal_to(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.not_equal_to(index);
    }

    pub fn starts_with(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.starts_with(index);
    }

    pub fn ends_with(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.ends_with(index);
    }

    pub fn contains(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.contains(index);
    }

    pub fn is_in(&mut self, indices: PyEdgeIndicesComparisonOperand) {
        self.0.is_in(indices);
    }

    pub fn is_not_in(&mut self, indices: PyEdgeIndicesComparisonOperand) {
        self.0.is_not_in(indices);
    }

    pub fn add(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.add(index);
    }

    pub fn sub(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.sub(index);
    }

    pub fn mul(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.mul(index);
    }

    pub fn pow(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.pow(index);
    }

    pub fn r#mod(&mut self, index: PyEdgeIndexComparisonOperand) {
        self.0.r#mod(index);
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyEdgeIndexOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyEdgeIndexOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyEdgeIndexOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn ungroup(&mut self) -> PyEdgeIndicesOperand {
        self.0.ungroup().into()
    }

    pub fn deep_clone(&self) -> Self {
        self.0.deep_clone().into()
    }
}
