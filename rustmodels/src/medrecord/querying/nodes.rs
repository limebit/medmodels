use super::{
    attributes::PyNodeAttributesTreeOperand, edges::PyEdgeOperand, PyGroupCardinalityWrapper,
    PyMedRecordAttributeCardinalityWrapper,
};
use crate::medrecord::{
    attribute::PyMedRecordAttribute,
    errors::PyMedRecordError,
    querying::{
        attributes::PyNodeAttributesTreeGroupOperand,
        edges::PyEdgeGroupOperand,
        values::{PyNodeMultipleValuesWithIndexGroupOperand, PyNodeMultipleValuesWithIndexOperand},
    },
    PyNodeIndex,
};
use medmodels_core::{
    errors::MedRecordError,
    medrecord::{
        querying::{
            group_by::GroupOperand,
            nodes::{
                self, EdgeDirection, NodeIndexComparisonOperand, NodeIndexOperand,
                NodeIndicesComparisonOperand, NodeIndicesOperand, NodeOperand,
            },
            wrapper::Wrapper,
            DeepClone,
        },
        NodeIndex,
    },
};
use pyo3::{
    pyclass, pymethods,
    types::{PyAnyMethods, PyFunction},
    Bound, FromPyObject, PyAny, PyResult,
};
use std::ops::Deref;

#[pyclass(eq, eq_int)]
#[derive(Clone, PartialEq)]
pub enum PyEdgeDirection {
    Incoming = 0,
    Outgoing = 1,
    Both = 2,
}

impl From<EdgeDirection> for PyEdgeDirection {
    fn from(value: EdgeDirection) -> Self {
        match value {
            EdgeDirection::Incoming => Self::Incoming,
            EdgeDirection::Outgoing => Self::Outgoing,
            EdgeDirection::Both => Self::Both,
        }
    }
}

impl From<PyEdgeDirection> for EdgeDirection {
    fn from(value: PyEdgeDirection) -> Self {
        match value {
            PyEdgeDirection::Incoming => Self::Incoming,
            PyEdgeDirection::Outgoing => Self::Outgoing,
            PyEdgeDirection::Both => Self::Both,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub enum NodeOperandGroupDiscriminator {
    Attribute(PyMedRecordAttribute),
}

impl From<NodeOperandGroupDiscriminator> for nodes::NodeOperandGroupDiscriminator {
    fn from(value: NodeOperandGroupDiscriminator) -> Self {
        match value {
            NodeOperandGroupDiscriminator::Attribute(attribute) => {
                Self::Attribute(attribute.into())
            }
        }
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyNodeOperand(Wrapper<NodeOperand>);

impl From<Wrapper<NodeOperand>> for PyNodeOperand {
    fn from(operand: Wrapper<NodeOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyNodeOperand> for Wrapper<NodeOperand> {
    fn from(operand: PyNodeOperand) -> Self {
        operand.0
    }
}

#[pymethods]
impl PyNodeOperand {
    pub fn attribute(
        &mut self,
        attribute: PyMedRecordAttribute,
    ) -> PyNodeMultipleValuesWithIndexOperand {
        self.0.attribute(attribute).into()
    }

    pub fn attributes(&mut self) -> PyNodeAttributesTreeOperand {
        self.0.attributes().into()
    }

    pub fn index(&mut self) -> PyNodeIndicesOperand {
        self.0.index().into()
    }

    pub fn in_group(&mut self, group: PyGroupCardinalityWrapper) {
        self.0.in_group(group);
    }

    pub fn has_attribute(&mut self, attribute: PyMedRecordAttributeCardinalityWrapper) {
        self.0.has_attribute(attribute);
    }

    pub fn edges(&mut self, direction: PyEdgeDirection) -> PyEdgeOperand {
        self.0.edges(direction.into()).into()
    }

    pub fn neighbors(&mut self, direction: PyEdgeDirection) -> PyNodeOperand {
        self.0.neighbors(direction.into()).into()
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyNodeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyNodeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyNodeOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn group_by(&mut self, discriminator: NodeOperandGroupDiscriminator) -> PyNodeGroupOperand {
        self.0.group_by(discriminator.into()).into()
    }

    pub fn deep_clone(&self) -> Self {
        self.0.deep_clone().into()
    }
}

#[pyclass]
#[repr(transparent)]
pub struct PyNodeGroupOperand(Wrapper<GroupOperand<NodeOperand>>);

impl From<Wrapper<GroupOperand<NodeOperand>>> for PyNodeGroupOperand {
    fn from(operand: Wrapper<GroupOperand<NodeOperand>>) -> Self {
        Self(operand)
    }
}

impl From<PyNodeGroupOperand> for Wrapper<GroupOperand<NodeOperand>> {
    fn from(operand: PyNodeGroupOperand) -> Self {
        operand.0
    }
}

#[pymethods]
impl PyNodeGroupOperand {
    pub fn attribute(
        &mut self,
        attribute: PyMedRecordAttribute,
    ) -> PyNodeMultipleValuesWithIndexGroupOperand {
        self.0.attribute(attribute).into()
    }

    pub fn attributes(&mut self) -> PyNodeAttributesTreeGroupOperand {
        self.0.attributes().into()
    }

    pub fn index(&mut self) -> PyNodeIndicesGroupOperand {
        self.0.index().into()
    }

    pub fn in_group(&mut self, group: PyGroupCardinalityWrapper) {
        self.0.in_group(group);
    }

    pub fn has_attribute(&mut self, attribute: PyMedRecordAttributeCardinalityWrapper) {
        self.0.has_attribute(attribute);
    }

    pub fn edges(&mut self, direction: PyEdgeDirection) -> PyEdgeGroupOperand {
        self.0.edges(direction.into()).into()
    }

    pub fn neighbors(&mut self, direction: PyEdgeDirection) -> PyNodeGroupOperand {
        self.0.neighbors(direction.into()).into()
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyNodeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyNodeOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyNodeOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn deep_clone(&self) -> Self {
        self.0.deep_clone().into()
    }
}

#[repr(transparent)]
pub struct PyNodeIndexComparisonOperand(NodeIndexComparisonOperand);

impl From<NodeIndexComparisonOperand> for PyNodeIndexComparisonOperand {
    fn from(operand: NodeIndexComparisonOperand) -> Self {
        Self(operand)
    }
}

impl From<PyNodeIndexComparisonOperand> for NodeIndexComparisonOperand {
    fn from(operand: PyNodeIndexComparisonOperand) -> Self {
        operand.0
    }
}

impl FromPyObject<'_> for PyNodeIndexComparisonOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(index) = ob.extract::<PyNodeIndex>() {
            Ok(NodeIndexComparisonOperand::Index(NodeIndex::from(index)).into())
        } else if let Ok(operand) = ob.extract::<PyNodeIndexOperand>() {
            Ok(PyNodeIndexComparisonOperand(operand.0.into()))
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {ob} into NodeIndex or NodeIndexOperand",
                )))
                .into(),
            )
        }
    }
}

#[repr(transparent)]
pub struct PyNodeIndicesComparisonOperand(NodeIndicesComparisonOperand);

impl From<NodeIndicesComparisonOperand> for PyNodeIndicesComparisonOperand {
    fn from(operand: NodeIndicesComparisonOperand) -> Self {
        Self(operand)
    }
}

impl From<PyNodeIndicesComparisonOperand> for NodeIndicesComparisonOperand {
    fn from(operand: PyNodeIndicesComparisonOperand) -> Self {
        operand.0
    }
}

impl FromPyObject<'_> for PyNodeIndicesComparisonOperand {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(indices) = ob.extract::<Vec<PyNodeIndex>>() {
            Ok(NodeIndicesComparisonOperand::Indices(
                indices.into_iter().map(NodeIndex::from).collect(),
            )
            .into())
        } else if let Ok(operand) = ob.extract::<PyNodeIndicesOperand>() {
            Ok(PyNodeIndicesComparisonOperand(operand.0.into()))
        } else {
            Err(
                PyMedRecordError::from(MedRecordError::ConversionError(format!(
                    "Failed to convert {ob} into List[NodeIndex] or NodeIndicesOperand",
                )))
                .into(),
            )
        }
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyNodeIndicesOperand(Wrapper<NodeIndicesOperand>);

impl From<Wrapper<NodeIndicesOperand>> for PyNodeIndicesOperand {
    fn from(operand: Wrapper<NodeIndicesOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyNodeIndicesOperand> for Wrapper<NodeIndicesOperand> {
    fn from(operand: PyNodeIndicesOperand) -> Self {
        operand.0
    }
}

impl Deref for PyNodeIndicesOperand {
    type Target = Wrapper<NodeIndicesOperand>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyNodeIndicesOperand {
    pub fn max(&mut self) -> PyNodeIndexOperand {
        self.0.max().into()
    }

    pub fn min(&mut self) -> PyNodeIndexOperand {
        self.0.min().into()
    }

    pub fn count(&mut self) -> PyNodeIndexOperand {
        self.0.count().into()
    }

    pub fn sum(&mut self) -> PyNodeIndexOperand {
        self.0.sum().into()
    }

    pub fn random(&mut self) -> PyNodeIndexOperand {
        self.0.random().into()
    }

    pub fn greater_than(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.greater_than(index);
    }

    pub fn greater_than_or_equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.greater_than_or_equal_to(index);
    }

    pub fn less_than(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.less_than(index);
    }

    pub fn less_than_or_equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.less_than_or_equal_to(index);
    }

    pub fn equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.equal_to(index);
    }

    pub fn not_equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.not_equal_to(index);
    }

    pub fn starts_with(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.starts_with(index);
    }

    pub fn ends_with(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.ends_with(index);
    }

    pub fn contains(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.contains(index);
    }

    pub fn is_in(&mut self, indices: PyNodeIndicesComparisonOperand) {
        self.0.is_in(indices);
    }

    pub fn is_not_in(&mut self, indices: PyNodeIndicesComparisonOperand) {
        self.0.is_not_in(indices);
    }

    pub fn add(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.add(index);
    }

    pub fn sub(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.sub(index);
    }

    pub fn mul(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.mul(index);
    }

    pub fn pow(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.pow(index);
    }

    pub fn r#mod(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.r#mod(index);
    }

    pub fn abs(&mut self) {
        self.0.abs();
    }

    pub fn trim(&mut self) {
        self.0.trim();
    }

    pub fn trim_start(&mut self) {
        self.0.trim_start();
    }

    pub fn trim_end(&mut self) {
        self.0.trim_end();
    }

    pub fn lowercase(&mut self) {
        self.0.lowercase();
    }

    pub fn uppercase(&mut self) {
        self.0.uppercase();
    }

    pub fn slice(&mut self, start: usize, end: usize) {
        self.0.slice(start, end);
    }

    pub fn is_string(&mut self) {
        self.0.is_string();
    }

    pub fn is_int(&mut self) {
        self.0.is_int();
    }

    pub fn is_max(&mut self) {
        self.0.is_max();
    }

    pub fn is_min(&mut self) {
        self.0.is_min();
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyNodeIndicesOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyNodeIndicesOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyNodeIndicesOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn deep_clone(&self) -> Self {
        self.0.deep_clone().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyNodeIndicesGroupOperand(Wrapper<GroupOperand<NodeIndicesOperand>>);

impl From<Wrapper<GroupOperand<NodeIndicesOperand>>> for PyNodeIndicesGroupOperand {
    fn from(operand: Wrapper<GroupOperand<NodeIndicesOperand>>) -> Self {
        Self(operand)
    }
}

impl From<PyNodeIndicesGroupOperand> for Wrapper<GroupOperand<NodeIndicesOperand>> {
    fn from(operand: PyNodeIndicesGroupOperand) -> Self {
        operand.0
    }
}

impl Deref for PyNodeIndicesGroupOperand {
    type Target = Wrapper<GroupOperand<NodeIndicesOperand>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyNodeIndicesGroupOperand {
    pub fn max(&mut self) -> PyNodeIndexGroupOperand {
        self.0.max().into()
    }

    pub fn min(&mut self) -> PyNodeIndexGroupOperand {
        self.0.min().into()
    }

    pub fn count(&mut self) -> PyNodeIndexGroupOperand {
        self.0.count().into()
    }

    pub fn sum(&mut self) -> PyNodeIndexGroupOperand {
        self.0.sum().into()
    }

    pub fn random(&mut self) -> PyNodeIndexGroupOperand {
        self.0.random().into()
    }

    pub fn greater_than(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.greater_than(index);
    }

    pub fn greater_than_or_equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.greater_than_or_equal_to(index);
    }

    pub fn less_than(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.less_than(index);
    }

    pub fn less_than_or_equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.less_than_or_equal_to(index);
    }

    pub fn equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.equal_to(index);
    }

    pub fn not_equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.not_equal_to(index);
    }

    pub fn starts_with(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.starts_with(index);
    }

    pub fn ends_with(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.ends_with(index);
    }

    pub fn contains(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.contains(index);
    }

    pub fn is_in(&mut self, indices: PyNodeIndicesComparisonOperand) {
        self.0.is_in(indices);
    }

    pub fn is_not_in(&mut self, indices: PyNodeIndicesComparisonOperand) {
        self.0.is_not_in(indices);
    }

    pub fn add(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.add(index);
    }

    pub fn sub(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.sub(index);
    }

    pub fn mul(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.mul(index);
    }

    pub fn pow(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.pow(index);
    }

    pub fn r#mod(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.r#mod(index);
    }

    pub fn abs(&mut self) {
        self.0.abs();
    }

    pub fn trim(&mut self) {
        self.0.trim();
    }

    pub fn trim_start(&mut self) {
        self.0.trim_start();
    }

    pub fn trim_end(&mut self) {
        self.0.trim_end();
    }

    pub fn lowercase(&mut self) {
        self.0.lowercase();
    }

    pub fn uppercase(&mut self) {
        self.0.uppercase();
    }

    pub fn slice(&mut self, start: usize, end: usize) {
        self.0.slice(start, end);
    }

    pub fn is_string(&mut self) {
        self.0.is_string();
    }

    pub fn is_int(&mut self) {
        self.0.is_int();
    }

    pub fn is_max(&mut self) {
        self.0.is_max();
    }

    pub fn is_min(&mut self) {
        self.0.is_min();
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyNodeIndicesOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyNodeIndicesOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyNodeIndicesOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn ungroup(&mut self) -> PyNodeIndicesOperand {
        self.0.ungroup().into()
    }

    pub fn deep_clone(&self) -> Self {
        self.0.deep_clone().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyNodeIndexOperand(Wrapper<NodeIndexOperand>);

impl From<Wrapper<NodeIndexOperand>> for PyNodeIndexOperand {
    fn from(operand: Wrapper<NodeIndexOperand>) -> Self {
        Self(operand)
    }
}

impl From<PyNodeIndexOperand> for Wrapper<NodeIndexOperand> {
    fn from(operand: PyNodeIndexOperand) -> Self {
        operand.0
    }
}

impl Deref for PyNodeIndexOperand {
    type Target = Wrapper<NodeIndexOperand>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyNodeIndexOperand {
    pub fn greater_than(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.greater_than(index);
    }

    pub fn greater_than_or_equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.greater_than_or_equal_to(index);
    }

    pub fn less_than(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.less_than(index);
    }

    pub fn less_than_or_equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.less_than_or_equal_to(index);
    }

    pub fn equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.equal_to(index);
    }

    pub fn not_equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.not_equal_to(index);
    }

    pub fn starts_with(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.starts_with(index);
    }

    pub fn ends_with(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.ends_with(index);
    }

    pub fn contains(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.contains(index);
    }

    pub fn is_in(&mut self, indices: PyNodeIndicesComparisonOperand) {
        self.0.is_in(indices);
    }

    pub fn is_not_in(&mut self, indices: PyNodeIndicesComparisonOperand) {
        self.0.is_not_in(indices);
    }

    pub fn add(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.add(index);
    }

    pub fn sub(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.sub(index);
    }

    pub fn mul(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.mul(index);
    }

    pub fn pow(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.pow(index);
    }

    pub fn r#mod(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.r#mod(index);
    }

    pub fn abs(&mut self) {
        self.0.abs();
    }

    pub fn trim(&mut self) {
        self.0.trim();
    }

    pub fn trim_start(&mut self) {
        self.0.trim_start();
    }

    pub fn trim_end(&mut self) {
        self.0.trim_end();
    }

    pub fn lowercase(&mut self) {
        self.0.lowercase();
    }

    pub fn uppercase(&mut self) {
        self.0.uppercase();
    }

    pub fn slice(&mut self, start: usize, end: usize) {
        self.0.slice(start, end);
    }

    pub fn is_string(&mut self) {
        self.0.is_string();
    }

    pub fn is_int(&mut self) {
        self.0.is_int();
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyNodeIndexOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyNodeIndexOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyNodeIndexOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn deep_clone(&self) -> Self {
        self.0.deep_clone().into()
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyNodeIndexGroupOperand(Wrapper<GroupOperand<NodeIndexOperand>>);

impl From<Wrapper<GroupOperand<NodeIndexOperand>>> for PyNodeIndexGroupOperand {
    fn from(operand: Wrapper<GroupOperand<NodeIndexOperand>>) -> Self {
        Self(operand)
    }
}

impl From<PyNodeIndexGroupOperand> for Wrapper<GroupOperand<NodeIndexOperand>> {
    fn from(operand: PyNodeIndexGroupOperand) -> Self {
        operand.0
    }
}

impl Deref for PyNodeIndexGroupOperand {
    type Target = Wrapper<GroupOperand<NodeIndexOperand>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pymethods]
impl PyNodeIndexGroupOperand {
    pub fn greater_than(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.greater_than(index);
    }

    pub fn greater_than_or_equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.greater_than_or_equal_to(index);
    }

    pub fn less_than(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.less_than(index);
    }

    pub fn less_than_or_equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.less_than_or_equal_to(index);
    }

    pub fn equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.equal_to(index);
    }

    pub fn not_equal_to(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.not_equal_to(index);
    }

    pub fn starts_with(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.starts_with(index);
    }

    pub fn ends_with(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.ends_with(index);
    }

    pub fn contains(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.contains(index);
    }

    pub fn is_in(&mut self, indices: PyNodeIndicesComparisonOperand) {
        self.0.is_in(indices);
    }

    pub fn is_not_in(&mut self, indices: PyNodeIndicesComparisonOperand) {
        self.0.is_not_in(indices);
    }

    pub fn add(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.add(index);
    }

    pub fn sub(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.sub(index);
    }

    pub fn mul(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.mul(index);
    }

    pub fn pow(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.pow(index);
    }

    pub fn r#mod(&mut self, index: PyNodeIndexComparisonOperand) {
        self.0.r#mod(index);
    }

    pub fn abs(&mut self) {
        self.0.abs();
    }

    pub fn trim(&mut self) {
        self.0.trim();
    }

    pub fn trim_start(&mut self) {
        self.0.trim_start();
    }

    pub fn trim_end(&mut self) {
        self.0.trim_end();
    }

    pub fn lowercase(&mut self) {
        self.0.lowercase();
    }

    pub fn uppercase(&mut self) {
        self.0.uppercase();
    }

    pub fn slice(&mut self, start: usize, end: usize) {
        self.0.slice(start, end);
    }

    pub fn is_string(&mut self) {
        self.0.is_string();
    }

    pub fn is_int(&mut self) {
        self.0.is_int();
    }

    pub fn either_or(&mut self, either: &Bound<'_, PyFunction>, or: &Bound<'_, PyFunction>) {
        self.0.either_or(
            |operand| {
                either
                    .call1((PyNodeIndexOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
            |operand| {
                or.call1((PyNodeIndexOperand::from(operand.clone()),))
                    .expect("Call must succeed");
            },
        );
    }

    pub fn exclude(&mut self, query: &Bound<'_, PyFunction>) {
        self.0.exclude(|operand| {
            query
                .call1((PyNodeIndexOperand::from(operand.clone()),))
                .expect("Call must succeed");
        });
    }

    pub fn ungroup(&mut self) -> PyNodeIndicesOperand {
        self.0.ungroup().into()
    }

    pub fn deep_clone(&self) -> Self {
        self.0.deep_clone().into()
    }
}
