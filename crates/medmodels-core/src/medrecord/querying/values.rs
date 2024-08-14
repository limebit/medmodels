use super::{
    edges::{EdgeValueOperand, EdgeValuesOperand},
    nodes::{NodeValueOperand, NodeValuesOperand},
    traits::{DeepClone, ReadWriteOrPanic},
    wrapper::Wrapper,
};
use crate::{
    medrecord::{EdgeIndex, MedRecordAttribute, MedRecordValue, NodeIndex},
    MedRecord,
};

#[derive(Debug, Clone)]
pub enum Value<'a> {
    AttributeValue(&'a MedRecordValue),
    NodeIndex(&'a MedRecordAttribute),
    EdgeIndex(&'a EdgeIndex),
}

impl<'a> PartialEq for Value<'a> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::AttributeValue(value1), Self::AttributeValue(value2)) => value1 == value2,
            (Self::NodeIndex(attr1), Self::NodeIndex(attr2)) => attr1 == attr2,
            (Self::EdgeIndex(index1), Self::EdgeIndex(index2)) => index1 == index2,
            _ => false, // TODO
        }
    }
}

impl<'a> PartialOrd for Value<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Self::AttributeValue(value1), Self::AttributeValue(value2)) => {
                value1.partial_cmp(value2)
            }
            (Self::NodeIndex(attr1), Self::NodeIndex(attr2)) => attr1.partial_cmp(attr2),
            (Self::EdgeIndex(index1), Self::EdgeIndex(index2)) => index1.partial_cmp(index2),
            _ => None, // TODO
        }
    }
}

#[derive(Debug, Clone)]
pub enum ValuesKind {
    Attribute(MedRecordAttribute),
    Index,
}

impl ValuesKind {
    pub fn evaluate_node_values<'a>(
        &'a self,
        medrecord: &'a MedRecord,
        node_indices: impl Iterator<Item = &'a NodeIndex> + 'a,
    ) -> Box<dyn Iterator<Item = (&'a NodeIndex, Value)> + 'a> {
        match self {
            Self::Attribute(attribute) => Box::new(node_indices.filter_map(move |node_index| {
                let value = medrecord
                    .node_attributes(node_index)
                    .expect("Node must exist")
                    .get(attribute)?;
                Some((node_index, Value::AttributeValue(value)))
            })),
            Self::Index => {
                Box::new(node_indices.map(move |index| (index, Value::NodeIndex(index))))
            }
        }
    }

    pub fn evaluate_edge_values<'a>(
        &'a self,
        medrecord: &'a MedRecord,
        edge_indices: impl Iterator<Item = &'a EdgeIndex> + 'a,
    ) -> Box<dyn Iterator<Item = (&'a EdgeIndex, Value)> + 'a> {
        match self {
            Self::Attribute(attribute) => Box::new(edge_indices.filter_map(move |edge_index| {
                let value = medrecord
                    .edge_attributes(edge_index)
                    .expect("Edge must exist")
                    .get(attribute)?;
                Some((edge_index, Value::AttributeValue(value)))
            })),
            Self::Index => {
                Box::new(edge_indices.map(move |index| (index, Value::EdgeIndex(index))))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum ValueKind {
    Max(ValuesKind),
    Min(ValuesKind),

    All(ValuesKind),
    Any(ValuesKind),
}

#[derive(Debug, Clone)]
pub enum ValuesOperand {
    Nodes(Wrapper<NodeValuesOperand>),
    Edges(Wrapper<EdgeValuesOperand>),
    Explicit(Vec<MedRecordValue>),
}

impl DeepClone for ValuesOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Nodes(context) => Self::Nodes(context.deep_clone()),
            Self::Edges(context) => Self::Edges(context.deep_clone()),
            Self::Explicit(values) => Self::Explicit(values.clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ValueOperand {
    Nodes(Wrapper<NodeValueOperand>),
    Edges(Wrapper<EdgeValueOperand>),
    Explicit(MedRecordValue),
}

impl DeepClone for ValueOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Nodes(context) => Self::Nodes(context.deep_clone()),
            Self::Edges(context) => Self::Edges(context.deep_clone()),
            Self::Explicit(value) => Self::Explicit(value.clone()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ComparisonOperand {
    Single(ValueOperand),
    Multiple(ValuesOperand),
}

impl DeepClone for ComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::Single(operand) => Self::Single(operand.deep_clone()),
            Self::Multiple(operand) => Self::Multiple(operand.deep_clone()),
        }
    }
}

impl From<Wrapper<NodeValuesOperand>> for ComparisonOperand {
    fn from(value: Wrapper<NodeValuesOperand>) -> Self {
        Self::Multiple(ValuesOperand::Nodes(value))
    }
}

impl From<Wrapper<EdgeValuesOperand>> for ComparisonOperand {
    fn from(value: Wrapper<EdgeValuesOperand>) -> Self {
        Self::Multiple(ValuesOperand::Edges(value.0.read_or_panic().clone().into()))
    }
}

impl<V> From<Vec<V>> for ComparisonOperand
where
    V: Into<MedRecordValue>,
{
    fn from(value: Vec<V>) -> Self {
        Self::Multiple(ValuesOperand::Explicit(
            value.into_iter().map(Into::into).collect(),
        ))
    }
}

impl<V, const N: usize> From<[V; N]> for ComparisonOperand
where
    V: Into<MedRecordValue> + Clone,
{
    fn from(value: [V; N]) -> Self {
        value.to_vec().into()
    }
}

impl From<Wrapper<NodeValueOperand>> for ComparisonOperand {
    fn from(value: Wrapper<NodeValueOperand>) -> Self {
        Self::Single(ValueOperand::Nodes(value))
    }
}

impl From<Wrapper<EdgeValueOperand>> for ComparisonOperand {
    fn from(value: Wrapper<EdgeValueOperand>) -> Self {
        Self::Single(ValueOperand::Edges(value))
    }
}

impl<V> From<V> for ComparisonOperand
where
    V: Into<MedRecordValue>,
{
    fn from(value: V) -> Self {
        Self::Single(ValueOperand::Explicit(value.into()))
    }
}
