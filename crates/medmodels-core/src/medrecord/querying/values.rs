use super::{
    edges::{EdgeValueOperand, EdgeValuesOperand},
    nodes::{NodeValueOperand, NodeValuesOperand},
    traits::{DeepClone, ReadWriteOrPanic},
    wrapper::Wrapper,
};
use crate::medrecord::MedRecordValue;

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
