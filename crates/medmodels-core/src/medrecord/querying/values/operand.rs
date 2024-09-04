use super::operation::{MedRecordValueOperation, MedRecordValuesOperation};
use crate::{
    medrecord::{
        querying::{
            edges::EdgeOperation,
            nodes::NodeOperation,
            traits::{DeepClone, ReadWriteOrPanic},
        },
        EdgeOperand, MedRecordAttribute, MedRecordValue, NodeOperand, Wrapper,
    },
    MedRecord,
};

#[derive(Debug, Clone)]
pub enum MedRecordValueComparisonOperand {
    SingleOperand(MedRecordValueOperand),
    SingleValue(MedRecordValue),
    MultipleOperand(MedRecordValuesOperand),
    MultipleValues(Vec<MedRecordValue>),
}

impl DeepClone for MedRecordValueComparisonOperand {
    fn deep_clone(&self) -> Self {
        match self {
            Self::SingleOperand(operand) => Self::SingleOperand(operand.deep_clone()),
            Self::SingleValue(value) => Self::SingleValue(value.clone()),
            Self::MultipleOperand(operand) => Self::MultipleOperand(operand.deep_clone()),
            Self::MultipleValues(values) => Self::MultipleValues(values.clone()),
        }
    }
}

impl From<Wrapper<MedRecordValueOperand>> for MedRecordValueComparisonOperand {
    fn from(value: Wrapper<MedRecordValueOperand>) -> Self {
        Self::SingleOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<MedRecordValueOperand>> for MedRecordValueComparisonOperand {
    fn from(value: &Wrapper<MedRecordValueOperand>) -> Self {
        Self::SingleOperand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<MedRecordValue>> From<V> for MedRecordValueComparisonOperand {
    fn from(value: V) -> Self {
        Self::SingleValue(value.into())
    }
}

impl From<Wrapper<MedRecordValuesOperand>> for MedRecordValueComparisonOperand {
    fn from(value: Wrapper<MedRecordValuesOperand>) -> Self {
        Self::MultipleOperand(value.0.read_or_panic().deep_clone())
    }
}

impl From<&Wrapper<MedRecordValuesOperand>> for MedRecordValueComparisonOperand {
    fn from(value: &Wrapper<MedRecordValuesOperand>) -> Self {
        Self::MultipleOperand(value.0.read_or_panic().deep_clone())
    }
}

impl<V: Into<MedRecordValue>> From<Vec<V>> for MedRecordValueComparisonOperand {
    fn from(value: Vec<V>) -> Self {
        Self::MultipleValues(value.into_iter().map(Into::into).collect())
    }
}

impl<V: Into<MedRecordValue> + Clone, const N: usize> From<[V; N]>
    for MedRecordValueComparisonOperand
{
    fn from(value: [V; N]) -> Self {
        value.to_vec().into()
    }
}

#[derive(Debug, Clone)]
pub enum ValueKind {
    Max,
    Min,
}

#[derive(Debug, Clone)]
pub enum Context {
    NodeOperand(NodeOperand),
    EdgeOperand(EdgeOperand),
}

impl Context {
    pub(crate) fn get_values<'a>(
        &self,
        medrecord: &'a MedRecord,
        attribute: MedRecordAttribute,
    ) -> Box<dyn Iterator<Item = &'a MedRecordValue> + 'a> {
        match self {
            Self::NodeOperand(node_operand) => {
                let node_indices = node_operand.evaluate(medrecord);

                Box::new(
                    NodeOperation::get_values(medrecord, node_indices, attribute)
                        .map(|(_, value)| value),
                )
            }
            Self::EdgeOperand(edge_operand) => {
                let edge_indices = edge_operand.evaluate(medrecord);

                Box::new(
                    EdgeOperation::get_values(medrecord, edge_indices, attribute)
                        .map(|(_, value)| value),
                )
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct MedRecordValuesOperand {
    pub(crate) context: Context,
    pub(crate) attribute: MedRecordAttribute,
    operations: Vec<MedRecordValuesOperation>,
}

impl DeepClone for MedRecordValuesOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            attribute: self.attribute.clone(),
            operations: self
                .operations
                .iter()
                .map(|operation| operation.deep_clone())
                .collect(),
        }
    }
}

impl MedRecordValuesOperand {
    pub(crate) fn new(context: Context, attribute: MedRecordAttribute) -> Self {
        Self {
            context,
            attribute,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate<'a, T: 'a>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a T, &'a MedRecordValue)> + 'a,
    ) -> impl Iterator<Item = (&'a T, &'a MedRecordValue)> {
        let values = Box::new(values) as Box<dyn Iterator<Item = (&'a T, &'a MedRecordValue)>>;

        self.operations
            .iter()
            .fold(values, |edge_indices, operation| {
                operation.evaluate(medrecord, edge_indices)
            })
    }

    pub fn max(&mut self) -> Wrapper<MedRecordValueOperand> {
        let operand = Wrapper::<MedRecordValueOperand>::new(self.deep_clone(), ValueKind::Max);

        self.operations
            .push(MedRecordValuesOperation::ValueOperand {
                operand: operand.clone(),
            });

        operand
    }

    pub fn min(&mut self) -> Wrapper<MedRecordValueOperand> {
        let operand = Wrapper::<MedRecordValueOperand>::new(self.deep_clone(), ValueKind::Min);

        self.operations
            .push(MedRecordValuesOperation::ValueOperand {
                operand: operand.clone(),
            });

        operand
    }

    pub fn less_than<V: Into<MedRecordValueComparisonOperand>>(&mut self, value: V) {
        self.operations.push(MedRecordValuesOperation::LessThan {
            value: value.into(),
        });
    }
}

impl Wrapper<MedRecordValuesOperand> {
    pub(crate) fn new(context: Context, attribute: MedRecordAttribute) -> Self {
        MedRecordValuesOperand::new(context, attribute).into()
    }

    pub(crate) fn evaluate<'a, T: 'a>(
        &self,
        medrecord: &'a MedRecord,
        values: impl Iterator<Item = (&'a T, &'a MedRecordValue)> + 'a,
    ) -> impl Iterator<Item = (&'a T, &'a MedRecordValue)> {
        self.0.read_or_panic().evaluate(medrecord, values)
    }

    pub fn max(&self) -> Wrapper<MedRecordValueOperand> {
        self.0.write_or_panic().max()
    }

    pub fn min(&self) -> Wrapper<MedRecordValueOperand> {
        self.0.write_or_panic().min()
    }

    pub fn less_than<V: Into<MedRecordValueComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().less_than(value)
    }
}

#[derive(Debug, Clone)]
pub struct MedRecordValueOperand {
    pub(crate) context: MedRecordValuesOperand,
    pub(crate) kind: ValueKind,
    operations: Vec<MedRecordValueOperation>,
}

impl DeepClone for MedRecordValueOperand {
    fn deep_clone(&self) -> Self {
        Self {
            context: self.context.deep_clone(),
            kind: self.kind.clone(),
            operations: self
                .operations
                .iter()
                .map(|operation| operation.deep_clone())
                .collect(),
        }
    }
}

impl MedRecordValueOperand {
    pub(crate) fn new(context: MedRecordValuesOperand, kind: ValueKind) -> Self {
        Self {
            context,
            kind,
            operations: Vec::new(),
        }
    }

    pub(crate) fn evaluate<'a>(&self, medrecord: &'a MedRecord, value: &'a MedRecordValue) -> bool {
        self.operations
            .iter()
            .all(|operation| operation.evaluate(medrecord, value))
    }

    pub fn less_than<V: Into<MedRecordValueComparisonOperand>>(&mut self, value: V) {
        self.operations.push(MedRecordValueOperation::LessThan {
            value: value.into(),
        });
    }
}

impl Wrapper<MedRecordValueOperand> {
    pub(crate) fn new(context: MedRecordValuesOperand, kind: ValueKind) -> Self {
        MedRecordValueOperand::new(context, kind).into()
    }

    pub(crate) fn evaluate<'a>(&self, medrecord: &'a MedRecord, value: &'a MedRecordValue) -> bool {
        self.0.read_or_panic().evaluate(medrecord, value)
    }

    pub fn less_than<V: Into<MedRecordValueComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().less_than(value)
    }
}
