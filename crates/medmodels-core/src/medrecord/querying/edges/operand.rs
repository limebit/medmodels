use super::{values::EdgeValuesOperand, EdgeOperation};
use crate::{
    medrecord::{
        querying::{
            evaluate::{EvaluateOperand, EvaluateOperation},
            nodes::NodeOperand,
            wrapper::{OperandContext, Wrapper},
        },
        EdgeIndex, MedRecordAttribute,
    },
    MedRecord,
};
use std::{cell::RefCell, fmt::Debug, rc::Rc};

#[derive(Debug, Clone)]
pub struct EdgeOperand {
    pub(crate) operations: Vec<EdgeOperation>,
}

impl EvaluateOperand for EdgeOperand {
    type Index = EdgeIndex;

    fn evaluate<'a>(
        &self,
        medrecord: &'a MedRecord,
        end_index: Option<usize>,
    ) -> Box<dyn Iterator<Item = &'a Self::Index> + 'a> {
        let edge_indices =
            Box::new(medrecord.edge_indices()) as Box<dyn Iterator<Item = &'a EdgeIndex>>;

        self.operations[0..end_index.unwrap_or(self.operations.len())]
            .iter()
            .fold(Box::new(edge_indices), |edge_indices, operation| {
                operation.evaluate(medrecord, edge_indices)
            })
    }
}

impl EdgeOperand {
    pub(crate) fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }

    pub fn connects_to<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<NodeOperand>),
    {
        let mut node_operand = Wrapper::<NodeOperand>::new();

        query(&mut node_operand);

        self.operations.push(EdgeOperation::ConnectsTo {
            operand: node_operand,
        });
    }

    pub fn attribute(
        &mut self,
        attribute: MedRecordAttribute,
        self_wrapper: &Wrapper<EdgeOperand>,
    ) -> Wrapper<EdgeValuesOperand> {
        let operand = Wrapper::<EdgeValuesOperand>::new(
            OperandContext::new(self_wrapper.clone(), self.operations.len()),
            attribute,
        );

        self.operations.push(EdgeOperation::Attribute {
            operand: operand.clone(),
        });

        operand
    }
}

impl Wrapper<EdgeOperand> {
    pub(crate) fn new() -> Self {
        Self(Rc::new(RefCell::new(EdgeOperand::new())))
    }

    pub fn connects_to<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<NodeOperand>),
    {
        self.0.borrow_mut().connects_to(query);
    }

    pub fn attribute<A>(&self, attribute: A) -> Wrapper<EdgeValuesOperand>
    where
        A: Into<MedRecordAttribute>,
    {
        self.0.borrow_mut().attribute(attribute.into(), self)
    }
}
