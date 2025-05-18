use crate::medrecord::querying::DeepClone;

use super::operand::GroupableOperand;

#[derive(Debug, Clone)]
pub enum GroupByOperation<O: GroupableOperand> {
    ForEach { operand: O },
}

impl<O: GroupableOperand> DeepClone for GroupByOperation<O> {
    fn deep_clone(&self) -> Self {
        match self {
            GroupByOperation::ForEach { operand } => GroupByOperation::ForEach {
                operand: operand.deep_clone(),
            },
        }
    }
}
