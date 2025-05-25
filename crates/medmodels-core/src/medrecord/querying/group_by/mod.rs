mod operand;

use super::DeepClone;
pub use operand::RootGroupOperand;
pub(crate) use operand::{
    EvaluateBackwardGrouped, GroupOperand, GroupableOperand, PartitionGroups,
};

#[derive(Debug, Clone)]
pub struct RootContext<CO: GroupableOperand> {
    operand: CO,
    discriminator: CO::Discriminator,
}

impl<CO: GroupableOperand> DeepClone for RootContext<CO> {
    fn deep_clone(&self) -> Self {
        Self {
            operand: self.operand.deep_clone(),
            discriminator: self.discriminator.clone(),
        }
    }
}

impl<CO: GroupableOperand> RootContext<CO> {
    pub(crate) fn new(operand: CO, discriminator: CO::Discriminator) -> Self {
        Self {
            operand,
            discriminator,
        }
    }
}
