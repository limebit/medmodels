mod operand;

use super::DeepClone;
pub use operand::GroupByOperand;
pub(crate) use operand::{GroupableOperand, PartitionGroups};

#[derive(Debug, Clone)]
pub struct Context<CO: GroupableOperand> {
    operand: CO,
    discriminator: CO::Discriminator,
}

impl<CO: GroupableOperand> DeepClone for Context<CO> {
    fn deep_clone(&self) -> Self {
        Self {
            operand: self.operand.deep_clone(),
            discriminator: self.discriminator.clone(),
        }
    }
}

impl<CO: GroupableOperand> Context<CO> {
    pub(crate) fn new(operand: CO, discriminator: CO::Discriminator) -> Self {
        Self {
            operand,
            discriminator,
        }
    }
}
