use crate::medrecord::querying::{
    group_by::{GroupOperand, GroupedOperand},
    wrapper::Wrapper,
    DeepClone, ReadWriteOrPanic,
};

pub trait Index {
    type ReturnOperand;

    fn index(&mut self) -> Wrapper<Self::ReturnOperand>;
}

impl<O: Index> Wrapper<O> {
    pub fn index(&self) -> Wrapper<O::ReturnOperand> {
        self.0.write_or_panic().index()
    }
}

impl<O: GroupedOperand + Index> Index for GroupOperand<O>
where
    Self: DeepClone,
    O::ReturnOperand: GroupedOperand,
    <O::ReturnOperand as GroupedOperand>::Context: From<Self>,
{
    type ReturnOperand = GroupOperand<O::ReturnOperand>;

    fn index(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.index();

        Wrapper::<GroupOperand<O::ReturnOperand>>::new(self.deep_clone().into(), operand)
    }
}
