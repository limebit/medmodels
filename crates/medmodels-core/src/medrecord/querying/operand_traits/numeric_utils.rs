use crate::medrecord::querying::{
    group_by::{GroupOperand, GroupedOperand},
    wrapper::Wrapper,
    ReadWriteOrPanic,
};

pub trait Round {
    fn round(&mut self);
}

impl<O: Round> Wrapper<O> {
    pub fn round(&self) {
        self.0.write_or_panic().round();
    }
}

impl<O: GroupedOperand + Round> Round for GroupOperand<O> {
    fn round(&mut self) {
        self.operand.round()
    }
}

pub trait Ceil {
    fn ceil(&mut self);
}

impl<O: Ceil> Wrapper<O> {
    pub fn ceil(&self) {
        self.0.write_or_panic().ceil();
    }
}

impl<O: GroupedOperand + Ceil> Ceil for GroupOperand<O> {
    fn ceil(&mut self) {
        self.operand.ceil()
    }
}

pub trait Floor {
    fn floor(&mut self);
}

impl<O: Floor> Wrapper<O> {
    pub fn floor(&self) {
        self.0.write_or_panic().floor();
    }
}

impl<O: GroupedOperand + Floor> Floor for GroupOperand<O> {
    fn floor(&mut self) {
        self.operand.floor()
    }
}

pub trait Sqrt {
    fn sqrt(&mut self);
}

impl<O: Sqrt> Wrapper<O> {
    pub fn sqrt(&self) {
        self.0.write_or_panic().sqrt();
    }
}

impl<O: GroupedOperand + Sqrt> Sqrt for GroupOperand<O> {
    fn sqrt(&mut self) {
        self.operand.sqrt()
    }
}
