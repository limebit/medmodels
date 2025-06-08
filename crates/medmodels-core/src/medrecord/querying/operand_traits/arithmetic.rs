use crate::medrecord::querying::{
    group_by::{GroupOperand, GroupedOperand},
    wrapper::Wrapper,
    ReadWriteOrPanic,
};

pub trait Add {
    type ComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: Add> Wrapper<O> {
    pub fn add<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().add(value);
    }
}

impl<O: GroupedOperand + Add> Add for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn add<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.add(value)
    }
}

pub trait Sub {
    type ComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: Sub> Wrapper<O> {
    pub fn sub<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().sub(value);
    }
}

impl<O: GroupedOperand + Sub> Sub for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn sub<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.sub(value)
    }
}

pub trait Mul {
    type ComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: Mul> Wrapper<O> {
    pub fn mul<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().mul(value);
    }
}

impl<O: GroupedOperand + Mul> Mul for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn mul<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.mul(value)
    }
}

pub trait Div {
    type ComparisonOperand;

    fn div<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: Div> Wrapper<O> {
    pub fn div<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().div(value);
    }
}

impl<O: GroupedOperand + Div> Div for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn div<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.div(value)
    }
}

pub trait Pow {
    type ComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: Pow> Wrapper<O> {
    pub fn pow<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().pow(value);
    }
}

impl<O: GroupedOperand + Pow> Pow for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn pow<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.pow(value)
    }
}

pub trait Mod {
    type ComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: Mod> Wrapper<O> {
    pub fn r#mod<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().r#mod(value);
    }
}

impl<O: GroupedOperand + Mod> Mod for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn r#mod<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.r#mod(value)
    }
}

pub trait Abs {
    fn abs(&mut self);
}

impl<O: Abs> Wrapper<O> {
    pub fn abs(&self) {
        self.0.write_or_panic().abs();
    }
}

impl<O: GroupedOperand + Abs> Abs for GroupOperand<O> {
    fn abs(&mut self) {
        self.operand.abs();
    }
}
