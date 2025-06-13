use crate::medrecord::querying::{
    group_by::{GroupOperand, GroupedOperand},
    wrapper::Wrapper,
    ReadWriteOrPanic,
};

pub trait IsIn {
    type ComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V);
}

impl<O: IsIn> Wrapper<O> {
    pub fn is_in<V: Into<O::ComparisonOperand>>(&self, values: V) {
        self.0.write_or_panic().is_in(values);
    }
}

impl<O: GroupedOperand + IsIn> IsIn for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn is_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operand.is_in(values)
    }
}

pub trait IsNotIn {
    type ComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V);
}

impl<O: IsNotIn> Wrapper<O> {
    pub fn is_not_in<V: Into<O::ComparisonOperand>>(&self, values: V) {
        self.0.write_or_panic().is_not_in(values);
    }
}

impl<O: GroupedOperand + IsNotIn> IsNotIn for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn is_not_in<V: Into<Self::ComparisonOperand>>(&mut self, values: V) {
        self.operand.is_not_in(values)
    }
}
