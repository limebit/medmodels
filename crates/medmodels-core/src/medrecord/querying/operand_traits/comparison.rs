use crate::medrecord::querying::{
    group_by::{GroupOperand, GroupedOperand},
    wrapper::Wrapper,
    ReadWriteOrPanic,
};

pub trait GreaterThan {
    type ComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: GreaterThan> Wrapper<O> {
    pub fn greater_than<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().greater_than(value);
    }
}

impl<O: GroupedOperand + GreaterThan> GreaterThan for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn greater_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.greater_than(value)
    }
}

pub trait GreaterThanOrEqualTo {
    type ComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: GreaterThanOrEqualTo> Wrapper<O> {
    pub fn greater_than_or_equal_to<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().greater_than_or_equal_to(value);
    }
}

impl<O: GroupedOperand + GreaterThanOrEqualTo> GreaterThanOrEqualTo for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn greater_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.greater_than_or_equal_to(value)
    }
}

pub trait LessThan {
    type ComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: LessThan> Wrapper<O> {
    pub fn less_than<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().less_than(value);
    }
}

impl<O: GroupedOperand + LessThan> LessThan for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn less_than<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.less_than(value)
    }
}

pub trait LessThanOrEqualTo {
    type ComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: LessThanOrEqualTo> Wrapper<O> {
    pub fn less_than_or_equal_to<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().less_than_or_equal_to(value);
    }
}

impl<O: GroupedOperand + LessThanOrEqualTo> LessThanOrEqualTo for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn less_than_or_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.less_than_or_equal_to(value)
    }
}

pub trait EqualTo {
    type ComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: EqualTo> Wrapper<O> {
    pub fn equal_to<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().equal_to(value);
    }
}

impl<O: GroupedOperand + EqualTo> EqualTo for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.equal_to(value)
    }
}

pub trait NotEqualTo {
    type ComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: NotEqualTo> Wrapper<O> {
    pub fn not_equal_to<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().not_equal_to(value);
    }
}

impl<O: GroupedOperand + NotEqualTo> NotEqualTo for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn not_equal_to<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.not_equal_to(value)
    }
}
