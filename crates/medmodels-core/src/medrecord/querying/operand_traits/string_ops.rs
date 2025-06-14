use crate::medrecord::querying::{
    group_by::{GroupOperand, GroupedOperand},
    wrapper::Wrapper,
    ReadWriteOrPanic,
};

pub trait Trim {
    fn trim(&mut self);
}

impl<O: Trim> Wrapper<O> {
    pub fn trim(&self) {
        self.0.write_or_panic().trim();
    }
}

impl<O: GroupedOperand + Trim> Trim for GroupOperand<O> {
    fn trim(&mut self) {
        self.operand.trim();
    }
}

pub trait TrimStart {
    fn trim_start(&mut self);
}

impl<O: TrimStart> Wrapper<O> {
    pub fn trim_start(&self) {
        self.0.write_or_panic().trim_start();
    }
}

impl<O: GroupedOperand + TrimStart> TrimStart for GroupOperand<O> {
    fn trim_start(&mut self) {
        self.operand.trim_start();
    }
}

pub trait TrimEnd {
    fn trim_end(&mut self);
}

impl<O: TrimEnd> Wrapper<O> {
    pub fn trim_end(&self) {
        self.0.write_or_panic().trim_end();
    }
}

impl<O: GroupedOperand + TrimEnd> TrimEnd for GroupOperand<O> {
    fn trim_end(&mut self) {
        self.operand.trim_end();
    }
}

pub trait Lowercase {
    fn lowercase(&mut self);
}

impl<O: Lowercase> Wrapper<O> {
    pub fn lowercase(&self) {
        self.0.write_or_panic().lowercase();
    }
}

impl<O: GroupedOperand + Lowercase> Lowercase for GroupOperand<O> {
    fn lowercase(&mut self) {
        self.operand.lowercase();
    }
}

pub trait Uppercase {
    fn uppercase(&mut self);
}

impl<O: Uppercase> Wrapper<O> {
    pub fn uppercase(&self) {
        self.0.write_or_panic().uppercase();
    }
}

impl<O: GroupedOperand + Uppercase> Uppercase for GroupOperand<O> {
    fn uppercase(&mut self) {
        self.operand.uppercase();
    }
}

pub trait Slice {
    fn slice(&mut self, start: usize, end: usize);
}

impl<O: Slice> Wrapper<O> {
    pub fn slice(&self, start: usize, end: usize) {
        self.0.write_or_panic().slice(start, end);
    }
}

impl<O: GroupedOperand + Slice> Slice for GroupOperand<O> {
    fn slice(&mut self, start: usize, end: usize) {
        self.operand.slice(start, end);
    }
}

pub trait StartsWith {
    type ComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: StartsWith> Wrapper<O> {
    pub fn starts_with<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().starts_with(value);
    }
}

impl<O: GroupedOperand + StartsWith> StartsWith for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn starts_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.starts_with(value);
    }
}

pub trait EndsWith {
    type ComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: EndsWith> Wrapper<O> {
    pub fn ends_with<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().ends_with(value);
    }
}

impl<O: GroupedOperand + EndsWith> EndsWith for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn ends_with<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.ends_with(value);
    }
}

pub trait Contains {
    type ComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V);
}

impl<O: Contains> Wrapper<O> {
    pub fn contains<V: Into<O::ComparisonOperand>>(&self, value: V) {
        self.0.write_or_panic().contains(value);
    }
}

impl<O: GroupedOperand + Contains> Contains for GroupOperand<O> {
    type ComparisonOperand = O::ComparisonOperand;

    fn contains<V: Into<Self::ComparisonOperand>>(&mut self, value: V) {
        self.operand.contains(value);
    }
}
