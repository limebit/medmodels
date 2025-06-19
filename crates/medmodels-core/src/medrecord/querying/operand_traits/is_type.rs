use crate::medrecord::querying::{
    group_by::{GroupOperand, GroupedOperand},
    wrapper::Wrapper,
    ReadWriteOrPanic,
};

pub trait IsFloat {
    fn is_float(&mut self);
}

impl<O: IsFloat> Wrapper<O> {
    pub fn is_float(&self) {
        self.0.write_or_panic().is_float();
    }
}

impl<O: GroupedOperand + IsFloat> IsFloat for GroupOperand<O> {
    fn is_float(&mut self) {
        self.operand.is_float()
    }
}

pub trait IsBool {
    fn is_bool(&mut self);
}

impl<O: IsBool> Wrapper<O> {
    pub fn is_bool(&self) {
        self.0.write_or_panic().is_bool();
    }
}

impl<O: GroupedOperand + IsBool> IsBool for GroupOperand<O> {
    fn is_bool(&mut self) {
        self.operand.is_bool()
    }
}

pub trait IsDateTime {
    fn is_datetime(&mut self);
}

impl<O: IsDateTime> Wrapper<O> {
    pub fn is_datetime(&self) {
        self.0.write_or_panic().is_datetime();
    }
}

impl<O: GroupedOperand + IsDateTime> IsDateTime for GroupOperand<O> {
    fn is_datetime(&mut self) {
        self.operand.is_datetime()
    }
}

pub trait IsDuration {
    fn is_duration(&mut self);
}

impl<O: IsDuration> Wrapper<O> {
    pub fn is_duration(&self) {
        self.0.write_or_panic().is_duration();
    }
}

impl<O: GroupedOperand + IsDuration> IsDuration for GroupOperand<O> {
    fn is_duration(&mut self) {
        self.operand.is_duration()
    }
}

pub trait IsNull {
    fn is_null(&mut self);
}

impl<O: IsNull> Wrapper<O> {
    pub fn is_null(&self) {
        self.0.write_or_panic().is_null();
    }
}

impl<O: GroupedOperand + IsNull> IsNull for GroupOperand<O> {
    fn is_null(&mut self) {
        self.operand.is_null()
    }
}

pub trait IsString {
    fn is_string(&mut self);
}

impl<O: IsString> Wrapper<O> {
    pub fn is_string(&self) {
        self.0.write_or_panic().is_string();
    }
}

impl<O: GroupedOperand + IsString> IsString for GroupOperand<O> {
    fn is_string(&mut self) {
        self.operand.is_string()
    }
}

pub trait IsInt {
    fn is_int(&mut self);
}

impl<O: IsInt> Wrapper<O> {
    pub fn is_int(&self) {
        self.0.write_or_panic().is_int();
    }
}

impl<O: GroupedOperand + IsInt> IsInt for GroupOperand<O> {
    fn is_int(&mut self) {
        self.operand.is_int()
    }
}
