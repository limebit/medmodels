use super::{wrapper::Wrapper, ReadWriteOrPanic};
use crate::prelude::MedRecordAttribute;

pub trait Attribute {
    type ReturnOperand;

    fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<Self::ReturnOperand>;
}

impl<O: Attribute> Wrapper<O> {
    pub fn attribute(&self, attribute: impl Into<MedRecordAttribute>) -> Wrapper<O::ReturnOperand> {
        self.0.write_or_panic().attribute(attribute.into())
    }
}

pub trait Max {
    type ReturnOperand;

    fn max(&mut self) -> Wrapper<Self::ReturnOperand>;
}

impl<O: Max> Wrapper<O> {
    pub fn max(&self) -> Wrapper<O::ReturnOperand> {
        self.0.write_or_panic().max()
    }
}

pub trait Count {
    type ReturnOperand;

    fn count(&mut self) -> Wrapper<Self::ReturnOperand>;
}

impl<O: Count> Wrapper<O> {
    pub fn count(&self) -> Wrapper<O::ReturnOperand> {
        self.0.write_or_panic().count()
    }
}

pub trait Index {
    type ReturnOperand;

    fn index(&mut self) -> Wrapper<Self::ReturnOperand>;
}

impl<O: Index> Wrapper<O> {
    pub fn index(&self) -> Wrapper<O::ReturnOperand> {
        self.0.write_or_panic().index()
    }
}
