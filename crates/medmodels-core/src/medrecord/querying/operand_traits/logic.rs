use crate::medrecord::querying::{
    group_by::{GroupOperand, GroupedOperand},
    wrapper::Wrapper,
    DeepClone, ReadWriteOrPanic,
};

pub trait EitherOr {
    type QueryOperand;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>);
}

impl<O: EitherOr> Wrapper<O> {
    pub fn either_or<EQ, OQ>(&self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<O::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<O::QueryOperand>),
    {
        self.0.write_or_panic().either_or(either_query, or_query);
    }
}

impl<O: GroupedOperand + EitherOr> EitherOr for GroupOperand<O> {
    type QueryOperand = O::QueryOperand;

    fn either_or<EQ, OQ>(&mut self, either_query: EQ, or_query: OQ)
    where
        EQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
        OQ: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        self.operand.either_or(either_query, or_query)
    }
}

pub trait Exclude {
    type QueryOperand;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>);
}

impl<O: Exclude> Wrapper<O> {
    pub fn exclude<Q>(&self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<O::QueryOperand>),
    {
        self.0.write_or_panic().exclude(query);
    }
}

impl<O: GroupedOperand + Exclude> Exclude for GroupOperand<O> {
    type QueryOperand = O::QueryOperand;

    fn exclude<Q>(&mut self, query: Q)
    where
        Q: FnOnce(&mut Wrapper<Self::QueryOperand>),
    {
        self.operand.exclude(query)
    }
}

pub trait Random {
    type ReturnOperand;

    fn random(&mut self) -> Wrapper<Self::ReturnOperand>;
}

impl<O: Random> Wrapper<O> {
    pub fn random(&self) -> Wrapper<O::ReturnOperand> {
        self.0.write_or_panic().random()
    }
}

impl<O: GroupedOperand + Random> Random for GroupOperand<O>
where
    Self: DeepClone,
    O::ReturnOperand: GroupedOperand,
    <O::ReturnOperand as GroupedOperand>::Context: From<Self>,
{
    type ReturnOperand = GroupOperand<O::ReturnOperand>;

    fn random(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.random();

        Wrapper::<GroupOperand<O::ReturnOperand>>::new(self.deep_clone().into(), operand)
    }
}

pub trait IsMax {
    fn is_max(&mut self);
}

impl<O: IsMax> Wrapper<O> {
    pub fn is_max(&self) {
        self.0.write_or_panic().is_max();
    }
}

impl<O: GroupedOperand + IsMax> IsMax for GroupOperand<O> {
    fn is_max(&mut self) {
        self.operand.is_max()
    }
}

pub trait IsMin {
    fn is_min(&mut self);
}

impl<O: IsMin> Wrapper<O> {
    pub fn is_min(&self) {
        self.0.write_or_panic().is_min();
    }
}

impl<O: GroupedOperand + IsMin> IsMin for GroupOperand<O> {
    fn is_min(&mut self) {
        self.operand.is_min()
    }
}
