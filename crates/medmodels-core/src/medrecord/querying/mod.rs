pub mod attributes;
pub mod edges;
pub mod group_by;
pub mod nodes;
mod operand_traits;
pub mod values;
pub mod wrapper;

use super::{EdgeIndex, MedRecord, MedRecordAttribute, MedRecordValue, NodeIndex, Wrapper};
use crate::{
    errors::MedRecordResult,
    medrecord::querying::group_by::{GroupBy, GroupKey, PartitionGroups},
};
use attributes::{
    EdgeAttributesTreeOperand, EdgeMultipleAttributesOperand, EdgeSingleAttributeOperand,
    GetAllAttributes, GetAttributes, NodeAttributesTreeOperand, NodeMultipleAttributesOperand,
    NodeSingleAttributeOperand,
};
use edges::{EdgeIndexOperand, EdgeIndicesOperand, EdgeOperand};
use group_by::{GroupOperand, GroupedOperand};
use nodes::{NodeIndexOperand, NodeIndicesOperand, NodeOperand};
use std::{
    fmt::{Debug, Display},
    hash::Hash,
    sync::{RwLock, RwLockReadGuard, RwLockWriteGuard},
};
use values::{
    EdgeMultipleValuesOperand, EdgeSingleValueOperand, GetValues, NodeMultipleValuesOperand,
    NodeSingleValueOperand,
};

macro_rules! impl_return_operand_for_tuples {
    ($($T:ident),+) => {
        impl<'a, $($T: ReturnOperand<'a>),+> ReturnOperand<'a> for ($($T,)+) {
            type ReturnValue = ($($T::ReturnValue,)+);

            #[allow(non_snake_case)]
            fn evaluate(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
                let ($($T,)+) = self;

                $(let $T = $T.evaluate(medrecord)?;)+

                Ok(($($T,)+))
            }
        }
    };
}

macro_rules! impl_iterator_return_operand {
    ($( $Operand:ident => $Item:ty ),* $(,)?) => {
        $(
            impl<'a> ReturnOperand<'a> for Wrapper<$Operand> {
                type ReturnValue = Box<dyn Iterator<Item = $Item> + 'a>;

                fn evaluate(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
                    Ok(Box::new(self.evaluate_backward(medrecord)?))
                }
            }
        )*
    };
}

macro_rules! impl_direct_return_operand {
    ($( $Operand:ident => $ReturnValue:ty ),* $(,)?) => {
        $(
            impl<'a> ReturnOperand<'a> for Wrapper<$Operand> {
                type ReturnValue = $ReturnValue;

                fn evaluate(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
                    self.evaluate_backward(medrecord)
                }
            }
        )*
    };
}

pub trait Index: Eq + Clone + Hash + Display + GetAttributes {}

impl Index for NodeIndex {}

impl Index for EdgeIndex {}

impl<I: Index> Index for &I {}

pub trait RootOperand:
    GetAllAttributes<Self::Index> + GetValues<Self::Index> + GroupedOperand + Debug + Clone + DeepClone
{
    type Index: Index;
    type Discriminator: Debug + Clone + DeepClone;

    fn _evaluate_forward<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: BoxedIterator<'a, &'a Self::Index>,
    ) -> MedRecordResult<BoxedIterator<'a, &'a Self::Index>>;

    fn _evaluate_forward_grouped<'a>(
        &self,
        medrecord: &'a MedRecord,
        indices: BoxedIterator<'a, BoxedIterator<'a, &'a Self::Index>>,
    ) -> MedRecordResult<BoxedIterator<'a, BoxedIterator<'a, &'a Self::Index>>>;

    fn _evaluate_backward<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, &'a Self::Index>>;

    fn _evaluate_backward_grouped_operand<'a>(
        group_operand: &GroupOperand<Self>,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, BoxedIterator<'a, &'a Self::Index>>>;

    fn _group_by(&mut self, discriminator: Self::Discriminator) -> Wrapper<GroupOperand<Self>>;

    fn _partition<'a>(
        medrecord: &'a MedRecord,
        indices: BoxedIterator<'a, &'a Self::Index>,
        discriminator: &Self::Discriminator,
    ) -> BoxedIterator<'a, (GroupKey<'a>, BoxedIterator<'a, &'a Self::Index>)>;

    fn _merge<'a>(
        indices: BoxedIterator<'a, BoxedIterator<'a, &'a Self::Index>>,
    ) -> BoxedIterator<'a, &'a Self::Index>;
}

impl<'a, O: RootOperand> EvaluateForward<'a> for O
where
    O: 'a,
{
    type InputValue = BoxedIterator<'a, &'a O::Index>;
    type ReturnValue = BoxedIterator<'a, &'a O::Index>;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        indices: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        self._evaluate_forward(medrecord, indices)
    }
}

impl<'a, O: RootOperand> EvaluateForwardGrouped<'a> for O
where
    O: 'a,
{
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        indices: BoxedIterator<'a, BoxedIterator<'a, &'a O::Index>>,
    ) -> MedRecordResult<BoxedIterator<'a, BoxedIterator<'a, &'a O::Index>>> {
        self._evaluate_forward_grouped(medrecord, indices)
    }
}

impl<'a, O: RootOperand> EvaluateBackward<'a> for O
where
    O: 'a,
{
    type ReturnValue = BoxedIterator<'a, &'a O::Index>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        self._evaluate_backward(medrecord)
    }
}

impl<'a, O: RootOperand> EvaluateBackward<'a> for GroupOperand<O>
where
    O: 'a,
{
    type ReturnValue = BoxedIterator<'a, BoxedIterator<'a, &'a O::Index>>;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        O::_evaluate_backward_grouped_operand(self, medrecord)
    }
}

impl<O: RootOperand> GroupBy for O {
    type Discriminator = <Self as RootOperand>::Discriminator;

    fn group_by(&mut self, discriminator: Self::Discriminator) -> Wrapper<GroupOperand<Self>>
    where
        Self: Sized,
    {
        self._group_by(discriminator)
    }
}

impl<'a, O: RootOperand> PartitionGroups<'a> for O
where
    O: 'a,
{
    type Values = BoxedIterator<'a, &'a O::Index>;

    fn partition(
        medrecord: &'a MedRecord,
        indices: Self::Values,
        discriminator: &Self::Discriminator,
    ) -> BoxedIterator<'a, (GroupKey<'a>, Self::Values)> {
        Self::_partition(medrecord, indices, discriminator)
    }

    fn merge(indices: BoxedIterator<'a, Self::Values>) -> Self::Values {
        Self::_merge(indices)
    }
}

pub trait EvaluateForward<'a> {
    type InputValue;
    type ReturnValue;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        values: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue>;
}

impl<'a, O: EvaluateForward<'a>> EvaluateForward<'a> for Wrapper<O> {
    type InputValue = O::InputValue;
    type ReturnValue = O::ReturnValue;

    fn evaluate_forward(
        &self,
        medrecord: &'a MedRecord,
        values: Self::InputValue,
    ) -> MedRecordResult<Self::ReturnValue> {
        self.0.read_or_panic().evaluate_forward(medrecord, values)
    }
}

pub trait EvaluateForwardGrouped<'a>: EvaluateForward<'a> {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        values: BoxedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<BoxedIterator<'a, Self::ReturnValue>>;
}

impl<'a, O: EvaluateForwardGrouped<'a>> EvaluateForwardGrouped<'a> for Wrapper<O> {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        values: BoxedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<BoxedIterator<'a, Self::ReturnValue>> {
        self.0
            .read_or_panic()
            .evaluate_forward_grouped(medrecord, values)
    }
}

pub trait EvaluateBackward<'a> {
    type ReturnValue;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue>;
}

impl<'a, O: EvaluateBackward<'a>> EvaluateBackward<'a> for Wrapper<O> {
    type ReturnValue = O::ReturnValue;

    fn evaluate_backward(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        self.0.read_or_panic().evaluate_backward(medrecord)
    }
}

pub trait ReduceInput<'a>: EvaluateForward<'a> {
    type Context: EvaluateBackward<'a>;

    fn reduce_input(
        &self,
        values: <Self::Context as EvaluateBackward<'a>>::ReturnValue,
    ) -> MedRecordResult<<Self as EvaluateForward<'a>>::InputValue>;
}

impl<'a, O> Wrapper<O> {
    pub(crate) fn reduce_input(
        &self,
        values: <<O as ReduceInput<'a>>::Context as EvaluateBackward<'a>>::ReturnValue,
    ) -> MedRecordResult<<O as EvaluateForward<'a>>::InputValue>
    where
        O: ReduceInput<'a>,
    {
        self.0.read_or_panic().reduce_input(values)
    }
}

pub trait DeepClone {
    fn deep_clone(&self) -> Self;
}

impl<T: DeepClone> DeepClone for Option<T> {
    fn deep_clone(&self) -> Self {
        self.as_ref().map(|value| value.deep_clone())
    }
}

impl<T: DeepClone> DeepClone for Box<T> {
    fn deep_clone(&self) -> Self {
        Box::new(T::deep_clone(self))
    }
}

impl<T: DeepClone> DeepClone for Vec<T> {
    fn deep_clone(&self) -> Self {
        self.iter().map(|value| value.deep_clone()).collect()
    }
}

pub(crate) trait ReadWriteOrPanic<T> {
    fn read_or_panic(&self) -> RwLockReadGuard<'_, T>;

    fn write_or_panic(&self) -> RwLockWriteGuard<'_, T>;
}

impl<T> ReadWriteOrPanic<T> for RwLock<T> {
    fn read_or_panic(&self) -> RwLockReadGuard<'_, T> {
        self.read().unwrap()
    }

    fn write_or_panic(&self) -> RwLockWriteGuard<'_, T> {
        self.write().unwrap()
    }
}

pub(crate) type BoxedIterator<'a, T> = Box<dyn Iterator<Item = T> + 'a>;

#[derive(Debug, Clone)]
pub enum OptionalIndexWrapper<I: Index, T> {
    WithIndex((I, T)),
    WithoutIndex(T),
}

impl<I: Index, T> OptionalIndexWrapper<I, T> {
    pub fn get_value(&self) -> &T {
        match self {
            OptionalIndexWrapper::WithIndex((_, value)) => value,
            OptionalIndexWrapper::WithoutIndex(value) => value,
        }
    }

    pub fn get_index(&self) -> Option<&I> {
        match self {
            OptionalIndexWrapper::WithIndex((index, _)) => Some(index),
            OptionalIndexWrapper::WithoutIndex(_) => None,
        }
    }

    pub fn unpack(self) -> (Option<I>, T) {
        match self {
            OptionalIndexWrapper::WithIndex((index, value)) => (Some(index), value),
            OptionalIndexWrapper::WithoutIndex(value) => (None, value),
        }
    }

    pub fn map<U, F>(self, f: F) -> OptionalIndexWrapper<I, U>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            OptionalIndexWrapper::WithIndex((index, value)) => {
                OptionalIndexWrapper::WithIndex((index, f(value)))
            }
            OptionalIndexWrapper::WithoutIndex(value) => {
                OptionalIndexWrapper::WithoutIndex(f(value))
            }
        }
    }
}

impl<I: Index, T> From<T> for OptionalIndexWrapper<I, T> {
    fn from(value: T) -> Self {
        OptionalIndexWrapper::WithoutIndex(value)
    }
}

impl<I: Index, T> From<(I, T)> for OptionalIndexWrapper<I, T> {
    fn from(value: (I, T)) -> Self {
        OptionalIndexWrapper::WithIndex(value)
    }
}

#[derive(Debug, Clone)]
pub struct Selection<'a, R: ReturnOperand<'a>> {
    medrecord: &'a MedRecord,
    return_operand: R,
}

impl<'a, R: ReturnOperand<'a>> Selection<'a, R> {
    pub fn new_node<Q>(medrecord: &'a MedRecord, query: Q) -> Self
    where
        Q: FnOnce(&mut Wrapper<NodeOperand>) -> R,
    {
        let mut operand = Wrapper::<NodeOperand>::new(None);

        Self {
            medrecord,
            return_operand: query(&mut operand),
        }
    }

    pub fn new_edge<Q>(medrecord: &'a MedRecord, query: Q) -> Self
    where
        Q: FnOnce(&mut Wrapper<EdgeOperand>) -> R,
    {
        let mut operand = Wrapper::<EdgeOperand>::new(None);

        Self {
            medrecord,
            return_operand: query(&mut operand),
        }
    }

    pub fn evaluate(&self) -> MedRecordResult<R::ReturnValue> {
        self.return_operand.evaluate(self.medrecord)
    }
}

pub trait ReturnOperand<'a> {
    type ReturnValue;

    fn evaluate(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue>;
}

impl_iterator_return_operand!(
    NodeAttributesTreeOperand     => (&'a NodeIndex, Vec<MedRecordAttribute>),
    EdgeAttributesTreeOperand     => (&'a EdgeIndex, Vec<MedRecordAttribute>),
    NodeMultipleAttributesOperand => (&'a NodeIndex, MedRecordAttribute),
    EdgeMultipleAttributesOperand => (&'a EdgeIndex, MedRecordAttribute),
    EdgeIndicesOperand            => EdgeIndex,
    NodeIndicesOperand            => NodeIndex,
    NodeMultipleValuesOperand     => (&'a NodeIndex, MedRecordValue),
    EdgeMultipleValuesOperand     => (&'a EdgeIndex, MedRecordValue),
);

impl_direct_return_operand!(
    NodeSingleAttributeOperand => Option<OptionalIndexWrapper<&'a NodeIndex, MedRecordAttribute>>,
    EdgeSingleAttributeOperand => Option<OptionalIndexWrapper<&'a EdgeIndex, MedRecordAttribute>>,
    EdgeIndexOperand           => Option<EdgeIndex>,
    NodeIndexOperand           => Option<NodeIndex>,
    NodeSingleValueOperand     => Option<OptionalIndexWrapper<&'a NodeIndex, MedRecordValue>>,
    EdgeSingleValueOperand     => Option<OptionalIndexWrapper<&'a EdgeIndex, MedRecordValue>>,
);

impl<'a, O: GroupedOperand> ReturnOperand<'a> for Wrapper<GroupOperand<O>>
where
    GroupOperand<O>: EvaluateBackward<'a>,
    Wrapper<O>: ReturnOperand<'a>,
{
    type ReturnValue = <Self as EvaluateBackward<'a>>::ReturnValue;

    fn evaluate(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        self.evaluate_backward(medrecord)
    }
}

impl_return_operand_for_tuples!(R1);
impl_return_operand_for_tuples!(R1, R2);
impl_return_operand_for_tuples!(R1, R2, R3);
impl_return_operand_for_tuples!(R1, R2, R3, R4);
impl_return_operand_for_tuples!(R1, R2, R3, R4, R5);
impl_return_operand_for_tuples!(R1, R2, R3, R4, R5, R6);
impl_return_operand_for_tuples!(R1, R2, R3, R4, R5, R6, R7);
impl_return_operand_for_tuples!(R1, R2, R3, R4, R5, R6, R7, R8);
impl_return_operand_for_tuples!(R1, R2, R3, R4, R5, R6, R7, R8, R9);
impl_return_operand_for_tuples!(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10);
impl_return_operand_for_tuples!(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11);
impl_return_operand_for_tuples!(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12);
impl_return_operand_for_tuples!(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13);
impl_return_operand_for_tuples!(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14);
impl_return_operand_for_tuples!(R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12, R13, R14, R15);

impl<'a, R: ReturnOperand<'a>> ReturnOperand<'a> for &R {
    type ReturnValue = R::ReturnValue;

    fn evaluate(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        R::evaluate(self, medrecord)
    }
}

impl<'a, R: ReturnOperand<'a>> ReturnOperand<'a> for &mut R {
    type ReturnValue = R::ReturnValue;

    fn evaluate(&self, medrecord: &'a MedRecord) -> MedRecordResult<Self::ReturnValue> {
        R::evaluate(self, medrecord)
    }
}
