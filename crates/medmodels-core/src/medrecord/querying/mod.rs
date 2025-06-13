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
    medrecord::querying::{
        attributes::{
            EdgeMultipleAttributesWithoutIndexOperand, EdgeSingleAttributeWithoutIndexOperand,
            NodeMultipleAttributesWithoutIndexOperand, NodeSingleAttributeWithoutIndexOperand,
        },
        group_by::{GroupBy, GroupKey, PartitionGroups},
        values::{EdgeSingleValueWithoutIndexOperand, NodeSingleValueWithoutIndexOperand},
    },
};
use attributes::{
    EdgeAttributesTreeOperand, EdgeMultipleAttributesWithIndexOperand,
    EdgeSingleAttributeWithIndexOperand, GetAllAttributes, GetAttributes,
    NodeAttributesTreeOperand, NodeMultipleAttributesWithIndexOperand,
    NodeSingleAttributeWithIndexOperand,
};
use edges::{EdgeIndexOperand, EdgeIndicesOperand, EdgeOperand};
use group_by::{GroupOperand, GroupedOperand};
use itertools::Itertools;
use nodes::{NodeIndexOperand, NodeIndicesOperand, NodeOperand};
use std::{
    fmt::{Debug, Display},
    hash::Hash,
    sync::{RwLock, RwLockReadGuard, RwLockWriteGuard},
};
use values::{
    EdgeMultipleValuesWithIndexOperand, EdgeMultipleValuesWithoutIndexOperand,
    EdgeSingleValueWithIndexOperand, GetValues, NodeMultipleValuesWithIndexOperand,
    NodeMultipleValuesWithoutIndexOperand, NodeSingleValueWithIndexOperand,
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
        indices: GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>>>;

    fn _evaluate_backward<'a>(
        &self,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<BoxedIterator<'a, &'a Self::Index>>;

    fn _evaluate_backward_grouped_operand<'a>(
        group_operand: &GroupOperand<Self>,
        medrecord: &'a MedRecord,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>>>;

    fn _group_by(&mut self, discriminator: Self::Discriminator) -> Wrapper<GroupOperand<Self>>;

    fn _partition<'a>(
        medrecord: &'a MedRecord,
        indices: BoxedIterator<'a, &'a Self::Index>,
        discriminator: &Self::Discriminator,
    ) -> GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>>;

    fn _merge<'a>(
        indices: GroupedIterator<'a, BoxedIterator<'a, &'a Self::Index>>,
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

pub type GroupedIterator<'a, O> = BoxedIterator<'a, (GroupKey<'a>, O)>;

pub(crate) fn tee_grouped_iterator<'a, O: 'a + Clone>(
    iterator: GroupedIterator<'a, BoxedIterator<'a, O>>,
) -> (
    GroupedIterator<'a, BoxedIterator<'a, O>>,
    GroupedIterator<'a, BoxedIterator<'a, O>>,
) {
    let mut iterators = (Vec::new(), Vec::new());

    iterator.for_each(|(key, inner_iterator)| {
        let (inner_iterator_1, inner_iterator_2) = Itertools::tee(inner_iterator);

        iterators
            .0
            .push((key.clone(), Box::new(inner_iterator_1) as BoxedIterator<_>));
        iterators
            .1
            .push((key, Box::new(inner_iterator_2) as BoxedIterator<_>));
    });

    (
        Box::new(iterators.0.into_iter()),
        Box::new(iterators.1.into_iter()),
    )
}

impl<'a, O: RootOperand> EvaluateForwardGrouped<'a> for O
where
    O: 'a,
{
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        indices: GroupedIterator<'a, BoxedIterator<'a, &'a O::Index>>,
    ) -> MedRecordResult<GroupedIterator<'a, BoxedIterator<'a, &'a O::Index>>> {
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
    type ReturnValue = GroupedIterator<'a, BoxedIterator<'a, &'a O::Index>>;

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
    ) -> GroupedIterator<'a, Self::Values> {
        Self::_partition(medrecord, indices, discriminator)
    }

    fn merge(indices: GroupedIterator<'a, Self::Values>) -> Self::Values {
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
        values: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>>;
}

impl<'a, O: EvaluateForwardGrouped<'a>> EvaluateForwardGrouped<'a> for Wrapper<O> {
    fn evaluate_forward_grouped(
        &self,
        medrecord: &'a MedRecord,
        values: GroupedIterator<'a, Self::InputValue>,
    ) -> MedRecordResult<GroupedIterator<'a, Self::ReturnValue>> {
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
    NodeAttributesTreeOperand                 => (&'a NodeIndex, Vec<MedRecordAttribute>),
    EdgeAttributesTreeOperand                 => (&'a EdgeIndex, Vec<MedRecordAttribute>),
    NodeMultipleAttributesWithIndexOperand    => (&'a NodeIndex, MedRecordAttribute),
    NodeMultipleAttributesWithoutIndexOperand => MedRecordAttribute,
    EdgeMultipleAttributesWithIndexOperand    => (&'a EdgeIndex, MedRecordAttribute),
    EdgeMultipleAttributesWithoutIndexOperand => MedRecordAttribute,
    EdgeIndicesOperand                        => EdgeIndex,
    NodeIndicesOperand                        => NodeIndex,
    NodeMultipleValuesWithIndexOperand        => (&'a NodeIndex, MedRecordValue),
    NodeMultipleValuesWithoutIndexOperand     => MedRecordValue,
    EdgeMultipleValuesWithIndexOperand        => (&'a EdgeIndex, MedRecordValue),
    EdgeMultipleValuesWithoutIndexOperand     => MedRecordValue,
);

impl_direct_return_operand!(
    NodeSingleAttributeWithIndexOperand    => Option<(&'a NodeIndex, MedRecordAttribute)>,
    NodeSingleAttributeWithoutIndexOperand => Option<MedRecordAttribute>,
    EdgeSingleAttributeWithIndexOperand    => Option<(&'a EdgeIndex, MedRecordAttribute)>,
    EdgeSingleAttributeWithoutIndexOperand => Option<MedRecordAttribute>,
    EdgeIndexOperand                       => Option<EdgeIndex>,
    NodeIndexOperand                       => Option<NodeIndex>,
    NodeSingleValueWithIndexOperand        => Option<(&'a NodeIndex, MedRecordValue)>,
    NodeSingleValueWithoutIndexOperand     => Option<MedRecordValue>,
    EdgeSingleValueWithIndexOperand        => Option<(&'a EdgeIndex, MedRecordValue)>,
    EdgeSingleValueWithoutIndexOperand     => Option<MedRecordValue>,
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
