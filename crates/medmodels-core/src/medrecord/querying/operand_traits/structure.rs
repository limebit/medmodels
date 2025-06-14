use crate::{
    medrecord::querying::{
        group_by::{GroupOperand, GroupedOperand},
        wrapper::{CardinalityWrapper, Wrapper},
        DeepClone, ReadWriteOrPanic,
    },
    prelude::{EdgeDirection, Group, MedRecordAttribute},
};

pub trait Attribute {
    type ReturnOperand;

    fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<Self::ReturnOperand>;
}

impl<O: Attribute> Wrapper<O> {
    pub fn attribute(&self, attribute: impl Into<MedRecordAttribute>) -> Wrapper<O::ReturnOperand> {
        self.0.write_or_panic().attribute(attribute.into())
    }
}

impl<O: GroupedOperand + Attribute> Attribute for GroupOperand<O>
where
    Self: DeepClone,
    O::ReturnOperand: GroupedOperand,
    <O::ReturnOperand as GroupedOperand>::Context: From<Self>,
{
    type ReturnOperand = GroupOperand<O::ReturnOperand>;

    fn attribute(&mut self, attribute: MedRecordAttribute) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.attribute(attribute);

        Wrapper::<GroupOperand<O::ReturnOperand>>::new(self.deep_clone().into(), operand)
    }
}

pub trait Attributes {
    type ReturnOperand;

    fn attributes(&mut self) -> Wrapper<Self::ReturnOperand>;
}

impl<O: Attributes> Wrapper<O> {
    pub fn attributes(&self) -> Wrapper<O::ReturnOperand> {
        self.0.write_or_panic().attributes()
    }
}

impl<O: GroupedOperand + Attributes> Attributes for GroupOperand<O>
where
    Self: DeepClone,
    O::ReturnOperand: GroupedOperand,
    <O::ReturnOperand as GroupedOperand>::Context: From<Self>,
{
    type ReturnOperand = GroupOperand<O::ReturnOperand>;

    fn attributes(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.attributes();

        Wrapper::<GroupOperand<O::ReturnOperand>>::new(self.deep_clone().into(), operand)
    }
}

pub trait HasAttribute {
    fn has_attribute<A: Into<CardinalityWrapper<MedRecordAttribute>>>(&mut self, attribute: A);
}

impl<O: HasAttribute> Wrapper<O> {
    pub fn has_attribute<A: Into<CardinalityWrapper<MedRecordAttribute>>>(&self, attribute: A) {
        self.0.write_or_panic().has_attribute(attribute);
    }
}

impl<O: GroupedOperand + HasAttribute> HasAttribute for GroupOperand<O> {
    fn has_attribute<A: Into<CardinalityWrapper<MedRecordAttribute>>>(&mut self, attribute: A) {
        self.operand.has_attribute(attribute);
    }
}

pub trait InGroup {
    fn in_group<G: Into<CardinalityWrapper<Group>>>(&mut self, group: G);
}

impl<O: InGroup> Wrapper<O> {
    pub fn in_group<G: Into<CardinalityWrapper<Group>>>(&self, group: G) {
        self.0.write_or_panic().in_group(group);
    }
}

impl<O: GroupedOperand + InGroup> InGroup for GroupOperand<O> {
    fn in_group<G: Into<CardinalityWrapper<Group>>>(&mut self, group: G) {
        self.operand.in_group(group);
    }
}

pub trait Edges {
    type ReturnOperand;

    fn edges(&mut self, edge_direction: EdgeDirection) -> Wrapper<Self::ReturnOperand>;
}

impl<O: Edges> Wrapper<O> {
    pub fn edges(&self, edge_direction: EdgeDirection) -> Wrapper<O::ReturnOperand> {
        self.0.write_or_panic().edges(edge_direction)
    }
}

impl<O: GroupedOperand + Edges> Edges for GroupOperand<O>
where
    Self: DeepClone,
    O::ReturnOperand: GroupedOperand,
    <O::ReturnOperand as GroupedOperand>::Context: From<Self>,
{
    type ReturnOperand = GroupOperand<O::ReturnOperand>;

    fn edges(&mut self, edge_direction: EdgeDirection) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.edges(edge_direction);

        Wrapper::<GroupOperand<O::ReturnOperand>>::new(self.deep_clone().into(), operand)
    }
}

pub trait Neighbors {
    type ReturnOperand;

    fn neighbors(&mut self, edge_direction: EdgeDirection) -> Wrapper<Self::ReturnOperand>;
}

impl<O: Neighbors> Wrapper<O> {
    pub fn neighbors(&self, edge_direction: EdgeDirection) -> Wrapper<O::ReturnOperand> {
        self.0.write_or_panic().neighbors(edge_direction)
    }
}

impl<O: GroupedOperand + Neighbors> Neighbors for GroupOperand<O>
where
    Self: DeepClone,
    O::ReturnOperand: GroupedOperand,
    <O::ReturnOperand as GroupedOperand>::Context: From<Self>,
{
    type ReturnOperand = GroupOperand<O::ReturnOperand>;

    fn neighbors(&mut self, edge_direction: EdgeDirection) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.neighbors(edge_direction);

        Wrapper::<GroupOperand<O::ReturnOperand>>::new(self.deep_clone().into(), operand)
    }
}

pub trait SourceNode {
    type ReturnOperand;

    fn source_node(&mut self) -> Wrapper<Self::ReturnOperand>;
}

impl<O: SourceNode> Wrapper<O> {
    pub fn source_node(&self) -> Wrapper<O::ReturnOperand> {
        self.0.write_or_panic().source_node()
    }
}

impl<O: GroupedOperand + SourceNode> SourceNode for GroupOperand<O>
where
    Self: DeepClone,
    O::ReturnOperand: GroupedOperand,
    <O::ReturnOperand as GroupedOperand>::Context: From<Self>,
{
    type ReturnOperand = GroupOperand<O::ReturnOperand>;

    fn source_node(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.source_node();

        Wrapper::<GroupOperand<O::ReturnOperand>>::new(self.deep_clone().into(), operand)
    }
}

pub trait TargetNode {
    type ReturnOperand;

    fn target_node(&mut self) -> Wrapper<Self::ReturnOperand>;
}

impl<O: TargetNode> Wrapper<O> {
    pub fn target_node(&self) -> Wrapper<O::ReturnOperand> {
        self.0.write_or_panic().target_node()
    }
}

impl<O: GroupedOperand + TargetNode> TargetNode for GroupOperand<O>
where
    Self: DeepClone,
    O::ReturnOperand: GroupedOperand,
    <O::ReturnOperand as GroupedOperand>::Context: From<Self>,
{
    type ReturnOperand = GroupOperand<O::ReturnOperand>;

    fn target_node(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.target_node();

        Wrapper::<GroupOperand<O::ReturnOperand>>::new(self.deep_clone().into(), operand)
    }
}

pub trait ToValues {
    type ReturnOperand;

    fn to_values(&mut self) -> Wrapper<Self::ReturnOperand>;
}

impl<O: ToValues> Wrapper<O> {
    pub fn to_values(&self) -> Wrapper<O::ReturnOperand> {
        self.0.write_or_panic().to_values()
    }
}

impl<O: GroupedOperand + ToValues> ToValues for GroupOperand<O>
where
    Self: DeepClone,
    O::ReturnOperand: GroupedOperand,
    <O::ReturnOperand as GroupedOperand>::Context: From<Self>,
{
    type ReturnOperand = GroupOperand<O::ReturnOperand>;

    fn to_values(&mut self) -> Wrapper<Self::ReturnOperand> {
        let operand = self.operand.to_values();

        Wrapper::<GroupOperand<O::ReturnOperand>>::new(self.deep_clone().into(), operand)
    }
}
