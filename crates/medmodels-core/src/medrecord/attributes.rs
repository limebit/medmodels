use crate::{
    errors::MedRecordError,
    prelude::{
        Attributes, EdgeIndex, Group, MedRecordAttribute, MedRecordValue, NodeIndex, SchemaType,
    },
    MedRecord,
};

macro_rules! impl_attributes_mut {
    (
        $struct_name:ident,
        $index_type:ty,
        $index_field:ident,
        $entity:literal,
        $contains_fn:ident,
        $groups_of_fn:ident,
        $get_attributes_fn:ident,
        $get_attributes_mut_fn:ident,
        $schema_update_fn:ident,
        $schema_validate_fn:ident
    ) => {
        pub struct $struct_name<'a> {
            $index_field: &'a $index_type,
            medrecord: &'a mut MedRecord,
        }

        impl<'a> $struct_name<'a> {
            pub(crate) fn new(
                $index_field: &'a $index_type,
                medrecord: &'a mut MedRecord,
            ) -> Result<Self, MedRecordError> {
                if !medrecord.$contains_fn($index_field) {
                    return Err(MedRecordError::IndexError(format!(
                        concat!("Cannot find ", $entity, " with index {}"),
                        $index_field
                    )));
                }

                Ok(Self {
                    $index_field,
                    medrecord,
                })
            }

            fn get_groups(&self) -> Vec<Group> {
                self.medrecord
                    .$groups_of_fn(self.$index_field)
                    .expect(concat!($entity, " must exist."))
                    .cloned()
                    .collect()
            }

            fn handle_schema(
                &mut self,
                attributes: &Attributes,
                groups: &[Group],
            ) -> Result<(), MedRecordError> {
                let schema = &mut self.medrecord.schema;

                match schema.schema_type() {
                    SchemaType::Inferred => {
                        if groups.is_empty() {
                            schema.$schema_update_fn(attributes, None, false);
                        } else {
                            for group in groups {
                                schema.$schema_update_fn(attributes, Some(group), false);
                            }
                        }
                    }
                    SchemaType::Provided => {
                        if groups.is_empty() {
                            schema.$schema_validate_fn(self.$index_field, attributes, None)?;
                        } else {
                            for group in groups {
                                schema.$schema_validate_fn(
                                    self.$index_field,
                                    attributes,
                                    Some(group),
                                )?;
                            }
                        }
                    }
                }

                Ok(())
            }

            fn set_attributes(&mut self, attributes: Attributes) {
                *self
                    .medrecord
                    .graph
                    .$get_attributes_mut_fn(self.$index_field)
                    .expect(concat!($entity, " must exist.")) = attributes;
            }

            pub fn replace_attributes(
                &mut self,
                attributes: Attributes,
            ) -> Result<(), MedRecordError> {
                let groups = self.get_groups();
                self.handle_schema(&attributes, &groups)?;
                self.set_attributes(attributes);
                Ok(())
            }

            pub fn update_attribute(
                &mut self,
                attribute: &MedRecordAttribute,
                value: MedRecordValue,
            ) -> Result<(), MedRecordError> {
                let groups = self.get_groups();

                let mut attributes = self
                    .medrecord
                    .$get_attributes_fn(self.$index_field)
                    .expect(concat!($entity, " must exist."))
                    .clone();
                attributes
                    .entry(attribute.clone())
                    .and_modify(|v| *v = value.clone())
                    .or_insert(value);

                self.handle_schema(&attributes, &groups)?;
                self.set_attributes(attributes);
                Ok(())
            }

            pub fn remove_attribute(
                &mut self,
                attribute: &MedRecordAttribute,
            ) -> Result<MedRecordValue, MedRecordError> {
                let groups = self.get_groups();

                let mut attributes = self
                    .medrecord
                    .$get_attributes_fn(self.$index_field)
                    .expect(concat!($entity, " must exist."))
                    .clone();
                let removed_value = attributes.remove(attribute);

                let Some(removed_value) = removed_value else {
                    return Err(MedRecordError::KeyError(format!(
                        concat!("Attribute {} does not exist on ", $entity, " {}"),
                        attribute, self.$index_field
                    )));
                };

                self.handle_schema(&attributes, &groups)?;
                self.set_attributes(attributes);
                Ok(removed_value)
            }
        }
    };
}

impl_attributes_mut!(
    NodeAttributesMut,
    NodeIndex,
    node_index,
    "node",
    contains_node,
    groups_of_node,
    node_attributes,
    node_attributes_mut,
    update_node,
    validate_node
);

impl_attributes_mut!(
    EdgeAttributesMut,
    EdgeIndex,
    edge_index,
    "edge",
    contains_edge,
    groups_of_edge,
    edge_attributes,
    edge_attributes_mut,
    update_edge,
    validate_edge
);
