pub mod inferred;
pub mod provided;

use std::collections::HashMap;

use super::{Attributes, EdgeIndex, MedRecord, NodeIndex};
use crate::medrecord::{datatypes::DataType, MedRecordAttribute};
use inferred::InferredSchema;
use provided::ProvidedSchema;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum Schema {
    Inferred(InferredSchema),
    Provided(ProvidedSchema),
}

impl Default for Schema {
    fn default() -> Self {
        Self::Inferred(InferredSchema::default())
    }
}

impl Schema {
    pub fn infer(medrecord: &MedRecord) -> Self {
        Self::Inferred(InferredSchema::infer(medrecord))
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum AttributeType {
    Categorical,
    Continuous,
    Temporal,
    Unstructured,
}

impl AttributeType {
    pub fn infer_from(data_type: &DataType) -> Self {
        match data_type {
            DataType::String => Self::Unstructured,
            DataType::Int => Self::Continuous,
            DataType::Float => Self::Continuous,
            DataType::Bool => Self::Categorical,
            DataType::DateTime => Self::Temporal,
            DataType::Duration => Self::Continuous,
            DataType::Null => Self::Unstructured,
            DataType::Any => Self::Unstructured,
            DataType::Union((first_dataype, second_dataype)) => {
                Self::infer_from(first_dataype).merge(&Self::infer_from(second_dataype))
            }
            DataType::Option(dataype) => Self::infer_from(dataype),
        }
    }

    fn merge(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Categorical, Self::Categorical) => Self::Categorical,
            (Self::Continuous, Self::Continuous) => Self::Continuous,
            (Self::Temporal, Self::Temporal) => Self::Temporal,
            _ => Self::Unstructured,
        }
    }
}

impl DataType {
    fn merge(&self, other: &Self) -> Self {
        if self.evaluate(other) {
            self.clone()
        } else {
            match (self, other) {
                (Self::Null, _) => Self::Option(Box::new(other.clone())),
                (_, Self::Null) => Self::Option(Box::new(self.clone())),
                (_, Self::Any) => Self::Any,
                _ => Self::Union((Box::new(self.clone()), Box::new(other.clone()))),
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AttributeDataType {
    pub data_type: DataType,
    pub attribute_type: AttributeType,
}

impl AttributeDataType {
    pub fn new(data_type: DataType, attribute_type: AttributeType) -> Self {
        Self {
            data_type,
            attribute_type,
        }
    }

    fn merge(&mut self, other: &Self) {
        self.data_type = self.data_type.merge(&other.data_type);
        self.attribute_type = self.attribute_type.merge(&other.attribute_type);
    }
}

impl From<DataType> for AttributeDataType {
    fn from(value: DataType) -> Self {
        let attribute_type = AttributeType::infer_from(&value);

        Self {
            data_type: value,
            attribute_type,
        }
    }
}

impl From<(DataType, AttributeType)> for AttributeDataType {
    fn from(value: (DataType, AttributeType)) -> Self {
        Self {
            data_type: value.0,
            attribute_type: value.1,
        }
    }
}

type AttributeSchema = HashMap<MedRecordAttribute, AttributeDataType>;
