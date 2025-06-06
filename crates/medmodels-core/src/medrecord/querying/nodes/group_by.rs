use super::NodeOperand;
use crate::medrecord::querying::group_by::GroupedOperand;

impl GroupedOperand for NodeOperand {
    type Context = Self;
}
