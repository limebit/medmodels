use super::operation::{EdgeOperation, NodeOperation, Operation};
use crate::medrecord::{EdgeIndex, MedRecord, NodeIndex};

#[derive(Debug)]
pub struct NodeSelection<'a> {
    medrecord: &'a MedRecord,
    operation: NodeOperation,
}

impl<'a> NodeSelection<'a> {
    pub fn new(medrecord: &'a MedRecord, operation: NodeOperation) -> Self {
        Self {
            medrecord,
            operation,
        }
    }

    pub fn iter(self) -> impl Iterator<Item = &'a NodeIndex> {
        self.operation
            .evaluate(self.medrecord, self.medrecord.node_indices())
    }

    pub fn collect<B: FromIterator<&'a NodeIndex>>(self) -> B {
        FromIterator::from_iter(self.iter())
    }
}

#[derive(Debug)]
pub struct EdgeSelection<'a> {
    medrecord: &'a MedRecord,
    operation: EdgeOperation,
}

impl<'a> EdgeSelection<'a> {
    pub fn new(medrecord: &'a MedRecord, operation: EdgeOperation) -> Self {
        Self {
            medrecord,
            operation,
        }
    }

    pub fn iter(self) -> impl Iterator<Item = &'a EdgeIndex> {
        self.operation
            .evaluate(self.medrecord, self.medrecord.edge_indices())
    }

    pub fn collect<B: FromIterator<&'a EdgeIndex>>(self) -> B {
        FromIterator::from_iter(self.iter())
    }
}

#[cfg(test)]
mod test {
    use crate::medrecord::{edge, node, Attributes, MedRecord, MedRecordAttribute, NodeIndex};
    use std::collections::HashMap;

    fn create_nodes() -> Vec<(NodeIndex, Attributes)> {
        vec![
            (
                "0".into(),
                HashMap::from([
                    ("lorem".into(), "ipsum".into()),
                    ("dolor".into(), "  ipsum  ".into()),
                    ("test".into(), "Ipsum".into()),
                    ("integer".into(), 1.into()),
                    ("float".into(), 0.5.into()),
                ]),
            ),
            (
                "1".into(),
                HashMap::from([("amet".into(), "consectetur".into())]),
            ),
            (
                "2".into(),
                HashMap::from([("adipiscing".into(), "elit".into())]),
            ),
            ("3".into(), HashMap::new()),
        ]
    }

    fn create_edges() -> Vec<(NodeIndex, NodeIndex, Attributes)> {
        vec![
            (
                "0".into(),
                "1".into(),
                HashMap::from([
                    ("sed".into(), "do".into()),
                    ("eiusmod".into(), "tempor".into()),
                    ("dolor".into(), "  do  ".into()),
                    ("test".into(), "DO".into()),
                ]),
            ),
            (
                "1".into(),
                "2".into(),
                HashMap::from([("incididunt".into(), "ut".into())]),
            ),
            (
                "0".into(),
                "2".into(),
                HashMap::from([
                    ("test".into(), 1.into()),
                    ("integer".into(), 1.into()),
                    ("float".into(), 0.5.into()),
                ]),
            ),
            (
                "0".into(),
                "2".into(),
                HashMap::from([("test".into(), 0.into())]),
            ),
        ]
    }

    fn create_medrecord() -> MedRecord {
        let nodes = create_nodes();
        let edges = create_edges();

        MedRecord::from_tuples(nodes, Some(edges), None).unwrap()
    }

    #[test]
    fn test_iter() {
        let medrecord = create_medrecord();

        assert_eq!(
            1,
            medrecord
                .select_nodes(node().has_attribute("lorem"))
                .iter()
                .count(),
        );

        assert_eq!(
            1,
            medrecord
                .select_edges(edge().has_attribute("sed"))
                .iter()
                .count(),
        );
    }

    #[test]
    fn test_collect() {
        let medrecord = create_medrecord();

        assert_eq!(
            vec![&MedRecordAttribute::from("0")],
            medrecord
                .select_nodes(node().has_attribute("lorem"))
                .collect::<Vec<_>>(),
        );

        assert_eq!(
            vec![&0],
            medrecord
                .select_edges(edge().has_attribute("sed"))
                .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_select_nodes_node() {
        let mut medrecord = create_medrecord();

        medrecord
            .add_group("test".into(), Some(vec!["0".into()]), None)
            .unwrap();

        // Node in group
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().in_group("test"))
                .iter()
                .count(),
        );

        // Node has attribute
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().has_attribute("lorem"))
                .iter()
                .count(),
        );

        // Node has outgoing edge with
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().has_outgoing_edge_with(edge().index().equal(0)))
                .iter()
                .count(),
        );

        // Node has incoming edge with
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().has_incoming_edge_with(edge().index().equal(0)))
                .iter()
                .count(),
        );

        // Node has edge with
        assert_eq!(
            2,
            medrecord
                .select_nodes(node().has_edge_with(edge().index().equal(0)))
                .iter()
                .count(),
        );

        // Node has neighbor with
        assert_eq!(
            2,
            medrecord
                .select_nodes(node().has_neighbor_with(node().index().equal("2")))
                .iter()
                .count(),
        );
    }

    #[test]
    fn test_select_nodes_node_index() {
        let medrecord = create_medrecord();

        // Index greater
        assert_eq!(
            2,
            medrecord
                .select_nodes(node().index().greater("1"))
                .iter()
                .count(),
        );

        // Index less
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().index().less("1"))
                .iter()
                .count(),
        );

        // Index greater or equal
        assert_eq!(
            3,
            medrecord
                .select_nodes(node().index().greater_or_equal("1"))
                .iter()
                .count(),
        );

        // Index less or equal
        assert_eq!(
            2,
            medrecord
                .select_nodes(node().index().less_or_equal("1"))
                .iter()
                .count(),
        );

        // Index equal
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().index().equal("1"))
                .iter()
                .count(),
        );

        // Index not equal
        assert_eq!(
            3,
            medrecord
                .select_nodes(node().index().not_equal("1"))
                .iter()
                .count(),
        );

        // Index in
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().index().r#in(vec!["1"]))
                .iter()
                .count(),
        );

        // Index in
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().index().r#in(node().has_attribute("lorem")))
                .iter()
                .count(),
        );

        // Index not in
        assert_eq!(
            3,
            medrecord
                .select_nodes(node().index().not_in(node().has_attribute("lorem")))
                .iter()
                .count(),
        );

        // Index starts with
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().index().starts_with("1"))
                .iter()
                .count(),
        );

        // Index ends with
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().index().ends_with("1"))
                .iter()
                .count(),
        );

        // Index contains
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().index().contains("1"))
                .iter()
                .count(),
        );
    }

    #[test]
    fn test_select_nodes_node_attribute() {
        let medrecord = create_medrecord();

        // Attribute greater
        assert_eq!(
            0,
            medrecord
                .select_nodes(node().attribute("lorem").greater("ipsum"))
                .iter()
                .count(),
        );

        // Attribute less
        assert_eq!(
            0,
            medrecord
                .select_nodes(node().attribute("lorem").less("ipsum"))
                .iter()
                .count(),
        );

        // Attribute greater or equal
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().attribute("lorem").greater_or_equal("ipsum"))
                .iter()
                .count(),
        );

        // Attribute less or equal
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().attribute("lorem").less_or_equal("ipsum"))
                .iter()
                .count(),
        );

        // Attribute equal
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().attribute("lorem").equal("ipsum"))
                .iter()
                .count(),
        );

        // Attribute not equal
        assert_eq!(
            0,
            medrecord
                .select_nodes(node().attribute("lorem").not_equal("ipsum"))
                .iter()
                .count(),
        );

        // Attribute in
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().attribute("lorem").r#in(vec!["ipsum"]))
                .iter()
                .count(),
        );

        // Attribute not in
        assert_eq!(
            0,
            medrecord
                .select_nodes(node().attribute("lorem").not_in(vec!["ipsum"]))
                .iter()
                .count(),
        );

        // Attribute starts with
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().attribute("lorem").starts_with("ip"))
                .iter()
                .count(),
        );

        // Attribute ends with
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().attribute("lorem").ends_with("um"))
                .iter()
                .count(),
        );

        // Attribute contains
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().attribute("lorem").contains("su"))
                .iter()
                .count(),
        );

        // Attribute compare to attribute
        assert_eq!(
            1,
            medrecord
                .select_nodes(node().attribute("lorem").equal(node().attribute("lorem")))
                .iter()
                .count(),
        );

        // Attribute compare to attribute
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .not_equal(node().attribute("lorem"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute add
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .equal(node().attribute("lorem").add("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute add
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .not_equal(node().attribute("lorem").add("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute sub
        // Returns nothing because can't sub a string
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .equal(node().attribute("lorem").sub("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute sub
        // Doesn't work because can't sub a string
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .not_equal(node().attribute("lorem").sub("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute sub
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("integer")
                        .equal(node().attribute("integer").sub(10))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute sub
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("integer")
                        .not_equal(node().attribute("integer").sub(10))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute mul
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .equal(node().attribute("lorem").mul(2))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute mul
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .not_equal(node().attribute("lorem").mul(2))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute div
        // Returns nothing because can't div a string
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .equal(node().attribute("lorem").div("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute div
        // Doesn't work because can't div a string
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .not_equal(node().attribute("lorem").div("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute div
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("integer")
                        .equal(node().attribute("integer").div(2))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute div
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("integer")
                        .not_equal(node().attribute("integer").div(2))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute pow
        // Returns nothing because can't pow a string
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .equal(node().attribute("lorem").pow("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute pow
        // Doesn't work because can't pow a string
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .not_equal(node().attribute("lorem").pow("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute pow
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("integer")
                        .equal(node().attribute("integer").pow(2)) // 1 ** 2 = 1
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute pow
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("integer")
                        .not_equal(node().attribute("integer").pow(2)) // 1 ** 2 = 1
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute mod
        // Returns nothing because can't mod a string
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .equal(node().attribute("lorem").r#mod("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute mod
        // Doesn't work because can't mod a string
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .not_equal(node().attribute("lorem").r#mod("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute mod
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("integer")
                        .equal(node().attribute("integer").r#mod(2)) // 1 % 2 = 1
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute mod
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("integer")
                        .not_equal(node().attribute("integer").r#mod(2)) // 1 % 2 = 1
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute round
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .equal(node().attribute("lorem").round())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute round
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .not_equal(node().attribute("lorem").round())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute round
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("integer")
                        .equal(node().attribute("float").round())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute round
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("float")
                        .not_equal(node().attribute("float").round())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute ceil
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("integer")
                        .equal(node().attribute("float").ceil())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute ceil
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("float")
                        .not_equal(node().attribute("float").ceil())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute floor
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("integer")
                        .equal(node().attribute("float").floor())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute floor
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("float")
                        .not_equal(node().attribute("float").floor())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute abs
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("integer")
                        .equal(node().attribute("integer").abs())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute sqrt
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("integer")
                        .equal(node().attribute("integer").sqrt()) // sqrt(1) = 1
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute trim
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .equal(node().attribute("dolor").trim())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute trim_start
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .equal(node().attribute("dolor").trim_start())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute trim_end
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .equal(node().attribute("dolor").trim_end())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute lowercase
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .equal(node().attribute("test").lowercase())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute uppercase
        assert_eq!(
            0,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .equal(node().attribute("test").uppercase())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute slice
        assert_eq!(
            1,
            medrecord
                .select_nodes(
                    node()
                        .attribute("lorem")
                        .equal(node().attribute("dolor").slice(2..7))
                )
                .iter()
                .count(),
        );
    }

    #[test]
    fn test_select_edges_edge() {
        let mut medrecord = create_medrecord();

        medrecord
            .add_group("test".into(), None, Some(vec![0]))
            .unwrap();

        // Edge connected to target
        assert_eq!(
            3,
            medrecord
                .select_edges(edge().connected_target("2"))
                .iter()
                .count(),
        );

        // Edge connected to source
        assert_eq!(
            3,
            medrecord
                .select_edges(edge().connected_source("0"))
                .iter()
                .count(),
        );

        // Edge connected
        assert_eq!(
            2,
            medrecord.select_edges(edge().connected("1")).iter().count(),
        );

        // Edge in group
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().in_group("test"))
                .iter()
                .count(),
        );

        // Edge has attribute
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().has_attribute("sed"))
                .iter()
                .count(),
        );

        // Edge connected to target with
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().connected_target_with(node().index().equal("1")))
                .iter()
                .count(),
        );

        // Edge connected to source with
        assert_eq!(
            3,
            medrecord
                .select_edges(edge().connected_source_with(node().index().equal("0")))
                .iter()
                .count(),
        );

        // Edge connected with
        assert_eq!(
            2,
            medrecord
                .select_edges(edge().connected_with(node().index().equal("1")))
                .iter()
                .count(),
        );

        // Edge has parallel edges with
        assert_eq!(
            2,
            medrecord
                .select_edges(edge().has_parallel_edges_with(edge().has_attribute("test")))
                .iter()
                .count(),
        );

        // Edge has parallel edges with self comparison
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge().has_parallel_edges_with_self_comparison(
                        edge()
                            .attribute("test")
                            .equal(edge().attribute("test").sub(1))
                    )
                )
                .iter()
                .count(),
        );
    }

    #[test]
    fn test_select_edges_edge_index() {
        let medrecord = create_medrecord();

        // Index greater
        assert_eq!(
            2,
            medrecord
                .select_edges(edge().index().greater(1))
                .iter()
                .count(),
        );

        // Index less
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().index().less(1))
                .iter()
                .count(),
        );

        // Index greater or equal
        assert_eq!(
            3,
            medrecord
                .select_edges(edge().index().greater_or_equal(1))
                .iter()
                .count(),
        );

        // Index less or equal
        assert_eq!(
            2,
            medrecord
                .select_edges(edge().index().less_or_equal(1))
                .iter()
                .count(),
        );

        // Index equal
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().index().equal(1))
                .iter()
                .count(),
        );

        // Index not equal
        assert_eq!(
            3,
            medrecord
                .select_edges(edge().index().not_equal(1))
                .iter()
                .count(),
        );

        // Index in
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().index().r#in(vec![1_usize]))
                .iter()
                .count(),
        );

        // Index in
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().index().r#in(edge().has_attribute("sed")))
                .iter()
                .count(),
        );

        // Index not in
        assert_eq!(
            3,
            medrecord
                .select_edges(edge().index().not_in(edge().has_attribute("sed")))
                .iter()
                .count(),
        );
    }

    #[test]
    fn test_select_edges_edge_attribute() {
        let medrecord = create_medrecord();

        // Attribute greater
        assert_eq!(
            0,
            medrecord
                .select_edges(edge().attribute("sed").greater("do"))
                .iter()
                .count(),
        );

        // Attribute less
        assert_eq!(
            0,
            medrecord
                .select_edges(edge().attribute("sed").less("do"))
                .iter()
                .count(),
        );

        // Attribute greater or equal
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().attribute("sed").greater_or_equal("do"))
                .iter()
                .count(),
        );

        // Attribute less or equal
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().attribute("sed").less_or_equal("do"))
                .iter()
                .count(),
        );

        // Attribute equal
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().attribute("sed").equal("do"))
                .iter()
                .count(),
        );

        // Attribute not equal
        assert_eq!(
            0,
            medrecord
                .select_edges(edge().attribute("sed").not_equal("do"))
                .iter()
                .count(),
        );

        // Attribute in
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().attribute("sed").r#in(vec!["do"]))
                .iter()
                .count(),
        );

        // Attribute not in
        assert_eq!(
            0,
            medrecord
                .select_edges(edge().attribute("sed").not_in(vec!["do"]))
                .iter()
                .count(),
        );

        // Attribute starts with
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().attribute("sed").starts_with("d"))
                .iter()
                .count(),
        );

        // Attribute ends with
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().attribute("sed").ends_with("o"))
                .iter()
                .count(),
        );

        // Attribute contains
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().attribute("sed").contains("do"))
                .iter()
                .count(),
        );

        // Attribute compare to attribute
        assert_eq!(
            1,
            medrecord
                .select_edges(edge().attribute("sed").equal(edge().attribute("sed")))
                .iter()
                .count(),
        );

        // Attribute compare to attribute
        assert_eq!(
            0,
            medrecord
                .select_edges(edge().attribute("sed").not_equal(edge().attribute("sed")))
                .iter()
                .count(),
        );

        // Attribute compare to attribute add
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .equal(edge().attribute("sed").add("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute add
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .not_equal(edge().attribute("sed").add("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute sub
        // Returns nothing because can't sub a string
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .equal(edge().attribute("sed").sub("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute sub
        // Doesn't work because can't sub a string
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .not_equal(edge().attribute("sed").sub("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute sub
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("integer")
                        .equal(edge().attribute("integer").sub(10))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute sub
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("integer")
                        .not_equal(edge().attribute("integer").sub(10))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute mul
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .equal(edge().attribute("sed").mul(2))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute mul
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .not_equal(edge().attribute("sed").mul(2))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute div
        // Returns nothing because can't div a string
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .equal(edge().attribute("sed").div("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute div
        // Doesn't work because can't div a string
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .not_equal(edge().attribute("sed").div("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute div
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("integer")
                        .equal(edge().attribute("integer").div(2))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute div
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("integer")
                        .not_equal(edge().attribute("integer").div(2))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute pow
        // Returns nothing because can't pow a string
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("lorem")
                        .equal(edge().attribute("lorem").pow("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute pow
        // Doesn't work because can't pow a string
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("lorem")
                        .not_equal(edge().attribute("lorem").pow("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute pow
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("integer")
                        .equal(edge().attribute("integer").pow(2)) // 1 ** 2 = 1
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute pow
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("integer")
                        .not_equal(edge().attribute("integer").pow(2)) // 1 ** 2 = 1
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute mod
        // Returns nothing because can't mod a string
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("lorem")
                        .equal(edge().attribute("lorem").r#mod("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute mod
        // Doesn't work because can't mod a string
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("lorem")
                        .not_equal(edge().attribute("lorem").r#mod("10"))
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute mod
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("integer")
                        .equal(edge().attribute("integer").r#mod(2)) // 1 % 2 = 1
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute mod
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("integer")
                        .not_equal(edge().attribute("integer").r#mod(2)) // 1 % 2 = 1
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute round
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .equal(edge().attribute("sed").round())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute round
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .not_equal(edge().attribute("sed").round())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute round
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("integer")
                        .equal(edge().attribute("float").round())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute round
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("float")
                        .not_equal(edge().attribute("float").round())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute ceil
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("integer")
                        .equal(edge().attribute("float").ceil())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute ceil
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("float")
                        .not_equal(edge().attribute("float").ceil())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute floor
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("integer")
                        .equal(edge().attribute("float").floor())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute floor
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("float")
                        .not_equal(edge().attribute("float").floor())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute abs
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("integer")
                        .equal(edge().attribute("integer").abs())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute sqrt
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("integer")
                        .equal(edge().attribute("integer").sqrt()) // sqrt(1) = 1
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute trim
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .equal(edge().attribute("dolor").trim())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute trim_start
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .equal(edge().attribute("dolor").trim_start())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute trim_end
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .equal(edge().attribute("dolor").trim_end())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute lowercase
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .equal(edge().attribute("test").lowercase())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute uppercase
        assert_eq!(
            0,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .equal(edge().attribute("test").uppercase())
                )
                .iter()
                .count(),
        );

        // Attribute compare to attribute slice
        assert_eq!(
            1,
            medrecord
                .select_edges(
                    edge()
                        .attribute("sed")
                        .equal(edge().attribute("dolor").slice(2..4))
                )
                .iter()
                .count(),
        );
    }
}
