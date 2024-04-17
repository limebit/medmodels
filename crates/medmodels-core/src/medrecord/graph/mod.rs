mod edge;
mod node;

use super::{MedRecordAttribute, MedRecordValue};
use crate::errors::GraphError;
use edge::Edge;
use medmodels_utils::aliases::MrHashMap;
use node::Node;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::atomic::AtomicUsize};

pub type NodeIndex = MedRecordAttribute;
pub type EdgeIndex = usize;
pub type Attributes = HashMap<MedRecordAttribute, MedRecordValue>;

#[derive(Serialize, Deserialize, Debug)]
pub(super) struct Graph {
    nodes: MrHashMap<NodeIndex, Node>,
    edges: MrHashMap<EdgeIndex, Edge>,
    edge_index_counter: AtomicUsize,
}

#[allow(dead_code)]
impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: MrHashMap::new(),
            edges: MrHashMap::new(),
            edge_index_counter: AtomicUsize::new(0),
        }
    }

    pub fn with_capacity(node_capacity: usize, edge_capacity: usize) -> Self {
        Self {
            nodes: MrHashMap::with_capacity(node_capacity),
            edges: MrHashMap::with_capacity(edge_capacity),
            edge_index_counter: AtomicUsize::new(0),
        }
    }

    pub fn from_tuples(
        nodes: Vec<(NodeIndex, Attributes)>,
        edges: Option<Vec<(NodeIndex, NodeIndex, Attributes)>>,
    ) -> Result<Self, GraphError> {
        let mut graph = Self::with_capacity(
            nodes.len(),
            edges.as_ref().map(|vec| vec.len()).unwrap_or(0),
        );

        for (node_index, attributes) in nodes {
            graph.add_node(node_index, attributes);
        }

        if let Some(edges) = edges {
            for (source_node_index, target_node_index, attributes) in edges {
                graph.add_edge(source_node_index, target_node_index, attributes)?;
            }
        }

        Ok(graph)
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();

        self.edge_index_counter = AtomicUsize::new(0);
    }

    pub fn clear_edges(&mut self) {
        self.edges.clear();

        self.edge_index_counter = AtomicUsize::new(0);
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn add_node(&mut self, node_index: NodeIndex, attributes: Attributes) {
        let node = Node::new(attributes);

        self.nodes.insert(node_index, node);
    }

    pub fn remove_node(&mut self, node_index: &NodeIndex) -> Result<Attributes, GraphError> {
        let node = self
            .nodes
            .remove(node_index)
            .ok_or(GraphError::IndexError(format!(
                "Cannot find node with index {}",
                node_index
            )))?;

        for edge_index in node.outgoing_edge_indices {
            self.edges.remove(&edge_index);
        }

        for edge_index in node.incoming_edge_indices {
            self.edges.remove(&edge_index);
        }

        Ok(node.attributes)
    }

    pub fn contains_node(&self, node_index: &NodeIndex) -> bool {
        self.nodes.contains_key(node_index)
    }

    fn get_edge_index(&self) -> EdgeIndex {
        self.edge_index_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn add_edge(
        &mut self,
        source_node_index: NodeIndex,
        target_node_index: NodeIndex,
        attributes: Attributes,
    ) -> Result<EdgeIndex, GraphError> {
        if !self.nodes.contains_key(&target_node_index) {
            return Err(GraphError::IndexError(format!(
                "Cannot find node with index {}",
                target_node_index
            )));
        }

        let edge_index = self.get_edge_index();

        let outgoing_node =
            self.nodes
                .get_mut(&source_node_index)
                .ok_or(GraphError::IndexError(format!(
                    "Cannot find node with index {}",
                    source_node_index
                )))?;

        outgoing_node.outgoing_edge_indices.insert(edge_index);

        let incoming_node = self
            .nodes
            .get_mut(&target_node_index)
            .expect("Node must exist");

        incoming_node.incoming_edge_indices.insert(edge_index);

        let edge = Edge::new(attributes, source_node_index, target_node_index);

        self.edges.insert(edge_index, edge);

        Ok(edge_index)
    }

    pub fn remove_edge(&mut self, edge_index: &EdgeIndex) -> Result<Attributes, GraphError> {
        let edge = self
            .edges
            .remove(edge_index)
            .ok_or(GraphError::IndexError(format!(
                "Cannot find edge with index {}",
                edge_index
            )))?;

        self.nodes
            .get_mut(&edge.target_node_index)
            .expect("Node must exist")
            .incoming_edge_indices
            .remove(edge_index);

        self.nodes
            .get_mut(&edge.source_node_index)
            .expect("Node must exist")
            .outgoing_edge_indices
            .remove(edge_index);

        Ok(edge.attributes)
    }

    pub fn node_attributes(&self, node_index: &NodeIndex) -> Result<&Attributes, GraphError> {
        Ok(&self
            .nodes
            .get(node_index)
            .ok_or(GraphError::IndexError(format!(
                "Cannot find node with index {}",
                node_index
            )))?
            .attributes)
    }

    pub fn node_attributes_mut(
        &mut self,
        node_index: &NodeIndex,
    ) -> Result<&mut Attributes, GraphError> {
        Ok(&mut self
            .nodes
            .get_mut(node_index)
            .ok_or(GraphError::IndexError(format!(
                "Cannot find node with index {}",
                node_index
            )))?
            .attributes)
    }

    pub fn nodes_attributes(&self) -> impl Iterator<Item = &Attributes> {
        self.nodes.values().map(|node| &node.attributes)
    }

    pub fn nodes_attributes_mut(&mut self) -> impl Iterator<Item = &mut Attributes> {
        self.nodes.iter_mut().map(|(_, node)| &mut node.attributes)
    }

    pub fn node_indices(&self) -> impl Iterator<Item = &NodeIndex> {
        self.nodes.keys()
    }

    pub fn edge_attributes(&self, edge_index: &EdgeIndex) -> Result<&Attributes, GraphError> {
        Ok(&self
            .edges
            .get(edge_index)
            .ok_or(GraphError::IndexError(format!(
                "Cannot find edge with index {}",
                edge_index
            )))?
            .attributes)
    }

    pub fn edge_attributes_mut(
        &mut self,
        edge_index: &EdgeIndex,
    ) -> Result<&mut Attributes, GraphError> {
        Ok(&mut self
            .edges
            .get_mut(edge_index)
            .ok_or(GraphError::IndexError(format!(
                "Cannot find edge with index {}",
                edge_index
            )))?
            .attributes)
    }

    pub fn edges_attributes(&self) -> impl Iterator<Item = &Attributes> {
        self.edges.values().map(|edge| &edge.attributes)
    }

    pub fn edges_attributes_mut(&mut self) -> impl Iterator<Item = &mut Attributes> {
        self.edges.values_mut().map(|edge| &mut edge.attributes)
    }

    pub fn edge_endpoints(
        &self,
        edge_index: &EdgeIndex,
    ) -> Result<(&NodeIndex, &NodeIndex), GraphError> {
        let edge = self
            .edges
            .get(edge_index)
            .ok_or(GraphError::IndexError(format!(
                "Cannot find edge with index {}",
                edge_index
            )))?;

        Ok((&edge.source_node_index, &edge.target_node_index))
    }

    pub fn outgoing_edges(
        &self,
        node_index: &NodeIndex,
    ) -> Result<impl Iterator<Item = &EdgeIndex>, GraphError> {
        Ok(self
            .nodes
            .get(node_index)
            .ok_or(GraphError::IndexError(format!(
                "Cannot find node with index {}",
                node_index
            )))?
            .outgoing_edge_indices
            .iter())
    }

    pub fn incoming_edges(
        &self,
        node_index: &NodeIndex,
    ) -> Result<impl Iterator<Item = &EdgeIndex>, GraphError> {
        Ok(self
            .nodes
            .get(node_index)
            .ok_or(GraphError::IndexError(format!(
                "Cannot find node with index {}",
                node_index
            )))?
            .incoming_edge_indices
            .iter())
    }

    pub fn edge_indices(&self) -> impl Iterator<Item = &EdgeIndex> {
        self.edges.keys()
    }

    pub fn edges_connecting<'a>(
        &'a self,
        source_node_index: &'a NodeIndex,
        target_node_index: &'a NodeIndex,
    ) -> impl Iterator<Item = &'a EdgeIndex> {
        self.edges
            .iter()
            .filter(move |(_, edge)| {
                &edge.source_node_index == source_node_index
                    && &edge.target_node_index == target_node_index
            })
            .map(|(edge_index, _)| edge_index)
    }

    pub fn contains_edge(&self, edge_index: &EdgeIndex) -> bool {
        self.edges.contains_key(edge_index)
    }

    pub fn neighbors(
        &self,
        node_index: &NodeIndex,
    ) -> Result<impl Iterator<Item = &NodeIndex>, GraphError> {
        Ok(self
            .nodes
            .get(node_index)
            .ok_or(GraphError::IndexError(format!(
                "Cannot find node with index {}",
                node_index
            )))?
            .outgoing_edge_indices
            .iter()
            .map(|edge_index| {
                &self
                    .edges
                    .get(edge_index)
                    .expect("Edge must exist")
                    .target_node_index
            }))
    }
}

#[cfg(test)]
mod test {
    use super::{Attributes, Graph, NodeIndex};
    use crate::errors::GraphError;
    use std::collections::HashMap;

    fn create_nodes() -> Vec<(NodeIndex, Attributes)> {
        vec![
            (
                "0".into(),
                HashMap::from([
                    ("lorem".into(), "ipsum".into()),
                    ("dolor".into(), "sit".into()),
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
                ]),
            ),
            (
                "1".into(),
                "2".into(),
                HashMap::from([("incididunt".into(), "ut".into())]),
            ),
            ("0".into(), "2".into(), HashMap::new()),
        ]
    }

    fn create_graph() -> Graph {
        let nodes = create_nodes();
        let edges = create_edges();

        Graph::from_tuples(nodes, Some(edges)).unwrap()
    }

    #[test]
    fn test_from_tuples() {
        let graph = create_graph();

        assert_eq!(4, graph.node_count());
        assert_eq!(3, graph.edge_count());
    }

    #[test]
    fn test_invalid_from_tuples() {
        let nodes = create_nodes();

        // Adding an edge pointing to a non-existing node should fail
        assert!(Graph::from_tuples(
            nodes.clone(),
            Some(vec![("0".into(), "50".into(), HashMap::new())])
        )
        .is_err_and(|e| matches!(e, GraphError::IndexError(_))));

        // Adding an edge from a non-existing should fail
        assert!(
            Graph::from_tuples(nodes, Some(vec![("50".into(), "0".into(), HashMap::new())]))
                .is_err_and(|e| matches!(e, GraphError::IndexError(_)))
        );
    }

    #[test]
    fn test_clear() {
        let mut graph = create_graph();

        graph.clear();

        assert_eq!(0, graph.node_count());
        assert_eq!(0, graph.edge_count());
    }

    #[test]
    fn test_clear_edges() {
        let mut graph = create_graph();

        graph.clear_edges();

        assert_eq!(4, graph.node_count());
        assert_eq!(0, graph.edge_count());
    }

    #[test]
    fn test_node_count() {
        let mut graph = Graph::new();

        assert_eq!(0, graph.node_count());

        graph.add_node("0".into(), HashMap::new());

        assert_eq!(1, graph.node_count());
    }

    #[test]
    fn test_edge_count() {
        let mut graph = Graph::new();

        graph.add_node("0".into(), HashMap::new());
        graph.add_node("1".into(), HashMap::new());

        assert_eq!(0, graph.edge_count());

        graph
            .add_edge("0".into(), "1".into(), HashMap::new())
            .unwrap();

        assert_eq!(1, graph.edge_count());
    }

    #[test]
    fn test_add_node() {
        let mut graph = Graph::new();

        assert_eq!(0, graph.node_count());

        graph.add_node("0".into(), HashMap::new());

        assert_eq!(1, graph.node_count());
    }

    #[test]
    fn test_remove_node() {
        let mut graph = create_graph();

        assert_eq!(4, graph.node_count());

        let attributes = graph.remove_node(&"0".into()).unwrap();

        assert_eq!(3, graph.node_count());

        assert_eq!(create_nodes()[0].1, attributes);
    }

    #[test]
    fn test_invalid_remove_node() {
        let mut graph = create_graph();

        assert!(graph
            .remove_node(&"50".into())
            .is_err_and(|e| matches!(e, GraphError::IndexError(_))));
    }

    #[test]
    fn test_contains_node() {
        let graph = create_graph();

        assert!(graph.contains_node(&"0".into()));

        assert!(!graph.contains_node(&"50".into()));
    }

    #[test]
    fn test_add_edge() {
        let mut graph = create_graph();

        assert_eq!(3, graph.edge_count());

        graph
            .add_edge("0".into(), "3".into(), HashMap::new())
            .unwrap();

        assert_eq!(4, graph.edge_count());
    }

    #[test]
    fn test_invalid_add_edge() {
        let nodes = create_nodes();
        let mut graph = Graph::from_tuples(nodes, None).unwrap();

        // Adding an edge pointing to a non-existing node should fail
        assert!(graph
            .add_edge("0".into(), "50".into(), HashMap::new())
            .is_err_and(|e| matches!(e, GraphError::IndexError(_))));

        // Adding an edge from a non-existing node should fail
        assert!(graph
            .add_edge("50".into(), "0".into(), HashMap::new())
            .is_err_and(|e| matches!(e, GraphError::IndexError(_))));
    }

    #[test]
    fn test_remove_edge() {
        let mut graph = create_graph();

        let attributes = graph.remove_edge(&0).unwrap();

        assert_eq!(2, graph.edge_count());

        assert_eq!(create_edges()[0].2, attributes);
    }

    #[test]
    fn test_invalid_remove_edge() {
        let mut graph = create_graph();

        // Removing an edge with a non-existing edge index should fail
        assert!(graph
            .remove_edge(&3)
            .is_err_and(|e| matches!(e, GraphError::IndexError(_))));
    }

    #[test]
    fn test_node_attributes() {
        let graph = create_graph();

        assert_eq!(
            &create_nodes()[0].1,
            graph.node_attributes(&"0".into()).unwrap()
        );
    }

    #[test]
    fn test_invalid_node_attributes() {
        let graph = create_graph();

        // Accessing the node attributes of a non-existing node should fail
        assert!(graph
            .node_attributes(&"50".into())
            .is_err_and(|e| matches!(e, GraphError::IndexError(_))));
    }

    #[test]
    fn test_node_attributes_mut() {
        let mut graph = create_graph();

        let attributes = graph.node_attributes_mut(&"0".into()).unwrap();

        assert_eq!(&create_nodes()[0].1, attributes);

        let new_attributes = HashMap::from([("0".into(), "1".into()), ("2".into(), "3".into())]);

        *attributes = new_attributes.clone();

        assert_eq!(&new_attributes, graph.node_attributes(&"0".into()).unwrap());
    }

    #[test]
    fn test_invalid_node_attributes_mut() {
        let mut graph = create_graph();

        // Accessing the node attributes of a non-existing node should fail
        assert!(graph
            .node_attributes_mut(&"50".into())
            .is_err_and(|e| matches!(e, GraphError::IndexError(_))));
    }

    #[test]
    fn test_nodes_attributes() {
        let graph = create_graph();

        let all_attributes = create_nodes()
            .into_iter()
            .map(|(_, attributes)| attributes)
            .collect::<Vec<_>>();

        for attributes in graph.nodes_attributes() {
            assert!(all_attributes.contains(attributes));
        }
    }

    #[test]
    fn test_nodes_attributes_mut() {
        let mut graph = create_graph();

        let all_attributes = create_nodes()
            .into_iter()
            .map(|(_, attributes)| attributes)
            .collect::<Vec<_>>();

        let new_attributes = HashMap::from([("0".into(), "1".into()), ("2".into(), "3".into())]);

        for attributes in graph.nodes_attributes_mut() {
            assert!(all_attributes.contains(attributes));

            *attributes = new_attributes.clone();
        }

        for attributes in graph.nodes_attributes() {
            assert_eq!(&new_attributes, attributes);
        }
    }

    #[test]
    fn test_node_indices() {
        let graph = create_graph();

        let node_indices = create_nodes()
            .into_iter()
            .map(|(node_index, _)| node_index)
            .collect::<Vec<_>>();

        for node_index in graph.node_indices() {
            assert!(node_indices.contains(node_index));
        }
    }

    #[test]
    fn test_edge_attributes() {
        let graph = create_graph();

        assert_eq!(&create_edges()[0].2, graph.edge_attributes(&0).unwrap());
    }

    #[test]
    fn test_invalid_edge_attributes() {
        let graph = create_graph();

        // Accessing the edge attributes of a non-existing edge should fail
        assert!(graph
            .edge_attributes(&50)
            .is_err_and(|e| matches!(e, GraphError::IndexError(_))));
    }

    #[test]
    fn test_edge_attributes_mut() {
        let mut graph = create_graph();

        // let attributes = graph.edge_attributes_mut(&0.into()).unwrap();
        let attributes = graph.edge_attributes_mut(&0).unwrap();

        assert_eq!(&create_edges()[0].2, attributes);

        let new_attributes = HashMap::from([("0".into(), "1".into()), ("2".into(), "3".into())]);

        *attributes = new_attributes.clone();

        assert_eq!(&new_attributes, graph.edge_attributes(&0).unwrap());
    }

    #[test]
    fn test_invalid_edge_attributes_mut() {
        let mut graph = create_graph();

        // Accessing the edge attributes of a non-existing edge should fail
        assert!(graph
            .edge_attributes_mut(&50)
            .is_err_and(|e| matches!(e, GraphError::IndexError(_))));
    }

    #[test]
    fn test_edges_attributes() {
        let graph = create_graph();

        let all_attributes = create_edges()
            .into_iter()
            .map(|(_, _, attributes)| attributes)
            .collect::<Vec<_>>();

        for attributes in graph.edges_attributes() {
            assert!(all_attributes.contains(attributes));
        }
    }

    #[test]
    fn test_edges_attributes_mut() {
        let mut graph = create_graph();

        let all_attributes = create_edges()
            .into_iter()
            .map(|(_, _, attributes)| attributes)
            .collect::<Vec<_>>();

        let new_attributes = HashMap::from([("0".into(), "1".into()), ("2".into(), "3".into())]);

        for attributes in graph.edges_attributes_mut() {
            assert!(all_attributes.contains(attributes));

            *attributes = new_attributes.clone();
        }

        for attributes in graph.edges_attributes() {
            assert_eq!(&new_attributes, attributes);
        }
    }

    #[test]
    fn test_edge_endpoints() {
        let graph = create_graph();

        let edge = &create_edges()[0];

        let endpoints = graph.edge_endpoints(&0).unwrap();

        assert_eq!(&edge.0, endpoints.0);

        assert_eq!(&edge.1, endpoints.1);
    }

    #[test]
    fn test_invalid_edge_endpoints() {
        let graph = create_graph();

        // Accessing the edge endpoints of a non-existing edge should fail
        assert!(graph
            .edge_endpoints(&50)
            .is_err_and(|e| matches!(e, GraphError::IndexError(_))));
    }

    #[test]
    fn test_edge_indices() {
        let graph = create_graph();

        let edge_indices = [0, 1, 2];

        for edge_index in graph.edge_indices() {
            assert!(edge_indices.contains(edge_index));
        }
    }

    #[test]
    fn test_edges_connecting() {
        let graph = create_graph();

        let first_index = "0".into();
        let second_index = "1".into();
        let edges_connecting = graph.edges_connecting(&first_index, &second_index);

        assert_eq!(vec![&0], edges_connecting.collect::<Vec<_>>());

        let first_index = "0".into();
        let second_index = "3".into();
        let edges_connecting = graph.edges_connecting(&first_index, &second_index);

        assert_eq!(0, edges_connecting.count());
    }

    #[test]
    fn test_contains_edge() {
        let graph = create_graph();

        assert!(graph.contains_edge(&0));

        assert!(!graph.contains_edge(&50));
    }

    #[test]
    fn test_neighbors() {
        let graph = create_graph();

        let neighbors = graph.neighbors(&"0".into()).unwrap();

        assert_eq!(2, neighbors.count());
    }

    #[test]
    fn test_invalid_neighbors() {
        let graph = create_graph();

        assert!(graph
            .neighbors(&"50".into())
            .is_err_and(|e| matches!(e, GraphError::IndexError(_))));
    }
}
