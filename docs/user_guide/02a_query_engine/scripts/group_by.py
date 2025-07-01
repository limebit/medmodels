# ruff: noqa: D100, D103
from medmodels import MedRecord
from medmodels.medrecord.querying import (
    EdgeIndexGroupOperand,
    EdgeMultipleValuesWithIndexGroupOperand,
    EdgeOperand,
    EdgeOperandGroupDiscriminator,
    NodeMultipleValuesWithIndexGroupOperand,
    NodeOperand,
    NodeOperandGroupDiscriminator,
    NodeSingleValueWithoutIndexGroupOperand,
)

medrecord = MedRecord().from_simple_example_dataset()


def query_node_group_by_gender(
    node: NodeOperand,
) -> NodeMultipleValuesWithIndexGroupOperand:
    grouped_nodes = node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))

    return grouped_nodes.attribute("age")


medrecord.query_nodes(query_node_group_by_gender)


def query_node_group_by_gender_mean(
    node: NodeOperand,
) -> NodeSingleValueWithoutIndexGroupOperand:
    grouped_nodes = node.group_by(NodeOperandGroupDiscriminator.Attribute("gender"))
    age_groups = grouped_nodes.attribute("age")

    return age_groups.mean()


medrecord.query_nodes(query_node_group_by_gender_mean)


def query_edge_group_by_source_node(
    edge: EdgeOperand,
) -> EdgeMultipleValuesWithIndexGroupOperand:
    edge.index().less_than(20)
    grouped_edges = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())

    return grouped_edges.attribute("time")


medrecord.query_edges(query_edge_group_by_source_node)


def query_edge_group_by_count_edges(edge: EdgeOperand) -> EdgeIndexGroupOperand:
    grouped_edges = edge.group_by(EdgeOperandGroupDiscriminator.SourceNode())

    return grouped_edges.index().count()


medrecord.query_edges(query_edge_group_by_count_edges)
