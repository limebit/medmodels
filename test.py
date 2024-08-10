from medmodels._medmodels import PyMedRecord, PyNodeOperand

nodes = [
    ("0", {}),
    ("1", {}),
    ("2", {}),
    ("3", {}),
]

edges = [
    ("0", "1", {"time": 0}),
    ("0", "1", {"time": 2}),
    ("0", "1", {"time": 3}),
    ("0", "1", {"time": 4}),
    ("0", "2", {"time": 5}),
    ("2", "0", {"time": 5}),
]

medrecord = PyMedRecord.from_tuples(nodes, edges)

medrecord.add_group("treatment", ["1"], None)
medrecord.add_group("outcome", ["2"], None)


def query(node: PyNodeOperand):
    edges_to_treatment = node.outgoing_edges()
    edges_to_treatment.target_node().in_group("treatment")

    edges_to_outcome = node.outgoing_edges()
    edges_to_outcome.target_node().in_group("outcome")

    max_time_edge = edges_to_treatment.attribute("time").max()

    max_time_edge.less_than(edges_to_outcome.attribute("time"))


print(medrecord.select_nodes(query))
