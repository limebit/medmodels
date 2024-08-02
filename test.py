from typing import Callable, List, Union


class EdgeOperation: ...


class EdgeIndex: ...


class EdgeOperandValue:
    def less_than(
        self, other: Union["EdgeOperandValues", "EdgeOperandValue"]
    ) -> None: ...
    def add(self, value: int) -> "EdgeOperandValue": ...


class EdgeOperandValues:
    def less_than(self, other: "EdgeOperandValues") -> None: ...
    def max(self) -> "EdgeOperandValue": ...


class EdgeOperand:
    def in_group(self, group: str) -> None: ...
    def attribute(self, attribute: str) -> EdgeOperandValues: ...
    def connects_to(self, query: Callable[["NodeOperand"], None]) -> "EdgeOperand": ...


class NodeOperation: ...


class NodeIndex: ...


class NodeOperand:
    def outgoing_edges(self) -> EdgeOperand: ...
    def in_group(self, group: str) -> None: ...


class MedRecord:
    def select_nodes(self, query: Callable[[NodeOperand], None]) -> List[NodeIndex]: ...


medrecord = MedRecord()


def query(node: NodeOperand):
    edges_to_treatment = node.outgoing_edges().connects_to(
        lambda node2: node2.in_group("treatment")
    )

    edges_to_outcome = node.outgoing_edges().connects_to(
        lambda node2: node2.in_group("outcome")
    )

    max_time_edge = edges_to_treatment.attribute("time").max().add(5)

    max_time_edge.less_than(edges_to_outcome.attribute("time"))


medrecord.select_nodes(query)
