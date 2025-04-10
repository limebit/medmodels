import unittest
from typing import List, Tuple
from datetime import datetime, timedelta

import pytest

from medmodels import MedRecord
from medmodels.medrecord.querying import (
    EdgeDirection,
    EdgeIndexOperand,
    EdgeIndicesOperand,
    EdgeOperand,
    NodeIndexOperand,
    NodeIndicesOperand,
    NodeOperand,
    MultipleValuesOperand,
    AttributesTreeOperand,
    MultipleAttributesOperand,
    SingleAttributeOperand,
    SingleValueOperand,
)
from medmodels.medrecord.types import Attributes, NodeIndex, EdgeIndex, MedRecordValue


def query_node(node: NodeOperand) -> None:
    node.index().equal_to("pat_1")

def query_edge(edge: EdgeOperand) -> None:
    edge.index().equal_to(0)


class TestNodeOperand(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the test environment with a diverse MedRecord."""
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_node_operand_attribute_simple(self):
        def query(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            return node.attribute("gender")

        assert self.medrecord.query_nodes(query) == {'pat_1': 'M'}

    def test_node_operand_attributes(self):
        def query(node: NodeOperand) -> AttributesTreeOperand:
            query_node(node)
            return node.attributes()

        result = {key: sorted(value) for key, value in self.medrecord.query_nodes(query).items()}
        assert result == {'pat_1': ['age', 'gender']}

    def test_node_operand_index(self):
        def query(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            return node.index()

        assert self.medrecord.query_nodes(query) == ['pat_1']

    def test_node_operand_in_group(self):
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group("patient")
            return node.index()

        assert sorted(self.medrecord.query_nodes(query1)) == ['pat_1', 'pat_2', 'pat_3', 'pat_4', 'pat_5']

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group(["patient", "diagnosis"]) # Must be in BOTH
            return node.index()

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes_to_group("diagnosis", "pat_1")

        assert sorted(self.medrecord.query_nodes(query2)) == ['pat_1']

    def test_node_operand_has_attribute(self):
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.has_attribute("age")
            return node.index()

        assert self.medrecord.query_nodes(query1) == ['pat_1']

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.has_attribute(["gender", "age"])
            return node.index()

        assert self.medrecord.query_nodes(query2) == ['pat_1']

    def test_node_operand_edges(self):
        def query1(node: NodeOperand) -> EdgeIndicesOperand:
            query_node(node)
            return node.edges(EdgeDirection.OUTGOING).index()

        # TODO: revisit after issue #376
        assert 45 in self.medrecord.query_nodes(query1)

        def query2(node: NodeOperand) -> EdgeIndicesOperand:
            query_node(node)
            return node.edges(EdgeDirection.INCOMING).index()

        # TODO: revisit after issue #376
        assert 0 in self.medrecord.query_nodes(query2)

        def query3(node: NodeOperand) -> EdgeIndicesOperand:
            query_node(node)
            return node.edges(EdgeDirection.BOTH).index()

        # TODO: revisit after issue #376
        assert 0 in self.medrecord.query_nodes(query3)

    def test_node_operand_neighbors(self):
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            neighbors = node.neighbors(EdgeDirection.INCOMING)
            neighbors.in_group("procedure")
            return neighbors.index()

        # TODO: revisit after issue #376
        assert 'procedure_301884003' in self.medrecord.query_nodes(query1)

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            neighbors = node.neighbors(EdgeDirection.OUTGOING)
            neighbors.in_group("procedure")
            return neighbors.index()

        # TODO: revisit after issue #376
        assert 'procedure_301884003' in self.medrecord.query_nodes(query2)

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            neighbors = node.neighbors(EdgeDirection.BOTH)
            neighbors.in_group("procedure")
            return neighbors.index()

        # TODO: revisit after issue #376
        assert 'procedure_301884003' in self.medrecord.query_nodes(query3)

    def test_node_operand_either_or(self):
        def query(node: NodeOperand) -> NodeIndicesOperand:
            node.either_or(
                lambda node: node.attribute("age").greater_than(90),
                lambda node: node.attribute("age").less_than(20),
            )
            return node.index()

        assert sorted(self.medrecord.query_nodes(query)) == ['pat_3', 'pat_4']

    def test_node_operand_exclude(self):
        def query(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group("patient")
            node.exclude(lambda node: node.attribute("age").greater_than(30))
            return node.index()

        assert sorted(self.medrecord.query_nodes(query)) == ['pat_2', 'pat_4']

    def test_node_operand_clone(self):
        def query(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group("patient")
            clone = node.clone()
            node.attribute("age").greater_than(30)
            return clone.index()

        assert sorted(self.medrecord.query_nodes(query)) == ['pat_1', 'pat_2', 'pat_3', 'pat_4', 'pat_5']

class TestEdgeOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_edge_operand_attribute_simple(self):
        def query(edge: EdgeOperand) -> MultipleValuesOperand:
            query_edge(edge)
            return edge.attribute("time")

        assert self.medrecord.query_edges(query) == {0: datetime(2014, 2, 6, 0, 0)}

    def test_edge_operand_attributes(self):
        def query(edge: EdgeOperand) -> AttributesTreeOperand:
            query_edge(edge)
            return edge.attributes()

        result = {key: sorted(value) for key, value in self.medrecord.query_edges(query).items()}
        assert result == {0: ['duration_days', 'time']}

    def test_edge_operand_index(self):
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_edge(edge)
            return edge.index()

        assert self.medrecord.query_edges(query) == [0]

    def test_edge_operand_in_group(self):
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_edge(edge)
            edge.in_group("patient_diagnosis")
            return edge.index()

        assert self.medrecord.query_edges(query) == [0]

        def query2(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_edge(edge)
            edge.in_group(["patient_diagnosis", "treatment"])
            return edge.index()
        
        self.medrecord.unfreeze_schema()
        self.medrecord.add_group("treatment")
        self.medrecord.add_edges_to_group("treatment", 0)

        assert self.medrecord.query_edges(query2) == [0]

    def test_edge_operand_has_attribute(self):
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_edge(edge)
            edge.has_attribute("time")
            return edge.index()
        
        assert self.medrecord.query_edges(query) == [0]

    def test_edge_operand_source_node(self):
        def query(edge: EdgeOperand) -> NodeIndicesOperand:
            query_edge(edge)
            return edge.source_node().index()

        # TODO: revisit after issue #376
        result = self.medrecord.query_edges(query)  # pyright: ignore[reportUnusedVariable]
        # assert result == ['pat_1']
    
    def test_edge_operand_target_node(self):
        def query(edge: EdgeOperand) -> NodeIndicesOperand:
            query_edge(edge)
            return edge.target_node().index()

        # TODO: revisit after issue #376
        result = self.medrecord.query_edges(query)  # pyright: ignore[reportUnusedVariable]
        # assert result == ['pat_2']

    def test_edge_operand_either_or(self):
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.either_or(
                lambda edge: query_edge(edge),
                lambda edge: edge.index().equal_to(1),
            )
            return edge.index()

        assert sorted(self.medrecord.query_edges(query)) == [0, 1]

    def test_edge_operand_exclude(self):
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(5)
            edge.exclude(lambda edge: edge.attribute("duration_days").greater_than(1))
            return edge.index()

        assert sorted(self.medrecord.query_edges(query)) == [0, 4]

    def test_edge_operand_clone(self):
        def query(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.index().less_than(5)
            clone = edge.clone()
            edge.attribute("duration_days").less_than(1)
            return clone.index()
        
        assert sorted(self.medrecord.query_edges(query)) == [0, 1, 2, 3, 4]


class TestMultipleValuesOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_multiple_values_operand_numeric(self):

        assert self.medrecord.query_nodes(lambda node: node.attribute("age").min()) == ('pat_4', 19)

        assert self.medrecord.query_nodes(lambda node: node.attribute("age").max()) == ('pat_3', 96)

        assert self.medrecord.query_nodes(lambda node: node.attribute("age").mean()) == 43.2

        assert self.medrecord.query_nodes(lambda node: node.attribute("age").median()) == 37

        medrecord_mode = self.medrecord.clone()
        medrecord_mode.unfreeze_schema()
        medrecord_mode.add_nodes(("pat_6", {"age": 22}))
        assert medrecord_mode.query_nodes(lambda node: node.attribute("age").mode()) == 22

        std = float(self.medrecord.query_nodes(lambda node: node.attribute("age").std()))  # pyright: ignore[reportArgumentType]
        assert round(std, 2) == 27.79

        var = float(self.medrecord.query_nodes(lambda node: node.attribute("age").var()))  # pyright: ignore[reportArgumentType]
        assert round(var, 2) == 772.56

        assert self.medrecord.query_nodes(lambda node: node.attribute("age").count()) == 5

        assert self.medrecord.query_nodes(lambda node: node.attribute("age").sum()) == 216

        def query_first(node: NodeOperand) -> SingleValueOperand:
            query_node(node)
            return node.attribute("age").first()
        
        # TODO: need to implement sorting for first
        assert self.medrecord.query_nodes(query_first) == ('pat_1', 42)

        def query_last(node: NodeOperand) -> SingleValueOperand:
            query_node(node)
            return node.attribute("age").last()

        # TODO: need to implement sorting for last
        assert self.medrecord.query_nodes(query_last) == ('pat_1', 42)

    def test_multiple_values_operand_datatypes(self):
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.attribute("gender").is_string()
            return node.index()
        
        assert self.medrecord.query_nodes(query1) == ['pat_1']

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.attribute("age").is_string()
            return node.index()

        assert self.medrecord.query_nodes(query2) == []

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.attribute("age").is_int()
            return node.index()
        
        assert self.medrecord.query_nodes(query3) == ['pat_1']

        def query4(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_edge(edge)
            edge.attribute("time").is_datetime()
            return edge.index()
        
        assert self.medrecord.query_edges(query4) == [0]

        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(("pat_1", "pat_2", {"float_attribute": 2.3}))

        def query5(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("float_attribute").is_float()
            return edge.index()
        
        assert self.medrecord.query_edges(query5) == [160]

        # def query6(edge: EdgeOperand) -> EdgeIndicesOperand:
        #     query_edge(edge)
        #     edge.attribute("duration_days").is_duration()
        #     return edge.index()
        
        # TODO: revisit after changing the duration type in example dataset
        # assert self.medrecord.query_edges(query6) == [1]

        self.medrecord.add_edges(("pat_1", "pat_3", {"bool_attribute": True}))
        
        def query7(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("bool_attribute").is_bool()
            return edge.index()
        
        assert self.medrecord.query_edges(query7) == [161]

        self.medrecord.add_edges(("pat_1", "pat_4", {"null_attribute": None}))

        def query8(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("null_attribute").is_null()
            return edge.index()
        
        assert self.medrecord.query_edges(query8) == [162]

    def test_multiple_values_operand_comparisons(self):
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").is_max()
            return node.index()
        
        assert self.medrecord.query_nodes(query1) == ['pat_3']

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").is_min()
            return node.index()
        
        assert self.medrecord.query_nodes(query2) == ['pat_4']

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").greater_than(90)
            return node.index()
        
        assert self.medrecord.query_nodes(query3) == ['pat_3']

        def query4(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").less_than(20)
            return node.index()
        
        assert self.medrecord.query_nodes(query4) == ['pat_4']

        def query5(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").equal_to(42)
            return node.index()
        
        assert self.medrecord.query_nodes(query5) == ['pat_1']
    
        def query6(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").not_equal_to(42)
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query6)) == ['pat_2', 'pat_3', 'pat_4', 'pat_5']

        def query7(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").is_in([42, 19])
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query7)) == ['pat_1', 'pat_4']

        def query8(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").is_not_in([42, 19])
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query8)) == ['pat_2', 'pat_3', 'pat_5']

        def query9(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").less_than_or_equal_to(42)
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query9)) == ['pat_1', 'pat_2', 'pat_4', 'pat_5']  

        def query10(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").greater_than_or_equal_to(42)
            return node.index()

        assert sorted(self.medrecord.query_nodes(query10)) == ['pat_1', 'pat_3']

        def query11(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").starts_with("1")
            return node.index()
        
        assert self.medrecord.query_nodes(query11) == ['pat_4']

        def query12(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").ends_with("9")
            return node.index()
        
        assert self.medrecord.query_nodes(query12) == ['pat_4']

        def query13(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("gender").contains("M")
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query13)) == ['pat_1', 'pat_4', 'pat_5']

    def test_multiple_values_operand_operations(self):
        def query1(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.add(10)
            return age
        
        assert self.medrecord.query_nodes(query1) == {'pat_1': 52}

        def query2(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.subtract(10)
            return age
        
        assert self.medrecord.query_nodes(query2) == {'pat_1': 32}

        def query3(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.multiply(10)
            return age
        
        assert self.medrecord.query_nodes(query3) == {'pat_1': 420}

        def query4(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.divide(10)
            return age
        
        assert self.medrecord.query_nodes(query4) == {'pat_1': 4.2}

        def query5(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.modulo(10)
            return age
        
        assert self.medrecord.query_nodes(query5) == {'pat_1': 2}

        def query6(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.power(2)
            return age
        
        assert self.medrecord.query_nodes(query6) == {'pat_1': 1764}

        def query7(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.divide(5)
            age.floor()
            return age
        
        assert self.medrecord.query_nodes(query7) == {'pat_1': 8}

        def query8(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.divide(5)
            age.ceil()
            return age

        assert self.medrecord.query_nodes(query8) == {'pat_1': 9}

        def query9(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.divide(5)
            age.round()
            return age
        
        assert self.medrecord.query_nodes(query9) == {'pat_1': 8}

        def query10(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.sqrt()
            return age
        
        result = self.medrecord.query_nodes(query10)
        result = {key: round(float(value), 2) for key, value in result.items()} # pyright: ignore[reportArgumentType]
        assert result == {'pat_1': 6.48}

        def query11(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            age = node.attribute("age")
            age.subtract(45)
            age.absolute()
            return age
        
        assert self.medrecord.query_nodes(query11) == {'pat_1': 3}

        def query12(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            age = node.attribute("gender")
            age.lowercase()
            return age
        
        assert self.medrecord.query_nodes(query12) == {'pat_1': 'm'}

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes(("pat_6", {"spacing": " hello "}))

        def query13(node: NodeOperand) -> MultipleValuesOperand:
            age = node.attribute("spacing")
            age.uppercase()
            return age
        
        assert self.medrecord.query_nodes(query13) == {'pat_6': ' HELLO '}

        def query14(node: NodeOperand) -> MultipleValuesOperand:
            age = node.attribute("spacing")
            age.trim()
            return age
        
        assert self.medrecord.query_nodes(query14) == {'pat_6': 'hello'}

        def query15(node: NodeOperand) -> MultipleValuesOperand:
            age = node.attribute("spacing")
            age.trim_start()
            return age
        
        assert self.medrecord.query_nodes(query15) == {'pat_6': 'hello '}

        def query16(node: NodeOperand) -> MultipleValuesOperand:
            age = node.attribute("spacing")
            age.trim_end()
            return age
        
        assert self.medrecord.query_nodes(query16) == {'pat_6': ' hello'}

        def query17(node: NodeOperand) -> MultipleValuesOperand:
            age = node.attribute("spacing")
            age.slice(0, 3)
            return age
        
        assert self.medrecord.query_nodes(query17) == {'pat_6': ' he'}

        def query18(node: NodeOperand) -> MultipleValuesOperand:
            node.attribute("age").either_or(
                lambda attribute: attribute.greater_than(90),
                lambda attribute: attribute.less_than(20),
            )
            return node.attribute("age")
        
        result = self.medrecord.query_nodes(query18)
        assert self.medrecord.query_nodes(query18) == {'pat_3': 96, 'pat_4': 19}

        def query19(node: NodeOperand) -> MultipleValuesOperand:
            node.attribute("age").exclude(
                lambda attribute: attribute.less_than(90),
            )
            return node.attribute("age")

        assert self.medrecord.query_nodes(query19) == {'pat_3': 96}

        def query20(node: NodeOperand) -> MultipleValuesOperand:
            query_node(node)
            clone = node.attribute("age").clone()
            node.attribute("age").add(10)
            return clone
        
        assert self.medrecord.query_nodes(query20) == {'pat_1': 42}

class TestSingleValueOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_single_value_operand_datatypes_max(self):
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.attribute("gender").max().is_string()
            return node.index()
        
        assert self.medrecord.query_nodes(query1) == ['pat_1']

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.attribute("age").max().is_string()
            return node.index()
        
        assert self.medrecord.query_nodes(query2) == []

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            query_node(node)
            node.attribute("age").max().is_int()
            return node.index()
        
        assert self.medrecord.query_nodes(query3) == ['pat_1']

        def query4(edge: EdgeOperand) -> EdgeIndicesOperand:
            query_edge(edge)
            edge.attribute("time").max().is_datetime()
            return edge.index()
        
        assert self.medrecord.query_edges(query4) == [0]

        self.medrecord.unfreeze_schema()
        self.medrecord.add_edges(("pat_1", "pat_2", {"float_attribute": 2.3}))

        def query5(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("float_attribute").max().is_float()
            return edge.index()
        
        assert self.medrecord.query_edges(query5) == [160]

        # TODO: revisit after changing the duration type in example dataset
        # def query6(edge: EdgeOperand) -> EdgeIndicesOperand:
        #     query_edge(edge)
        #     edge.attribute("duration_days").max().is_duration()
        #     return edge.index()
        # assert self.medrecord.query_edges(query6) == [1]

        self.medrecord.add_edges(("pat_1", "pat_3", {"bool_attribute": True}))

        def query7(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("bool_attribute").max().is_bool()
            return edge.index()
        
        assert self.medrecord.query_edges(query7) == [161]

        self.medrecord.add_edges(("pat_1", "pat_4", {"null_attribute": None}))

        def query8(edge: EdgeOperand) -> EdgeIndicesOperand:
            edge.attribute("null_attribute").max().is_null()
            return edge.index()
        
        assert self.medrecord.query_edges(query8) == [162]

    def test_single_node_operand_comparisons_max(self):
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").max().greater_than(90)
            return node.index()
        
        assert self.medrecord.query_nodes(query1) == ['pat_3']

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").max().less_than(20)
            return node.index()
        
        assert self.medrecord.query_nodes(query2) == ['pat_4']

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").max().equal_to(42)
            return node.index()
        
        assert self.medrecord.query_nodes(query3) == ['pat_1']

        def query4(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").max().not_equal_to(42)
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query4)) == ['pat_2', 'pat_3', 'pat_4', 'pat_5']

        def query5(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").max().is_in([42, 19])
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query5)) == ['pat_1', 'pat_4']

        def query6(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").max().is_not_in([42, 19])
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query6)) == ['pat_2', 'pat_3', 'pat_5']

        def query7(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").max().less_than_or_equal_to(42)
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query7)) == ['pat_1', 'pat_2', 'pat_4', 'pat_5']  

        def query8(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").max().greater_than_or_equal_to(42)
            return node.index()

        assert sorted(self.medrecord.query_nodes(query8)) == ['pat_1', 'pat_3']

        def query9(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").max().starts_with("1")
            return node.index()
        
        assert self.medrecord.query_nodes(query9) == ['pat_4']

        def query10(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("age").max().ends_with("9")
            return node.index()
        
        assert self.medrecord.query_nodes(query10) == ['pat_4']

        def query11(node: NodeOperand) -> NodeIndicesOperand:
            node.attribute("gender").max().contains("M")
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query11)) == ['pat_1', 'pat_4', 'pat_5']


class TestNodeIndexOperand(unittest.TestCase):
    def setUp(self) -> None:
        self.medrecord = MedRecord.from_simple_example_dataset()

    def test_node_index_operand_comparisons(self):
        def query1(node: NodeOperand) -> NodeIndicesOperand:
            node.index().equal_to("pat_1")
            return node.index()
        
        assert self.medrecord.query_nodes(query1) == ['pat_1']

        def query2(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group("patient")
            node.index().not_equal_to("pat_1")
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query2)) == ['pat_2', 'pat_3', 'pat_4', 'pat_5']

        self.medrecord.unfreeze_schema()
        self.medrecord.add_nodes((0, {}))
        self.medrecord.add_nodes((1, {}))

        def query3(node: NodeOperand) -> NodeIndicesOperand:
            node.index().is_in([0, 1])
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query3)) == [0, 1]

        def query4(node: NodeOperand) -> NodeIndicesOperand:
            node.in_group("patient")
            node.index().is_not_in(["pat_1", "pat_2"])
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query4)) == ['pat_3', 'pat_4', 'pat_5']

        def query5(node: NodeOperand) -> NodeIndicesOperand:
            node.index().less_than(1)
            return node.index()
        
        assert self.medrecord.query_nodes(query5) == [0]

        def query6(node: NodeOperand) -> NodeIndicesOperand:
            node.index().less_than_or_equal_to(1)
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query6)) == [0, 1]

        def query7(node: NodeOperand) -> NodeIndicesOperand:
            node.index().greater_than(0)
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query7)) == [1]

        def query8(node: NodeOperand) -> NodeIndicesOperand:
            node.index().greater_than_or_equal_to(0)
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query8)) == [0, 1]

        def query9(node: NodeOperand) -> NodeIndicesOperand:
            node.index().starts_with("pat_")
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query9)) == ['pat_1', 'pat_2', 'pat_3', 'pat_4', 'pat_5']

        def query10(node: NodeOperand) -> NodeIndicesOperand:
            node.index().ends_with("_1")
            return node.index()
        
        assert self.medrecord.query_nodes(query10) == ['pat_1']

        def query11(node: NodeOperand) -> NodeIndicesOperand:
            node.index().contains("at")
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query11)) == ['pat_1', 'pat_2', 'pat_3', 'pat_4', 'pat_5']

        def query12(node: NodeOperand) -> NodeIndicesOperand:
        
            node.index().add(2)
            return node.index()
        
        assert sorted(self.medrecord.query_nodes(query12)) == [2,3]
        



        







if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNodeOperand))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEdgeOperand))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMultipleValuesOperand))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSingleValueOperand))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNodeIndexOperand))


    unittest.TextTestRunner(verbosity=2).run(suite)
