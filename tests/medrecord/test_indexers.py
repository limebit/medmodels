import unittest

import pytest

from medmodels import MedRecord
from medmodels.medrecord.querying import (
    EdgeIndexOperand,
    EdgeIndicesOperand,
    EdgeOperand,
    NodeIndexOperand,
    NodeIndicesOperand,
    NodeOperand,
)


def create_medrecord() -> MedRecord:
    return MedRecord.from_tuples(
        [
            (0, {"foo": "bar", "bar": "foo", "lorem": "ipsum"}),
            (1, {"foo": "bar", "bar": "foo"}),
            (2, {"foo": "bar", "bar": "foo"}),
            (3, {"foo": "bar", "bar": "test"}),
        ],
        [
            (0, 1, {"foo": "bar", "bar": "foo", "lorem": "ipsum"}),
            (1, 2, {"foo": "bar", "bar": "foo"}),
            (2, 3, {"foo": "bar", "bar": "foo"}),
            (3, 0, {"foo": "bar", "bar": "test"}),
        ],
    )


def node_greater_than_or_equal_two(node: NodeOperand) -> NodeIndicesOperand:
    node.index().greater_than_or_equal_to(2)

    return node.index()


def node_max(node: NodeOperand) -> NodeIndexOperand:
    node.index().greater_than(2)

    return node.index().max()


def node_max_greater_than_3(node: NodeOperand) -> NodeIndexOperand:
    max_index = node.index().max()
    max_index.greater_than(3)

    return max_index


def node_greater_than_three(node: NodeOperand) -> NodeIndicesOperand:
    node.index().greater_than(3)

    return node.index()


def node_less_than_two(node: NodeOperand) -> NodeIndicesOperand:
    node.index().less_than(2)

    return node.index()


def edge_greater_than_or_equal_two(edge: EdgeOperand) -> EdgeIndicesOperand:
    edge.index().greater_than_or_equal_to(2)

    return edge.index()


def edge_greater_than_three(edge: EdgeOperand) -> EdgeIndicesOperand:
    edge.index().greater_than(3)

    return edge.index()


def edge_less_than_two(edge: EdgeOperand) -> EdgeIndicesOperand:
    edge.index().less_than(2)

    return edge.index()


def edge_max(edge: EdgeOperand) -> EdgeIndexOperand:
    edge.index().greater_than(2)

    return edge.index().max()


def edge_max_greater_than_3(edge: EdgeOperand) -> EdgeIndexOperand:
    max_index = edge.index().max()
    max_index.greater_than(3)

    return max_index


class TestIndexers(unittest.TestCase):
    def test_node_getitem(self) -> None:
        medrecord = create_medrecord()

        assert medrecord.node[0] == {"foo": "bar", "bar": "foo", "lorem": "ipsum"}

        # Accessing a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.node[50]

        assert medrecord.node[0, "foo"] == "bar"

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.node[0, "test"]

        assert medrecord.node[0, ["foo", "bar"]] == {"foo": "bar", "bar": "foo"}

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.node[0, ["foo", "test"]]

        assert medrecord.node[0, :] == {"foo": "bar", "bar": "foo", "lorem": "ipsum"}

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[0, 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[0, :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[0, ::1]

        assert medrecord.node[[0, 1]] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
        }

        with pytest.raises(IndexError):
            medrecord.node[[0, 50]]

        assert medrecord.node[[0, 1], "foo"] == {0: "bar", 1: "bar"}

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.node[[0, 1], "test"]

        # Accessing a key that doesn't exist in all nodes should fail
        with pytest.raises(KeyError):
            medrecord.node[[0, 1], "lorem"]

        assert medrecord.node[[0, 1], ["foo", "bar"]] == {
            0: {"foo": "bar", "bar": "foo"},
            1: {"foo": "bar", "bar": "foo"},
        }

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.node[[0, 1], ["foo", "test"]]

        # Accessing a key that doesn't exist in all nodes should fail
        with pytest.raises(KeyError):
            medrecord.node[[0, 1], ["foo", "lorem"]]

        assert medrecord.node[[0, 1], :] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[[0, 1], 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[[0, 1], :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[[0, 1], ::1]

        assert medrecord.node[node_greater_than_or_equal_two] == {
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }
        assert medrecord.node[node_max] == {"foo": "bar", "bar": "test"}

        # Empty query should not fail when using a NodeIndicesOperand
        assert medrecord.node[node_greater_than_three] == {}

        # Query should fail when using a NodeIndexOperand with no return value
        with pytest.raises(IndexError, match="The query returned no results"):
            medrecord.node[node_max_greater_than_3]

        assert medrecord.node[node_greater_than_or_equal_two, "foo"] == {
            2: "bar",
            3: "bar",
        }

        assert medrecord.node[node_max, "foo"] == "bar"

        with pytest.raises(IndexError, match="The query returned no results"):
            medrecord.node[node_max_greater_than_3, "foo"]

        # Accessing a non-existing key should fail
        with pytest.raises(
            KeyError,
        ):
            medrecord.node[node_greater_than_or_equal_two, "test"]

        assert medrecord.node[node_greater_than_or_equal_two, ["foo", "bar"]] == {
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        assert medrecord.node[node_max, ["foo", "bar"]] == {"foo": "bar", "bar": "test"}

        with pytest.raises(IndexError, match="The query returned no results"):
            medrecord.node[node_max_greater_than_3, ["foo", "bar"]]

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.node[node_greater_than_or_equal_two, ["foo", "test"]]

        # Accessing a key that doesn't exist in all nodes should fail
        with pytest.raises(KeyError):
            medrecord.node[node_less_than_two, ["foo", "lorem"]]

        assert medrecord.node[node_greater_than_or_equal_two, :] == {
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }
        assert medrecord.node[node_max, :] == {"foo": "bar", "bar": "test"}

        with pytest.raises(IndexError, match="The query returned no results"):
            medrecord.node[node_max_greater_than_3, :]

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[node_greater_than_or_equal_two, 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[node_greater_than_or_equal_two, :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[node_greater_than_or_equal_two, ::1]

        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[:1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[::1]

        assert medrecord.node[:, "foo"] == {0: "bar", 1: "bar", 2: "bar", 3: "bar"}

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.node[:, "test"]

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[1:, "foo"]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[:1, "foo"]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[::1, "foo"]

        assert medrecord.node[:, ["foo", "bar"]] == {
            0: {"foo": "bar", "bar": "foo"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.node[:, ["foo", "test"]]

        # Accessing a key that doesn't exist in all nodes should fail
        with pytest.raises(KeyError):
            medrecord.node[:, ["foo", "lorem"]]

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[1:, ["foo", "bar"]]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[:1, ["foo", "bar"]]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[::1, ["foo", "bar"]]

        assert medrecord.node[:, :] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[1:, :]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[:1, :]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[::1, :]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[:, 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[:, :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[:, ::1]

    def test_node_setitem(self) -> None:
        # Updating existing attributes

        medrecord = create_medrecord()
        medrecord.node[0] = {"foo": "bar", "bar": "test"}
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "test"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[[0, 1]] = {"foo": "bar", "bar": "test"}
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "test"},
            1: {"foo": "bar", "bar": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_greater_than_or_equal_two] = {"foo": "test", "bar": "test2"}
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "test", "bar": "test2"},
            3: {"foo": "test", "bar": "test2"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_max] = {"foo": "test", "bar": "test2"}
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "test", "bar": "test2"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_max_greater_than_3] = {"foo": "test", "bar": "test2"}
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[1:] = {"foo": "bar", "bar": "test"}

        medrecord = create_medrecord()
        medrecord.node[:] = {"foo": "bar", "bar": "test"}
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "test"},
            1: {"foo": "bar", "bar": "test"},
            2: {"foo": "bar", "bar": "test"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Updating a non-existing node should fail
        with pytest.raises(IndexError):
            medrecord.node[50] = {"foo": "bar", "test": "test"}

        medrecord = create_medrecord()
        medrecord.node[0, "foo"] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "test", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[0, ["foo", "bar"]] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "test", "bar": "test", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[0, :] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "test", "bar": "test", "lorem": "test"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[0, 1:] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[0, :1] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[0, ::1] = "test"

        medrecord = create_medrecord()
        medrecord.node[[0, 1], "foo"] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "test", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "test", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[[0, 1], ["foo", "bar"]] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "test", "bar": "test", "lorem": "ipsum"},
            1: {"foo": "test", "bar": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[[0, 1], :] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "test", "bar": "test", "lorem": "test"},
            1: {"foo": "test", "bar": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[[0, 1], 1:] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[[0, 1], :1] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[[0, 1], ::1] = "test"

        medrecord = create_medrecord()
        medrecord.node[node_greater_than_or_equal_two] = {"foo": "bar", "bar": "test"}
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "test"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Empty query should not fail
        medrecord.node[node_greater_than_three] = {"foo": "bar", "bar": "test"}

        medrecord = create_medrecord()
        medrecord.node[node_greater_than_or_equal_two, "foo"] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "test", "bar": "foo"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_max, "foo"] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_max_greater_than_3, "foo"] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_greater_than_or_equal_two, ["foo", "bar"]] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "test", "bar": "test"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_max, ["foo", "bar"]] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord.node[node_max_greater_than_3, ["foo", "bar"]] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_greater_than_or_equal_two, :] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "test", "bar": "test"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_max, :] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_max_greater_than_3, :] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[node_greater_than_or_equal_two, 1:] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[node_greater_than_or_equal_two, :1] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[node_greater_than_or_equal_two, ::1] = "test"

        medrecord = create_medrecord()
        medrecord.node[:, "foo"] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "test", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "test", "bar": "foo"},
            2: {"foo": "test", "bar": "foo"},
            3: {"foo": "test", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[1:, "foo"] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[:1, "foo"] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[::1, "foo"] = "test"

        medrecord = create_medrecord()
        medrecord.node[:, ["foo", "bar"]] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "test", "bar": "test", "lorem": "ipsum"},
            1: {"foo": "test", "bar": "test"},
            2: {"foo": "test", "bar": "test"},
            3: {"foo": "test", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[1:, ["foo", "bar"]] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[:1, ["foo", "bar"]] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[::1, ["foo", "bar"]] = "test"

        medrecord = create_medrecord()
        medrecord.node[:, :] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "test", "bar": "test", "lorem": "test"},
            1: {"foo": "test", "bar": "test"},
            2: {"foo": "test", "bar": "test"},
            3: {"foo": "test", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[1:, :] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[:1, :] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[::1, :] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[:, 1:] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[:, :1] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.node[:, ::1] = "test"

        # Adding new attributes

        medrecord = create_medrecord()
        medrecord.node[0, "test"] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum", "test": "test"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[0, ["test", "test2"]] = "test"
        assert medrecord.node[:] == {
            0: {
                "foo": "bar",
                "bar": "foo",
                "lorem": "ipsum",
                "test": "test",
                "test2": "test",
            },
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[[0, 1], "test"] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum", "test": "test"},
            1: {"foo": "bar", "bar": "foo", "test": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[[0, 1], ["test", "test2"]] = "test"
        assert medrecord.node[:] == {
            0: {
                "foo": "bar",
                "bar": "foo",
                "lorem": "ipsum",
                "test": "test",
                "test2": "test",
            },
            1: {"foo": "bar", "bar": "foo", "test": "test", "test2": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_greater_than_or_equal_two, "test"] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo", "test": "test"},
            3: {"foo": "bar", "bar": "test", "test": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_greater_than_or_equal_two, ["test", "test2"]] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo", "test": "test", "test2": "test"},
            3: {"foo": "bar", "bar": "test", "test": "test", "test2": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[:, "test"] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum", "test": "test"},
            1: {"foo": "bar", "bar": "foo", "test": "test"},
            2: {"foo": "bar", "bar": "foo", "test": "test"},
            3: {"foo": "bar", "bar": "test", "test": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[:, ["test", "test2"]] = "test"
        assert medrecord.node[:] == {
            0: {
                "foo": "bar",
                "bar": "foo",
                "lorem": "ipsum",
                "test": "test",
                "test2": "test",
            },
            1: {"foo": "bar", "bar": "foo", "test": "test", "test2": "test"},
            2: {"foo": "bar", "bar": "foo", "test": "test", "test2": "test"},
            3: {"foo": "bar", "bar": "test", "test": "test", "test2": "test"},
        }

        # Adding and updating attributes

        medrecord = create_medrecord()
        medrecord.node[[0, 1], "lorem"] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "test"},
            1: {"foo": "bar", "bar": "foo", "lorem": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[[0, 1], ["lorem", "test"]] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            1: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_less_than_two, "lorem"] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "test"},
            1: {"foo": "bar", "bar": "foo", "lorem": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[node_less_than_two, ["lorem", "test"]] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            1: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[:, "lorem"] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "test"},
            1: {"foo": "bar", "bar": "foo", "lorem": "test"},
            2: {"foo": "bar", "bar": "foo", "lorem": "test"},
            3: {"foo": "bar", "bar": "test", "lorem": "test"},
        }

        medrecord = create_medrecord()
        medrecord.node[:, ["lorem", "test"]] = "test"
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            1: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            2: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            3: {"foo": "bar", "bar": "test", "lorem": "test", "test": "test"},
        }

    def test_node_delitem(self) -> None:
        medrecord = create_medrecord()
        del medrecord.node[0, "foo"]
        assert medrecord.node[:] == {
            0: {"bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Removing from a non-existing node should fail
        with pytest.raises(IndexError):
            del medrecord.node[50, "foo"]

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.node[0, "test"]

        medrecord = create_medrecord()
        del medrecord.node[0, ["foo", "bar"]]
        assert medrecord.node[:] == {
            0: {"lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.node[0, ["foo", "test"]]

        medrecord = create_medrecord()
        del medrecord.node[0, :]
        assert medrecord.node[:] == {
            0: {},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[0, 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[0, :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[0, ::1]

        medrecord = create_medrecord()
        del medrecord.node[[0, 1], "foo"]
        assert medrecord.node[:] == {
            0: {"bar": "foo", "lorem": "ipsum"},
            1: {"bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Removing from a non-existing node should fail
        with pytest.raises(IndexError):
            del medrecord.node[[0, 50], "foo"]

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.node[[0, 1], "test"]

        medrecord = create_medrecord()
        del medrecord.node[[0, 1], ["foo", "bar"]]
        assert medrecord.node[:] == {
            0: {"lorem": "ipsum"},
            1: {},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.node[[0, 1], ["foo", "test"]]

        medrecord = create_medrecord()
        # Removing a key that doesn't exist in all nodes should fail
        with pytest.raises(KeyError):
            del medrecord.node[[0, 1], ["foo", "lorem"]]

        medrecord = create_medrecord()
        del medrecord.node[[0, 1], :]
        assert medrecord.node[:] == {
            0: {},
            1: {},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[[0, 1], 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[[0, 1], :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[[0, 1], ::1]

        medrecord = create_medrecord()
        del medrecord.node[node_greater_than_or_equal_two, "foo"]
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"bar": "foo"},
            3: {"bar": "test"},
        }

        medrecord = create_medrecord()
        del medrecord.node[node_max, "foo"]
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"bar": "test"},
        }

        medrecord = create_medrecord()
        del medrecord.node[node_max_greater_than_3, "foo"]
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Empty query should not fail
        del medrecord.node[node_greater_than_three, "foo"]
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.node[node_greater_than_or_equal_two, "test"]

        medrecord = create_medrecord()
        del medrecord.node[node_greater_than_or_equal_two, ["foo", "bar"]]
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {},
            3: {},
        }

        medrecord = create_medrecord()
        del medrecord.node[node_max, ["foo", "bar"]]
        assert medrecord.node[:] == {
            1: {"foo": "bar", "bar": "foo"},
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            2: {"foo": "bar", "bar": "foo"},
            3: {},
        }

        medrecord = create_medrecord()
        del medrecord.node[node_max_greater_than_3, ["foo", "bar"]]
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.node[node_greater_than_or_equal_two, ["foo", "test"]]

        medrecord = create_medrecord()
        # Removing a key that doesn't exist in all nodes should fail
        with pytest.raises(KeyError):
            del medrecord.node[node_less_than_two, ["foo", "lorem"]]

        medrecord = create_medrecord()
        del medrecord.node[node_greater_than_or_equal_two, :]
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {},
            3: {},
        }

        medrecord = create_medrecord()
        del medrecord.node[node_max, :]
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {},
        }

        medrecord = create_medrecord()
        del medrecord.node[node_max_greater_than_3, :]
        assert medrecord.node[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[node_greater_than_or_equal_two, 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[node_greater_than_or_equal_two, :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[node_greater_than_or_equal_two, ::1]

        medrecord = create_medrecord()
        del medrecord.node[:, "foo"]
        assert medrecord.node[:] == {
            0: {"bar": "foo", "lorem": "ipsum"},
            1: {"bar": "foo"},
            2: {"bar": "foo"},
            3: {"bar": "test"},
        }

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.node[:, "test"]

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[1:, "foo"]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[:1, "foo"]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[::1, "foo"]

        medrecord = create_medrecord()
        del medrecord.node[:, ["foo", "bar"]]
        assert medrecord.node[:] == {0: {"lorem": "ipsum"}, 1: {}, 2: {}, 3: {}}

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.node[:, ["foo", "test"]]

        medrecord = create_medrecord()
        # Removing a key that doesn't exist in all nodes should fail
        with pytest.raises(KeyError):
            del medrecord.node[:, ["foo", "lorem"]]

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[1:, ["foo", "bar"]]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[:1, ["foo", "bar"]]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[::1, ["foo", "bar"]]

        medrecord = create_medrecord()
        del medrecord.node[:, :]
        assert medrecord.node[:] == {0: {}, 1: {}, 2: {}, 3: {}}

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[1:, :]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[:1, :]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[::1, :]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[:, 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[:, :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.node[:, ::1]

    def test_edge_getitem(self) -> None:
        medrecord = create_medrecord()

        assert medrecord.edge[0] == {"foo": "bar", "bar": "foo", "lorem": "ipsum"}

        # Accessing a non-existing edge should fail
        with pytest.raises(IndexError):
            medrecord.edge[50]

        assert medrecord.edge[0, "foo"] == "bar"

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.edge[0, "test"]

        assert medrecord.edge[0, ["foo", "bar"]] == {"foo": "bar", "bar": "foo"}

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.edge[0, ["foo", "test"]]

        assert medrecord.edge[0, :] == {"foo": "bar", "bar": "foo", "lorem": "ipsum"}

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[0, 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[0, :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[0, ::1]

        assert medrecord.edge[[0, 1]] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
        }

        with pytest.raises(IndexError):
            medrecord.edge[[0, 50]]

        assert medrecord.edge[[0, 1], "foo"] == {0: "bar", 1: "bar"}

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.edge[[0, 1], "test"]

        # Accessing a key that doesn't exist in all edges should fail
        with pytest.raises(KeyError):
            medrecord.edge[[0, 1], "lorem"]

        assert medrecord.edge[[0, 1], ["foo", "bar"]] == {
            0: {"foo": "bar", "bar": "foo"},
            1: {"foo": "bar", "bar": "foo"},
        }

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.edge[[0, 1], ["foo", "test"]]

        # Accessing a key that doesn't exist in all edges should fail
        with pytest.raises(KeyError):
            medrecord.edge[[0, 1], ["foo", "lorem"]]

        assert medrecord.edge[[0, 1], :] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[[0, 1], 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[[0, 1], :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[[0, 1], ::1]

        assert medrecord.edge[edge_greater_than_or_equal_two] == {
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        assert medrecord.edge[edge_max] == {"foo": "bar", "bar": "test"}

        with pytest.raises(IndexError, match="The query returned no results"):
            medrecord.edge[edge_max_greater_than_3]

        # Empty query should not fail
        assert medrecord.edge[edge_greater_than_three] == {}

        assert medrecord.edge[edge_greater_than_or_equal_two, "foo"] == {
            2: "bar",
            3: "bar",
        }

        assert medrecord.edge[edge_max, "foo"] == "bar"

        with pytest.raises(IndexError, match="The query returned no results"):
            medrecord.edge[edge_max_greater_than_3, "foo"]

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.edge[edge_greater_than_or_equal_two, "test"]

        assert medrecord.edge[edge_greater_than_or_equal_two, ["foo", "bar"]] == {
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        assert medrecord.edge[edge_max, ["foo", "bar"]] == {"foo": "bar", "bar": "test"}

        with pytest.raises(IndexError, match="The query returned no results"):
            medrecord.edge[edge_max_greater_than_3, ["foo", "bar"]]

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.edge[edge_greater_than_or_equal_two, ["foo", "test"]]

        # Accessing a key that doesn't exist in all edges should fail
        with pytest.raises(KeyError):
            medrecord.edge[edge_less_than_two, ["foo", "lorem"]]

        assert medrecord.edge[edge_greater_than_or_equal_two, :] == {
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        assert medrecord.edge[edge_max, :] == {"foo": "bar", "bar": "test"}

        with pytest.raises(IndexError, match="The query returned no results"):
            medrecord.edge[edge_max_greater_than_3, :]

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[edge_greater_than_or_equal_two, 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[edge_greater_than_or_equal_two, :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[edge_greater_than_or_equal_two, ::1]

        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[:1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[::1]

        assert medrecord.edge[:, "foo"] == {0: "bar", 1: "bar", 2: "bar", 3: "bar"}

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.edge[:, "test"]

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[1:, "foo"]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[:1, "foo"]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[::1, "foo"]

        assert medrecord.edge[:, ["foo", "bar"]] == {
            0: {"foo": "bar", "bar": "foo"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        # Accessing a non-existing key should fail
        with pytest.raises(KeyError):
            medrecord.edge[:, ["foo", "test"]]

        # Accessing a key that doesn't exist in all edges should fail
        with pytest.raises(KeyError):
            medrecord.edge[:, ["foo", "lorem"]]

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[1:, ["foo", "bar"]]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[:1, ["foo", "bar"]]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[::1, ["foo", "bar"]]

        assert medrecord.edge[:, :] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[1:, :]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[:1, :]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[::1, :]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[:, 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[:, :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[:, ::1]

    def test_edge_setitem(self) -> None:
        # Updating existing attributes

        medrecord = create_medrecord()
        medrecord.edge[0] = {"foo": "bar", "bar": "test"}
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "test"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[[0, 1]] = {"foo": "test", "bar": "test1"}
        assert medrecord.edge[:] == {
            0: {"foo": "test", "bar": "test1"},
            1: {"foo": "test", "bar": "test1"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Updating a non-existing edge should fail
        with pytest.raises(IndexError):
            medrecord.edge[50] = {"foo": "bar", "test": "test"}

        medrecord = create_medrecord()
        medrecord.edge[0, "foo"] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "test", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[0, ["foo", "bar"]] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "test", "bar": "test", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[0, :] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "test", "bar": "test", "lorem": "test"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[0, 1:] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[0, :1] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[0, ::1] = "test"

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], "foo"] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "test", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "test", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], ["foo", "bar"]] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "test", "bar": "test", "lorem": "ipsum"},
            1: {"foo": "test", "bar": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], :] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "test", "bar": "test", "lorem": "test"},
            1: {"foo": "test", "bar": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[[0, 1], 1:] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[[0, 1], :1] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[[0, 1], ::1] = "test"

        medrecord = create_medrecord()
        medrecord.edge[edge_greater_than_or_equal_two] = {"foo": "bar", "bar": "test"}
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "test"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_max] = {"foo": "test", "bar": "test1"}
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "test", "bar": "test1"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_max_greater_than_3] = {"foo": "test", "bar": "test1"}
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Empty query should not fail
        medrecord.edge[edge_greater_than_three] = {"foo": "bar", "bar": "test"}

        medrecord = create_medrecord()
        medrecord.edge[edge_greater_than_or_equal_two, "foo"] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "test", "bar": "foo"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_max, "foo"] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_max_greater_than_3, "foo"] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_greater_than_or_equal_two, ["foo", "bar"]] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "test", "bar": "test"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_max, ["foo", "bar"]] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_max_greater_than_3, ["foo", "bar"]] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_greater_than_or_equal_two, :] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "test", "bar": "test"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_max, :] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_max_greater_than_3, :] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[edge_greater_than_or_equal_two, 1:] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[edge_greater_than_or_equal_two, :1] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[edge_greater_than_or_equal_two, ::1] = "test"

        medrecord = create_medrecord()
        medrecord.edge[:, "foo"] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "test", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "test", "bar": "foo"},
            2: {"foo": "test", "bar": "foo"},
            3: {"foo": "test", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[:] = {"foo": "bar", "bar": "test"}
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "test"},
            1: {"foo": "bar", "bar": "test"},
            2: {"foo": "bar", "bar": "test"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[1:] = {"foo": "bar", "bar": "test"}

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[1:, "foo"] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[:1, "foo"] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[::1, "foo"] = "test"

        medrecord = create_medrecord()
        medrecord.edge[:, ["foo", "bar"]] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "test", "bar": "test", "lorem": "ipsum"},
            1: {"foo": "test", "bar": "test"},
            2: {"foo": "test", "bar": "test"},
            3: {"foo": "test", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[1:, ["foo", "bar"]] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[:1, ["foo", "bar"]] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[::1, ["foo", "bar"]] = "test"

        medrecord = create_medrecord()
        medrecord.edge[:, :] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "test", "bar": "test", "lorem": "test"},
            1: {"foo": "test", "bar": "test"},
            2: {"foo": "test", "bar": "test"},
            3: {"foo": "test", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[1:, :] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[:1, :] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[::1, :] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[:, 1:] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[:, :1] = "test"
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            medrecord.edge[:, ::1] = "test"

        # Adding new attributes

        medrecord = create_medrecord()
        medrecord.edge[0, "test"] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum", "test": "test"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[0, ["test", "test2"]] = "test"
        assert medrecord.edge[:] == {
            0: {
                "foo": "bar",
                "bar": "foo",
                "lorem": "ipsum",
                "test": "test",
                "test2": "test",
            },
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], "test"] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum", "test": "test"},
            1: {"foo": "bar", "bar": "foo", "test": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], ["test", "test2"]] = "test"
        assert medrecord.edge[:] == {
            0: {
                "foo": "bar",
                "bar": "foo",
                "lorem": "ipsum",
                "test": "test",
                "test2": "test",
            },
            1: {"foo": "bar", "bar": "foo", "test": "test", "test2": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_greater_than_or_equal_two, "test"] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo", "test": "test"},
            3: {"foo": "bar", "bar": "test", "test": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_greater_than_or_equal_two, ["test", "test2"]] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo", "test": "test", "test2": "test"},
            3: {"foo": "bar", "bar": "test", "test": "test", "test2": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[:, "test"] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum", "test": "test"},
            1: {"foo": "bar", "bar": "foo", "test": "test"},
            2: {"foo": "bar", "bar": "foo", "test": "test"},
            3: {"foo": "bar", "bar": "test", "test": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[:, ["test", "test2"]] = "test"
        assert medrecord.edge[:] == {
            0: {
                "foo": "bar",
                "bar": "foo",
                "lorem": "ipsum",
                "test": "test",
                "test2": "test",
            },
            1: {"foo": "bar", "bar": "foo", "test": "test", "test2": "test"},
            2: {"foo": "bar", "bar": "foo", "test": "test", "test2": "test"},
            3: {"foo": "bar", "bar": "test", "test": "test", "test2": "test"},
        }

        # Adding and updating attributes

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], "lorem"] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "test"},
            1: {"foo": "bar", "bar": "foo", "lorem": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], ["lorem", "test"]] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            1: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_less_than_two, "lorem"] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "test"},
            1: {"foo": "bar", "bar": "foo", "lorem": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[edge_less_than_two, ["lorem", "test"]] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            1: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[:, "lorem"] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "test"},
            1: {"foo": "bar", "bar": "foo", "lorem": "test"},
            2: {"foo": "bar", "bar": "foo", "lorem": "test"},
            3: {"foo": "bar", "bar": "test", "lorem": "test"},
        }

        medrecord = create_medrecord()
        medrecord.edge[:, ["lorem", "test"]] = "test"
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            1: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            2: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
            3: {"foo": "bar", "bar": "test", "lorem": "test", "test": "test"},
        }

    def test_edge_delitem(self) -> None:
        medrecord = create_medrecord()
        del medrecord.edge[0, "foo"]
        assert medrecord.edge[:] == {
            0: {"bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Removing from a non-existing edge should fail
        with pytest.raises(IndexError):
            del medrecord.edge[50, "foo"]

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.edge[0, "test"]

        medrecord = create_medrecord()
        del medrecord.edge[0, ["foo", "bar"]]
        assert medrecord.edge[:] == {
            0: {"lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.edge[0, ["foo", "test"]]

        medrecord = create_medrecord()
        del medrecord.edge[0, :]
        assert medrecord.edge[:] == {
            0: {},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[0, 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[0, :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[0, ::1]

        medrecord = create_medrecord()
        del medrecord.edge[[0, 1], "foo"]
        assert medrecord.edge[:] == {
            0: {"bar": "foo", "lorem": "ipsum"},
            1: {"bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Removing from a non-existing edge should fail
        with pytest.raises(IndexError):
            del medrecord.edge[[0, 50], "foo"]

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.edge[[0, 1], "test"]

        medrecord = create_medrecord()
        del medrecord.edge[[0, 1], ["foo", "bar"]]
        assert medrecord.edge[:] == {
            0: {"lorem": "ipsum"},
            1: {},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.edge[[0, 1], ["foo", "test"]]

        medrecord = create_medrecord()
        # Removing a key that doesn't exist in all edges should fail
        with pytest.raises(KeyError):
            del medrecord.edge[[0, 1], ["foo", "lorem"]]

        medrecord = create_medrecord()
        del medrecord.edge[[0, 1], :]
        assert medrecord.edge[:] == {
            0: {},
            1: {},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[[0, 1], 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[[0, 1], :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[[0, 1], ::1]

        medrecord = create_medrecord()
        del medrecord.edge[edge_greater_than_or_equal_two, "foo"]
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"bar": "foo"},
            3: {"bar": "test"},
        }

        medrecord = create_medrecord()
        del medrecord.edge[edge_max, "foo"]
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"bar": "test"},
        }

        medrecord = create_medrecord()
        with pytest.raises(IndexError, match="The query returned no results"):
            del medrecord.edge[edge_max_greater_than_3, "foo"]

        medrecord = create_medrecord()
        # Empty query should not fail
        del medrecord.edge[edge_greater_than_three, "foo"]
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.edge[edge_greater_than_or_equal_two, "test"]

        medrecord = create_medrecord()
        del medrecord.edge[edge_greater_than_or_equal_two, ["foo", "bar"]]
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {},
            3: {},
        }

        medrecord = create_medrecord()
        del medrecord.edge[edge_max, ["foo", "bar"]]
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {},
        }

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.edge[edge_greater_than_or_equal_two, ["foo", "test"]]

        medrecord = create_medrecord()
        # Removing a key that doesn't exist in all edges should fail
        with pytest.raises(KeyError):
            del medrecord.edge[edge_less_than_two, ["foo", "lorem"]]

        medrecord = create_medrecord()
        del medrecord.edge[edge_greater_than_or_equal_two, :]
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {},
            3: {},
        }

        medrecord = create_medrecord()
        del medrecord.edge[edge_max, :]
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {},
        }

        medrecord = create_medrecord()
        del medrecord.edge[edge_max_greater_than_3, :]
        assert medrecord.edge[:] == {
            0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
            1: {"foo": "bar", "bar": "foo"},
            2: {"foo": "bar", "bar": "foo"},
            3: {"foo": "bar", "bar": "test"},
        }

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[edge_greater_than_or_equal_two, 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[edge_greater_than_or_equal_two, :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[edge_greater_than_or_equal_two, ::1]

        medrecord = create_medrecord()
        del medrecord.edge[:, "foo"]
        assert medrecord.edge[:] == {
            0: {"bar": "foo", "lorem": "ipsum"},
            1: {"bar": "foo"},
            2: {"bar": "foo"},
            3: {"bar": "test"},
        }

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.edge[:, "test"]

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[1:, "foo"]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[:1, "foo"]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[::1, "foo"]

        medrecord = create_medrecord()
        del medrecord.edge[:, ["foo", "bar"]]
        assert medrecord.edge[:] == {0: {"lorem": "ipsum"}, 1: {}, 2: {}, 3: {}}

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with pytest.raises(KeyError):
            del medrecord.edge[:, ["foo", "test"]]

        medrecord = create_medrecord()
        # Removing a key that doesn't exist in all edges should fail
        with pytest.raises(KeyError):
            del medrecord.edge[:, ["foo", "lorem"]]

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[1:, ["foo", "bar"]]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[:1, ["foo", "bar"]]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[::1, ["foo", "bar"]]

        medrecord = create_medrecord()
        del medrecord.edge[:, :]
        assert medrecord.edge[:] == {0: {}, 1: {}, 2: {}, 3: {}}

        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[1:, :]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[:1, :]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[::1, :]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[:, 1:]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[:, :1]
        with pytest.raises(ValueError, match="Invalid slice, only ':' is allowed"):
            del medrecord.edge[:, ::1]


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestIndexers)
    unittest.TextTestRunner(verbosity=2).run(run_test)
