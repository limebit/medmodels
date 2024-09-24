import unittest

from medmodels import MedRecord
from medmodels.medrecord.querying import EdgeOperand, NodeOperand


def create_medrecord():
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


def node_greater_than_or_equal_two(node: NodeOperand):
    node.index().greater_than_or_equal_to(2)


def node_greater_than_three(node: NodeOperand):
    node.index().greater_than(3)


def node_less_than_two(node: NodeOperand):
    node.index().less_than(2)


def edge_greater_than_or_equal_two(edge: EdgeOperand):
    edge.index().greater_than_or_equal_to(2)


def edge_greater_than_three(edge: EdgeOperand):
    edge.index().greater_than(3)


def edge_less_than_two(edge: EdgeOperand):
    edge.index().less_than(2)


class TestMedRecord(unittest.TestCase):
    def test_node_getitem(self):
        medrecord = create_medrecord()

        self.assertEqual(
            {"foo": "bar", "bar": "foo", "lorem": "ipsum"}, medrecord.node[0]
        )

        # Accessing a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.node[50]

        self.assertEqual("bar", medrecord.node[0, "foo"])

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.node[0, "test"]

        self.assertEqual(
            {"foo": "bar", "bar": "foo"}, medrecord.node[0, ["foo", "bar"]]
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.node[0, ["foo", "test"]]

        self.assertEqual(
            {"foo": "bar", "bar": "foo", "lorem": "ipsum"}, medrecord.node[0, :]
        )

        with self.assertRaises(ValueError):
            medrecord.node[0, 1:]
        with self.assertRaises(ValueError):
            medrecord.node[0, :1]
        with self.assertRaises(ValueError):
            medrecord.node[0, ::1]

        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
            },
            medrecord.node[[0, 1]],
        )

        with self.assertRaises(IndexError):
            medrecord.node[[0, 50]]

        self.assertEqual(
            {
                0: "bar",
                1: "bar",
            },
            medrecord.node[[0, 1], "foo"],
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.node[[0, 1], "test"]

        # Accessing a key that doesn't exist in all nodes should fail
        with self.assertRaises(KeyError):
            medrecord.node[[0, 1], "lorem"]

        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo"},
                1: {"foo": "bar", "bar": "foo"},
            },
            medrecord.node[[0, 1], ["foo", "bar"]],
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.node[[0, 1], ["foo", "test"]]

        # Accessing a key that doesn't exist in all nodes should fail
        with self.assertRaises(KeyError):
            medrecord.node[[0, 1], ["foo", "lorem"]]

        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
            },
            medrecord.node[[0, 1], :],
        )

        with self.assertRaises(ValueError):
            medrecord.node[[0, 1], 1:]
        with self.assertRaises(ValueError):
            medrecord.node[[0, 1], :1]
        with self.assertRaises(ValueError):
            medrecord.node[[0, 1], ::1]

        self.assertEqual(
            {2: {"foo": "bar", "bar": "foo"}, 3: {"foo": "bar", "bar": "test"}},
            medrecord.node[node_greater_than_or_equal_two],
        )

        # Empty query should not fail
        self.assertEqual(
            {},
            medrecord.node[node_greater_than_three],
        )

        self.assertEqual(
            {2: "bar", 3: "bar"},
            medrecord.node[node_greater_than_or_equal_two, "foo"],
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.node[node_greater_than_or_equal_two, "test"]

        self.assertEqual(
            {
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[node_greater_than_or_equal_two, ["foo", "bar"]],
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.node[node_greater_than_or_equal_two, ["foo", "test"]]

        # Accessing a key that doesn't exist in all nodes should fail
        with self.assertRaises(KeyError):
            medrecord.node[node_less_than_two, ["foo", "lorem"]]

        self.assertEqual(
            {
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[node_greater_than_or_equal_two, :],
        )

        with self.assertRaises(ValueError):
            medrecord.node[node_greater_than_or_equal_two, 1:]
        with self.assertRaises(ValueError):
            medrecord.node[node_greater_than_or_equal_two, :1]
        with self.assertRaises(ValueError):
            medrecord.node[node_greater_than_or_equal_two, ::1]

        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        with self.assertRaises(ValueError):
            medrecord.node[1:]
        with self.assertRaises(ValueError):
            medrecord.node[:1]
        with self.assertRaises(ValueError):
            medrecord.node[::1]

        self.assertEqual(
            {
                0: "bar",
                1: "bar",
                2: "bar",
                3: "bar",
            },
            medrecord.node[:, "foo"],
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.node[:, "test"]

        with self.assertRaises(ValueError):
            medrecord.node[1:, "foo"]
        with self.assertRaises(ValueError):
            medrecord.node[:1, "foo"]
        with self.assertRaises(ValueError):
            medrecord.node[::1, "foo"]

        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:, ["foo", "bar"]],
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.node[:, ["foo", "test"]]

        # Accessing a key that doesn't exist in all nodes should fail
        with self.assertRaises(KeyError):
            medrecord.node[:, ["foo", "lorem"]]

        with self.assertRaises(ValueError):
            medrecord.node[1:, ["foo", "bar"]]
        with self.assertRaises(ValueError):
            medrecord.node[:1, ["foo", "bar"]]
        with self.assertRaises(ValueError):
            medrecord.node[::1, ["foo", "bar"]]

        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:, :],
        )

        with self.assertRaises(ValueError):
            medrecord.node[1:, :]
        with self.assertRaises(ValueError):
            medrecord.node[:1, :]
        with self.assertRaises(ValueError):
            medrecord.node[::1, :]
        with self.assertRaises(ValueError):
            medrecord.node[:, 1:]
        with self.assertRaises(ValueError):
            medrecord.node[:, :1]
        with self.assertRaises(ValueError):
            medrecord.node[:, ::1]

    def test_node_setitem(self):
        # Updating existing attributes

        medrecord = create_medrecord()
        medrecord.node[0] = {"foo": "bar", "bar": "test"}
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "test"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        # Updating a non-existing node should fail
        with self.assertRaises(IndexError):
            medrecord.node[50] = {"foo": "bar", "test": "test"}

        medrecord = create_medrecord()
        medrecord.node[0, "foo"] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[0, ["foo", "bar"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "test", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[0, :] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "test", "lorem": "test"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        with self.assertRaises(ValueError):
            medrecord.node[0, 1:] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[0, :1] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[0, ::1] = "test"

        medrecord = create_medrecord()
        medrecord.node[[0, 1], "foo"] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "test", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[[0, 1], ["foo", "bar"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "test", "lorem": "ipsum"},
                1: {"foo": "test", "bar": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[[0, 1], :] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "test", "lorem": "test"},
                1: {"foo": "test", "bar": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        with self.assertRaises(ValueError):
            medrecord.node[[0, 1], 1:] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[[0, 1], :1] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[[0, 1], ::1] = "test"

        medrecord = create_medrecord()
        medrecord.node[node_greater_than_or_equal_two] = {"foo": "bar", "bar": "test"}
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "test"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        # Empty query should not fail
        medrecord.node[node_greater_than_three] = {"foo": "bar", "bar": "test"}

        medrecord = create_medrecord()
        medrecord.node[node_greater_than_or_equal_two, "foo"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "test", "bar": "foo"},
                3: {"foo": "test", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[node_greater_than_or_equal_two, ["foo", "bar"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "test", "bar": "test"},
                3: {"foo": "test", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[node_greater_than_or_equal_two, :] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "test", "bar": "test"},
                3: {"foo": "test", "bar": "test"},
            },
            medrecord.node[:],
        )

        with self.assertRaises(ValueError):
            medrecord.node[node_greater_than_or_equal_two, 1:] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[node_greater_than_or_equal_two, :1] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[node_greater_than_or_equal_two, ::1] = "test"

        medrecord = create_medrecord()
        medrecord.node[:, "foo"] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "test", "bar": "foo"},
                2: {"foo": "test", "bar": "foo"},
                3: {"foo": "test", "bar": "test"},
            },
            medrecord.node[:],
        )

        with self.assertRaises(ValueError):
            medrecord.node[1:, "foo"] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[:1, "foo"] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[::1, "foo"] = "test"

        medrecord = create_medrecord()
        medrecord.node[:, ["foo", "bar"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "test", "lorem": "ipsum"},
                1: {"foo": "test", "bar": "test"},
                2: {"foo": "test", "bar": "test"},
                3: {"foo": "test", "bar": "test"},
            },
            medrecord.node[:],
        )

        with self.assertRaises(ValueError):
            medrecord.node[1:, ["foo", "bar"]] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[:1, ["foo", "bar"]] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[::1, ["foo", "bar"]] = "test"

        medrecord = create_medrecord()
        medrecord.node[:, :] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "test", "lorem": "test"},
                1: {"foo": "test", "bar": "test"},
                2: {"foo": "test", "bar": "test"},
                3: {"foo": "test", "bar": "test"},
            },
            medrecord.node[:],
        )

        with self.assertRaises(ValueError):
            medrecord.node[1:, :] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[:1, :] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[::1, :] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[:, 1:] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[:, :1] = "test"
        with self.assertRaises(ValueError):
            medrecord.node[:, ::1] = "test"

        # Adding new attributes

        medrecord = create_medrecord()
        medrecord.node[0, "test"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum", "test": "test"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[0, ["test", "test2"]] = "test"
        self.assertEqual(
            {
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
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[[0, 1], "test"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum", "test": "test"},
                1: {"foo": "bar", "bar": "foo", "test": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[[0, 1], ["test", "test2"]] = "test"
        self.assertEqual(
            {
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
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[node_greater_than_or_equal_two, "test"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo", "test": "test"},
                3: {"foo": "bar", "bar": "test", "test": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[node_greater_than_or_equal_two, ["test", "test2"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {
                    "foo": "bar",
                    "bar": "foo",
                    "test": "test",
                    "test2": "test",
                },
                3: {
                    "foo": "bar",
                    "bar": "test",
                    "test": "test",
                    "test2": "test",
                },
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[:, "test"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum", "test": "test"},
                1: {"foo": "bar", "bar": "foo", "test": "test"},
                2: {"foo": "bar", "bar": "foo", "test": "test"},
                3: {"foo": "bar", "bar": "test", "test": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[:, ["test", "test2"]] = "test"
        self.assertEqual(
            {
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
            },
            medrecord.node[:],
        )

        # Adding and updating attributes

        medrecord = create_medrecord()
        medrecord.node[[0, 1], "lorem"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "test"},
                1: {"foo": "bar", "bar": "foo", "lorem": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[[0, 1], ["lorem", "test"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                1: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[node_less_than_two, "lorem"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "test"},
                1: {"foo": "bar", "bar": "foo", "lorem": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[node_less_than_two, ["lorem", "test"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                1: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[:, "lorem"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "test"},
                1: {"foo": "bar", "bar": "foo", "lorem": "test"},
                2: {"foo": "bar", "bar": "foo", "lorem": "test"},
                3: {"foo": "bar", "bar": "test", "lorem": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        medrecord.node[:, ["lorem", "test"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                1: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                2: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                3: {"foo": "bar", "bar": "test", "lorem": "test", "test": "test"},
            },
            medrecord.node[:],
        )

    def test_node_delitem(self):
        medrecord = create_medrecord()
        del medrecord.node[0, "foo"]
        self.assertEqual(
            {
                0: {"bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        # Removing from a non-existing node should fail
        with self.assertRaises(IndexError):
            del medrecord.node[50, "foo"]

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.node[0, "test"]

        medrecord = create_medrecord()
        del medrecord.node[0, ["foo", "bar"]]
        self.assertEqual(
            {
                0: {"lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.node[0, ["foo", "test"]]

        medrecord = create_medrecord()
        del medrecord.node[0, :]
        self.assertEqual(
            {
                0: {},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        with self.assertRaises(ValueError):
            del medrecord.node[0, 1:]
        with self.assertRaises(ValueError):
            del medrecord.node[0, :1]
        with self.assertRaises(ValueError):
            del medrecord.node[0, ::1]

        medrecord = create_medrecord()
        del medrecord.node[[0, 1], "foo"]
        self.assertEqual(
            {
                0: {"bar": "foo", "lorem": "ipsum"},
                1: {"bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        # Removing from a non-existing node should fail
        with self.assertRaises(IndexError):
            del medrecord.node[[0, 50], "foo"]

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.node[[0, 1], "test"]

        medrecord = create_medrecord()
        del medrecord.node[[0, 1], ["foo", "bar"]]
        self.assertEqual(
            {
                0: {"lorem": "ipsum"},
                1: {},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.node[[0, 1], ["foo", "test"]]

        medrecord = create_medrecord()
        # Removing a key that doesn't exist in all nodes should fail
        with self.assertRaises(KeyError):
            del medrecord.node[[0, 1], ["foo", "lorem"]]

        medrecord = create_medrecord()
        del medrecord.node[[0, 1], :]
        self.assertEqual(
            {
                0: {},
                1: {},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        with self.assertRaises(ValueError):
            del medrecord.node[[0, 1], 1:]
        with self.assertRaises(ValueError):
            del medrecord.node[[0, 1], :1]
        with self.assertRaises(ValueError):
            del medrecord.node[[0, 1], ::1]

        medrecord = create_medrecord()
        del medrecord.node[node_greater_than_or_equal_two, "foo"]
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"bar": "foo"},
                3: {"bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        # Empty query should not fail
        del medrecord.node[node_greater_than_three, "foo"]
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.node[node_greater_than_or_equal_two, "test"]

        medrecord = create_medrecord()
        del medrecord.node[node_greater_than_or_equal_two, ["foo", "bar"]]
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {},
                3: {},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.node[node_greater_than_or_equal_two, ["foo", "test"]]

        medrecord = create_medrecord()
        # Removing a key that doesn't exist in all nodes should fail
        with self.assertRaises(KeyError):
            del medrecord.node[node_less_than_two, ["foo", "lorem"]]

        medrecord = create_medrecord()
        del medrecord.node[node_greater_than_or_equal_two, :]
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {},
                3: {},
            },
            medrecord.node[:],
        )

        with self.assertRaises(ValueError):
            del medrecord.node[node_greater_than_or_equal_two, 1:]
        with self.assertRaises(ValueError):
            del medrecord.node[node_greater_than_or_equal_two, :1]
        with self.assertRaises(ValueError):
            del medrecord.node[node_greater_than_or_equal_two, ::1]

        medrecord = create_medrecord()
        del medrecord.node[:, "foo"]
        self.assertEqual(
            {
                0: {"bar": "foo", "lorem": "ipsum"},
                1: {"bar": "foo"},
                2: {"bar": "foo"},
                3: {"bar": "test"},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.node[:, "test"]

        with self.assertRaises(ValueError):
            del medrecord.node[1:, "foo"]
        with self.assertRaises(ValueError):
            del medrecord.node[:1, "foo"]
        with self.assertRaises(ValueError):
            del medrecord.node[::1, "foo"]

        medrecord = create_medrecord()
        del medrecord.node[:, ["foo", "bar"]]
        self.assertEqual(
            {
                0: {"lorem": "ipsum"},
                1: {},
                2: {},
                3: {},
            },
            medrecord.node[:],
        )

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.node[:, ["foo", "test"]]

        medrecord = create_medrecord()
        # Removing a key that doesn't exist in all nodes should fail
        with self.assertRaises(KeyError):
            del medrecord.node[:, ["foo", "lorem"]]

        with self.assertRaises(ValueError):
            del medrecord.node[1:, ["foo", "bar"]]
        with self.assertRaises(ValueError):
            del medrecord.node[:1, ["foo", "bar"]]
        with self.assertRaises(ValueError):
            del medrecord.node[::1, ["foo", "bar"]]

        medrecord = create_medrecord()
        del medrecord.node[:, :]
        self.assertEqual(
            {
                0: {},
                1: {},
                2: {},
                3: {},
            },
            medrecord.node[:],
        )

        with self.assertRaises(ValueError):
            del medrecord.node[1:, :]
        with self.assertRaises(ValueError):
            del medrecord.node[:1, :]
        with self.assertRaises(ValueError):
            del medrecord.node[::1, :]
        with self.assertRaises(ValueError):
            del medrecord.node[:, 1:]
        with self.assertRaises(ValueError):
            del medrecord.node[:, :1]
        with self.assertRaises(ValueError):
            del medrecord.node[:, ::1]

    def test_edge_getitem(self):
        medrecord = create_medrecord()

        self.assertEqual(
            {"foo": "bar", "bar": "foo", "lorem": "ipsum"}, medrecord.edge[0]
        )

        # Accessing a non-existing edge should fail
        with self.assertRaises(IndexError):
            medrecord.edge[50]

        self.assertEqual("bar", medrecord.edge[0, "foo"])

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.edge[0, "test"]

        self.assertEqual(
            {"foo": "bar", "bar": "foo"}, medrecord.edge[0, ["foo", "bar"]]
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.edge[0, ["foo", "test"]]

        self.assertEqual(
            {"foo": "bar", "bar": "foo", "lorem": "ipsum"}, medrecord.edge[0, :]
        )

        with self.assertRaises(ValueError):
            medrecord.edge[0, 1:]
        with self.assertRaises(ValueError):
            medrecord.edge[0, :1]
        with self.assertRaises(ValueError):
            medrecord.edge[0, ::1]

        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
            },
            medrecord.edge[[0, 1]],
        )

        with self.assertRaises(IndexError):
            medrecord.edge[[0, 50]]

        self.assertEqual(
            {
                0: "bar",
                1: "bar",
            },
            medrecord.edge[[0, 1], "foo"],
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.edge[[0, 1], "test"]

        # Accessing a key that doesn't exist in all edges should fail
        with self.assertRaises(KeyError):
            medrecord.edge[[0, 1], "lorem"]

        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo"},
                1: {"foo": "bar", "bar": "foo"},
            },
            medrecord.edge[[0, 1], ["foo", "bar"]],
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.edge[[0, 1], ["foo", "test"]]

        # Accessing a key that doesn't exist in all edges should fail
        with self.assertRaises(KeyError):
            medrecord.edge[[0, 1], ["foo", "lorem"]]

        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
            },
            medrecord.edge[[0, 1], :],
        )

        with self.assertRaises(ValueError):
            medrecord.edge[[0, 1], 1:]
        with self.assertRaises(ValueError):
            medrecord.edge[[0, 1], :1]
        with self.assertRaises(ValueError):
            medrecord.edge[[0, 1], ::1]

        self.assertEqual(
            {2: {"foo": "bar", "bar": "foo"}, 3: {"foo": "bar", "bar": "test"}},
            medrecord.edge[edge_greater_than_or_equal_two],
        )

        # Empty query should not fail
        self.assertEqual(
            {},
            medrecord.edge[edge_greater_than_three],
        )

        self.assertEqual(
            {2: "bar", 3: "bar"},
            medrecord.edge[edge_greater_than_or_equal_two, "foo"],
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.edge[edge_greater_than_or_equal_two, "test"]

        self.assertEqual(
            {
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[edge_greater_than_or_equal_two, ["foo", "bar"]],
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.edge[edge_greater_than_or_equal_two, ["foo", "test"]]

        # Accessing a key that doesn't exist in all edges should fail
        with self.assertRaises(KeyError):
            medrecord.edge[edge_less_than_two, ["foo", "lorem"]]

        self.assertEqual(
            {
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[edge_greater_than_or_equal_two, :],
        )

        with self.assertRaises(ValueError):
            medrecord.edge[edge_greater_than_or_equal_two, 1:]
        with self.assertRaises(ValueError):
            medrecord.edge[edge_greater_than_or_equal_two, :1]
        with self.assertRaises(ValueError):
            medrecord.edge[edge_greater_than_or_equal_two, ::1]

        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        with self.assertRaises(ValueError):
            medrecord.edge[1:]
        with self.assertRaises(ValueError):
            medrecord.edge[:1]
        with self.assertRaises(ValueError):
            medrecord.edge[::1]

        self.assertEqual(
            {
                0: "bar",
                1: "bar",
                2: "bar",
                3: "bar",
            },
            medrecord.edge[:, "foo"],
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.edge[:, "test"]

        with self.assertRaises(ValueError):
            medrecord.edge[1:, "foo"]
        with self.assertRaises(ValueError):
            medrecord.edge[:1, "foo"]
        with self.assertRaises(ValueError):
            medrecord.edge[::1, "foo"]

        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:, ["foo", "bar"]],
        )

        # Accessing a non-existing key should fail
        with self.assertRaises(KeyError):
            medrecord.edge[:, ["foo", "test"]]

        # Accessing a key that doesn't exist in all edges should fail
        with self.assertRaises(KeyError):
            medrecord.edge[:, ["foo", "lorem"]]

        with self.assertRaises(ValueError):
            medrecord.edge[1:, ["foo", "bar"]]
        with self.assertRaises(ValueError):
            medrecord.edge[:1, ["foo", "bar"]]
        with self.assertRaises(ValueError):
            medrecord.edge[::1, ["foo", "bar"]]

        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:, :],
        )

        with self.assertRaises(ValueError):
            medrecord.edge[1:, :]
        with self.assertRaises(ValueError):
            medrecord.edge[:1, :]
        with self.assertRaises(ValueError):
            medrecord.edge[::1, :]
        with self.assertRaises(ValueError):
            medrecord.edge[:, 1:]
        with self.assertRaises(ValueError):
            medrecord.edge[:, :1]
        with self.assertRaises(ValueError):
            medrecord.edge[:, ::1]

    def test_edge_setitem(self):
        # Updating existing attributes

        medrecord = create_medrecord()
        medrecord.edge[0] = {"foo": "bar", "bar": "test"}
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "test"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        # Updating a non-existing edge should fail
        with self.assertRaises(IndexError):
            medrecord.edge[50] = {"foo": "bar", "test": "test"}

        medrecord = create_medrecord()
        medrecord.edge[0, "foo"] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[0, ["foo", "bar"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "test", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[0, :] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "test", "lorem": "test"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        with self.assertRaises(ValueError):
            medrecord.edge[0, 1:] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[0, :1] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[0, ::1] = "test"

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], "foo"] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "test", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], ["foo", "bar"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "test", "lorem": "ipsum"},
                1: {"foo": "test", "bar": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], :] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "test", "lorem": "test"},
                1: {"foo": "test", "bar": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        with self.assertRaises(ValueError):
            medrecord.edge[[0, 1], 1:] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[[0, 1], :1] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[[0, 1], ::1] = "test"

        medrecord = create_medrecord()
        medrecord.edge[edge_greater_than_or_equal_two] = {"foo": "bar", "bar": "test"}
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "test"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        # Empty query should not fail
        medrecord.edge[edge_greater_than_three] = {"foo": "bar", "bar": "test"}

        medrecord = create_medrecord()
        medrecord.edge[edge_greater_than_or_equal_two, "foo"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "test", "bar": "foo"},
                3: {"foo": "test", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[edge_greater_than_or_equal_two, ["foo", "bar"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "test", "bar": "test"},
                3: {"foo": "test", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[edge_greater_than_or_equal_two, :] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "test", "bar": "test"},
                3: {"foo": "test", "bar": "test"},
            },
            medrecord.edge[:],
        )

        with self.assertRaises(ValueError):
            medrecord.edge[edge_greater_than_or_equal_two, 1:] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[edge_greater_than_or_equal_two, :1] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[edge_greater_than_or_equal_two, ::1] = "test"

        medrecord = create_medrecord()
        medrecord.edge[:, "foo"] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "test", "bar": "foo"},
                2: {"foo": "test", "bar": "foo"},
                3: {"foo": "test", "bar": "test"},
            },
            medrecord.edge[:],
        )

        with self.assertRaises(ValueError):
            medrecord.edge[1:, "foo"] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[:1, "foo"] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[::1, "foo"] = "test"

        medrecord = create_medrecord()
        medrecord.edge[:, ["foo", "bar"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "test", "lorem": "ipsum"},
                1: {"foo": "test", "bar": "test"},
                2: {"foo": "test", "bar": "test"},
                3: {"foo": "test", "bar": "test"},
            },
            medrecord.edge[:],
        )

        with self.assertRaises(ValueError):
            medrecord.edge[1:, ["foo", "bar"]] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[:1, ["foo", "bar"]] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[::1, ["foo", "bar"]] = "test"

        medrecord = create_medrecord()
        medrecord.edge[:, :] = "test"
        self.assertEqual(
            {
                0: {"foo": "test", "bar": "test", "lorem": "test"},
                1: {"foo": "test", "bar": "test"},
                2: {"foo": "test", "bar": "test"},
                3: {"foo": "test", "bar": "test"},
            },
            medrecord.edge[:],
        )

        with self.assertRaises(ValueError):
            medrecord.edge[1:, :] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[:1, :] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[::1, :] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[:, 1:] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[:, :1] = "test"
        with self.assertRaises(ValueError):
            medrecord.edge[:, ::1] = "test"

        # Adding new attributes

        medrecord = create_medrecord()
        medrecord.edge[0, "test"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum", "test": "test"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[0, ["test", "test2"]] = "test"
        self.assertEqual(
            {
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
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], "test"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum", "test": "test"},
                1: {"foo": "bar", "bar": "foo", "test": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], ["test", "test2"]] = "test"
        self.assertEqual(
            {
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
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[edge_greater_than_or_equal_two, "test"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo", "test": "test"},
                3: {"foo": "bar", "bar": "test", "test": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[edge_greater_than_or_equal_two, ["test", "test2"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {
                    "foo": "bar",
                    "bar": "foo",
                    "test": "test",
                    "test2": "test",
                },
                3: {
                    "foo": "bar",
                    "bar": "test",
                    "test": "test",
                    "test2": "test",
                },
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[:, "test"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum", "test": "test"},
                1: {"foo": "bar", "bar": "foo", "test": "test"},
                2: {"foo": "bar", "bar": "foo", "test": "test"},
                3: {"foo": "bar", "bar": "test", "test": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[:, ["test", "test2"]] = "test"
        self.assertEqual(
            {
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
            },
            medrecord.edge[:],
        )

        # Adding and updating attributes

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], "lorem"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "test"},
                1: {"foo": "bar", "bar": "foo", "lorem": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[[0, 1], ["lorem", "test"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                1: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[edge_less_than_two, "lorem"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "test"},
                1: {"foo": "bar", "bar": "foo", "lorem": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[edge_less_than_two, ["lorem", "test"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                1: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[:, "lorem"] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "test"},
                1: {"foo": "bar", "bar": "foo", "lorem": "test"},
                2: {"foo": "bar", "bar": "foo", "lorem": "test"},
                3: {"foo": "bar", "bar": "test", "lorem": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        medrecord.edge[:, ["lorem", "test"]] = "test"
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                1: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                2: {"foo": "bar", "bar": "foo", "lorem": "test", "test": "test"},
                3: {"foo": "bar", "bar": "test", "lorem": "test", "test": "test"},
            },
            medrecord.edge[:],
        )

    def test_edge_delitem(self):
        medrecord = create_medrecord()
        del medrecord.edge[0, "foo"]
        self.assertEqual(
            {
                0: {"bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        # Removing from a non-existing edge should fail
        with self.assertRaises(IndexError):
            del medrecord.edge[50, "foo"]

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.edge[0, "test"]

        medrecord = create_medrecord()
        del medrecord.edge[0, ["foo", "bar"]]
        self.assertEqual(
            {
                0: {"lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.edge[0, ["foo", "test"]]

        medrecord = create_medrecord()
        del medrecord.edge[0, :]
        self.assertEqual(
            {
                0: {},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        with self.assertRaises(ValueError):
            del medrecord.edge[0, 1:]
        with self.assertRaises(ValueError):
            del medrecord.edge[0, :1]
        with self.assertRaises(ValueError):
            del medrecord.edge[0, ::1]

        medrecord = create_medrecord()
        del medrecord.edge[[0, 1], "foo"]
        self.assertEqual(
            {
                0: {"bar": "foo", "lorem": "ipsum"},
                1: {"bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        # Removing from a non-existing edge should fail
        with self.assertRaises(IndexError):
            del medrecord.edge[[0, 50], "foo"]

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.edge[[0, 1], "test"]

        medrecord = create_medrecord()
        del medrecord.edge[[0, 1], ["foo", "bar"]]
        self.assertEqual(
            {
                0: {"lorem": "ipsum"},
                1: {},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.edge[[0, 1], ["foo", "test"]]

        medrecord = create_medrecord()
        # Removing a key that doesn't exist in all edges should fail
        with self.assertRaises(KeyError):
            del medrecord.edge[[0, 1], ["foo", "lorem"]]

        medrecord = create_medrecord()
        del medrecord.edge[[0, 1], :]
        self.assertEqual(
            {
                0: {},
                1: {},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        with self.assertRaises(ValueError):
            del medrecord.edge[[0, 1], 1:]
        with self.assertRaises(ValueError):
            del medrecord.edge[[0, 1], :1]
        with self.assertRaises(ValueError):
            del medrecord.edge[[0, 1], ::1]

        medrecord = create_medrecord()
        del medrecord.edge[edge_greater_than_or_equal_two, "foo"]
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"bar": "foo"},
                3: {"bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        # Empty query should not fail
        del medrecord.edge[edge_greater_than_three, "foo"]
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {"foo": "bar", "bar": "foo"},
                3: {"foo": "bar", "bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.edge[edge_greater_than_or_equal_two, "test"]

        medrecord = create_medrecord()
        del medrecord.edge[edge_greater_than_or_equal_two, ["foo", "bar"]]
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {},
                3: {},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.edge[edge_greater_than_or_equal_two, ["foo", "test"]]

        medrecord = create_medrecord()
        # Removing a key that doesn't exist in all edges should fail
        with self.assertRaises(KeyError):
            del medrecord.edge[edge_less_than_two, ["foo", "lorem"]]

        medrecord = create_medrecord()
        del medrecord.edge[edge_greater_than_or_equal_two, :]
        self.assertEqual(
            {
                0: {"foo": "bar", "bar": "foo", "lorem": "ipsum"},
                1: {"foo": "bar", "bar": "foo"},
                2: {},
                3: {},
            },
            medrecord.edge[:],
        )

        with self.assertRaises(ValueError):
            del medrecord.edge[edge_greater_than_or_equal_two, 1:]
        with self.assertRaises(ValueError):
            del medrecord.edge[edge_greater_than_or_equal_two, :1]
        with self.assertRaises(ValueError):
            del medrecord.edge[edge_greater_than_or_equal_two, ::1]

        medrecord = create_medrecord()
        del medrecord.edge[:, "foo"]
        self.assertEqual(
            {
                0: {"bar": "foo", "lorem": "ipsum"},
                1: {"bar": "foo"},
                2: {"bar": "foo"},
                3: {"bar": "test"},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.edge[:, "test"]

        with self.assertRaises(ValueError):
            del medrecord.edge[1:, "foo"]
        with self.assertRaises(ValueError):
            del medrecord.edge[:1, "foo"]
        with self.assertRaises(ValueError):
            del medrecord.edge[::1, "foo"]

        medrecord = create_medrecord()
        del medrecord.edge[:, ["foo", "bar"]]
        self.assertEqual(
            {
                0: {"lorem": "ipsum"},
                1: {},
                2: {},
                3: {},
            },
            medrecord.edge[:],
        )

        medrecord = create_medrecord()
        # Removing a non-existing key should fail
        with self.assertRaises(KeyError):
            del medrecord.edge[:, ["foo", "test"]]

        medrecord = create_medrecord()
        # Removing a key that doesn't exist in all edges should fail
        with self.assertRaises(KeyError):
            del medrecord.edge[:, ["foo", "lorem"]]

        with self.assertRaises(ValueError):
            del medrecord.edge[1:, ["foo", "bar"]]
        with self.assertRaises(ValueError):
            del medrecord.edge[:1, ["foo", "bar"]]
        with self.assertRaises(ValueError):
            del medrecord.edge[::1, ["foo", "bar"]]

        medrecord = create_medrecord()
        del medrecord.edge[:, :]
        self.assertEqual(
            {
                0: {},
                1: {},
                2: {},
                3: {},
            },
            medrecord.edge[:],
        )

        with self.assertRaises(ValueError):
            del medrecord.edge[1:, :]
        with self.assertRaises(ValueError):
            del medrecord.edge[:1, :]
        with self.assertRaises(ValueError):
            del medrecord.edge[::1, :]
        with self.assertRaises(ValueError):
            del medrecord.edge[:, 1:]
        with self.assertRaises(ValueError):
            del medrecord.edge[:, :1]
        with self.assertRaises(ValueError):
            del medrecord.edge[:, ::1]
