import unittest

import pytest

import medmodels.medrecord as mr
from medmodels._medmodels import (
    PyAttributeDataType,
    PyAttributeType,
    PyGroupSchema,
    PySchema,
)


def create_medrecord() -> mr.MedRecord:
    return mr.MedRecord.from_simple_example_dataset()


class TestAttributeType(unittest.TestCase):
    def test_from_py_attribute_type(self) -> None:
        assert (
            mr.AttributeType._from_py_attribute_type(PyAttributeType.Categorical)
            == mr.AttributeType.Categorical
        )
        assert (
            mr.AttributeType._from_py_attribute_type(PyAttributeType.Continuous)
            == mr.AttributeType.Continuous
        )
        assert (
            mr.AttributeType._from_py_attribute_type(PyAttributeType.Temporal)
            == mr.AttributeType.Temporal
        )
        assert (
            mr.AttributeType._from_py_attribute_type(PyAttributeType.Unstructured)
            == mr.AttributeType.Unstructured
        )

    def test_infer(self) -> None:
        assert mr.AttributeType.infer(mr.String()) == mr.AttributeType.Unstructured
        assert mr.AttributeType.infer(mr.Int()) == mr.AttributeType.Continuous
        assert mr.AttributeType.infer(mr.Float()) == mr.AttributeType.Continuous
        assert mr.AttributeType.infer(mr.Bool()) == mr.AttributeType.Categorical
        assert mr.AttributeType.infer(mr.DateTime()) == mr.AttributeType.Temporal
        assert mr.AttributeType.infer(mr.Duration()) == mr.AttributeType.Temporal
        assert mr.AttributeType.infer(mr.Null()) == mr.AttributeType.Unstructured
        assert mr.AttributeType.infer(mr.Any()) == mr.AttributeType.Unstructured
        assert (
            mr.AttributeType.infer(mr.Union(mr.Int(), mr.Float()))
            == mr.AttributeType.Continuous
        )
        assert (
            mr.AttributeType.infer(mr.Option(mr.Int())) == mr.AttributeType.Continuous
        )

    def test_into_py_attribute_type(self) -> None:
        assert (
            mr.AttributeType.Categorical._into_py_attribute_type()
            == PyAttributeType.Categorical
        )
        assert (
            mr.AttributeType.Continuous._into_py_attribute_type()
            == PyAttributeType.Continuous
        )
        assert (
            mr.AttributeType.Temporal._into_py_attribute_type()
            == PyAttributeType.Temporal
        )
        assert (
            mr.AttributeType.Unstructured._into_py_attribute_type()
            == PyAttributeType.Unstructured
        )

    def test_repr(self) -> None:
        assert repr(mr.AttributeType.Categorical) == "AttributeType.Categorical"
        assert repr(mr.AttributeType.Continuous) == "AttributeType.Continuous"
        assert repr(mr.AttributeType.Temporal) == "AttributeType.Temporal"
        assert repr(mr.AttributeType.Unstructured) == "AttributeType.Unstructured"

    def test_str(self) -> None:
        assert str(mr.AttributeType.Categorical) == "Categorical"
        assert str(mr.AttributeType.Continuous) == "Continuous"
        assert str(mr.AttributeType.Temporal) == "Temporal"
        assert str(mr.AttributeType.Unstructured) == "Unstructured"

    def test_hash(self) -> None:
        assert hash(mr.AttributeType.Categorical) == hash("Categorical")
        assert hash(mr.AttributeType.Continuous) == hash("Continuous")
        assert hash(mr.AttributeType.Temporal) == hash("Temporal")
        assert hash(mr.AttributeType.Unstructured) == hash("Unstructured")

    def test_eq(self) -> None:
        assert mr.AttributeType.Categorical == mr.AttributeType.Categorical
        assert mr.AttributeType.Categorical == PyAttributeType.Categorical
        assert mr.AttributeType.Categorical != mr.AttributeType.Continuous
        assert mr.AttributeType.Categorical != mr.AttributeType.Temporal
        assert mr.AttributeType.Categorical != mr.AttributeType.Unstructured
        assert mr.AttributeType.Categorical != PyAttributeType.Continuous
        assert mr.AttributeType.Categorical != PyAttributeType.Temporal
        assert mr.AttributeType.Categorical != PyAttributeType.Unstructured

        assert mr.AttributeType.Continuous == mr.AttributeType.Continuous
        assert mr.AttributeType.Continuous == PyAttributeType.Continuous
        assert mr.AttributeType.Continuous != mr.AttributeType.Categorical
        assert mr.AttributeType.Continuous != mr.AttributeType.Temporal
        assert mr.AttributeType.Continuous != mr.AttributeType.Unstructured
        assert mr.AttributeType.Continuous != PyAttributeType.Categorical
        assert mr.AttributeType.Continuous != PyAttributeType.Temporal
        assert mr.AttributeType.Continuous != PyAttributeType.Unstructured

        assert mr.AttributeType.Temporal == mr.AttributeType.Temporal
        assert mr.AttributeType.Temporal == PyAttributeType.Temporal
        assert mr.AttributeType.Temporal != mr.AttributeType.Categorical
        assert mr.AttributeType.Temporal != mr.AttributeType.Continuous
        assert mr.AttributeType.Temporal != mr.AttributeType.Unstructured
        assert mr.AttributeType.Temporal != PyAttributeType.Categorical
        assert mr.AttributeType.Temporal != PyAttributeType.Continuous
        assert mr.AttributeType.Temporal != PyAttributeType.Unstructured

        assert mr.AttributeType.Unstructured == mr.AttributeType.Unstructured
        assert mr.AttributeType.Unstructured == PyAttributeType.Unstructured
        assert mr.AttributeType.Unstructured != mr.AttributeType.Categorical
        assert mr.AttributeType.Unstructured != mr.AttributeType.Continuous
        assert mr.AttributeType.Unstructured != mr.AttributeType.Temporal
        assert mr.AttributeType.Unstructured != PyAttributeType.Categorical
        assert mr.AttributeType.Unstructured != PyAttributeType.Continuous
        assert mr.AttributeType.Unstructured != PyAttributeType.Temporal


class TestGroupSchema(unittest.TestCase):
    def test_from_py_group_schema(self) -> None:
        assert mr.GroupSchema._from_py_group_schema(
            PyGroupSchema(
                nodes={
                    "test": PyAttributeDataType(
                        mr.String()._inner(), PyAttributeType.Unstructured
                    )
                },
                edges={},
            )
        ).nodes == {"test": (mr.String(), mr.AttributeType.Unstructured)}

    def test_nodes(self) -> None:
        group_schema = mr.GroupSchema(nodes={"test": mr.String()}, edges={})

        assert group_schema.nodes == {
            "test": (mr.String(), mr.AttributeType.Unstructured)
        }

    def test_edges(self) -> None:
        group_schema = mr.GroupSchema(nodes={}, edges={"test": mr.String()})

        assert group_schema.edges == {
            "test": (mr.String(), mr.AttributeType.Unstructured)
        }

    def test_validate_node(self) -> None:
        group_schema = mr.GroupSchema(
            nodes={
                "key1": (mr.Int(), mr.AttributeType.Categorical),
                "key2": (mr.Float(), mr.AttributeType.Continuous),
            },
            edges={},
        )

        group_schema.validate_node("0", {"key1": 0, "key2": 0.0})

        with pytest.raises(
            ValueError,
            match=r"Attribute key1 of node with index 0 is of type Float. Expected Int.",
        ):
            group_schema.validate_node("0", {"key1": 0.0, "key2": 0.0})

    def test_validate_edge(self) -> None:
        group_schema = mr.GroupSchema(
            nodes={},
            edges={
                "key1": (mr.Int(), mr.AttributeType.Categorical),
                "key2": (mr.Float(), mr.AttributeType.Continuous),
            },
        )

        group_schema.validate_edge(0, {"key1": 0, "key2": 0.0})

        with pytest.raises(
            ValueError,
            match=r"Attribute key1 of edge with index 0 is of type Float. Expected Int.",
        ):
            group_schema.validate_edge(0, {"key1": 0.0, "key2": 0.0})


class TestSchema(unittest.TestCase):
    def test_infer(self) -> None:
        medrecord = mr.MedRecord()
        medrecord.add_nodes([(0, {"key1": 0}), (1, {"key2": 0.0})])
        medrecord.add_edges((0, 1, {"key3": True}))

        schema = mr.Schema.infer(medrecord)

        assert len(schema.ungrouped.nodes) == 2
        assert len(schema.ungrouped.edges) == 1

        medrecord.add_group("test", [0, 1], [0])

        schema = mr.Schema.infer(medrecord)

        assert len(schema.ungrouped.nodes) == 0
        assert len(schema.ungrouped.edges) == 0
        assert len(schema.groups) == 1
        assert len(schema.group("test").nodes) == 2
        assert len(schema.group("test").edges) == 1

    def test_from_py_schema(self) -> None:
        assert mr.Schema._from_py_schema(
            PySchema(
                groups={},
                ungrouped=PyGroupSchema(
                    nodes={
                        "test": PyAttributeDataType(
                            mr.String()._inner(), PyAttributeType.Unstructured
                        )
                    },
                    edges={},
                ),
            )
        ).ungrouped.nodes == {"test": (mr.String(), mr.AttributeType.Unstructured)}

    def test_groups(self) -> None:
        schema = mr.Schema(
            groups={"test": mr.GroupSchema()}, ungrouped=mr.GroupSchema()
        )

        assert schema.groups == ["test"]

    def test_group(self) -> None:
        schema = mr.Schema(
            groups={"test": mr.GroupSchema(nodes={"test": mr.String()}, edges={})},
            ungrouped=mr.GroupSchema(),
        )

        assert schema.group("test").nodes == {
            "test": (mr.String(), mr.AttributeType.Unstructured)
        }

        with pytest.raises(ValueError, match=r"Group invalid not found in schema."):
            schema.group("invalid")

    def test_default(self) -> None:
        schema = mr.Schema(
            groups={}, ungrouped=mr.GroupSchema(nodes={"test": mr.String()}, edges={})
        )

        assert schema.ungrouped.nodes == {
            "test": (mr.String(), mr.AttributeType.Unstructured)
        }

    def test_schema_type(self) -> None:
        schema = mr.Schema(groups={}, ungrouped=mr.GroupSchema())

        assert schema.schema_type == mr.SchemaType.Provided

        schema = mr.Schema(
            groups={"test": mr.GroupSchema()},
            ungrouped=mr.GroupSchema(),
            schema_type=mr.SchemaType.Inferred,
        )

        assert schema.schema_type == mr.SchemaType.Inferred

    def test_validate_node(self) -> None:
        schema = mr.Schema(groups={}, ungrouped=mr.GroupSchema())

        schema.set_node_attribute("key1", mr.Int(), mr.AttributeType.Continuous)

        schema.validate_node("0", {"key1": 0})

        with pytest.raises(
            ValueError,
            match=r"Attribute key1 of node with index 0 is of type String. Expected Int.",
        ):
            schema.validate_node("0", {"key1": "invalid"})

    def test_validate_edge(self) -> None:
        schema = mr.Schema(groups={}, ungrouped=mr.GroupSchema())

        schema.set_edge_attribute("key1", mr.Bool(), mr.AttributeType.Categorical)

        schema.validate_edge(0, {"key1": True})

        with pytest.raises(
            ValueError,
            match=r"Attribute key1 of edge with index 0 is of type Int. Expected Bool.",
        ):
            schema.validate_edge(0, {"key1": 0})

    def test_set_node_attribute(self) -> None:
        schema = mr.Schema(groups={}, ungrouped=mr.GroupSchema())

        schema.set_node_attribute("key1", mr.Int())

        assert schema.ungrouped.nodes["key1"][0] == mr.Int()

        schema.set_node_attribute("key1", mr.Float(), mr.AttributeType.Continuous)

        assert schema.ungrouped.nodes["key1"][0] == mr.Float()

        schema.set_node_attribute(
            "key1", mr.Float(), mr.AttributeType.Continuous, "group1"
        )

        assert schema.group("group1").nodes["key1"][0] == mr.Float()

    def test_set_edge_attribute(self) -> None:
        schema = mr.Schema(groups={}, ungrouped=mr.GroupSchema())

        schema.set_edge_attribute("key1", mr.Bool())

        assert schema.ungrouped.edges["key1"][0] == mr.Bool()

        schema.set_edge_attribute("key1", mr.Float(), mr.AttributeType.Continuous)

        assert schema.ungrouped.edges["key1"][0] == mr.Float()

        schema.set_edge_attribute(
            "key1", mr.Float(), mr.AttributeType.Continuous, "group1"
        )

        assert schema.group("group1").edges["key1"][0] == mr.Float()

    def test_update_node_attribute(self) -> None:
        schema = mr.Schema(groups={}, ungrouped=mr.GroupSchema())

        schema.set_node_attribute("key1", mr.Int(), mr.AttributeType.Continuous)

        schema.update_node_attribute("key1", mr.Float())

        assert schema.ungrouped.nodes["key1"][0] == mr.Union(mr.Int(), mr.Float())

        schema.set_node_attribute(
            "key1", mr.Int(), mr.AttributeType.Continuous, "group1"
        )

        schema.update_node_attribute(
            "key1", mr.Float(), mr.AttributeType.Continuous, "group1"
        )

        assert schema.group("group1").nodes["key1"][0] == mr.Union(mr.Int(), mr.Float())

    def test_update_edge_attribute(self) -> None:
        schema = mr.Schema(groups={}, ungrouped=mr.GroupSchema())

        schema.set_edge_attribute("key1", mr.Bool(), mr.AttributeType.Categorical)

        schema.update_edge_attribute("key1", mr.String())

        assert schema.ungrouped.edges["key1"][0] == mr.Union(mr.Bool(), mr.String())

        schema.set_edge_attribute(
            "key1", mr.Bool(), mr.AttributeType.Categorical, "group1"
        )

        schema.update_edge_attribute(
            "key1", mr.String(), mr.AttributeType.Unstructured, "group1"
        )

        assert schema.group("group1").edges["key1"][0] == mr.Union(
            mr.Bool(), mr.String()
        )

    def test_remove_node_attribute(self) -> None:
        schema = mr.Schema(groups={}, ungrouped=mr.GroupSchema())

        schema.set_node_attribute("key1", mr.Int(), mr.AttributeType.Continuous)

        assert "key1" in schema.ungrouped.nodes

        schema.remove_node_attribute("key1")

        assert "key1" not in schema.ungrouped.nodes

        schema.set_node_attribute(
            "key1", mr.Int(), mr.AttributeType.Continuous, "group1"
        )

        assert "key1" in schema.group("group1").nodes

        schema.remove_node_attribute("key1", "group1")

        assert "key1" not in schema.group("group1").nodes

    def test_remove_edge_attribute(self) -> None:
        schema = mr.Schema(groups={}, ungrouped=mr.GroupSchema())

        schema.set_edge_attribute("key1", mr.Bool(), mr.AttributeType.Categorical)

        assert "key1" in schema.ungrouped.edges

        schema.remove_edge_attribute("key1")

        assert "key1" not in schema.ungrouped.edges

        schema.set_edge_attribute(
            "key1", mr.Bool(), mr.AttributeType.Categorical, "group1"
        )

        assert "key1" in schema.group("group1").edges

        schema.remove_edge_attribute("key1", "group1")

        assert "key1" not in schema.group("group1").edges

    def test_add_group(self) -> None:
        schema = mr.Schema(groups={}, ungrouped=mr.GroupSchema())

        schema.add_group(
            "group1",
            mr.GroupSchema(nodes={"key1": mr.Int()}, edges={"key1": mr.Float()}),
        )

        assert {"key1": (mr.Int(), mr.AttributeType.Continuous)} == schema.group(
            "group1"
        ).nodes
        assert {"key1": (mr.Float(), mr.AttributeType.Continuous)} == schema.group(
            "group1"
        ).edges

        with pytest.raises(ValueError, match=r"Group group1 already exists in schema."):
            schema.add_group("group1", mr.GroupSchema())

    def test_remove_group(self) -> None:
        schema = mr.Schema(
            groups={"group1": mr.GroupSchema()}, ungrouped=mr.GroupSchema()
        )

        assert "group1" in schema.groups

        schema.remove_group("group1")

        assert "group1" not in schema.groups

    def test_freeze_unfreeze(self) -> None:
        schema = mr.Schema(groups={}, ungrouped=mr.GroupSchema())

        schema.freeze()

        assert schema.schema_type == mr.SchemaType.Provided

        schema.unfreeze()

        assert schema.schema_type == mr.SchemaType.Inferred


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestAttributeType))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGroupSchema))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSchema))

    unittest.TextTestRunner(verbosity=2).run(suite)
