import unittest

import pytest

import medmodels.medrecord as mr


class TestMedRecordBuilder(unittest.TestCase):
    def test_add_nodes(self) -> None:
        builder = mr.MedRecord.builder().add_nodes([("node1", {})])

        assert len(builder._nodes) == 1

        builder.add_nodes([("node2", {})], group="group")

        assert len(builder._nodes) == 2

        medrecord = builder.build()

        assert len(medrecord.nodes) == 2
        assert len(medrecord.groups) == 1
        assert medrecord.groups_of_node("node2") == ["group"]

    def test_add_edges(self) -> None:
        builder = (
            mr.MedRecord.builder()
            .add_nodes([("node1", {}), ("node2", {})])
            .add_edges([("node1", "node2", {})])
        )

        assert len(builder._edges) == 1

        builder.add_edges([("node2", "node1", {})], group="group")

        medrecord = builder.build()

        assert len(medrecord.nodes) == 2
        assert len(medrecord.edges) == 2
        assert len(medrecord.groups) == 1
        assert medrecord.neighbors("node1") == ["node2"]
        assert medrecord.groups_of_edge(1) == ["group"]

    def test_add_group(self) -> None:
        builder = (
            mr.MedRecord.builder().add_nodes(("0", {})).add_group("group", nodes=["0"])
        )

        assert len(builder._groups) == 1

        medrecord = builder.build()

        assert len(medrecord.nodes) == 1
        assert len(medrecord.edges) == 0
        assert len(medrecord.groups) == 1
        assert medrecord.groups[0] == "group"
        assert medrecord.nodes_in_group("group") == ["0"]

    def test_with_schema(self) -> None:
        schema = mr.Schema(default=mr.GroupSchema(nodes={"attribute": mr.Int()}))

        medrecord = mr.MedRecord.builder().with_schema(schema).build()

        medrecord.add_nodes(("node1", {"attribute": 1}))

        with pytest.raises(
            ValueError,
            match=r"Attribute attribute of node with index node2 is of type String\. Expected Int\.",
        ):
            medrecord.add_nodes(("node2", {"attribute": "1"}))


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestMedRecordBuilder)
    unittest.TextTestRunner(verbosity=2).run(run_test)
