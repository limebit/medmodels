import unittest

import medmodels.medrecord as mr


class TestMedRecordBuilder(unittest.TestCase):
    def test_add_nodes(self):
        builder = mr.MedRecord.builder().add_nodes([("node1", {})])

        self.assertEqual(len(builder._nodes), 1)

        builder.add_nodes([("node2", {})], group="group")

        self.assertEqual(len(builder._nodes), 2)

        medrecord = builder.build()

        self.assertEqual(2, len(medrecord.nodes))
        self.assertEqual(1, len(medrecord.groups))
        self.assertEqual(["group"], medrecord.groups_of_node("node2"))

    def test_add_edges(self):
        builder = (
            mr.MedRecord.builder()
            .add_nodes([("node1", {}), ("node2", {})])
            .add_edges([("node1", "node2", {})])
        )

        self.assertEqual(len(builder._edges), 1)

        builder.add_edges([("node2", "node1", {})], group="group")

        medrecord = builder.build()

        self.assertEqual(2, len(medrecord.nodes))
        self.assertEqual(2, len(medrecord.edges))
        self.assertEqual(1, len(medrecord.groups))
        self.assertEqual(["node2"], medrecord.neighbors("node1"))
        self.assertEqual(["group"], medrecord.groups_of_edge(1))

    def test_add_group(self):
        builder = (
            mr.MedRecord.builder().add_nodes(("0", {})).add_group("group", nodes=["0"])
        )

        self.assertEqual(len(builder._groups), 1)

        medrecord = builder.build()

        self.assertEqual(1, len(medrecord.nodes))
        self.assertEqual(0, len(medrecord.edges))
        self.assertEqual(1, len(medrecord.groups))
        self.assertEqual("group", medrecord.groups[0])
        self.assertEqual(["0"], medrecord.nodes_in_group("group"))

    def test_with_schema(self):
        schema = mr.Schema(default=mr.GroupSchema(nodes={"attribute": mr.Int()}))

        medrecord = mr.MedRecord.builder().with_schema(schema).build()

        medrecord.add_node("node", {"attribute": 1})

        with self.assertRaises(ValueError):
            medrecord.add_node("node", {"attribute": "1"})
