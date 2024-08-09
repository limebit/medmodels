import unittest

import medmodels.medrecord as mr
from medmodels._medmodels import PyAttributeType
from medmodels.medrecord.schema import GroupSchema, Schema


def create_medrecord() -> mr.MedRecord:
    return mr.MedRecord.from_example_dataset()


class TestSchema(unittest.TestCase):
    def setUp(self):
        self.schema = create_medrecord().schema

    def test_groups(self):
        self.assertEqual(
            sorted(
                [
                    "diagnosis",
                    "drug",
                    "patient_diagnosis",
                    "patient_drug",
                    "patient_procedure",
                    "patient",
                    "procedure",
                ]
            ),
            sorted(self.schema.groups),
        )

    def test_group(self):
        self.assertTrue(isinstance(self.schema.group("patient"), mr.GroupSchema))  # pyright: ignore[reportUnnecessaryIsInstance]

        with self.assertRaises(ValueError):
            self.schema.group("nonexistent")

    def test_default(self):
        self.assertEqual(None, self.schema.default)
        schema = Schema(default=GroupSchema(nodes={"description": mr.String()}))

        self.assertTrue(isinstance(schema.default, mr.GroupSchema))

    def test_strict(self):
        self.assertEqual(True, self.schema.strict)


class TestGroupSchema(unittest.TestCase):
    def setUp(self):
        self.schema = create_medrecord().schema

    def test_nodes(self):
        self.assertEqual(
            {
                "age": (mr.Int(), mr.AttributeType.Continuous),
                "gender": (mr.String(), mr.AttributeType.Categorical),
            },
            self.schema.group("patient").nodes,
        )

    def test_edges(self):
        self.assertEqual(
            {
                "diagnosis_time": (mr.DateTime(), mr.AttributeType.Temporal),
                "duration_days": (mr.Option(mr.Float()), mr.AttributeType.Continuous),
            },
            self.schema.group("patient_diagnosis").edges,
        )

    def test_strict(self):
        self.assertEqual(True, self.schema.group("patient").strict)


class TestAttributesSchema(unittest.TestCase):
    def setUp(self):
        self.attributes_schema = (
            Schema(
                groups={"diagnosis": GroupSchema(nodes={"description": mr.String()})},
                strict=False,
            )
            .group("diagnosis")
            .nodes
        )

    def test_repr(self):
        self.assertEqual(
            "{'description': (DataType.String, None)}",
            repr(self.attributes_schema),
        )

        second_attributes_schema = (
            Schema(
                groups={
                    "diagnosis": GroupSchema(
                        nodes={
                            "description": (mr.String(), mr.AttributeType.Categorical)
                        }
                    )
                },
                strict=False,
            )
            .group("diagnosis")
            .nodes
        )

        self.assertEqual(
            "{'description': (DataType.String, AttributeType.Categorical)}",
            repr(second_attributes_schema),
        )

    def test_getitem(self):
        self.assertEqual(
            (mr.String(), None),
            self.attributes_schema["description"],
        )

        with self.assertRaises(KeyError):
            self.attributes_schema["nonexistent"]

    def test_contains(self):
        self.assertTrue("description" in self.attributes_schema)
        self.assertFalse("nonexistent" in self.attributes_schema)

    def test_len(self):
        self.assertEqual(1, len(self.attributes_schema))

    def test_eq(self):
        comparison_attributes_schema = (
            Schema(
                groups={"diagnosis": GroupSchema(nodes={"description": mr.String()})},
                strict=False,
            )
            .group("diagnosis")
            .nodes
        )

        self.assertEqual(self.attributes_schema, comparison_attributes_schema)

        comparison_attributes_schema = (
            Schema(
                groups={"diagnosis": GroupSchema(nodes={"description": mr.Int()})},
                strict=False,
            )
            .group("diagnosis")
            .nodes
        )

        self.assertNotEqual(self.attributes_schema, comparison_attributes_schema)

        comparison_attributes_schema = (
            Schema(
                groups={
                    "diagnosis": GroupSchema(
                        nodes={
                            "description": (mr.String(), mr.AttributeType.Categorical)
                        }
                    )
                },
                strict=False,
            )
            .group("diagnosis")
            .nodes
        )

        self.assertNotEqual(self.attributes_schema, comparison_attributes_schema)

        comparison_attributes_schema = (
            Schema(
                groups={
                    "diagnosis": GroupSchema(
                        nodes={
                            "description2": (mr.String(), mr.AttributeType.Categorical)
                        }
                    )
                },
                strict=False,
            )
            .group("diagnosis")
            .nodes
        )

        self.assertNotEqual(self.attributes_schema, comparison_attributes_schema)

        self.assertNotEqual(self.attributes_schema, None)

    def test_keys(self):
        self.assertEqual(["description"], list(self.attributes_schema.keys()))

    def test_values(self):
        self.assertEqual(
            [(mr.String(), None)],
            list(self.attributes_schema.values()),
        )

    def test_items(self):
        self.assertEqual(
            [("description", (mr.String(), None))],
            list(self.attributes_schema.items()),
        )

    def test_get(self):
        self.assertEqual(
            (mr.String(), None),
            self.attributes_schema.get("description"),
        )

        self.assertEqual(
            None,
            self.attributes_schema.get("nonexistent"),
        )

        self.assertEqual(
            (mr.String(), None),
            self.attributes_schema.get("nonexistent", (mr.String(), None)),
        )


class TestAttributeType(unittest.TestCase):
    def test_str(self):
        self.assertEqual("Categorical", str(mr.AttributeType.Categorical))
        self.assertEqual("Continuous", str(mr.AttributeType.Continuous))
        self.assertEqual("Temporal", str(mr.AttributeType.Temporal))

    def test_eq(self):
        self.assertEqual(mr.AttributeType.Categorical, mr.AttributeType.Categorical)
        self.assertEqual(mr.AttributeType.Categorical, PyAttributeType.Categorical)
        self.assertNotEqual(mr.AttributeType.Categorical, mr.AttributeType.Continuous)
        self.assertNotEqual(mr.AttributeType.Categorical, PyAttributeType.Continuous)

        self.assertEqual(mr.AttributeType.Continuous, mr.AttributeType.Continuous)
        self.assertEqual(mr.AttributeType.Continuous, PyAttributeType.Continuous)
        self.assertNotEqual(mr.AttributeType.Continuous, mr.AttributeType.Categorical)
        self.assertNotEqual(mr.AttributeType.Continuous, PyAttributeType.Categorical)

        self.assertEqual(mr.AttributeType.Temporal, mr.AttributeType.Temporal)
        self.assertEqual(mr.AttributeType.Temporal, PyAttributeType.Temporal)
        self.assertNotEqual(mr.AttributeType.Temporal, mr.AttributeType.Categorical)
        self.assertNotEqual(mr.AttributeType.Temporal, PyAttributeType.Categorical)
