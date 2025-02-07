import unittest

import pytest

import medmodels.medrecord as mr
from medmodels._medmodels import PyAttributeType
from medmodels.medrecord.schema import GroupSchema, Schema


def create_medrecord() -> mr.MedRecord:
    return mr.MedRecord.from_simple_example_dataset()


class TestSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = create_medrecord().schema

    def test_groups(self) -> None:
        assert sorted(
            [
                "diagnosis",
                "drug",
                "patient_diagnosis",
                "patient_drug",
                "patient_procedure",
                "patient",
                "procedure",
            ]
        ) == sorted(self.schema.groups)

    def test_group(self) -> None:
        assert isinstance(self.schema.group("patient"), mr.GroupSchema)  # pyright: ignore[reportUnnecessaryIsInstance]

        with pytest.raises(ValueError, match="No schema found for group: nonexistent"):
            self.schema.group("nonexistent")


class TestGroupSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = create_medrecord().schema

    def test_nodes(self) -> None:
        assert self.schema.group("patient").nodes == {
            "age": (mr.Int(), mr.AttributeType.Continuous),
            "gender": (mr.String(), mr.AttributeType.Categorical),
        }

    def test_edges(self) -> None:
        assert self.schema.group("patient_diagnosis").edges == {
            "time": (mr.DateTime(), mr.AttributeType.Temporal),
            "duration_days": (mr.Option(mr.Float()), mr.AttributeType.Continuous),
        }


class TestAttributesSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.attributes_schema = (
            Schema(
                groups={"diagnosis": GroupSchema(nodes={"description": mr.String()})},
            )
            .group("diagnosis")
            .nodes
        )

    def test_repr(self) -> None:
        assert (
            repr(self.attributes_schema)
            == "{'description': (DataType.String, AttributeType.Unstructured)}"
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
            )
            .group("diagnosis")
            .nodes
        )

        assert (
            repr(second_attributes_schema)
            == "{'description': (DataType.String, AttributeType.Categorical)}"
        )

    def test_getitem(self) -> None:
        assert (mr.String(), mr.AttributeType.Unstructured) == self.attributes_schema[
            "description"
        ]

        with pytest.raises(KeyError):
            self.attributes_schema["nonexistent"]

    def test_contains(self) -> None:
        assert "description" in self.attributes_schema
        assert "nonexistent" not in self.attributes_schema

    def test_len(self) -> None:
        assert len(self.attributes_schema) == 1

    def test_eq(self) -> None:
        comparison_attributes_schema = (
            Schema(
                groups={"diagnosis": GroupSchema(nodes={"description": mr.String()})},
            )
            .group("diagnosis")
            .nodes
        )

        assert self.attributes_schema == comparison_attributes_schema

        comparison_attributes_schema = (
            Schema(
                groups={"diagnosis": GroupSchema(nodes={"description": mr.Int()})},
            )
            .group("diagnosis")
            .nodes
        )

        assert self.attributes_schema != comparison_attributes_schema

        comparison_attributes_schema = (
            Schema(
                groups={
                    "diagnosis": GroupSchema(
                        nodes={
                            "description": (mr.String(), mr.AttributeType.Categorical)
                        }
                    )
                },
            )
            .group("diagnosis")
            .nodes
        )

        assert self.attributes_schema != comparison_attributes_schema

        comparison_attributes_schema = (
            Schema(
                groups={
                    "diagnosis": GroupSchema(
                        nodes={
                            "description2": (mr.String(), mr.AttributeType.Categorical)
                        }
                    )
                },
            )
            .group("diagnosis")
            .nodes
        )

        assert self.attributes_schema != comparison_attributes_schema

        assert self.attributes_schema is not None

    def test_keys(self) -> None:
        assert list(self.attributes_schema.keys()) == ["description"]

    def test_values(self) -> None:
        assert [(mr.String(), mr.AttributeType.Unstructured)] == list(
            self.attributes_schema.values()
        )

    def test_items(self) -> None:
        assert [("description", (mr.String(), mr.AttributeType.Unstructured))] == list(
            self.attributes_schema.items()
        )


class TestAttributeType(unittest.TestCase):
    def test_str(self) -> None:
        assert str(mr.AttributeType.Categorical) == "Categorical"
        assert str(mr.AttributeType.Continuous) == "Continuous"
        assert str(mr.AttributeType.Temporal) == "Temporal"

    def test_eq(self) -> None:
        assert mr.AttributeType.Categorical == mr.AttributeType.Categorical
        assert mr.AttributeType.Categorical == PyAttributeType.Categorical
        assert mr.AttributeType.Categorical != mr.AttributeType.Continuous
        assert mr.AttributeType.Categorical != PyAttributeType.Continuous

        assert mr.AttributeType.Continuous == mr.AttributeType.Continuous
        assert mr.AttributeType.Continuous == PyAttributeType.Continuous
        assert mr.AttributeType.Continuous != mr.AttributeType.Categorical
        assert mr.AttributeType.Continuous != PyAttributeType.Categorical

        assert mr.AttributeType.Temporal == mr.AttributeType.Temporal
        assert mr.AttributeType.Temporal == PyAttributeType.Temporal
        assert mr.AttributeType.Temporal != mr.AttributeType.Categorical
        assert mr.AttributeType.Temporal != PyAttributeType.Categorical


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestSchema)
    unittest.TextTestRunner(verbosity=2).run(run_test)

    run_test = unittest.TestLoader().loadTestsFromTestCase(TestGroupSchema)
    unittest.TextTestRunner(verbosity=2).run(run_test)

    run_test = unittest.TestLoader().loadTestsFromTestCase(TestAttributesSchema)
    unittest.TextTestRunner(verbosity=2).run(run_test)

    run_test = unittest.TestLoader().loadTestsFromTestCase(TestAttributeType)
    unittest.TextTestRunner(verbosity=2).run(run_test)
