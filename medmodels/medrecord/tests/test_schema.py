import unittest

import medmodels.medrecord as mr
from medmodels._medmodels import PyAttributeType


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


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestAttributeType)
    unittest.TextTestRunner(verbosity=2).run(run_test)
