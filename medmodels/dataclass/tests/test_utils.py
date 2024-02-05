import numpy as np
import pandas as pd
from medmodels.dataclass.utils import (
    df_to_nodes,
    df_to_edges,
    align_types,
    _larger,
    _larger_equals,
    _smaller,
    _smaller_equals,
    _not_equals,
    _equals,
    _anyof,
    _noneof,
    _startwith,
    _startwithany,
    _not_startwith,
    parse_criteria,
)
import unittest


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.sample_data1 = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "attr1": ["a", "b", "c"],
                "attr2": ["d", "e", "f"],
                "attr3": ["g", "h", "i"],
            }
        )  # Standard data

        self.sample_data2 = pd.DataFrame(
            {
                "id": ["1", "2", "2"],
                "attr1": ["a", "b", "c"],
                "attr2": ["d", "e", "f"],
                "attr3": ["g", "h", "i"],
            }
        )  # Data with duplicate ID

        self.sample_data3 = pd.DataFrame(
            {
                "id": ["1", "2", np.nan],
                "attr1": ["a", "b", "c"],
                "attr2": ["d", "e", "f"],
                "attr3": ["g", "h", "i"],
            }
        )  # Data with NaN in ID column

        self.sample_data4 = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "attr1": ["a", "b", "a"],
                "attr2": ["d", "e", "f"],
                "attr3": ["g", "h", "i"],
            }
        )  # Duplicating "a" attribute: two nodes connecting an edge to the same node

    def test_nodes(self):

        output = df_to_nodes(self.sample_data1, "id", ["attr1", "attr2"])
        self.assertTrue(
            np.array_equal(
                output,
                np.array(
                    [
                        ["1", {"attr1": "a", "attr2": "d"}],
                        ["2", {"attr1": "b", "attr2": "e"}],
                        ["3", {"attr1": "c", "attr2": "f"}],
                    ]
                ),
            )
        )

        output = df_to_nodes(self.sample_data2, "id", ["attr1", "attr2"])

        # We see that the last column is not added because the id is repeated
        self.assertTrue(
            np.array_equal(
                output,
                np.array(
                    [
                        ["1", {"attr1": "a", "attr2": "d"}],
                        ["2", {"attr1": "b", "attr2": "e"}],
                    ]
                ),
            )
        )

        # We see that the 3rd node is not added, because there is a NaN in the id column
        output = df_to_nodes(self.sample_data3, "id", ["attr2", "attr3"])

        self.assertTrue(
            np.array_equal(
                output,
                np.array(
                    [
                        ["1", {"attr2": "d", "attr3": "g"}],
                        ["2", {"attr2": "e", "attr3": "h"}],
                    ]
                ),
            )
        )

    def test_edges(self):

        output = df_to_edges(self.sample_data4, "id", "attr1", ["attr2", "attr3"])
        self.assertTrue(
            np.array_equal(
                output,
                np.array(
                    [
                        ["1", "a", {"attr2": "d", "attr3": "g"}],
                        ["2", "b", {"attr2": "e", "attr3": "h"}],
                        ["3", "a", {"attr2": "f", "attr3": "i"}],
                    ]
                ),
            )
        )

        # Here, we allow the id to be repeated, as nodes can have multiple edges
        output = df_to_edges(self.sample_data2, "id", "attr1", ["attr2", "attr3"])
        self.assertTrue(
            np.array_equal(
                output,
                np.array(
                    [
                        ["1", "a", {"attr2": "d", "attr3": "g"}],
                        ["2", "b", {"attr2": "e", "attr3": "h"}],
                        ["2", "c", {"attr2": "f", "attr3": "i"}],
                    ]
                ),
            )
        )

        # We see that the 3rd edge is added, even though there is a NaN in the id column
        output = df_to_edges(self.sample_data3, "id", "attr2", ["attr1", "attr3"])

        self.assertTrue(
            np.array_equal(
                output,
                np.array(
                    [
                        ["1", "d", {"attr1": "a", "attr3": "g"}],
                        ["2", "e", {"attr1": "b", "attr3": "h"}],
                        ["nan", "f", {"attr1": "c", "attr3": "i"}],
                    ]
                ),
            )
        )

    def test_align_types(self):
        @align_types
        def a(x, y):
            return type(x), type(y)

        self.assertEqual(a(1, 2), (int, int))
        self.assertEqual(a("1", 2), (str, str))
        self.assertEqual(a(1, "2"), (int, int))

    def test_parse_criteria(self):

        dim, attr, f, param = parse_criteria(["patients age > 10"])[0]

        self.assertEqual(dim, "patients")
        self.assertEqual(attr, "age")
        self.assertEqual(param, "10")
        self.assertIs(f, _larger)

    def test_criteria_filtering(self):
        self.assertTrue(_larger(7, 5))
        self.assertFalse(_larger(5, 7))
        self.assertFalse(_larger(7, 7))

        self.assertTrue(_larger_equals(7, 7))
        self.assertTrue(_larger_equals(7, 5))
        self.assertFalse(_larger_equals(5, 7))

        self.assertTrue(_smaller(5, 7))
        self.assertFalse(_smaller(7, 5))
        self.assertFalse(_smaller(7, 7))

        self.assertTrue(_smaller_equals(7, 7))
        self.assertTrue(_smaller_equals(5, 7))
        self.assertFalse(_smaller_equals(7, 5))

        self.assertFalse(_not_equals("a", "a"))
        self.assertTrue(_not_equals("a", "b"))
        self.assertFalse(_not_equals(1, 1))
        self.assertTrue(_not_equals(1, 2))

        self.assertFalse(_equals("a", "b"))
        self.assertTrue(_equals("a", "a"))
        self.assertFalse(_equals(1, 2))
        self.assertTrue(_equals(1, 1))

        self.assertTrue(_anyof("1", "[1,2,3]"))
        self.assertFalse(_anyof("4", "[1,2,3]"))

        self.assertFalse(_noneof("1", "[1,2,3]"))
        self.assertTrue(_noneof("4", "[1,2,3]"))

        self.assertTrue(_startwith("ABC", "AB"))
        self.assertFalse(_startwith("ABC", "ABB"))

        self.assertFalse(_not_startwith("ABC", "AB"))
        self.assertTrue(_not_startwith("ABC", "ABB"))

        self.assertTrue(_startwithany("ABCDE", "[A,B,C]"))
        self.assertFalse(_startwithany("ABCDE", "[B,C]"))


if __name__ == "__main__":

    run_test = unittest.TestLoader().loadTestsFromTestCase(TestUtils)
    unittest.TextTestRunner(verbosity=2).run(run_test)
