import unittest
from datetime import datetime

import pandas as pd
import polars as pl

from medmodels.medrecord.types import (
    is_attributes,
    is_edge_index,
    is_edge_tuple,
    is_edge_tuple_list,
    is_group,
    is_medrecord_attribute,
    is_medrecord_value,
    is_node_index,
    is_node_tuple,
    is_node_tuple_list,
    is_pandas_edge_dataframe_input,
    is_pandas_edge_dataframe_input_list,
    is_pandas_node_dataframe_input,
    is_pandas_node_dataframe_input_list,
    is_polars_edge_dataframe_input,
    is_polars_edge_dataframe_input_list,
    is_polars_node_dataframe_input,
    is_polars_node_dataframe_input_list,
)


class TestTypeAssertions(unittest.TestCase):
    def test_is_medrecord_attribute(self) -> None:
        assert is_medrecord_attribute("test")
        assert is_medrecord_attribute(123)
        assert not is_medrecord_attribute(12.34)
        assert not is_medrecord_attribute(None)

    def test_is_medrecord_value(self) -> None:
        assert is_medrecord_value("test")
        assert is_medrecord_value(123)
        assert is_medrecord_value(12.34)
        assert is_medrecord_value(value=True)
        assert is_medrecord_value(datetime.now())
        assert is_medrecord_value(None)
        assert not is_medrecord_value([])
        assert not is_medrecord_value({})

    def test_is_node_index(self) -> None:
        assert is_node_index("node")
        assert is_node_index(123)
        assert not is_node_index(12.34)

    def test_is_edge_index(self) -> None:
        assert is_edge_index(123)
        assert not is_edge_index("edge")
        assert not is_edge_index(12.34)

    def test_is_group(self) -> None:
        assert is_group("group")
        assert is_group(123)
        assert not is_group(12.34)

    def test_is_attributes(self) -> None:
        assert is_attributes({"key": "value"})
        assert is_attributes({"key": 123})
        assert not is_attributes(["key", "value"])
        assert not is_attributes("string")

    def test_is_node_tuple(self) -> None:
        assert is_node_tuple(("node", {"key": "value"}))
        assert not is_node_tuple(("node", "value"))
        assert not is_node_tuple(("node",))
        assert not is_node_tuple("node")

    def test_is_node_tuple_list(self) -> None:
        assert is_node_tuple_list(
            [("node1", {"key": "value"}), ("node2", {"key": 123})]
        )
        assert not is_node_tuple_list([("node1", {"key": "value"}), "invalid"])

    def test_is_edge_tuple(self) -> None:
        assert is_edge_tuple(("node1", "node2", {"key": "value"}))
        assert not is_edge_tuple(("node1", "node2"))
        assert not is_edge_tuple(("node1", "node2", "value"))

    def test_is_edge_tuple_list(self) -> None:
        assert is_edge_tuple_list(
            [
                ("node1", "node2", {"key": "value"}),
                ("node3", "node4", {"key": 123}),
            ]
        )
        assert not is_edge_tuple_list(
            [
                ("node1", "node2", {"key": "value"}),
                "invalid",
            ]
        )

    def test_is_polars_node_dataframe_input(self) -> None:
        df = pl.DataFrame({"col1": [1, 2, 3]})
        assert is_polars_node_dataframe_input((df, "col1"))
        assert not is_polars_node_dataframe_input((df, 123))
        assert not is_polars_node_dataframe_input(("invalid", "col1"))

    def test_is_polars_node_dataframe_input_list(self) -> None:
        df = pl.DataFrame({"col1": [1, 2, 3]})
        assert is_polars_node_dataframe_input_list([(df, "col1"), (df, "col2")])
        assert not is_polars_node_dataframe_input_list([(df, "col1"), "invalid"])

    def test_is_polars_edge_dataframe_input(self) -> None:
        df = pl.DataFrame({"col1": [1, 2, 3]})
        assert is_polars_edge_dataframe_input((df, "col1", "col2"))
        assert not is_polars_edge_dataframe_input((df, "col1", 123))
        assert not is_polars_edge_dataframe_input(("invalid", "col1", "col2"))

    def test_is_polars_edge_dataframe_input_list(self) -> None:
        df = pl.DataFrame({"col1": [1, 2, 3]})
        assert is_polars_edge_dataframe_input_list(
            [(df, "col1", "col2"), (df, "col3", "col4")]
        )
        assert not is_polars_edge_dataframe_input_list(
            [(df, "col1", "col2"), "invalid"]
        )

    def test_is_pandas_node_dataframe_input(self) -> None:
        df = pd.DataFrame({"col1": [1, 2, 3]})
        assert is_pandas_node_dataframe_input((df, "col1"))
        assert not is_pandas_node_dataframe_input((df, 123))
        assert not is_pandas_node_dataframe_input(("invalid", "col1"))

    def test_is_pandas_node_dataframe_input_list(self) -> None:
        df = pd.DataFrame({"col1": [1, 2, 3]})
        assert is_pandas_node_dataframe_input_list([(df, "col1"), (df, "col2")])
        assert not is_pandas_node_dataframe_input_list([(df, "col1"), "invalid"])

    def test_is_pandas_edge_dataframe_input(self) -> None:
        df = pd.DataFrame({"col1": [1, 2, 3]})
        assert is_pandas_edge_dataframe_input((df, "col1", "col2"))
        assert not is_pandas_edge_dataframe_input((df, "col1", 123))
        assert not is_pandas_edge_dataframe_input(("invalid", "col1", "col2"))

    def test_is_pandas_edge_dataframe_input_list(self) -> None:
        df = pd.DataFrame({"col1": [1, 2, 3]})
        assert is_pandas_edge_dataframe_input_list(
            [(df, "col1", "col2"), (df, "col3", "col4")]
        )
        assert not is_pandas_edge_dataframe_input_list(
            [(df, "col1", "col2"), "invalid"]
        )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestTypeAssertions)
    unittest.TextTestRunner(verbosity=2).run(run_test)
