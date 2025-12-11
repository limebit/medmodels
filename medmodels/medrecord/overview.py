"""Overview functions and classes for the medrecord module."""

from typing import TYPE_CHECKING, Dict, Final, Union

from medmodels._medmodels import (
    PY_DEFAULT_TRUNCATE_DETAILS,
    PyAttributeType,
    PyEdgeGroupOverview,
    PyGroupOverview,
    PyNodeGroupOverview,
    PyOverview,
)
from medmodels.medrecord.datatype import DataType
from medmodels.medrecord.schema import AttributeType
from medmodels.medrecord.types import MedRecordAttribute

if TYPE_CHECKING:
    from medmodels._medmodels import PyAttributeOverview
    from medmodels.medrecord.types import (
        CategoricalAttributeOverview,
        ContinuousAttributeOverview,
        TemporalAttributeOverview,
        UnstructuredAttributeOverview,
    )

DEFAULT_TRUNCATE_DETAILS: Final[int] = PY_DEFAULT_TRUNCATE_DETAILS


class AttributeOverview:
    """Overview data of an attribute."""

    _py_attribute_overview: "PyAttributeOverview"

    @classmethod
    def _from_py_attribute_overview(
        cls, py_attribute_overview: "PyAttributeOverview"
    ) -> "AttributeOverview":
        """Create an AttributeOverview from a PyAttributeOverview.

        Args:
            py_attribute_overview (PyAttributeOverview): The PyAttributeOverview
                to convert.

        Returns:
            AttributeOverview: The converted AttributeOverview.
        """
        attribute_overview = cls()
        attribute_overview._py_attribute_overview = py_attribute_overview
        return attribute_overview

    @property
    def data_type(self) -> DataType:
        """The data type of the attribute.

        Returns:
            DataType: The data type of the attribute.
        """
        return DataType._from_py_data_type(self._py_attribute_overview.data_type)

    @property
    def data(
        self,
    ) -> Union[
        "CategoricalAttributeOverview",
        "ContinuousAttributeOverview",
        "TemporalAttributeOverview",
        "UnstructuredAttributeOverview",
    ]:
        """The overview data of the attribute.

        Returns:
            Union[
                CategoricalAttributeOverview,
                ContinuousAttributeOverview,
                TemporalAttributeOverview,
                UnstructuredAttributeOverview,
            ]: The overview data of the attribute.
        """
        if (
            self._py_attribute_overview.data["attribute_type"]
            == PyAttributeType.Categorical
        ):
            return {
                "attribute_type": AttributeType.Categorical,
                "distinct_values": self._py_attribute_overview.data["distinct_values"],
            }

        if (
            self._py_attribute_overview.data["attribute_type"]
            == PyAttributeType.Continuous
        ):
            return {
                "attribute_type": AttributeType.Continuous,
                "min": self._py_attribute_overview.data["min"],
                "mean": self._py_attribute_overview.data["mean"],
                "max": self._py_attribute_overview.data["max"],
            }

        if (
            self._py_attribute_overview.data["attribute_type"]
            == PyAttributeType.Temporal
        ):
            return {
                "attribute_type": AttributeType.Temporal,
                "min": self._py_attribute_overview.data["min"],
                "max": self._py_attribute_overview.data["max"],
            }

        return {
            "attribute_type": AttributeType.Unstructured,
            "distinct_count": self._py_attribute_overview.data["distinct_count"],
        }


class NodeGroupOverview:
    """Overview data of a node group."""

    _py_node_group_overview: "PyNodeGroupOverview"

    @classmethod
    def _from_py_node_group_overview(
        cls, py_node_group_overview: "PyNodeGroupOverview"
    ) -> "NodeGroupOverview":
        """Create a NodeGroupOverview from a PyNodeGroupOverview.

        Args:
            py_node_group_overview (PyNodeGroupOverview): The PyNodeGroupOverview
                to convert.

        Returns:
            NodeGroupOverview: The converted NodeGroupOverview.
        """
        node_group_overview = cls()
        node_group_overview._py_node_group_overview = py_node_group_overview
        return node_group_overview

    @property
    def count(self) -> int:
        """The number of nodes in the group.

        Returns:
            int: The number of nodes in the group.
        """
        return self._py_node_group_overview.count

    @property
    def attributes(self) -> Dict[MedRecordAttribute, AttributeOverview]:
        """The attribute overviews of the node group.

        Returns:
            Dict[MedRecordAttribute, AttributeOverview]: The attribute overviews
                of the node group.
        """
        return {
            attribute: AttributeOverview._from_py_attribute_overview(py_overview)
            for attribute, py_overview in self._py_node_group_overview.attributes.items()
        }

    def __repr__(self) -> str:
        """Return the string representation of the NodeGroupOverview.

        Returns:
            str: The string representation of the NodeGroupOverview.
        """
        return self._py_node_group_overview.__repr__()


class EdgeGroupOverview:
    """Overview data of an edge group."""

    _py_edge_group_overview: "PyEdgeGroupOverview"

    @classmethod
    def _from_py_edge_group_overview(
        cls, py_edge_group_overview: "PyEdgeGroupOverview"
    ) -> "EdgeGroupOverview":
        """Create an EdgeGroupOverview from a PyEdgeGroupOverview.

        Args:
            py_edge_group_overview (PyEdgeGroupOverview): The PyEdgeGroupOverview
                to convert.

        Returns:
            EdgeGroupOverview: The converted EdgeGroupOverview.
        """
        edge_group_overview = cls()
        edge_group_overview._py_edge_group_overview = py_edge_group_overview
        return edge_group_overview

    @property
    def count(self) -> int:
        """The number of edges in the group.

        Returns:
            int: The number of edges in the group.
        """
        return self._py_edge_group_overview.count

    @property
    def attributes(self) -> Dict[MedRecordAttribute, AttributeOverview]:
        """The attribute overviews of the edge group.

        Returns:
            Dict[MedRecordAttribute, AttributeOverview]: The attribute overviews
                of the edge group.
        """
        return {
            attribute: AttributeOverview._from_py_attribute_overview(py_overview)
            for attribute, py_overview in self._py_edge_group_overview.attributes.items()
        }

    def __repr__(self) -> str:
        """Return the string representation of the EdgeGroupOverview.

        Returns:
            str: The string representation of the EdgeGroupOverview.
        """
        return self._py_edge_group_overview.__repr__()


class GroupOverview:
    """Overview data of a group (node and/or edge)."""

    _py_group_overview: PyGroupOverview

    @classmethod
    def _from_py_group_overview(
        cls, py_group_overview: PyGroupOverview
    ) -> "GroupOverview":
        """Create a GroupOverview from a PyGroupOverview.

        Args:
            py_group_overview (PyGroupOverview): The PyGroupOverview
                to convert.

        Returns:
            GroupOverview: The converted GroupOverview.
        """
        group_overview = cls()
        group_overview._py_group_overview = py_group_overview
        return group_overview

    @property
    def node_overview(self) -> NodeGroupOverview:
        """The node group overview.

        Returns:
            NodeGroupOverview: The node group overview.
        """
        return NodeGroupOverview._from_py_node_group_overview(
            self._py_group_overview.node_overview
        )

    @property
    def edge_overview(self) -> EdgeGroupOverview:
        """The edge group overview.

        Returns:
            EdgeGroupOverview: The edge group overview.
        """
        return EdgeGroupOverview._from_py_edge_group_overview(
            self._py_group_overview.edge_overview
        )

    def __repr__(self) -> str:
        """Return the string representation of the GroupOverview.

        Returns:
            str: The string representation of the GroupOverview.
        """
        return self._py_group_overview.__repr__()


class Overview:
    """Overview functions for the medrecord module."""

    _py_overview: PyOverview

    @classmethod
    def _from_py_overview(cls, py_overview: PyOverview) -> "Overview":
        """Create an Overview from a PyOverview.

        Args:
            py_overview (PyOverview): The PyOverview to convert.

        Returns:
            Overview: The converted Overview.
        """
        overview = cls()
        overview._py_overview = py_overview
        return overview

    @property
    def ungrouped_verview(self) -> GroupOverview:
        """The overview of ungrouped nodes/edges.

        Returns:
            GroupOverview: The overview of ungrouped nodes/edges.
        """
        return GroupOverview._from_py_group_overview(
            self._py_overview.ungrouped_overview
        )

    @property
    def grouped_overviews(self) -> Dict[MedRecordAttribute, GroupOverview]:
        """The overviews of grouped nodes/edges.

        Returns:
            Dict[MedRecordAttribute, GroupOverview]: The overviews of grouped
                nodes/edges.
        """
        return {
            attribute: GroupOverview._from_py_group_overview(py_overview)
            for attribute, py_overview in self._py_overview.grouped_overviews.items()
        }

    def __repr__(self) -> str:
        """Return the string representation of the Overview.

        Returns:
            str: The string representation of the Overview.
        """
        return self._py_overview.__repr__()
