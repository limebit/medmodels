import networkx as nx
import numpy as np
import itertools
import logging
import pandas as pd
from collections import defaultdict
from typing import List, Dict
from dataclasses import dataclass, make_dataclass, field
from medmodels.dataclass.utils import parse_criteria


class MedRecord:
    def __init__(self):
        """Initilize the MedRecord Class, initialize a node mapping. This may later be
        in the format of {'dimension_name':[node_id1, node_id2,..]}"""
        self.G = nx.MultiGraph()
        self._node_mapping = {}
        self.reserved_namespace = ["__group__"]

    @property
    def groups(self) -> List[str]:
        """Return all available groups

        :return: List of groups
        :rtype: list
        """
        if "__group__" in self._node_mapping.keys():
            return self._node_mapping["__group__"]
        else:
            return []

    def group(self, *name: str) -> List[str]:
        """Get ids of nodes, that belong to a group

        :param name: Name of the group
        :type name: str
        :return: List of nodes in that group
        :rtype: list
        """
        assert "__group__" in self._node_mapping.keys() and set(name).issubset(
            self._node_mapping["__group__"]
        ), "This group does not exist"
        return list(itertools.chain(*[self.G[key].keys() for key in name]))

    @property
    def nodes(self) -> List[str]:
        """Get a list of all nodes in the Graph

        :return: List of node ids
        :rtype: list
        """
        return list(self.G.nodes.keys())

    def node(self, *node_id: str) -> Dict[str, dict]:
        """Get node by id

        :param node_id: Identifier of the node
        :type node_id: str
        :return: Node as dictionary
        :rtype: dict
        """
        return {key: self.G.nodes[key] for key in node_id}

    @property
    def dimensions(self) -> List[str]:
        """Get a list of dimensions

        :return: List of dimensions in medrecord
        :rtype: list
        """
        return [
            key
            for key in self._node_mapping.keys()
            if key not in self.reserved_namespace
        ]

    def dimension(self, *dimension_name: str) -> List[str]:
        """Get all nodes within given dimensions

        :param dimension_name: Name of the dimension/s
        :type dimension_name: str
        :return: List of nodes
        :rtype: list
        """
        return list(
            itertools.chain(*[self._node_mapping[key] for key in dimension_name])
        )

    @property
    def edges(self) -> List[str]:
        """Get all edges

        :return: List of all edges
        :rtype: list
        """
        return list(nx.generate_edgelist(self.G))

    def edge(self, start_node: str, end_node: str) -> Dict[str, dict]:
        """Get all direct edges between a start and endpoint

        :param start_node: Node id of start point
        :type start_node: str
        :param end_node: Node id of end point
        :type end_node: str
        :return: Dict of edges
        :rtype: dict
        """
        return self.G.get_edge_data(start_node, end_node)

    def neighbors(self, *nodes: str, dimension_filter: List[str] = []) -> List[str]:
        """Get all direct neighbors of certain nodes
        :param nodes: Nodes to find neighbors for
        :type *nodes: str
        :return: List of nodes that neighbor the startpoints
        :rtype: list
        """
        assert set(nodes).issubset(set(self.nodes)), "Found non-existing node"
        neighbors = itertools.chain(*(list(self.G.neighbors(node)) for node in nodes))
        if len(dimension_filter) > 0:
            neighbors = list(
                set(neighbors).intersection(self.dimension(*dimension_filter))
            )
        return list(set(neighbors))

    def add_nodes(self, nodes: np.ndarray, dimension_name: str) -> None:
        """Add a dimension (number of nodes) to the dataset

        :param dimension: Array with dimension identifier and attributes in
        format [(id, {key:value}),]
        :type dimension: np.ndarray
        :param dimension_name: Name of the dimension (e.g. 'medication', 'diagnosis',..)
        :type dimension_name: str
        """
        assert self.is_unique(
            nodes[:, :1]
        ), "Dimension identifier are required to be unique, found duplicates"

        if (
            dimension_name in self._node_mapping.keys()
            and dimension_name not in self.reserved_namespace
        ):
            logging.info(f"Info: Dimension {dimension_name} in use, will append data.")

        self.G.add_nodes_from(nodes)
        n = nodes.copy()
        nodes = nodes[:, :1].squeeze().tolist()
        # just in case that there is only one node for a dimension
        if type(nodes) is not list:
            nodes = [nodes]

        # in case the dimension is new
        if dimension_name not in self._node_mapping.keys():
            self._node_mapping[dimension_name] = nodes
        else:
            self._node_mapping[dimension_name] += nodes

        unique_attributes = set([])
        for _, data in n:
            unique_attributes |= set(data.keys())

        self.update_explorer(dimension_name, unique_attributes)

    def add_edges(self, relation: np.ndarray) -> None:
        """Add edges between nodes to the graph

        :param relation: Edges with optional attributes in format
        [(identifier_node_1, identifier_node_2, {key:value})]
        :type relation: np.ndarray
        """
        self.G.add_edges_from(relation)

    def remove_node_from_group(
        self,
        group_name: str,
        *identifier: List[str],
    ) -> None:
        """Remove a node from a group

        :param group_name: Name of the group to remove frmo
        :type group_name: str
        :param identifier: List of node identifiers to be removed
        :type identifier: list
        """
        assert group_name in self.groups, "This group does not exist"
        self.G.remove_edges_from([(i, group_name) for i in identifier])

    def get_dimension_name(self, node_id: str) -> str:
        """Get dimension name of a node

        :param node_id: Identifier of the node
        :type node_id: str
        :return: Name of the dimension
        :rtype: str
        """
        for dim, node_list in self._node_mapping.items():
            if node_id in node_list and dim not in self.reserved_namespace:
                return dim
        raise KeyError("Node not found")

    def is_unique(self, x: np.ndarray) -> bool:
        """Check if some given data has unique identifiers and
        is not conflicting with other ids

        :param x: Array to be checked
        :type x: np.ndarray
        :return: True if idenfiers are unique
        :rtype: bool
        """
        unique_local = len(np.unique(x)) == len(x)

        node_ids = set(itertools.chain(*x.tolist()))
        existing_node_ids = list(itertools.chain(*self._node_mapping.values()))
        unique_global = node_ids.isdisjoint(existing_node_ids)
        return unique_local and unique_global

    def __repr__(self) -> str:
        """Represent the object

        :return: Object description
        :rtype: _type_
        """
        dims = ", ".join(
            [f"{key} ({len(value)} Nodes)" for key, value in self._node_mapping.items()]
        )
        return f"MedicalRecord dimensions: {dims}"

    def edges_to_df(self) -> pd.DataFrame:
        """Convert the edges to a pandas dataframe

        :return: Pandas dataframe containing edges
        :rtype: pd.DataFrame
        """
        edges = pd.DataFrame(
            [
                {
                    f"{e[2]['relation_type'].split('_')[0]}_id": e[0],
                    f"{e[2]['relation_type'].split('_')[1]}_id": e[1],
                    **e[2],
                }
                if "relation_type" in e[2]
                else {"id1": e[0], "id2": e[1], **e[2]}
                for e in self.G.edges(data=True)
            ]
        )
        return edges

    def nodes_to_df(
        self, dimension_name: str, dimension_id: str = None
    ) -> pd.DataFrame:
        """Convert the nodes to a pandas dataframe.

        :param dimension_name: The name of the dimension of the nodes
        :type dimension: str
        :param dimension_id: The name of the id of the nodes Defaults to "id".
        :type dimension_id: str, optional.
        :return: A pandas dataframe with the nodes
        :rtype: pd.DataFrame
        """
        if not dimension_id:
            dimension_id = dimension_name + "_id"
        return pd.DataFrame(
            [
                {dimension_id: d, **self.G.nodes[d]}
                for d in self._node_mapping[dimension_name]
            ]
        )

    def dimensions_to_dict(
        self,
        patients_dim_name: str = "patients",
    ) -> Dict[str, pd.DataFrame]:
        """Convert the medical record to a dictionary containing DataFrames with the
        information of each dimension.

        :param patients_dim_name: Name of the patients dimension.
            Defaults to "patients".
        :type patients_dim_name: str, optional
        :return: Dictionary with the information for each dimension.
        :rtype: Dict[str, pd.DataFrame]
        """
        edges = self.edges_to_df()
        assert (
            "relation_type" in edges.columns
        ), "relation_type column not found in edges dataframe, needed for DataFrame."
        info = {}

        for dimension in self.dimensions:
            nodes = self.nodes_to_df(dimension)
            nodes = nodes.rename(columns={"id": dimension + "_id"})

            if dimension == patients_dim_name:
                info[dimension] = nodes
                continue

            edges_dim = edges[
                edges["relation_type"] == patients_dim_name + "_" + dimension
            ]

            edges_dim = edges_dim.drop(columns=["relation_type"])
            edges_dim = edges_dim.dropna(how="all", axis=1).reset_index(drop=True)

            shared_cols = list(set(nodes.columns).intersection(edges_dim.columns))
            nodes = nodes.merge(edges_dim, on=shared_cols, how="inner")
            info[dimension] = nodes

        return info

    def to_df(
        self,
        unique_dimensions: List[str] = [],
        patients_dim_name: str = "patients",
    ) -> pd.DataFrame:
        """Convert the medical record to a pandas DataFrame with all data.

        :param unique_dimensions: Name of the dimension in case we want to have only
            one dimension in the dictionary. Defaults to None.
        :type unique_dimensions: List[str], optional
        :param patients_dim_name: Name of the patients dimension.
            Defaults to "patients".
        :type patients_dim_name: str, optional
        :return: DataFrame with the events and all their informaiton.
        :rtype: pd.DataFrame
        """
        edges = self.edges_to_df()
        assert (
            "relation_type" in edges.columns
        ), "relation_type column not found in edges dataframe, needed for DataFrame."
        assert isinstance(
            unique_dimensions, list
        ), "unique_dimensions must be a list of strings"

        info = {}
        dimensions = self.dimensions if not unique_dimensions else unique_dimensions

        for dimension in dimensions:
            nodes = self.nodes_to_df(dimension)
            nodes = nodes.rename(columns={"id": dimension + "_id"})
            nodes.insert(0, "type", dimension)

            if dimension == patients_dim_name:
                info[dimension] = nodes
                continue

            edges_dim = edges[
                edges["relation_type"] == patients_dim_name + "_" + dimension
            ]

            edges_dim = edges_dim.drop(columns=["relation_type"])
            edges_dim = edges_dim.dropna(how="all", axis=1).reset_index(drop=True)

            shared_cols = list(set(nodes.columns).intersection(edges_dim.columns))
            nodes = nodes.merge(edges_dim, on=shared_cols, how="inner")
            info[dimension] = nodes

        return pd.concat(info.values(), ignore_index=True)

    def add_group(
        self,
        name: str,
        identifier: List[str] = [],
        criteria: List[str] = [],
        attributes: dict = {},
        ignore_unknown_nodes: bool = False,
    ) -> None:
        """Group nodes by identifier and/or criteria

        :param name: Name of the group to be created
        :type name: str
        :param identifier: List of identifiers for further matching with criteria.
        If list is empty, consider all nodes for further criteria matching,
        defaults to []
        :type identifier: list, optional
        :param criteria: List of criteria for matching, each list entry in the format
        "{dimension} {attribute} {function, e.g. >} {parameter}", defaults to []
        :type criteria: list, optional
        :param attributes: Attributes of this group, defaults to {}
        :type attributes: dict, optional
        :param ignore_unknown_nodes: Flag that determines how to deal with unknown node
        identifer. If True, unknown nodes in identifier are ignored, defaults to False
        :type ignore_unknown_nodes: bool, optional
        :return: None
        :rtype: None
        """

        assert all(
            [isinstance(identifier, list), isinstance(criteria, list)]
        ), "Expected identifier and/or critieria to be lists"
        assert len(identifier + criteria) > 0, "Expected list of identifier or criteria"
        assert (
            set(identifier).issubset(set(self.nodes)) or ignore_unknown_nodes is True
        ), str(
            "Can not add non-existing nodes to a group. "
            "Remove unknown nodes from identifier list or set. "
            "ignore_unknown_nodes=True"
        )

        # ignore unknown nodes (flag tested in asserts)
        if not set(identifier).issubset(set(self.nodes)):
            identifier = [i for i in set(identifier) if i in self.nodes]

        criteria_list = parse_criteria(criteria)

        hit_nodes = []

        nodes_to_iterate = defaultdict(list)
        if len(identifier) == 0:
            nodes_to_iterate = self._node_mapping
        else:
            for i in identifier:
                dimension_name = self.get_dimension_name(i)
                nodes_to_iterate[dimension_name].append(i)

        # if there are are specified search criteria, search for them
        if len(criteria_list) > 0:
            sorted_criteria = {}
            for c in criteria_list:
                # add dimension to sorting
                if c[0] not in sorted_criteria:
                    sorted_criteria[c[0]] = []
                sorted_criteria[c[0]].append([c[1], c[2], c[3]])

            for dimension, node_list in nodes_to_iterate.items():
                if dimension not in sorted_criteria.keys():
                    continue
                for node, data in self.node(*node_list).items():
                    if all(
                        [f(data[attr], p) for attr, f, p in sorted_criteria[dimension]]
                    ):
                        hit_nodes.append(node)

        else:
            hit_nodes = identifier

        # if we are adding to a non-existing group
        if name not in self.groups:
            self.add_nodes(np.array([[name, attributes]]), "__group__")

        self.add_edges(np.array([[name, n, {}] for n in hit_nodes]))

    def remove_group(self, group_name: str) -> None:
        """Remove a group from the MedRecord

        :param group_name: Name of the group to remove
        :type group_name: str
        """
        assert group_name in self.groups, "This group does not exist"

        # Remove edges connected to the group
        self.G.remove_node(group_name)

        # Remove the group from the list of groups in __group__ category
        self._node_mapping["__group__"] = [
            group for group in self._node_mapping["__group__"] if group != group_name
        ]

        # Remove the category of groups if there are no groups left
        if not self._node_mapping["__group__"]:
            del self._node_mapping["__group__"]

    def update_explorer(self, dim: str, attributes: List[str]) -> None:
        """Update the explorer with new dimensions or attribues

        :param dim: Dimension to be updated
        :type dim: str
        :param attributes: Attributes of the dimensions
        :type attributes: list
        """
        dimension = make_dataclass(
            dim,
            [(name, str, field(default=dim + " " + name)) for name in attributes],
        )

        # if there is no navigator yet
        if not hasattr(self, "explorer"):
            self.explorer = make_dataclass("explorer", [(dim, dataclass, dimension)])

        # if there is already a navigator
        else:
            setattr(self.explorer, dim, dimension)
