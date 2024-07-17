from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm

from medmodels import MedRecord
from medmodels.medrecord.types import Group, NodeIndex


class GraphPreprocessor:
    """
    This class implements the preprocessing steps for the HSGNN model.
    It computes meta-paths of length 3 and generates sparse matrices with similarity
    scores for each meta-path.
    The class implements slow naive and faster vectorized methods for creating the
    sparse matrices.
    """

    def __init__(self, medrecord: MedRecord, length_paths: int = 3):
        """Initializes the HSGNNPreprocessor class with a MedRecord object.

        Args:
            medrecord (MedRecord): MedRecord object
            length_paths (int): length of the meta-paths to be computed
        """
        self.medrecord = medrecord
        self.length_paths = length_paths

    def _find_group_connections(self) -> List[Tuple[Group, Group]]:
        """
        Find all group connections in the MedRecord graph.

        Args:
            None

        Returns:
            List[Tuple[Group, Group]]: List of tuples of group connections

        Example:
            >>> find_group_connections()
            [('patient', 'diagnoses')]
        """
        groups = self.medrecord.groups
        group_connections = [
            (group1, group2)
            for group1 in groups
            for group2 in groups
            if self.medrecord.edges_connecting(
                self.medrecord.nodes_in_group(group1),
                self.medrecord.nodes_in_group(group2),
            )
        ]

        return group_connections

    def _find_individual_metapath(
        self, connections: List[Tuple[Group, Group]], starting_group: Group
    ) -> List[str]:
        """
        Find all unique meta-paths of length self.length_paths starting from a given group.

        Args:
            connections(List[Tuple[Group, Group]]): List of group connections
            starting_group(Group): Group to start the search from

        Returns:
            List[str]:List of unique meta-paths of length self.length_paths starting from the given group
        """

        def depth_first_search(current_node: Group, current_path: List[Group]) -> None:
            """
            Depth-first search to find meta-paths of length self.length_paths.

            Args:
                current_node (Group): Current node in the search
                current_path (List[Group]): Current path in the search
            """
            # If the current path has reached the required length, add it to the results
            if len(current_path) == self.length_paths:
                paths.append(current_path)
                return

            # Continue exploring connections if not
            for connection in connections:
                if connection[0] == current_node:
                    depth_first_search(connection[1], current_path + [connection[1]])
                elif connection[1] == current_node:
                    depth_first_search(connection[0], current_path + [connection[0]])
                elif connection[0] == connection[1] == current_node:
                    depth_first_search(connection[0], current_path + [connection[0]])

        paths = []
        depth_first_search(starting_group, [starting_group])
        return paths

    def find_metapaths(self) -> List[List[str]]:
        """
        Find all unique meta-paths of length self.length_paths in the MedRecord graph.

        :return: A list of unique meta-paths of length 3:
                 e.g. [['patient', 'diagnoses', 'patient']]
        :rtype: List[List[str]]
        """
        group_connections = self._find_group_connections()
        all_metapaths = []

        for group in self.medrecord.groups:
            metapaths = self._find_individual_metapath(group_connections, group)
            all_metapaths.extend(metapaths)

        return all_metapaths

    def compute_matrix_indices(
        self, meta_path: List[str], similarity_scores: coo_matrix
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the rows, cols indices, reshapes the similarity
        scores array of the meta-path to build the sparse matrix of size (N x N),
        where N is the number of all nodes in MedRecord.

        :param meta_path: meta-path
        :type meta_path: List[str]
        :param similarity_scores: similarity scores of the meta-path
        :type similarity_scores: coo_matrix
        :return: rows, columns indices, reshaped similarity scores array
        :rtype: List[torch.tensor]
        """
        first_group = meta_path[0]
        last_group = meta_path[-1]
        similarity_scores_values = similarity_scores.data
        nodes_first_dimension = self.medrecord.nodes_in_group(first_group)
        nodes_last_dimension = self.medrecord.nodes_in_group(last_group)

        all_nodes = {node: index for index, node in enumerate(self.medrecord.nodes)}

        start_index_fisrt_dimension, end_index_first_dimension = (
            all_nodes[nodes_first_dimension[0]],
            all_nodes[nodes_first_dimension[-1]] + 1,
        )
        full_to_scores_mapping_first_dimension = {
            scores_index: full_index
            for scores_index, full_index in enumerate(
                range(start_index_fisrt_dimension, end_index_first_dimension)
            )
        }
        rows_first_dimension = np.vectorize(full_to_scores_mapping_first_dimension.get)(
            similarity_scores.row
        )

        # symmetric meta-paths
        if meta_path[-1] == meta_path[0]:
            rows = torch.tensor(rows_first_dimension)
            columns_first_dimension = np.vectorize(
                full_to_scores_mapping_first_dimension.get
            )(similarity_scores.col)
            columns = torch.tensor(columns_first_dimension)

        # non-symmetric meta-paths
        else:
            start_index_last_dimension, end_index_last_dimension = (
                all_nodes[nodes_last_dimension[0]],
                all_nodes[nodes_last_dimension[-1]] + 1,
            )
            full_to_scores_mapping_last_dimension = {
                scores_index: full_index
                for scores_index, full_index in enumerate(
                    range(start_index_last_dimension, end_index_last_dimension)
                )
            }
            columns_last_dimension = np.vectorize(
                full_to_scores_mapping_last_dimension.get
            )(similarity_scores.col)
            rows = torch.tensor(
                np.concatenate((rows_first_dimension, columns_last_dimension), axis=0)
            )
            columns = torch.tensor(
                np.concatenate((columns_last_dimension, rows_first_dimension), axis=0)
            )
            similarity_scores_values = np.tile(similarity_scores_values, 2)
        similarity_scores_values = torch.tensor(
            similarity_scores_values.flatten(), dtype=torch.float16
        )
        return rows, columns, similarity_scores_values

    def _get_adjacency_matrix(self, nodelist: List[NodeIndex]) -> csr_matrix:
        """
        Get the adjacency matrix of a list of nodes.

        :param nodelist: list of nodes
        :type nodelist: List[str]
        :return: adjacency matrix
        :rtype: csr_matrix
        """
        n = len(nodelist)
        rows = []
        cols = []
        data = []

        for i, node1 in enumerate(nodelist):
            for j, node2 in enumerate(nodelist):
                edges = self.medrecord.edges_connecting(node1, node2, directed=False)
                if edges:
                    rows.append(i)
                    cols.append(j)
                    data.append(len(edges))

        return csr_matrix((data, (rows, cols)), shape=(n, n))

    def path_count(
        self, nodes_groups: List[List[NodeIndex]]
    ) -> Tuple[csr_matrix, Dict[NodeIndex, int], List[range]]:
        """
        Computes PathCount matrix by multiplying matrix by itself,
        each cell matrix[i,j] will contain the number of shared neighbors by i and j.

        :param nodelist: list of nodes of a certain type defined in the meta_path
        :type nodelist: List[str]
        :return: PathCount matrix
        :rtype: csr_matrix
        """
        # TODO: raise an error if there is a node pertaining to multiple groups
        node_indices, nodes_ranges = self.create_node_indices_and_ranges(nodes_groups)

        nodelist = list(node_indices.keys())
        full_adjacency_matrix = self._get_adjacency_matrix(nodelist)

        path_count_matrix = csr_matrix((len(nodes_groups[0]), len(nodes_groups[0])))
        path_count_matrix.setdiag(1)

        for i in range(len(nodes_groups) - 1):
            current_nodes = nodes_groups[i]
            next_nodes = nodes_groups[i + 1]

            # Get the indices for the current and next node groups
            current_indices = [node_indices[node] for node in current_nodes]
            next_indices = [node_indices[node] for node in next_nodes]

            # Extract the relevant submatrix from the full adjacency matrix
            adj_submatrix = full_adjacency_matrix[current_indices][:, next_indices]
            path_count_matrix = path_count_matrix @ adj_submatrix

        return path_count_matrix, node_indices, nodes_ranges

    def symmetric_pathsim(
        self,
        path_count_matrix: csr_matrix,
        nodes_range_first_dimension: range,
        nodes_range_last_dimension: range,
    ) -> coo_matrix:
        """
        Computes Symmetric PathSim (SPS) matrix.
        SPS = (PC(i,j) + PC(j,i)) / (PC(i,i) + PC(j,j)) =
              2 * PC(i,j) / (PC(i,i) + PC(j,j))

        :param path_count_matrix: PathCount matrix
        :type path_count_matrix: csr_matrix
        :param nodes_ranges: List of ranges for each dimension in the meta-path
        :type nodes_ranges: List[range]
        :return: Symmetric PathSim COO matrix
        :rtype: coo_matrix
        """
        # If the path count matrix is a scalar, return a scalar
        if path_count_matrix.shape == (1, 1):
            value = path_count_matrix[0, 0]
            return coo_matrix([[1.0 if value > 0 else 0.0]])

        indices_first = np.arange(
            nodes_range_first_dimension.start, nodes_range_first_dimension.stop
        )
        indices_last = np.arange(
            nodes_range_last_dimension.start, nodes_range_last_dimension.stop
        )

        # Calculate numerator
        path_count_numerator = path_count_matrix[indices_first][:, indices_last].power(
            2
        )

        diag_first = path_count_matrix.diagonal()[indices_first]
        diag_last = path_count_matrix.diagonal()[indices_last]
        denominator = diag_first[:, np.newaxis] + diag_last
        # instead of division by zero np.inf instead
        denominator[denominator == 0] = np.inf
        # Compute the outer sum using sparse operations
        symmetric_pathsim = path_count_numerator.multiply(2 / denominator)

        return symmetric_pathsim.tocoo()

    def symmetric_similarity_matrix(self, meta_path: List[str]) -> torch.Tensor:
        """
        Obtain a symmetric similarity matrix for a meta-path.

        :param meta_path: a meta-path of length 3
        :type meta_path: List[str], e.g., ['node_type_1', 'node_type_2', 'node_type_3']
        :return: symmetric similarity sparse matrix of size (N x N), where N is the
                 number of all nodes
        :rtype: torch.tensor (torch.float16)
        """
        nodes_groups = [self.medrecord.nodes_in_group(group) for group in meta_path]

        path_count_matrix, nodes_indices, nodes_ranges = self.path_count(nodes_groups)
        symmetric_pathsim = self.symmetric_pathsim(
            path_count_matrix, nodes_ranges[0], nodes_ranges[-1]
        )
        rows, columns, similarity_scores = self.compute_matrix_indices(
            meta_path, symmetric_pathsim
        )
        num_nodes = len(nodes_indices)
        full_matrix = torch.sparse_coo_tensor(
            torch.stack([rows, columns]),
            similarity_scores,
            size=(num_nodes, num_nodes),
            dtype=torch.float16,
        )
        return full_matrix

    def compute_all_subgraphs(self) -> List[torch.Tensor]:
        """
        Create a series of symmetric similarity matrices for each meta-path (subgraphs).

        :return: symmetric similarity matrix
        :rtype: List[toch.tensor]
        """
        meta_paths = self.find_metapaths()
        subgraphs = []
        for meta_path in tqdm(
            meta_paths,
            desc="Computing similarity subgraphs",
            bar_format="{l_bar}{bar}| meta-path {r_bar}",
        ):
            matrix = self.symmetric_similarity_matrix(meta_path)

            subgraphs.append(matrix)
        return subgraphs

    def create_node_indices_and_ranges(
        self, nodes_groups: List[List[NodeIndex]]
    ) -> Tuple[Dict[NodeIndex, int], List[range]]:
        node_indices = {}
        current_index = 0
        group_to_range = {}
        nodes_ranges = []

        # First pass: create node indices and identify unique groups
        for group in nodes_groups:
            group_key = tuple(sorted(group))
            if group_key not in group_to_range:
                start_index = current_index
                for node in group:
                    if node not in node_indices:
                        node_indices[node] = current_index
                        current_index += 1
                group_to_range[group_key] = range(start_index, current_index)

        # Second pass: create the list of ranges
        for group in nodes_groups:
            group_key = tuple(sorted(group))
            nodes_ranges.append(group_to_range[group_key])

        return node_indices, nodes_ranges
