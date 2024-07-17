import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd
import torch
from node2vec import Node2Vec
from torch_geometric.data import Data
from tqdm import tqdm

from medmodels.concepts.utils_concepts import medrecord_loader
from medmodels import MedRecord
from medmodels.predictive_modelling.hsgnn.graph_preprocessing import GraphPreprocessor
from medmodels.predictive_modelling.hsgnn.loaders import load_hyperparameters, logger

# Constants
REQUIRED_HYPERPARAMETERS = {
    "WORKERS",
    "DIMENSIONS_NODE2VEC",
    "DIMENSIONS_MCE",
    "HSGNN_APPROACH",
}


class HSGNNPreprocessor:
    """
    Class for preprocessing the data for HSGNN model, including:
    - computes node embeddings using Node2Vec or MCEwTAA or loads precomputed embeddings
    - computes HSGNN subgraph similarity matrices
    - converts data to PyG Data format
    """

    def __init__(
        self,
        medrecord: MedRecord,
        data_path: Path,
        hyperparams_path: Path,
        embeddings_path: Optional[Path] = None,
        mce_path: Optional[Path] = None,
    ) -> None:
        """
        Initializes the ModelPreprocessor.

        Args:
            medrecord (MedRecord): The medrecord.
            data_path (Path): The path to the data.
            hyperparams_path (Path): The path to the hyperparameters.
            embeddings_path (Path): The path to the embeddings.
            mce_path (Path): The path to the MCEwTAA model.

        Raises:
            FileNotFoundError: If the hyperparameters file does not exist.
            KeyError: If the hyperparameters type is not found in the file.
        """
        self.medrecord = medrecord
        self.data_path = data_path
        self.embeddings_path = embeddings_path
        self.mce_path = mce_path

        self.hyperparameters = load_hyperparameters(
            hyperparams_path, "preprocessing"
        )
        self.workers = self.hyperparameters.get("WORKERS", 1)
        self.dimensions_node2vec = self.hyperparameters.get("DIMENSIONS_NODE2VEC", 128)
        self.dimensions_mce = self.hyperparameters.get("DIMENSIONS_MCE", 1024)

    def compute_embeddings(self) -> pd.DataFrame:
        """
        Computes embeddings using Node2Vec.

        :return: embeddings dataframe
        :rtype: pd.DataFrame
        """
        logger.info("Computing embeddings using Node2Vec")
        node2vec_model = Node2Vec(
            self.medrecord.G, dimensions=self.dimensions_node2vec, workers=self.workers
        )
        node2vec_model = node2vec_model.fit(window=10)
        embeddings_dataframe = pd.DataFrame(
            [
                node2vec_model.wv.get_vector(str(node))
                for node in self.medrecord.G.nodes()
            ],
            index=self.medrecord.G.nodes,
        )
        return embeddings_dataframe

    def compute_embeddings_mce(self) -> pd.DataFrame:
        """
        Computes embeddings for non-medical concepts using Node2Vec and combines them
        with embeddings for medical concepts precomputed with MCEwTAA.

        :return: embeddings dataframe
        :rtype: pd.DataFrame
        """
        logger.info("Computing embeddings using Node2Vec and MCE")
        if not self.mce_path.exists():
            message = f"Could not find MCEwTAA model at {self.mce_path}"
            logger.error(message)
            raise FileNotFoundError(message)
        try:
            mce_model = torch.load(self.mce_path, map_location=torch.device("cpu"))
            logger.info(f"MCEwTAA model loaded from '{self.mce_path}'")
        except Exception as e:
            message = f"Could not load MCEwTAA model from '{self.mce_path}': {e}"
            logger.error(message)
            raise

        # prepare medrecord to compute embeddings
        medrecord = medrecord_loader(self.medrecord)
        medical_concepts = []
        mce_ids_dict = {}
        for concept in medrecord_mce["event"].unique():
            concept_name_medrecord = self._correct_node_name(concept)
            if concept in mce_model["token2id_dict"]:
                medical_concepts.append(concept)
                mce_ids_dict[concept_name_medrecord] = mce_model["token2id_dict"][
                    concept
                ]
            # remove nodes with occurrences less than min_total_events defined in mce
            else:
                self.medrecord.remove_node(concept_name_medrecord)
        node2vec_nodes = self.medrecord.dimension(
            "patients"
        ) + self.medrecord.dimension("admissions")

        node2vec_model = Node2Vec(
            self.medrecord.G, dimensions=self.dimensions_mce, workers=self.workers
        )
        node2vec_model = node2vec_model.fit(window=10)

        final_embeddings = {}
        for node in self.medrecord.G.nodes():
            if node in node2vec_nodes:
                final_embeddings[node] = node2vec_model.wv.get_vector(str(node))
            else:
                final_embeddings[node] = (
                    mce_model["emb_params"]["context_embedding.weight"][
                        mce_ids_dict[node]
                    ]
                    .detach()
                    .numpy()
                    .reshape(
                        self.dimensions_mce,
                    )
                )

        embeddings_dataframe = pd.DataFrame(
            [final_embeddings[node] for node in self.medrecord.G.nodes()],
            index=self.medrecord.G.nodes,
        )
        return embeddings_dataframe

    def save_embeddings(self, embeddings: pd.DataFrame, path: Path) -> None:
        """
        Saves embeddings to a pickle file.

        :param embeddings: The embeddings to save.
        :type embeddings: pandas.DataFrame
        :param path: The path to save the embeddings to.
        :type path: Path
        """
        try:
            embeddings.to_pickle(path)
            logger.info(f"Embeddings saved to '{path}'")
        except Exception as e:
            message = f"Could not save embeddings to '{path}': {e}"
            logger.error(message)
            raise

    def to_PyG_Data(
        self,
        embeddings: pd.DataFrame,
        subgraphs: List[torch.Tensor],
    ) -> Tuple[Data, torch.Tensor]:
        """
        Converts data to PyG Data format for training a predictive model.

        :param embeddings: The node embeddings
        :type embeddings: pandas.DataFrame
        :param subgraphs: The list of subgraph adjacency matrices.
        :type subgraphs: List[torch.tensor]
        :return: Data in the PyG Data format and node types mapping tensor.
        :rtype: Tuple[torch_geometric.data.Data, torch.Tensor]
        """
        nodes_list = list(self.medrecord.G.nodes())
        node_index_mapping = {node: index for index, node in enumerate(nodes_list)}

        edge_index = torch.tensor(
            [
                [node_index_mapping[edge[0]], node_index_mapping[edge[1]]]
                for edge in self.medrecord.G.edges()
            ],
            dtype=torch.int64,
        ).T
        node_features = torch.tensor(embeddings.values)

        # Prepare edge attributes (weights)
        # TODO: find a better way to compute edge weights
        # (sparse to dense is not efficient)
        subgraph_arrays = [subgraph.to_dense() for subgraph in subgraphs]
        edge_weights = [
            torch.sum(
                torch.stack(
                    [
                        subgraph_array[
                            node_index_mapping[edge[0]], node_index_mapping[edge[1]]
                        ]
                        for subgraph_array in subgraph_arrays
                    ]
                )
            )
            for edge in tqdm(
                self.medrecord.G.edges(), desc="Converting data to PyG Data format"
            )
        ]
        edge_attribute = torch.stack(edge_weights)

        data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attribute)

        return data

    def node_types_mapping(self) -> torch.Tensor:
        """
        Returns a tensor with node types mapping.
        E.g., if there are 3 nodes of type A, 2 nodes of type B and 1 node of type C,
              the tensor will be [0, 0, 0, 1, 1, 2].
        :return: node types mapping tensor
        :rtype: torch.Tensor
        """
        return torch.tensor([
            len(self.medrecord.nodes_in_group(group)) * [idx]
            for group, idx in enumerate(self.medrecord.groups)
        ]).flatten()

    def prepare_data_for_model(self) -> Tuple[Data, List[torch.Tensor], torch.Tensor]:
        """
        Prepares data for training HSGNN predictive model:
        - computes node embeddings using Node2Vec or MCEwTAA
          or load precomputed embeddings
        - computes HSGNN subgraph similarity matrices
        - converts data to PyG Data format

        :return: MedRecord in PyG Data format
                 the list of subgraph adjacency matrices,
                 node types.
        :rtype: torch_geometric.data.Data, List[torch.tensor], torch.Tensor
        """
        if not self.embeddings_path:
            logger.info("Computing node embeddings")
            start_time = time.time()
            file_name = os.path.basename(self.data_path)
            dir_name = os.path.dirname(self.data_path) + "/embeddings/"
            os.makedirs(os.path.dirname(dir_name), exist_ok=True)
            if not self.mce_path:
                embeddings_path = dir_name + "emb_" + file_name
                embeddings_df = self.compute_embeddings()
            else:
                embeddings_path = dir_name + "emb_mce_" + file_name
                embeddings_df = self.compute_embeddings_mce()
            end_time = time.time()
            logger.info(f"Computed node embeddings in {end_time - start_time} seconds")
            self.save_embeddings(embeddings_df, embeddings_path)
        else:
            logger.info("Loading node embeddings")
            if not os.path.exists(embeddings_path):
                message = f"Embeddings file '{embeddings_path}' not found"
                logger.error(message)
                raise FileNotFoundError(message)
            try:
                embeddings_df = pd.read_pickle(embeddings_path)
                logger.info(f"Loaded embeddings from {embeddings_path}")
            except Exception as e:
                message = f"Failed to load embeddings from {embeddings_path}: {e}"
                logger.error(message)
                raise

        logger.info("Calculating subgraphs using HSGNNPreprocessor")
        start_time = time.time()
        hsgnn_preprocessor = GraphPreprocessor(self.medrecord)
        subgraphs = hsgnn_preprocessor.compute_all_subgraphs()
        end_time = time.time()
        logger.info(f"Calculated subgraphs in {end_time - start_time} seconds")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Converting data to PyG Data format")
        start_time = time.time()
        data = self.to_PyG_Data(embeddings_df, subgraphs)
        data = data.to(device)
        node_types_tensor = self.node_types_mapping()
        node_types_tensor = node_types_tensor.to(device)
        end_time = time.time()
        logger.info(
            f"Converted data to PyG Data format in {end_time - start_time} seconds"
        )

        return data, subgraphs, node_types_tensor
