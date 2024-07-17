from typing import List, Tuple

import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from tqdm import tqdm


class SimplyWeightedSum(nn.Module):
    """
    A module that calculates a weighted sum of input tensors.

    A_meta = sum(w_k * A_k), where A_meta is the fused adjacency matrix from all k
    input adjacency matrices (similarity subgraphs), and w_k is a traiable scalar weight
    of the k-th subgraph and sum(w_k) = 1.
    """

    def __init__(self, number_of_subgraphs: int):
        """
        Initializes the SimplyWeightedSum module.

        :param number_of_subgraphs: Number of input subgraph tensors.
        :type number_of_subgraphs: int
        """
        super().__init__()
        self.weights = self._initialize_norm_weights(number_of_subgraphs)

    def _initialize_norm_weights(self, number_of_subgraphs: int) -> nn.ParameterList:
        """
        Initializes normalized weights for each subgraph for the HSGNN model.

        :param number_of_subgraphs: The number of subgraphs.
        :type number_of_subgraphs: int
        :return: The normalized weights for each subgraph.
        :rtype: nn.ParameterList
        """
        weights = [torch.randn(1) for _ in range(number_of_subgraphs)]
        normalized_weights = F.softmax(torch.cat(weights), dim=0)
        return nn.ParameterList(
            [nn.Parameter(weight.view(1, 1)) for weight in normalized_weights]
        )

    def forward(self, subgraphs: List[torch.Tensor]) -> torch.Tensor:
        """
        Calculates the weighted sum of the input tensors.
        A_meta - is a fused adjacency matrix from all subgraph matrices.
        A_meta = sum(w_k * A_k)
        sum(w_k) = 1

        :param input: List of input tensors.
        :type input: List[torch.Tensor]
        :return: fused adjacency matrix from all subgraph matrices by weighted sum of
                 input subgraph tensors.
        :rtype: torch.Tensor
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weighted_subgraphs = [
            torch.mul(subgraph.to(device).to_dense(), weight.to(device))
            for subgraph, weight in zip(subgraphs, self.weights)
        ]
        fused_matrix = torch.sum(
            torch.stack(weighted_subgraphs), dim=0, dtype=torch.float32
        )
        return fused_matrix


class HSGNNModel(torch.nn.Module):
    """
    The Heterogeneous Similarity Graph Neural Network (HSGNN) model.

    Consists of two GCN layers and a weighted sum layer.
    It is a link prediction model that predicts the existence of edges between nodes
    and its probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        number_of_subgraphs: int,
        nodes_types: torch.Tensor,
    ):
        """
        Initializes the HSGNNModel.

        :param in_channels: Number of input channels.
        :type in_channels: int
        :param hidden_channels: Number of hidden channels.
        :type hidden_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param number_of_subgraphs: Number of subgraphs.
        :type number_of_subgraphs: int
        :param nodes_types: A tensor containing the node types mapping.
        :type nodes_types: torch.Tensor
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.weighted_sum = SimplyWeightedSum(number_of_subgraphs)
        self.nodes_types = nodes_types

    def encode(
        self, node_features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Encodes the input data using two GCN layers.

        :param node_features: Node features.
        :type node_features: torch.Tensor
        :param edge_index: Edge indices.
        :type edge_index: torch.Tensor
        :return: Encoded node features.
        :rtype: torch.Tensor
        """
        node_features = self.conv1(node_features, edge_index).relu()
        return self.conv2(node_features, edge_index)

    def decode(
        self,
        node_embeddings: torch.Tensor,
        full_edge_index: torch.Tensor,
        fused_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the embeddings to calculate the edge scores.

        :param node_embeddings: Node embeddings.
        :type node_embeddings: torch.Tensor
        :param full_edge_index: Edge label indices - exsistent and non-existent edges.
        :type full_edge_index: torch.Tensor
        :param fused_matrix: The fused adjacency matrix from all subgraph matrices by
                             weighted sum of input subgraph tensors.
        :type fused_matrix: torch.Tensor
        :return: Edge scores.
        :rtype: torch.Tensor
        """
        edge_weights = fused_matrix[full_edge_index[0], full_edge_index[1]]
        edge_weights = edge_weights.unsqueeze(-1)
        edge_scores = (
            node_embeddings[full_edge_index[0]]
            * node_embeddings[full_edge_index[1]]
            * edge_weights
        ).sum(dim=-1)

        # Set edge score to a very low value if the nodes have the same type.
        # TODO: how can we avoid predictions of e.g. patient-admission links without
        #       explicitly defining them?
        same_node_type_mask = (
            self.nodes_types[full_edge_index[0]] == self.nodes_types[full_edge_index[1]]
        )
        edge_scores[same_node_type_mask] = -1e9

        return edge_scores

    def decode_all(
        self, node_embeddings: torch.Tensor, fused_matrix: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the embeddings to calculate the probabilistic adjacency matrix.

        :param node_embeddings: Node embeddings.
        :type node_embeddings: torch.Tensor
        :param fused_matrix: The fused adjacency matrix from all subgraph matrices by
                             weighted sum of input subgraph tensors.
        :type fused_matrix: torch.Tensor
        :return: Adjacency matrix and probabilities.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        adjacency_probability = node_embeddings @ node_embeddings.t() * fused_matrix
        # TODO: Instead of 0, create a hyperparameter for the threshold.
        adjacency = adjacency_probability > 0

        # Create a same node_type mask.
        same_node_type_mask = self.nodes_types.view(-1, 1) == self.nodes_types.view(
            1, -1
        )
        # Set elements to 0 where nodes have the same type.
        # TODO: how can we avoid predictions of e.g. patient-admission links without
        #       explicitly defining them?
        adjacency[same_node_type_mask] = 0
        return adjacency.nonzero(as_tuple=False).t(), adjacency_probability

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        subgraphs: List[torch.FloatTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the HSGNN model.

        :param node_features: Node features.
        :type node_features: torch.Tensor
        :param edge_index: Edge indices.
        :type edge_index: torch.Tensor
        :param subgraphs: List of subgraph adjacency matrices.
        :type subgraphs: List[torch.FloatTensor]
        :return: Tuple of embeddings and meta adjacency matrix.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        node_embeddings = self.encode(node_features, edge_index)
        fused_matrix = self.weighted_sum(subgraphs)
        return node_embeddings, fused_matrix


def train_link_predictor(
    model: HSGNNModel,
    train_data: Data,
    val_data: Data,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    criterion: torch.nn.Module,
    subgraphs: List[torch.Tensor],
    num_epochs: int = 100,
) -> HSGNNModel:
    """
    Train the link predicton model.

    :param model: The HSGNN model.
    :type model: HSGNNModel
    :param train_data: The training data.
    :type train_data: Data
    :param val_data: The validation data.
    :type val_data: Data
    :param optimizer: The optimizer.
    :type optimizer: torch.optim.Optimizer
    :param criterion: The loss criterion.
    :type criterion: torch.nn.Module
    :param subgraph_adj_matrices: The list of subgraph adjacency matrices.
    :type subgraph_adj_matrices: List[torch.Tensor]
    :param n_epochs: Number of training epochs.
    :type n_epochs: int
    :return: The trained model.
    :rtype: HSGNNModel
    """
    best_val_auc = 0
    # GradScaler for mixed precision training to prevent underflow.
    # Was designed for cuda and creates a warning when used on cpu.
    scaler = GradScaler()
    progress_bar = tqdm(total=num_epochs, desc="Training progress", leave=False)
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        with autocast():
            node_embeddings, fused_matrix = model(
                train_data.x, train_data.edge_index, subgraphs
            )

            # Sampling training negatives for every training epoch.
            negative_edge_index = negative_sampling(
                edge_index=train_data.edge_index,
                num_nodes=train_data.num_nodes,
                num_neg_samples=train_data.edge_label_index.size(1),
                method="sparse",
            )

            full_edge_index = torch.cat(
                [train_data.edge_label_index, negative_edge_index], dim=-1
            )
            edge_label = torch.cat(
                [
                    train_data.edge_label,
                    train_data.edge_label.new_zeros(negative_edge_index.size(1)),
                ],
                dim=0,
            )
            out = model.decode(node_embeddings, full_edge_index, fused_matrix).view(-1)
            loss = criterion(out, edge_label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        val_auc = eval_link_predictor(model, val_data, subgraphs)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_model.pt")

        scheduler.step(val_auc)

        if epoch % 10 == 0:
            progress_bar.set_postfix({"Train Loss": loss, "Val AUC": val_auc})
    progress_bar.close()
    model.load_state_dict(torch.load("best_model.pt"))
    return model


@torch.no_grad()
def eval_link_predictor(
    model: HSGNNModel,
    data: Data,
    subgraphs: List[torch.FloatTensor],
) -> float:
    """
    Evaluate the link predicton model.

    :param model: The HSGNN model.
    :type model: HSGNNModel
    :param data: The data for evaluation.
    :type data: Data
    :param subgraphs: The list of subgraph adjacency matrices.
    :type subgraphs: List[torch.FloatTensor]
    :return: The ROC AUC score.
    :rtype: float
    """
    model.eval()
    with autocast():
        node_embeddings, fused_matrix = model(data.x, data.edge_index, subgraphs)
        out = (
            model.decode(node_embeddings, data.edge_label_index, fused_matrix)
            .view(-1)
            .sigmoid()
        )
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy().astype(float))
