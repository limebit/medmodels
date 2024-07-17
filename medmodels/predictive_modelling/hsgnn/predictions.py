from pathlib import Path
from typing import List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import torch

from medmodels.dataclass.dataclass import MedRecord
from medmodels.predictive_modeling.hsgnn.hsgnn_model import HSGNNModel
from medmodels.predictive_modeling.hsgnn.model_preprocessing import ModelPreprocessor


# TODO: add error handling:
# - when pat_id is not in the data, we still get predictions we should not
# - when pat doesn't have any diagnoses
class HSGNNPredictor:
    """
    Class for making predictions using the HSGNN model on medical records.
    """

    def __init__(self, medrecord: MedRecord, data_path: Path, hyperparams_path: Path):
        """
        Initializes the HSGNNPredictor class.

        :param medrecord: MedRecord object.
        :type medrecord: MedRecord
        :param data_path: The path to the data.
        :type data_path: Path
        :param hyperparams_path: The path to the hyperparameters.
        :type hyperparams_path: Path
        """
        self.medrecord = medrecord
        self.data_path = data_path
        self.hyperparams_path = hyperparams_path
        self.index_node_mapping = {
            index: node for index, node in enumerate(list(medrecord.G.nodes()))
        }

    def find_node_type_by_index(self, index: int) -> str:
        """
        Finds the node type based on the given index.

        :param index: The index of the node.
        :type index: int
        :return: The type of the node.
        :rtype: str
        """
        assert index in range(len(self.medrecord.G.nodes())), "Index out of range."
        for dim in self.medrecord.dimensions:
            if self.index_node_mapping[index] in self.medrecord.dimension(dim):
                return dim

    def get_real_diagnoses(self, patient_id: int) -> List[str]:
        """
        Retrieves the diagnoses for the given patient.

        :param patient_id: Patient id.
        :type id: int
        :return: The list of diagnoses.
        :rtype: List[str]
        """
        assert patient_id in self.medrecord.dimension("patients"), "Patient not found."
        edges = self.medrecord.G.edges(self.index_node_mapping[patient_id])
        diagnoses = set()
        for edge in edges:
            if self.medrecord.get_dimension_name(edge[1]) in ["diagnoses"]:
                diagnoses.add(edge[1])
        return sorted(list(diagnoses))

    def load_model(
        self, model_path: str, prob_threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads the HSGNN model and makes predictions.

        :param model_path: The path to the trained model.
        :type model_path: str
        :param prob_threshold: The probability threshold for edge prediction.
        :type prob_threshold: float
        :return: The predicted edges and corresponding probabilities.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        model_preprocessor = ModelPreprocessor(
            medrecord=self.medrecord,
            data_path=self.data_path,
            hyperparams_path=self.hyperparams_path,
            embeddings_path=None,
            mce_path=None,
        )
        data, subgraphs, node_types = model_preprocessor.prepare_data_for_model()
        model = HSGNNModel(data.num_features, 128, 64, len(subgraphs), node_types)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(model_path, map_location=device))
        with torch.no_grad():
            node_embeddings, fused_matrix = model(data.x, data.edge_index, subgraphs)
        _, adjacency_probabilities = model.decode_all(node_embeddings, fused_matrix)
        filtered_probabilities = adjacency_probabilities > prob_threshold
        predicted_edges = filtered_probabilities.nonzero(as_tuple=False).t()
        probabilities = adjacency_probabilities[predicted_edges[0], predicted_edges[1]]
        return predicted_edges, probabilities

    def get_predictions(
        self,
        predicted_edges: torch.Tensor,
        probabilities: torch.Tensor,
        patient_id: int,
    ) -> List[Tuple[str, float]]:
        """
        Retrieves the predicted diagnoses and probabilities for a patient.

        :param predicted_edges: The predicted edges.
        :type predicted_edges: torch.Tensor
        :param probabilities: The probabilities corresponding to the predicted edges.
        :type probabilities: torch.Tensor
        :param patient_id: The ID of the patient.
        :type patient_id: int
        :return: The sorted predictions (diagnoses and probabilities) for the patient
                 in descending order.
        :rtype: List[Tuple[str, float]]
        """
        sorted_edges = (predicted_edges[0] == patient_id).nonzero()
        result = predicted_edges[1, sorted_edges]
        result = result.flatten().numpy()

        predicted_diagnoses = []
        predicted_probabilities = []

        for index in result:
            if self.find_node_type_by_index(index) in ["diagnoses"]:
                node_name = self.index_node_mapping[index]
                probability = probabilities[index].item() * 100
                predicted_diagnoses.append(node_name)
                predicted_probabilities.append(probability)

        # Sort the predicted diagnoses based on probabilities in descending order
        sorted_predictions = sorted(
            zip(predicted_diagnoses, predicted_probabilities),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_predictions

    def visualize(
        self,
        diagnoses: List[str],
        predictions: List[Tuple[str, float]],
        patient_id: str,
    ) -> None:
        """
        Visualizes the diagnoses and predictions using a graph.

        :param diagnoses: List of real diagnoses.
        :type diagnoses: List[str]
        :param predictions: List of predicted diagnoses with probabilities.
        :type predictions: List[Tuple[str, float]]
        :param patient_id: ID of the patient.
        :type patient_id: str
        :return: None
        """
        G = nx.Graph()
        G.add_node(patient_id, type="patient", size=300)
        for diagnosis in diagnoses:
            G.add_node(diagnosis, type="diagnosis", size=300)
            G.add_edge(patient_id, diagnosis)

        for prediction in predictions:
            if prediction[0] in diagnoses:
                G.add_node(
                    prediction[0],
                    type="shared",
                    size=(300 * prediction[1]),
                    prob=round(prediction[1], 2),
                )
            else:
                G.add_node(
                    prediction[0],
                    type="prediction",
                    size=(300 * prediction[1]),
                    prob=round(prediction[1], 2),
                )
            G.add_edge(patient_id, prediction[0])

        sns.set_color_codes("muted")
        color_mapping = {
            "prediction": "r",
            "patient": "b",
            "diagnosis": "g",
            "shared": "y",
        }

        for node in G.nodes:
            node_type = G.nodes[node]["type"]
            G.nodes[node]["color"] = color_mapping[node_type]

        node_colors = [G.nodes[node]["color"] for node in G.nodes]
        node_sizes = [G.nodes[node]["size"] for node in G.nodes]
        pos = nx.spring_layout(G)
        node_labels = {}
        for node in G.nodes:
            if G.nodes[node]["type"] != "patient":
                node_labels[node] = node
        plt.figure(figsize=(10, 8))
        nx.draw(
            G, pos, node_size=node_sizes, node_color=node_colors, labels=node_labels
        )

        legend_handles = []
        for node_type, color in color_mapping.items():
            legend_handles.append(
                mpatches.Patch(color=color, label=node_type.capitalize())
            )

        plt.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.95, 0.9),
            ncol=1,
        )

        diag_text = "REAL DIAGNOSES:\n" + "\n".join(
            [diagnosis for diagnosis in diagnoses]
        )
        plt.text(
            1.1,
            0.9,
            diag_text,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
            transform=plt.gca().transAxes,
        )
        pred_text = "PREDICTED DIAGNOSES:\n" + "\n".join(
            [f"{diagnosis} : {prob:.2f}%" for diagnosis, prob in predictions]
        )
        plt.text(
            1.1,
            0.6,
            pred_text,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round"),
            transform=plt.gca().transAxes,
        )

        plt.show()
