import pandas as pd
import torch

from medmodels.dataclass.dataclass import MedRecord
from medmodels.dataclass.utils import df_to_edges, df_to_nodes


class MockMedRecord:
    def __init__(self):
        self.medrecord = MedRecord()

    def create_mock_medrecord(self):
        patients = pd.DataFrame(
            {
                "patients_id": ["P-1", "P-2", "P-3"],
                "age": [30, 40, 20],
                "sex": ["male", "female", "female"],
            }
        )
        patient_nodes = df_to_nodes(patients, "patients_id", ["age", "sex"])

        diagnoses = pd.DataFrame(
            {
                "diagnoses_id": ["1", "D-2", "D-2"],
                "code": [1, 2, 2],
                "patients_id": ["P-1", "P-1", "P-2"],
            }
        )
        diagnoses["relation_type"] = "patients_diagnoses"
        diagnoses_nodes = df_to_nodes(diagnoses, "diagnoses_id", ["code"])
        diagnoses_patient_edges = df_to_edges(
            diagnoses, "diagnoses_id", "patients_id", ["relation_type"]
        )

        prescriptions = pd.DataFrame(
            {
                "prescriptions_id": ["prescriptions_1", "M-2", "M-2"],
                "code": [1, 2, 2],
                "patients_id": ["P-1", "P-1", "P-2"],
            }
        )
        prescriptions["relation_type"] = "patients_prescriptions"
        prescriptions_nodes = df_to_nodes(prescriptions, "prescriptions_id", ["code"])
        prescriptions_patient_edges = df_to_edges(
            prescriptions, "prescriptions_id", "patients_id", ["relation_type"]
        )

        self.medrecord.add_nodes(patient_nodes, "patients")
        self.medrecord.add_nodes(diagnoses_nodes, "diagnoses")
        self.medrecord.add_nodes(prescriptions_nodes, "prescriptions")
        self.medrecord.add_edges(diagnoses_patient_edges)
        self.medrecord.add_edges(prescriptions_patient_edges)
        return self.medrecord


class MockSubgraphs:
    def __init__(self):
        self.subgraphs = []

    def create_mock_subgraphs(self):
        subgraph_patient_symmetric = torch.sparse_coo_tensor(
            torch.stack([torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1])]),
            torch.tensor([1, 2 / 3, 2 / 3, 1]),
            size=(7, 7),
            dtype=torch.float16,
        )
        subgraph_diagnoses_symmetric = torch.sparse_coo_tensor(
            torch.stack([torch.tensor([3, 3, 4, 4]), torch.tensor([3, 4, 3, 4])]),
            torch.tensor([1, 2 / 3, 2 / 3, 1]),
            size=(7, 7),
            dtype=torch.float16,
        )

        subgraph_diagnoses_prescriptions = torch.sparse_coo_tensor(
            torch.stack(
                [
                    torch.tensor([3, 3, 4, 4, 5, 6, 5, 6]),
                    torch.tensor([5, 6, 5, 6, 3, 3, 4, 4]),
                ]
            ),
            torch.tensor([1, 2 / 3, 2 / 3, 1, 1, 2 / 3, 2 / 3, 1]),
            size=(7, 7),
            dtype=torch.float16,
        )

        subgraph_precriptions_symmetric = torch.sparse_coo_tensor(
            torch.stack([torch.tensor([5, 5, 6, 6]), torch.tensor([5, 6, 5, 6])]),
            torch.tensor([1, 2 / 3, 2 / 3, 1]),
            size=(7, 7),
            dtype=torch.float16,
        )
        self.subgraphs = [
            subgraph_patient_symmetric,
            subgraph_patient_symmetric,
            subgraph_diagnoses_symmetric,
            subgraph_diagnoses_prescriptions,
            subgraph_precriptions_symmetric,
        ]
        return self.subgraphs
