# ruff: noqa: D100, D103
from typing import Tuple

import pandas as pd

from medmodels import MedRecord

medrecord = MedRecord().from_simple_example_dataset()


# Showing example dataset
def retrieve_example_dataset(
    medrecord: MedRecord,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patients = pd.DataFrame(
        medrecord.node[medrecord.nodes_in_group("patient")]
    ).T.sort_index()
    drugs = pd.DataFrame(
        medrecord.node[medrecord.nodes_in_group("drug")]
    ).T.sort_index()

    patients_drugs_edges = medrecord.edge[medrecord.edges_in_group("patient_drug")]
    for edge in patients_drugs_edges:
        patients_drugs_edges[edge]["source"], patients_drugs_edges[edge]["target"] = (
            medrecord.edge_endpoints(edge)
        )

    patients_drugs = pd.DataFrame(patients_drugs_edges).T.sort_index()
    patients_drugs = patients_drugs[
        ["source", "target"]
        + [col for col in patients_drugs.columns if col not in ["source", "target"]]
    ]

    return patients, drugs, patients_drugs


patients, drugs, patients_drugs_edges = retrieve_example_dataset(medrecord)

patients.head(10)
drugs.head(10)
patients_drugs_edges.head(10)
