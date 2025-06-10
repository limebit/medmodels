# ruff: noqa: D100, D103
from typing import Tuple

import polars as pl

from medmodels import MedRecord

medrecord = MedRecord().from_advanced_example_dataset()


# Showing example dataset
def retrieve_example_dataset(
    medrecord: MedRecord,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    patients = pl.DataFrame(
        [
            {"patient_id": k, **v}
            for k, v in medrecord.node[medrecord.nodes_in_group("patient")].items()
        ]
    )
    diagnoses = pl.DataFrame(
        [
            {"diagnosis_id": k, **v}
            for k, v in medrecord.node[medrecord.nodes_in_group("diagnosis")].items()
        ]
    )
    drugs = pl.DataFrame(
        [
            {"drug_id": k, **v}
            for k, v in medrecord.node[medrecord.nodes_in_group("drug")].items()
        ]
    )

    patients_diagnoses_edges = medrecord.edge[
        medrecord.edges_in_group("patient_diagnosis")
    ]
    for edge in patients_diagnoses_edges:
        (
            patients_diagnoses_edges[edge]["source"],
            patients_diagnoses_edges[edge]["target"],
        ) = medrecord.edge_endpoints(edge)
    patients_diagnoses = pl.DataFrame(list(patients_diagnoses_edges.values()))
    patients_diagnoses = patients_diagnoses[
        ["source", "target"]
        + [col for col in patients_diagnoses.columns if col not in ["source", "target"]]
    ]

    patients_drugs_edges = medrecord.edge[medrecord.edges_in_group("patient_drug")]
    for edge in patients_drugs_edges:
        patients_drugs_edges[edge]["source"], patients_drugs_edges[edge]["target"] = (
            medrecord.edge_endpoints(edge)
        )

    patients_drugs = pl.DataFrame(list(patients_drugs_edges.values()))
    patients_drugs = patients_drugs[
        ["source", "target"]
        + [col for col in patients_drugs.columns if col not in ["source", "target"]]
    ]

    return patients, diagnoses, drugs, patients_diagnoses, patients_drugs


patients, diagnoses, drugs, patients_diagnoses_edges, patients_drugs_edges = (
    retrieve_example_dataset(medrecord)
)

patients.head(5)
diagnoses.head(5)
patients_diagnoses_edges.head(5)
drugs.head(5)
patients_drugs_edges.head(5)
