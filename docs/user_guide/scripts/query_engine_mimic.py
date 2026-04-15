# ruff: noqa: D100, D103, B018
import pandas as pd
import polars as pl

from medmodels import MedRecord
from medmodels.medrecord.querying import (
    EdgeIndicesOperand,
    EdgeOperand,
    NodeIndicesOperand,
    NodeOperand,
)

# Load MIMIC-IV demo subset
patients = pd.read_csv("patients.csv")
admissions = pd.read_csv("admissions.csv")

# Keep only the columns used in this guide
patients_df = patients[
    ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"]
]
admissions_df = admissions[
    ["subject_id", "hadm_id", "admittime", "dischtime", "hospital_expire_flag"]
]

# Convert to Polars
patients_pl = pl.from_pandas(patients_df)
admissions_pl = pl.from_pandas(admissions_df)

# Build MedRecord
record = (
    MedRecord.builder()
    .add_nodes((patients_pl, "subject_id"), group="Patients")
    .build()
)
record.add_nodes((admissions_pl, "hadm_id"), group="Admissions")
record.add_edges((admissions_pl, "subject_id", "hadm_id"), group="Patient→Admission")


# Basic node query
def query_patients_over_65(node: NodeOperand) -> NodeIndicesOperand:
    node.in_group("Patients")
    node.attribute("anchor_age").greater_than(65)

    return node.index()


record.query_nodes(query_patients_over_65)


# Admissions query
def query_mortality_admissions(node: NodeOperand) -> NodeIndicesOperand:
    node.in_group("Admissions")
    node.attribute("hospital_expire_flag").equal_to(1)

    return node.index()


record.query_nodes(query_mortality_admissions)


# Reusing node query
def query_patients_over_80(node: NodeOperand) -> NodeIndicesOperand:
    query_patients_over_65(node)
    node.attribute("anchor_age").greater_than(80)

    return node.index()


record.query_nodes(query_patients_over_80)


# Edge query
def query_patient_admission_edges(edge: EdgeOperand) -> EdgeIndicesOperand:
    edge.in_group("Patient→Admission")

    return edge.index()


record.query_edges(query_patient_admission_edges)


# Queries as function arguments
record.add_group("PatientsOver65", nodes=query_patients_over_65)
record.groups
