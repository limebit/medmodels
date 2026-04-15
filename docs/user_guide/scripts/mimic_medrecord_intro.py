# ruff: noqa: D100, D103, B018

import pandas as pd
import polars as pl
import medmodels as mm

# --- Load data ---
patients = pd.read_csv("patients.csv")
admissions = pd.read_csv("admissions.csv")

# --- Prepare data (select relevant columns) ---
patients_df = patients[
    ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"]
]

admissions_df = admissions[
    ["subject_id", "hadm_id", "admittime", "dischtime", "hospital_expire_flag"]
]

# --- Convert to Polars ---
patients_pl = pl.from_pandas(patients_df)
admissions_pl = pl.from_pandas(admissions_df)

# --- Build MedRecord ---
record = (
    mm.MedRecord.builder()
    .add_nodes((patients_pl, "subject_id"), group="Patients")
    .build()
)

# --- Add additional nodes ---
record.add_nodes((admissions_pl, "hadm_id"), group="Admissions")

# --- Add edges (relationships) ---
record.add_edges(
    (admissions_pl, "subject_id", "hadm_id"),
    group="Patient→Admission"
)

# --- Add groups (cohorts) ---
older_patients = patients.loc[
    patients["anchor_age"] > 65, "subject_id"
].tolist()

mortality_admissions = admissions.loc[
    admissions["hospital_expire_flag"] == 1, "hadm_id"
].tolist()

record.add_group("PatientsOver65", nodes=older_patients)
record.add_group("MortalityAdmissions", nodes=mortality_admissions)

# --- Overview ---
record.overview()

# --- Access elements ---
record.nodes
record.node[patients.iloc[0]["subject_id"]]
record.edge[0]
record.groups
record.nodes_in_group("PatientsOver65")

# --- Save and load ---
record.to_ron("mimic_record.ron")
new_record = mm.MedRecord.from_ron("mimic_record.ron")
