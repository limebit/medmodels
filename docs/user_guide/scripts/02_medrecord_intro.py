# ruff: noqa: D100, B018
import pandas as pd
import polars as pl

import medmodels as mm

# Patients DataFrame (Nodes)
patients = pd.DataFrame(
    [
        ["Patient 01", 72, "M", "USA"],
        ["Patient 02", 74, "M", "USA"],
        ["Patient 03", 64, "F", "GER"],
    ],
    columns=["ID", "Age", "Sex", "Loc"],
)

# Medications DataFrame (Nodes)
medications = pd.DataFrame(
    [["Med 01", "Insulin"], ["Med 02", "Warfarin"]], columns=["ID", "Name"]
)

# Patients-Medication Relation (Edges)
patient_medication = pd.DataFrame(
    [
        ["Patient 02", "Med 01", pd.Timestamp("20200607")],
        ["Patient 02", "Med 02", pd.Timestamp("20180202")],
        ["Patient 03", "Med 02", pd.Timestamp("20190302")],
    ],
    columns=["Pat_ID", "Med_ID", "Date"],
)

record = mm.MedRecord.builder().add_nodes((patients, "ID"), group="Patients").build()

record.add_nodes((medications, "ID"), group="Medications")

patient_tuples = [
    ("Patient 04", {"Age": 45, "Sex": "F", "Loc": "CHI"}),
    ("Patient 05", {"Age": 26, "Sex": "M", "Loc": "SPA"}),
]
record.add_nodes(patient_tuples, group="Patients")

patient_polars = pl.DataFrame(
    [
        ["Patient 06", 55, "F", "GER"],
        ["Patient 07", 61, "F", "USA"],
        ["Patient 08", 73, "M", "CHI"],
    ],
    schema=["ID", "Age", "Sex", "Loc"],
    orient="row",
)
record.add_nodes((patient_polars, "ID"), group="Patients")

record.add_edges((patient_medication, "Pat_ID", "Med_ID"))

record.add_group("US-Patients", nodes=["Patient 01", "Patient 02"])

record.add_nodes(
    (
        pd.DataFrame(
            [["Patient 09", 65, "M", "USA"]], columns=["ID", "Age", "Sex", "Loc"]
        ),
        "ID",
    ),
)

record.overview()

# Adding edges to a certain group
record.add_group("Patient-Medication", edges=record.edges)

# Getting all available nodes
record.nodes

# Accessing a certain node
record.node["Patient 01"]

# Accessing a certain edge
record.edge[0]

# Getting all available groups
record.groups

# Getting the nodes that are within a certain group
record.nodes_in_group("Medications")

record.to_ron("record.ron")
new_record = mm.MedRecord.from_ron("record.ron")
