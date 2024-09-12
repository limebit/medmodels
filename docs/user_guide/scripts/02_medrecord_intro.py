import pandas as pd

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
    [["Med 01", "Insulin"], ["Med 02", "Wararin"]], columns=["ID", "Name"]
)

# Patients-Medication Relation (Edges)
pat_med = pd.DataFrame(
    [
        ["Patient 02", "Med 01", pd.Timestamp("20200607")],
        ["Patient 02", "Med 02", pd.Timestamp("20180202")],
        ["Patient 03", "Med 02", pd.Timestamp("20190302")],
    ],
    columns=["Pat_ID", "Med_ID", "Date"],
)

# Diagnoses DataFrame (Nodes)
diagnoses = pd.DataFrame([["Diag 01", "I.21", 10]], columns=["ID", "ICD", "ICDv"])

# Patients-Diagnoses Relation (edges)
pat_diag = pd.DataFrame(
    [
        ["Patient 01", "Diag 01", pd.Timestamp("20210901")],
        ["Patient 01", "Diag 01", pd.Timestamp("20220806")],
        ["Patient 03", "Diag 01", pd.Timestamp("20180607")],
    ],
    columns=["Pat_ID", "Diag_ID", "Date"],
)

record = mm.MedRecord.builder().add_nodes((patients, "ID"), group="Patients").build()

# TODO: Adding new nodes to group must be added as issue in next MedModels weekly
record.add_nodes((medications, "ID"))
record.add_nodes((diagnoses, "ID"))

# TODO: Delete these groupings once ther grouping is added to add_nodes
record.add_group("Medications", list(medications["ID"].unique()))
record.add_group("Diagnoses", list(diagnoses["ID"].unique()))

# TODO_ Add groups to Edges in order to visualize
record.add_edges((pat_med, "Pat_ID", "Med_ID"))
record.add_edges((pat_diag, "Diag_ID", "Diag_ID"))


record.add_group("US-Patients", ["Patient 01", "Patient 02"])

record.print_attribute_table_nodes()

record.print_attribute_table_edges()

# Getting all available nodes
record.nodes
# ['Patient 03', 'Med 01', 'Diag 01', 'Med 02', 'Patient 01', 'Patient 02']

# Accessing a certain node
record.node["Patient 01"]
# {'Age': 72, 'Loc': 'USA', 'Sex': 'M'}

# Getting all available groups
record.groups
# ['Medications', 'Diagnoses', 'Patients', 'US-Patients']

# Getting the nodes that are within a certain groups
record.nodes_in_group("Medications")
# ['Med 02', 'Med 01']

record.to_ron("record.ron")
new_record = mm.MedRecord.from_ron("record.ron")
