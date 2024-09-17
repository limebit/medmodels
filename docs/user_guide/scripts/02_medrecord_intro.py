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

record.add_edges((patient_medication, "Pat_ID", "Med_ID"))

record.add_group("US-Patients", ["Patient 01", "Patient 02"])

record.print_attribute_table_nodes()

record.print_attribute_table_edges()

# Getting all available nodes
record.nodes
# ['Patient 03', 'Med 01', 'Med 02', 'Patient 01', 'Patient 02']

# Accessing a certain node
record.node["Patient 01"]
# {'Age': 72, 'Loc': 'USA', 'Sex': 'M'}

# Getting all available groups
record.groups
# ['Medications', 'Patients', 'US-Patients']

# Getting the nodes that are within a certain group
record.nodes_in_group("Medications")
# ['Med 02', 'Med 01']

record.to_ron("record.ron")
new_record = mm.MedRecord.from_ron("record.ron")
