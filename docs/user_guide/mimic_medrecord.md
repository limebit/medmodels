# MedRecord

## Preface

This guide shows how to transform the **MIMIC-IV Clinical Database Demo** into a MedRecord.

We focus on three steps:

1. Download the dataset  
2. Load it into pandas / polars  
3. Build a MedRecord  

This reflects a realistic workflow for healthcare data.

---

## Downloading the dataset

Download the MIMIC-IV Demo from:

https://physionet.org/content/mimic-iv-demo/2.2/

Steps:

1. Download the dataset archive  
2. Extract the archive  
3. Navigate to the `hosp/` folder  
4. Locate:

- `patients.csv.gz`  
- `admissions.csv.gz`  

These files are compressed.

### Option A — Extract files

Double-click to extract:

- `patients.csv`  
- `admissions.csv`  

### Option B — Load directly (recommended)

```python
import pandas as pd

patients = pd.read_csv("patients.csv.gz", compression="gzip")
admissions = pd.read_csv("admissions.csv.gz", compression="gzip")
```

---

## Loading and transforming data

In this guide:

- pandas → loading / selection  
- polars → MedModels input  

```python
import pandas as pd
import polars as pl

patients = pd.read_csv("patients.csv")
admissions = pd.read_csv("admissions.csv")

patients_pl = pl.from_pandas(patients)
admissions_pl = pl.from_pandas(admissions)
```

---

## Creating the MedRecord

We build the MedRecord using:

- Patients → nodes  
- Admissions → nodes  
- subject_id → relationships  

```python
import medmodels as mm

record = (
    mm.MedRecord.builder()
    .add_nodes((patients_pl, "subject_id"), group="Patients")
    .build()
)

record.add_nodes((admissions_pl, "hadm_id"), group="Admissions")

record.add_edges((admissions_pl, "subject_id", "hadm_id"), group="Patient→Admission")
```

---

## Groups (important concept)

Groups define reusable subsets of data.

In this example:

- `Patients` → all patient nodes  
- `Admissions` → all admission nodes  

We can also define new groups:

```python
older_patients = patients.loc[patients["anchor_age"] > 65, "subject_id"].tolist()

record.add_group("PatientsOver65", nodes=older_patients)
```

Groups are used later for:

- cohort definition  
- querying  
- treatment effect analysis  

---

## Summary

We performed:

- dataset download  
- loading with pandas  
- conversion to polars  
- MedRecord creation  

This transforms tabular healthcare data into a structured graph ready for analysis.
