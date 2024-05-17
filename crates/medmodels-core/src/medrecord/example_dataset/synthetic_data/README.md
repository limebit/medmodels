# Attribution

This dataset was created using the [Synthea™ Patient Generator](https://github.com/synthetichealth/synthea).

# Dataset Description

The dataset is a synthetic dataset created to simulate a small-scale medical record system containing comprehensive patient data. The dataset encompasses details for five hypothetical patients, capturing key medical attributes including patient demographics, diagnosis, medical procedures undertaken, and medications prescribed.

## Data Schema Nodes

### 1. `patient.csv`

The patient data consists of five hypothetical patients.

- **patient_id:** A unique identifier assigned each patient
- **gender:** Recorded as Male, Female, or Non-Binary
- **age:** Patient's age in years

### 2. `diagnosis.csv`

There are 24 unique diagnosis in the dataset.

- **diagnosis_code:** Unique code identifiying a specific diagnosis
- **description:** Description of the diagnosis

### 3. `procedure.csv`

The procedure data contains 24 different medical procedures.

- **procedure_code:** Unique identifier for each procedure
- **description:** Detailed procedural information

### 4. `drug.csv`

The drug data has 19 different prescribed medications.

- **drug_code:** Unique identifier for the medication prescribed
- **description:** Generic name of the drug and dosage

## Data Schema Edges

### 1. `patient_diagnosis.csv`

There are 60 diagnosis instances for the five patients.

- **patient_id:** Linked to the Patient Demographics
- **diagnosis_code:** Unique code identifiying a specific diagnosis
- **diagnosis_time:** Date of the diagnosis
- **duration_days** Duration in days in which the diagnosis held true

### 2. `patient_procedure.csv`

For all five patients there are 50 procedures in the dataset.

- **patient_id:** Linked to the Patient Demographics
- **procedure_code:** Unique identifier for each procedure
- **procedure_time:** Exact time and date for the procedure
- **duration_minutes:** How long the procedure needed in minutes

### 3. `patient_drug.csv`

There are also 50 prescribed medications for the five patients combined.

- **patient_id:** Linked to the Patient Demographics
- **drug_code:** Unique identifier for the medication prescribed
- **start_time:** Time and date when the medication was prescribed
- **quantity:** how many doses were prescribed
- **cost:** Cost of the medication order in Dollar
