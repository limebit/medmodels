# Attribution

This dataset was created using the [Synthea™ Patient Generator](https://github.com/synthetichealth/synthea).

## Dataset Description

The dataset is a synthetic dataset created to simulate a small-scale medical record system containing comprehensive patient data. The dataset encompasses details for five hypothetical patients, capturing key medical attributes including patient demographics, diagnosis, medical procedures undertaken, and medications prescribed.

## Data Schema Nodes

### 1. `patient.csv`

The patient data consists of 5 hypothetical patients.

- **patient_id:** Unique identifier assigned to each patient
- **gender:** Recorded as Male or Female
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

There are 60 diagnosis instances for the 5 patients.

- **patient_id:** Unique identifier assigned to each patient
- **diagnosis_code:** Unique code identifiying a specific diagnosis
- **time:** Date of the diagnosis
- **duration_days** Duration in days in which the diagnosis held true

### 2. `patient_procedure.csv`

For all 5 patients there are 50 procedures in the dataset.

- **patient_id:** Unique identifier assigned to each patient
- **procedure_code:** Unique identifier for each procedure
- **time:** Exact time and date for the procedure
- **duration_minutes:** How long the procedure needed in minutes

### 3. `patient_drug.csv`

There are also 50 prescribed medications for the 5 patients combined.

- **patient_id:** Unique identifier assigned to each patient
- **drug_code:** Unique identifier for the medication prescribed
- **time:** Time and date when the medication was prescribed
- **quantity:** The number of doses prescribed
- **cost:** Cost of the medication order in dollars
