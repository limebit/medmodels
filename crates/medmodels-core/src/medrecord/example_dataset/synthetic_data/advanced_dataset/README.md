# Attribution

This dataset was created using the [Synthea™ Patient Generator](https://github.com/synthetichealth/synthea).

## Dataset Description

The dataset is a synthetic dataset created to simulate a medium-scale medical record system containing comprehensive patient data. The dataset encompasses details for six-hundred hypothetical patients, capturing key medical attributes including patient demographics, diagnosis, medical procedures undertaken, medications prescribed and whether any of them have died and when.

## Data Schema Nodes

### 1. `patient.csv`

The patient data consists of 600 hypothetical patients.

- **patient_id:** Unique identifier assigned to each patient
- **gender:** Recorded as Male or Female
- **age:** Patient's age in years

### 2. `diagnosis.csv`

There are 206 unique diagnosis in the dataset.

- **diagnosis_code:** Unique code identifiying a specific diagnosis
- **description:** Description of the diagnosis

### 3. `procedure.csv`

The procedure data contains 96 different medical procedures.

- **procedure_code:** Unique identifier for each procedure
- **description:** Detailed procedural information

### 4. `drug.csv`

The drug data has 185 different prescribed medications.

- **drug_code:** Unique identifier for the medication prescribed
- **description:** Generic name of the drug and dosage

### 5. `event.csv`

The event data has 1 possible event.

- **event:** Event name (e.g., death)

## Data Schema Edges

### 1. `patient_diagnosis.csv`

There are 5741 diagnosis instances for the 600 patients.

- **patient_id:** Unique identifier assigned to each patient
- **diagnosis_code:** Unique code identifiying a specific diagnosis
- **time:** Date of the diagnosis
- **duration_days** Duration in days in which the diagnosis held true

### 2. `patient_procedure.csv`

For all 600 patients there are 677 procedures in the dataset.

- **patient_id:** Unique identifier assigned to each patient
- **procedure_code:** Unique identifier for each procedure
- **time:** Exact time and date for the procedure
- **duration_minutes:** How long the procedure needed in minutes

### 3. `patient_drug.csv`

There are also 10373 prescribed medications for the 600 patients combined.

- **patient_id:** Unique identifier assigned to each patient
- **drug_code:** Unique identifier for the medication prescribed
- **time:** Time and date when the medication was prescribed
- **quantity:** The number of doses prescribed
- **cost:** Cost of the medication order in dollars

### 4. `patient_event.csv`

There are also 92 events for the 600 patients combined.

- **patient_id:** Unique identifier assigned to each patient
- **event:** Name of the event linked to the patient
- **time:** Date when the event occurred
