<div align="center">
  <img alt="MedModels Logo" src="https://raw.githubusercontent.com/limebit/medmodels-static/refs/heads/main/logos/logo_black.svg">
</div>

<br>

<div align="center">
  <a href="https://github.com/astral-sh/ruff">
    <img alt="Code Style" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json">
  </a>
  <img alt="Python Versions" src="https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue">
  <a href="https://github.com/limebit/medmodels/blob/main/LICENSE">
    <img alt="MedModels License" src="https://img.shields.io/github/license/limebit/medmodels.svg">
  </a>
  <a href="https://github.com/limebit/medmodels/actions/workflows/testing.yml">
    <img src="https://github.com/limebit/medmodels/actions/workflows/testing.yml/badge.svg?branch=main" alt="Tests">
  </a>
  <a href="https://pypi.org/project/medmodels/">
    <img src="https://img.shields.io/pypi/v/medmodels" alt="PyPI Version">
  </a>
</div>


<h2 align="center">
  MedModels: A Rust powered Python Framework for Modern Healthcare Research
</h2>

## Motivation

Analyzing real-world evidence, especially patient data, is a complex task demanding accuracy and reproducibility.  Currently, research teams often re-implement the same statistical methods and data processing pipelines, leading to:

* **Inefficient codebases:** Duplication of effort and potential inconsistencies.
* **Faulty implementations:** Increased risk of errors in custom code.
* **Technical debt:**  Maintaining and updating numerous, disparate codebases becomes challenging.

MedModels addresses these challenges by providing a standardized, reliable, and efficient framework for handling, processing, and analyzing electronic health records (EHR) and claims data.

**Target Audience:**

MedModels is designed for a wide range of users working with real-world data and electronic health records, including:

* (Pharmaco-)Epidemiologists
* Real-World Data Analysts
* Health Economists
* Clinicians
* Data scientists
* Software developers

## Key Features

* **Rust-based Data Class:** Enables efficient transformation of patient data into flexible and extensible network graph structures.
* **High-Performance Computing:**  Handle large datasets in memory while maintaining fast processing speeds due to the underlying Rust implementation.
* **Standardized Workflows:** Streamline common tasks in real-world evidence analysis, reducing the need for custom code.
* **Interoperability:**  Facilitates collaboration and data sharing by providing a common data structure and analysis framework.

## Key Components

* **MedRecord Data Structure:**
    * **Graph-based Representation:**  Organizes medical data using nodes (e.g., patients, medications, diagnoses) and edges (e.g., date, dosage, duration) to capture complex interactions and dependencies.
    * **Dynamic Management:**  Provides methods to add, remove, and modify nodes and edges, as well as their associated attributes, allowing for flexible data manipulation.
    * **Efficient Querying:**  Enables efficient querying and retrieval of information from the graph structure, supporting various analytical tasks.
    * **Grouping and Filtering:**  Allows grouping of nodes and edges for simplified management and targeted analysis of specific subsets of data.
    * **High-Performance Backend:**  Built on a Rust backend for optimal performance and efficient handling of large-scale medical datasets.
    * **Easy Sharing and Storage:** MedRecords can be exported as `.ron` files (Rusty Object Notation), a human-readable format that is easy to share, store, and version control.

 * **Query Engine**:
    * **Flexible Querying:**  Efficiently query nodes and edges of the MedRecord based on various criteria, including attributes, labels, and relationships.
    * **High-Performance Implementation:**  Leverages a Rust backend for fast and efficient query execution, even on large datasets.

* **Treatment Effect Analysis:**

    *  **Estimating Treatment Effects:**  Provides a range of methods for estimating treatment effects from observational data, including:
        *  **Continuous Outcomes:** Analyze treatment effects on continuous outcomes
        *  **Binary Outcomes:**  Estimate odds ratios, risk ratios, and other metrics for binary outcomes.
        *  **Time-to-Event Outcomes:**  Perform survival analysis and estimate hazard ratios for time-to-event outcomes.
        *  **Effect Size Metrics:** Calculate standardized effect size metrics like Cohen's d and Hedges' g.

    * **Matching:**
        * **(High Dimensional) Propensity Score Matching:**  Reduce confounding bias by matching treated and untreated individuals based on their propensity scores.
        * **Nearest Neighbor Matching:**  Match individuals based on similarity in their observed characteristics.

## Getting Started

**Installation:**

MedModels can be installed from PyPI using the `pip` command:

```bash
pip install medmodels
```

**Quick Start:**

Here's a quick start guide showing an example of how to use MedModels to create a `MedRecord` object, add nodes and edges, and perform basic operations.

```python
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

# Create a MedRecord object using the builder pattern
record = mm.MedRecord.builder().add_nodes((patients, "ID"), group="Patients").build()

# Add medications as nodes to the MedRecord
record.add_nodes((medications, "ID"), group="Medications")

# Add edges representing the patient-medication relationships
record.add_edges((patient_medication, "Pat_ID", "Med_ID"))

# Create a group of US patients
record.add_group("US-Patients", ["Patient 01", "Patient 02"])

# Print an overview of the nodes and edges in the MedRecord
record.overview_nodes()
record.overview_edges()

# Accessing all available nodes
print(record.nodes)
# Output: ['Patient 03', 'Med 01', 'Med 02', 'Patient 01', 'Patient 02']

# Accessing a certain node and its attributes
print(record.node["Patient 01"])
# Output: {'Age': 72, 'Loc': 'USA', 'Sex': 'M'}

# Getting all available groups
print(record.groups)
# Output: ['Medications', 'Patients', 'US-Patients']

# Getting the nodes that are within a certain group
print(record.nodes_in_group("Medications"))
# Output: ['Med 02', 'Med 01']

# Save the MedRecord to a file in RON format
record.to_ron("record.ron")

# Load the MedRecord from the RON file
new_record = mm.MedRecord.from_ron("record.ron")
```
