# User Guide

```{toctree}
:maxdepth: 1
:hidden:

self
```

```{toctree}
:maxdepth: 1
:caption: Contents:
:hidden:

getstarted
```

<h2 class="no-number">Welcome to the MedModels User Guide!</h2>

MedModels is a powerful and versatile open-source Python package designed to streamline the analysis of real-world evidence data within the healthcare domain. This user guide is your one-stop shop for everything you need to know about leveraging MedModels' functionalities to unlock valuable insights from your medical data.

Whether you're a seasoned data scientist or just starting your foray into medical data analysis, MedModels empowers you with a comprehensive toolkit to tackle various challenges. Here's a glimpse into what MedModels can offer:

:::{admonition} `MedRecord`
:class: seealso, dropdown

The {py:class}`MedRecord <medmodels.medrecord.medrecord.MedRecord>` class provides a comprehensive interface for managing medical records using node and edge data structures. This class offers various methods to create, modify, and query these records. Here's a brief summary of its capabilities:

1. **Initialization and Schema Management**:

   - The class can be initialized with or without a schema.
   - Supports setting and getting schema information.

2. **Data Import and Export**:

   - Can create instances from tuples, Pandas, or Polars DataFrames.
   - Supports importing data from example datasets and RON files.
   - Can export the record to RON files.

3. **Node and Edge Operations**:

   - Add, remove, and query nodes and edges.
   - Supports operations like adding nodes from different data formats (tuples, DataFrames).
   - Provides methods to get node and edge attributes, and manage their connections.

4. **Grouping and Indexing**:

   - Allows creating and managing groups of nodes and edges.
   - Can add or remove nodes and edges to/from groups.
   - Provides methods to query nodes and edges in specific groups.

5. **Querying and Selection**:

   - Supports complex querying using node and edge operations.
   - Can select nodes and edges based on various conditions and operations.
   - Allows direct selection using indexing with operations.

6. **Connectivity and Relationships**:

   - Methods to find incoming and outgoing edges for nodes.
   - Can determine edge endpoints and find edges connecting specific nodes.
   - Supports both directed and undirected neighbor queries.

7. **Clearing and Counting**:

   - Can clear all data in the record.
   - Provides methods to count nodes, edges, and groups.

8. **Properties and Accessors**:
   - Properties to get node, edge, and group lists.
   - Provides indexers for node and edge attributes, allowing flexible querying and manipulation.

Overall, the `MedRecord` class is a robust tool for handling complex medical data with a flexible and efficient approach to manage nodes (patients, conditions, etc.) and edges (relationships, treatments, etc.) in a graph-based structure.

:::

:::{admonition} `TreatmentEffect`
:class: seealso, dropdown

The {py:class}`TreatmentEffect <medmodels.treatment_effect.treatment_effect.TreatmentEffect>` class is designed for analyzing treatment effects within medical records. The class has the following capabilities:

1. **Initialization and Configuration**:

   - Initializes with groups of treatment and outcome nodes.
   - Configurable parameters include patient groups, time attributes, washout periods, grace periods, follow-up periods, and criteria for filtering control groups.

2. **Group Identification and Filtering**:

   - Identifies patients who underwent treatment and experienced outcomes.
   - Finds control groups with similar criteria but without undergoing the treatment.
   - Applies customizable criteria filtering and time constraints between treatment and outcome.

3. **Matching and Comparison**:

   - Supports optional matching of control groups to treatment groups using specified matching methods and covariates.
   - Parameters for matching include essential covariates, one-hot covariates, matching models, number of neighbors, and hyperparameters.

4. **Outcome Analysis**:

   - Determines patients who had the outcome after the treatment, and optionally before the treatment if configured.
   - Applies washout periods to exclude patients who experienced outcomes during specified times.

5. **Control Group Handling**:

   - Identifies control patients and applies filters to ensure appropriate comparison groups.
   - Distinguishes between control patients who experienced the outcome and those who did not.

6. **Estimation and Reporting**:

   - Provides properties to create `Estimate` and `Report` objects for detailed analysis and reporting of treatment effects.

7. **Support for Temporal Analysis**:
   - Utilizes functions like `find_node_in_time_window` for temporal analysis of treatment and outcome relationships.

Overall, the `TreatmentEffect` class offers a robust framework for analyzing the impact of treatments in medical records, with flexible configurations for group identification, filtering, matching, and outcome analysis.

:::
