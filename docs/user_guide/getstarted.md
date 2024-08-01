# Getting started

## MedRecord

A _MedRecord_ is a data class that contains medical data in a network structure. It is based on _nodes_ and _edges_, which are connections between nodes. The MedRecord makes it easy to connect a dataset with different medical data tables or DataFrames into one structure with the necessary relationships.
The MedModels framework is based on the MedRecord class and all MedModels methods take a MedRecord as input.

## Nodes

Nodes are the core components of a MedRecord. Each data entry, such as patient, diagnoses or procedure entries, is an indivual node in the MedRecord. Each node needs to have a unique identifier and can have different attributes associated to it. For example the patient data would have the _patient_id_ column as the unique identifier, and _gender_ and _age_ could be additional attributes for each patient.

```python
# nodes - patient information
patient
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>gender</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>pat_1</td>
      <td>M</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pat_2</td>
      <td>F</td>
      <td>22</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pat_3</td>
      <td>F</td>
      <td>96</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pat_4</td>
      <td>M</td>
      <td>19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>pat_5</td>
      <td>M</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>

```python
diagnosis.head(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diagnosis_code</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>diagnosis_314529007</td>
      <td>Medication review due (situation)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>diagnosis_10509002</td>
      <td>Acute bronchitis (disorder)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>diagnosis_444814009</td>
      <td>Viral sinusitis (disorder)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>diagnosis_160968000</td>
      <td>Risk activity involvement (finding)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>diagnosis_48333001</td>
      <td>Burn injury (morphologic abnormality)</td>
    </tr>
  </tbody>
</table>
</div>

## Edges

Edges are what connects these individual nodes with eachother. They are always directed and based on one source and one target node. Edges can also have attributes for the specific connection of source and target node. For example a patient could be connected to all their diagnosis and the charasteristics of the diagnosis could be the attributes for that edge.

```python
# edges - patient source node and diagnosis target node
patient_diagnosis.sample(5)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>patient_id</th>
      <th>diagnosis_code</th>
      <th>diagnosis_time</th>
      <th>duration_days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>58</th>
      <td>pat_5</td>
      <td>diagnosis_73595000</td>
      <td>2023-04-07</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>pat_3</td>
      <td>diagnosis_73595000</td>
      <td>1995-03-26</td>
      <td>371.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>pat_1</td>
      <td>diagnosis_195662009</td>
      <td>2014-10-18</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>54</th>
      <td>pat_5</td>
      <td>diagnosis_314529007</td>
      <td>2019-03-15</td>
      <td>742.0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>pat_3</td>
      <td>diagnosis_399261000</td>
      <td>2001-05-20</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

## Creating a MedRecord

MedRecords can be created from Tuples, Pandas DataFrames or Polars DataFrames.

### Tuples

Lists containing tuples representing nodes with a optional list of tuples representing edges can be used to create a MedRecord. A node tuple consists of the node index and a dictionary of the node's attributes. An edge tuple consists of the source node index, the target node index, and the edge's attributes.

```python
nodes = [
    ("pat_1", {"gender": "M", "age": 42}),
    ("pat_2", {"gender": "F", "age": 22}),
    ("pat_3", {"gender": "F", "age": 96}),
    ("pat_4", {"gender": "M", "age": 19}),
    ("pat_5", {"gender": "M", "age": 37}),
    ("diagnosis_428251008", {"description": "History of appendectomy"}),
    ("diagnosis_314529007", {"description": "Medication review due (situation)"}),
    ("diagnosis_59621000", {"description": "Essential hypertension (disorder)"}),
]

edges = [
    (
        "pat_3",
        "diagnosis_428251008",
        {"diagnosis_time": "1962-10-21", "duration_days": None},
    ),
    (
        "pat_2",
        "diagnosis_314529007",
        {"diagnosis_time": "2015-01-13", "duration_days": 0},
    ),
    (
        "pat_5",
        "diagnosis_59621000",
        {"diagnosis_time": "2018-03-09", "duration_days": None},
    ),
    ("pat_3", "diagnosis_314529007", {"diagnosis_time": "1999-04-18"}),
]

medrecord = MedRecord.from_tuples(nodes, edges)
```

### Pandas DataFrames

If the MedRecord is created from a Pandas DataFrame, nodes and edges can be either a single DataFrame or a list of DataFrames. Edges are optional, but nodes need to be created to continue.

For nodes, the Pandas DataFrame should always have the column of the unique indentifiers as its index column.
Each edge DataFrame should have a multi-index with exactly two index columns. The index columns should be the same as the unique identifiers of the source nodes and the target nodes of the connection.

```python
# patient nodes
patient.set_index("patient_id", inplace=True)

# diagnosis nodes
diagnosis.set_index("diagnosis_code", inplace=True)
# patient diagnosis edges
patient_diagnosis.set_index(["patient_id", "diagnosis_code"], inplace=True)

# procedure nodes
procedure.set_index("procedure_code", inplace=True)
# patient procedure edges
patient_procedure.set_index(["patient_id", "procedure_code"], inplace=True)
```

```python
medrecord = MedRecord.from_pandas(
    nodes=[patient, diagnosis, procedure], edges=[patient_diagnosis, patient_procedure]
)
```

#### Adding Nodes

Nodes and Edges can be added to an existing MedRecord later, either as single DataFrames or a list of DataFrames.

```python
# add drug nodes to medrecord from pandas
drug.set_index("drug_code", inplace=True)
medrecord.add_nodes(nodes=drug)
```

### Polars Dataframes

When adding a Polars DataFrame to a MedRecord, the index columns must be specified with the DataFrame because there are no index columns in a Polars DataFrame.

The input format for MedRecord nodes from a Polars DataFrame is a tuple consisting of the DataFrame and the index column.
Edges need a tuple consisting of the DataFrame, the source node's index column and the target node's index column.

```python
# add edges or nodes directly from Polars data frame
patient_drug = pl.from_dataframe(patient_drug)

# specify index column if data frame is a polars data frame
patient_drug_edges = medrecord.add_edges_polars(
    edges=(patient_drug, "patient_id", "drug_code")
)
```

### Adding single entries

Single nodes can be added or removed to an existing MedRecord by their unique identifier. Attributes can also be added during that process.

Single edges between a source node and target node can be added to the MedRecord instance by specifiying the source and the target node identifier. Attributes for the connection can also be included.

```python
medrecord.add_node(node="pat_6", attributes={"age": 67, "gender": "F"})
# add connection between nodes, will return the edge identifier
edge_pat6_pat2_id = medrecord.add_edge(
    source_node="pat_6", target_node="pat_2", attributes={"relationship": "Mother"}
)
```

Nodes and edges can be easily removed by their identifier. To check if a node or edge exists, the `contain_node()` or `contain_edge()` functions can be used. If a node is deleted from the MedRecord, its corresponding edges will also be removed.

```python
# returns attributes for the node that will be removed
medrecord.remove_node("pat_6")
medrecord.contains_node("pat_6") or medrecord.contains_edge(edge_pat6_pat2_id)
```

    False

### Size of a MedRecord

The size of a MedRecord instance is determined by the number of nodes and their connecting edges.

```python
print(
    f"The medrecord has {medrecord.node_count()} nodes and {medrecord.edge_count()} edges."
)
```

    The medrecord has 73 nodes and 160 edges.

## Nodes

### Getting node attributes

The attributes belonging to nodes and edges can be retrieved via an indexer. The indexer returns a dictionary of nodes with the corresponding attributes and their respective values. The nodes can be selected through various indexing methods.

```python
first_patient = "pat_1"

# return node attributes
print(f"First patient attributes: {medrecord.node[first_patient]}")
print(f"Gender of first patient: {medrecord.node[first_patient, 'gender']}")
print(f"Age of multiple patients: {medrecord.node[['pat_2', 'pat_3', 'pat_4'], 'age']}")
```

    First patient attributes: {'gender': 'M', 'age': 42}
    Gender of first patient: M
    Age of multiple patients: {'pat_4': 19, 'pat_2': 22, 'pat_3': 96}

### Setting and updating node attributes

With the same indexing concept, attributes can also be updated or new attributes can be added.

```python
# updating node attributes
medrecord.node[first_patient, "gender"] = "F"
print(f"Gender of first patient: {medrecord.node[first_patient, 'gender']}")
# add new attributes for nodes
medrecord.node[first_patient, "death"] = True
print(f"First patient attributes: {medrecord.node[first_patient]}")
# deleting attributes
del medrecord.node[first_patient, "death"]
print(f"First patient attributes: {medrecord.node[first_patient]}")
```

    Gender of first patient: F
    First patient attributes: {'gender': 'F', 'death': True, 'age': 42}
    First patient attributes: {'gender': 'F', 'age': 42}

### Selecting nodes and grouping

Nodes can be selected using the MedRecords query engine. The `select_nodes()` function works with logical operators on node properties, attributes or the node index and returns a list of node indices.

Nodes and edges can be organized in groups for easier access. Nodes can be added to a group by their indices.

```python
# select all indeces for node
patient_ids = medrecord.select_nodes(node().index().starts_with("pat"))
medrecord.add_group(group="Patient", node=patient_ids)

print(f"Patients: {medrecord.select_nodes(node().in_group('Patient'))}")
```

    Patients: ['pat_5', 'pat_1', 'pat_4', 'pat_3', 'pat_2']

### Creating sub populations

Grouping can also be used to make sub populations that share the same properties. The nodes can be added to a group either by their indices or directly by giving a node operation to the node parameter.

```python
young_age = 20
# query and get node indices
young_id = medrecord.select_nodes(node().attribute("age") < young_age)
medrecord.add_group(group="Young", node=young_id)

# node operation
medrecord.add_group(group="Woman", node=node().attribute("gender").equal("F"))
```

The nodes of a group or a list of groups can be easily accessed with `group()`. The return is either a list of node indices for a single group or a dictionary with each group name,
mapping to its list of node indices in case of multiple groups.

To get all groups in which a node or a list of nodes is categorized, the function `groups_of_nodes()` can be used.

```python
print(
    f"Patients in Group 'Young' if threshold age is {young_age}: {medrecord.group('Young')}"
)
print(
    f"Patient {young_id[0]} is in the following groups: {medrecord.groups_of_node(young_id[0])}"
)
medrecord.group(["Young", "Woman"])
```

    Patients in Group 'Young' if threshold age is 20: ['pat_4']
    Patient pat_4 is in the following groups: ['Young', 'Patient']





    {'Young': ['pat_4'], 'Woman': ['pat_3', 'pat_1', 'pat_2']}

Nodes can also be added to an existing group later.

```python
higher_age = 25
additional_young_id = medrecord.select_nodes(
    node().attribute("age").greater_or_equal(young_age)
    & node().attribute("age").less(higher_age)
)
medrecord.add_node_to_group(group="Young", node=additional_young_id)

print(
    f"Patients in Group 'Young' if threshold age is {higher_age}: {medrecord.group('Young')}"
)
```

    Patients in Group 'Young' if threshold age is 25: ['pat_4', 'pat_2']

It is possible to remove nodes from groups and to remove groups entirely from the MedRecord.

```python
medrecord.remove_node_from_group(group="Young", node=additional_young_id)
print(f"Patients in group 'Young': {medrecord.select_nodes(node().in_group('Young'))}")


print(f"The MedRecord contains {medrecord.group_count()} groups.")
medrecord.remove_group("Woman")
print(
    f"After the removal operation, the MedRecord contains {medrecord.group_count()} groups."
)
print(
    f"Group 'Woman' is included in the MedRecord: {medrecord.contains_group('Woman')}"
)
```

    Patients in group 'Young': ['pat_4']
    The MedRecord contains 3 groups.
    After the removal operation, the MedRecord contains 2 groups.
    Group 'Woman' is included in the MedRecord: False

```python
# add different node types as groups
medrecord.add_group(group="Diagnosis", node=node().index().starts_with("diagnosis"))
```

## Edges

### Getting edge indices

Edges are assigned a unique index when they are added to the MedRecord. To retrieve the indices for a specific edge, the corresponding source and target node have to be specified in `edges_connecting()`. The same concept can also be used to get a list of all edge indices that are connecting a group of source nodes to a group of target nodes.

```python
# next PR
patient_diagnosis_edges = medrecord.edges_connecting(
    source_node=medrecord.group("Patient"), target_node=medrecord.group("Diagnosis")
)
```

All outgoing or incoming edges of a node or a list of nodes can be retrieved with the functions `outgoing_edges()` or `incoming_edges()` respectively. If the edges of a list of nodes is requested, the return will be a dictionary with the nodes as keys and their edges as values in lists. Otherwise for a single node, the return will be a simple list.

The outgoing edges of a node are only the ones where the node is defined as the source node, while incoming edges of a node are the edges, where the specific node is defined as a target node.

```python
# patient edges
print(
    f"The first patient has {len(medrecord.outgoing_edges(first_patient))} outgoing edges and {len(medrecord.incoming_edges(first_patient))} incoming edges."
)
```

    The first patient has 24 outgoing edges and 0 incoming edges.

```python
# diagnosis edges
diabetes_diagnosis = medrecord.select_nodes(
    node().attribute("description").contains("diabetes")
)
diabetes_incoming_edges = medrecord.incoming_edges(diabetes_diagnosis[0])
print(
    f"The diabetes diagnosis has {len(medrecord.outgoing_edges(diabetes_diagnosis[0]))} outgoing edges and the following incoming edges: {diabetes_incoming_edges}."
)
```

    The diabetes diagnosis has 0 outgoing edges and the following incoming edges: [8, 25].

From the edge indices, the source and target nodes can be retrieved with `edge_endpoints()`.

```python
medrecord.edge_endpoints(diabetes_incoming_edges)
```

    {25: ('pat_3', 'diagnosis_15777000'), 8: ('pat_1', 'diagnosis_15777000')}

### Getting edge attributes

Retrieving attributes for edges works with the same indexing principles as retrieving attributes for nodes.

```python
print(medrecord.edge[diabetes_incoming_edges])
print(
    f"The first drug of the first patient costs {medrecord.edge[patient_drug_edges[0], 'cost']}$."
)
```

    {25: {'diagnosis_time': '1981-01-04', 'duration_days': None}, 8: {'diagnosis_time': '2020-05-12', 'duration_days': None}}
    The first drug of the first patient costs 215.58$.

### Setting and updating attributes

New attributes for edges can be created or existing attributes can be updated with the indexing method.

```python
# updating existing attribute
medrecord.edge[patient_drug_edges[0], "cost"] = 100
# setting new attribute
medrecord.edge[patient_drug_edges[0], "price_changed"] = True
print(medrecord.edge[patient_drug_edges[0]])
```

    {'start_time': '2014-04-08T12:54:59Z', 'cost': 100, 'price_changed': True, 'quantity': 3}

### Selecting edges

Edges can also be selected using the query engine. The logic operators and functions are similar to the ones used for `select_nodes()`.

```python
medrecord.select_edges(edge().attribute("cost").greater(500))
```

    [114, 117, 124]

## Saving the MedRecord

A MedRecord instance and all its data can be saved as a RON (Rusty Object Notation) file. From there, it can also be loaded and a new MedRecord instance can be created from an existing RON file.

```python
medrecord.to_ron("medrecord.ron")

medrecord_loaded = MedRecord.from_ron("medrecord.ron")
```

## Clearing the MedRecord

All data can be removed from the MedRecord with the `clear()` function.

```python
medrecord.clear()
medrecord.node_count()
```

    0
