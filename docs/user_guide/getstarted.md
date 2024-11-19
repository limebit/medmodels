# Getting started

## MedRecord

A [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="_blank"} is a data class that contains medical data in a network structure. It is based on _nodes_ and _edges_, which are connections between nodes. The MedRecord makes it easy to connect a dataset with different medical data tables or DataFrames into one structure with the necessary relationships.
The MedModels framework is based on the MedRecord class and all MedModels methods take a MedRecord as input.

## Nodes

Nodes are the core components of a [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="_blank"}. Each data entry, such as patient, diagnoses or procedure entries, is an indivual node in the MedRecord. Each node needs to have a unique identifier and can have different attributes associated to it. For example the patient data would have the _patient_id_ column as the unique identifier, and _gender_ and _age_ could be additional attributes for each patient.

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

:::{dropdown} Methods used in the snippet

* [`from_tuples()`](medmodels.medrecord.medrecord.MedRecord.from_tuples){target="_blank"} : Creates a MedRecord instance from lists of node and edge tuples.

:::

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

:::{dropdown} Methods used in the snippet

* [`from_pandas()`](medmodels.medrecord.medrecord.MedRecord.from_pandas){target="_blank"} : Creates a MedRecord from Pandas DataFrames of nodes and optionally edges.

:::

#### Adding Nodes

Nodes and Edges can be added to an existing MedRecord later, either as single DataFrames or a list of DataFrames.

```python
# add drug nodes to medrecord from pandas
drug.set_index("drug_code", inplace=True)
medrecord.add_nodes(nodes=drug)
```

:::{dropdown} Methods used in the snippet

* [`add_nodes()`](medmodels.medrecord.medrecord.MedRecord.add_nodes){target="_blank"} : Adds nodes to the MedRecord from different data formats and optionally assigns them to a group.

:::

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

:::{dropdown} Methods used in the snippet

* [`add_edges_polars()`](medmodels.medrecord.medrecord.MedRecord.add_edges_polars){target="_blank"} : Adds nodes to the MedRecord from different data formats and optionally assigns them to a group.

:::

### Removing entries

Nodes and edges can be easily removed by their identifier. To check if a node or edge exists, the [`contains_node()`](medmodels.medrecord.medrecord.MedRecord.contains_node){target="_blank"} or [`contains_edge()`](medmodels.medrecord.medrecord.MedRecord.contains_edge){target="_blank"} functions can be used. If a node is deleted from the MedRecord, its corresponding edges will also be removed.

```python
# returns attributes for the node that will be removed
medrecord.remove_nodes("pat_6")
medrecord.contains_node("pat_6") or medrecord.contains_edge(edge_pat6_pat2_id)
```

    False

:::{dropdown} Methods used in the snippet

* [`remove_nodes()`](medmodels.medrecord.medrecord.MedRecord.remove_nodes){target="_blank"} : Removes a node or multiple nodes from the MedRecord and returns their attributes.
* [`contains_node()`](medmodels.medrecord.medrecord.MedRecord.contains_node){target="_blank"} : Checks whether a specific node exists in the MedRecord.
* [`contains_edge()`](medmodels.medrecord.medrecord.MedRecord.contains_edge){target="_blank"} : Checks whether a specific edge exists in the MedRecord.

:::

### Size of a MedRecord

The size of a MedRecord instance is determined by the number of nodes and their connecting edges.

```python
print(
    f"The medrecord has {medrecord.node_count()} nodes and {medrecord.edge_count()} edges."
)
```

    The medrecord has 73 nodes and 160 edges.

:::{dropdown} Methods used in the snippet

* [`node_count()`](medmodels.medrecord.medrecord.MedRecord.node_count){target="_blank"} : Returns the total number of nodes currently managed by the MedRecord.
* [`edge_count()`](medmodels.medrecord.medrecord.MedRecord.edge_count){target="_blank"} : Returns the total number of edges currently managed by the MedRecord.

:::

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

:::{dropdown} Methods used in the snippet

* [`node[]`](medmodels.medrecord.medrecord.MedRecord.node){target="_blank"} : Provides access to node attributes within the MedRecord instance via an indexer.
:::

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

:::{dropdown} Methods used in the snippet

* [`node[]`](medmodels.medrecord.medrecord.MedRecord.node){target="_blank"} : Provides access to node attributes within the MedRecord instance via an indexer.
:::

### Selecting nodes and grouping

Nodes can be selected using the MedRecords query engine. The [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="_blank"} function works with logical operators on node properties, attributes or the node index and returns a list of node indices.

Nodes and edges can be organized in groups for easier access. Nodes can be added to a group by their indices.

```python
# select all indices for node
patient_ids = medrecord.select_nodes(node().index().starts_with("pat"))
medrecord.add_group(group="Patient", node=patient_ids)

print(f"Patients: {medrecord.select_nodes(node().in_group('Patient'))}")
```

    Patients: ['pat_5', 'pat_1', 'pat_4', 'pat_3', 'pat_2']

:::{dropdown} Methods used in the snippet

* [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="_blank"} : Select nodes that match that query.
* [`node()`](medmodels.medrecord.querying.NodeOperand){target="_blank"} : Returns a [`NodeOperand`](medmodels.medrecord.querying.NodeOperand){target="_blank"} to query on the nodes of the MedRecord.
* [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"} : Returns a [`NodeIndexOperand`](medmodels.medrecord.querying.NodeIndexOperand){target="_blank"} to query on the node indices of the node operand.
* [`starts_with()`](medmodels.medrecord.querying.NodeIndexOperand.starts_with){target="_blank"} : Query the node indices that start with that argument.
* [`add_group()`](medmodels.medrecord.medrecord.MedRecord.add_group){target="_blank"} : Adds a group to the MedRecord, optionally with node and edge indices.
* [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="_blank"} : Query the nodes that are in that given group.
:::

### Creating sub populations

Grouping can also be used to make sub populations that share the same properties. The nodes can be added to a group either by their indices or directly by giving a node operation to the node parameter.

```python
young_age = 20
# query and get node indices
young_id = medrecord.select_nodes(node().attribute("age").less_than(young_age))
medrecord.add_group(group="Young", node=young_id)

# node operation
medrecord.add_group(group="Woman", node=node().attribute("gender").equal_to("F"))
```

:::{dropdown} Methods used in the snippet

* [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="_blank"} : Select nodes that match that query.
* [`node()`](medmodels.medrecord.querying.NodeOperand){target="_blank"} : Returns a [`NodeOperand`](medmodels.medrecord.querying.NodeOperand){target="_blank"} to query on the nodes of the MedRecord.
* [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`MultipleValuesOperand`](medmodels.medrecord.querying.MultipleValuesOperand){target="_blank"} to query on the values of the nodes for that attribute.
* [`less_than()`](medmodels.medrecord.querying.MultipleValuesOperand.less_than){target="_blank"} : Query values that are less than that value.
* [`equal_to()`](medmodels.medrecord.querying.MultipleValuesOperand.equal_to){target="_blank"} : Query values equal to that value.
* [`add_group()`](medmodels.medrecord.medrecord.MedRecord.add_group){target="_blank"} : Adds a group to the MedRecord, optionally with node and edge indices.
:::

The nodes of a group or a list of groups can be easily accessed with [`group()`](medmodels.medrecord.medrecord.MedRecord.group){target="_blank"}. The return is either a list of node indices for a single group or a dictionary with each group name,
mapping to its list of node indices in case of multiple groups.

To get all groups in which a node or a list of nodes is categorized, the function [`groups_of_node()`](medmodels.medrecord.medrecord.MedRecord.groups_of_node){target="_blank"} can be used.

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

:::{dropdown} Methods used in the snippet

* [`group()`](medmodels.medrecord.medrecord.MedRecord.group){target="_blank"} : Returns the node and edge indices associated with the specified group/s in the MedRecord.
* [`groups_of_node()`](medmodels.medrecord.medrecord.MedRecord.groups_of_node){target="_blank"} : Retrieves the groups associated with the specified node(s) in the MedRecord.
:::

Nodes can also be added to an existing group later.

```python
higher_age = 25
additional_young_id = medrecord.select_nodes(
    node().attribute("age").greater_than_or_equal_to(young_age)
    & node().attribute("age").less_than(higher_age)
)
medrecord.add_nodes_to_group(group="Young", nodes=additional_young_id)

print(
    f"Patients in Group 'Young' if threshold age is {higher_age}: {medrecord.group('Young')}"
)
```

    Patients in Group 'Young' if threshold age is 25: ['pat_4', 'pat_2']

:::{dropdown} Methods used in the snippet

* [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="_blank"} : Select nodes that match that query.
* [`node()`](medmodels.medrecord.querying.NodeOperand){target="_blank"} : Returns a [`NodeOperand`](medmodels.medrecord.querying.NodeOperand){target="_blank"} to query on the nodes of the MedRecord.
* [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`MultipleValuesOperand`](medmodels.medrecord.querying.MultipleValuesOperand){target="_blank"} to query on the values of the nodes for that attribute.
* [`greater_than_or_equal_to()`](medmodels.medrecord.querying.MultipleValuesOperand.greater_than_or_equal_to){target="_blank"} : Query values that are greater or equal to a specific value.
* [`less_than()`](medmodels.medrecord.querying.MultipleValuesOperand.less_than){target="_blank"} : Query values that are less than that value.
* [`group()`](medmodels.medrecord.medrecord.MedRecord.group){target="_blank"} : Returns the node and edge indices associated with the specified group/s in the MedRecord.
* [`add_nodes_to_group()`](medmodels.medrecord.medrecord.MedRecord.add_nodes_to_group){target="_blank"} : Retrieves the groups associated with the specified node(s) in the MedRecord.
:::

It is possible to remove nodes from groups and to remove groups entirely from the MedRecord.

```python
medrecord.remove_nodes_from_group(group="Young", nodes=additional_young_id)
print(f"Patients in group 'Young': {medrecord.select_nodes(node().in_group('Young'))}")


print(f"The MedRecord contains {medrecord.group_count()} groups.")
medrecord.remove_groups("Woman")
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

:::{dropdown} Methods used in the snippet

* [`remove_nodes_from_group()`](medmodels.medrecord.medrecord.MedRecord.remove_nodes_from_group){target="_blank"} : Select nodes that match that query.
* [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="_blank"} : Select nodes that match that query.
* [`node()`](medmodels.medrecord.querying.NodeOperand){target="_blank"} : Returns a [`NodeOperand`](medmodels.medrecord.querying.NodeOperand){target="_blank"} to query on the nodes of the MedRecord.
* [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="_blank"} : Query the nodes that are in that given group.
* [`group_count()`](medmodels.medrecord.medrecord.MedRecord.group_count){target="_blank"} : Returns the total number of groups currently defined within the MedRecord.
* [`remove_groups()`](medmodels.medrecord.medrecord.MedRecord.remove_groups){target="_blank"} : Removes one or more groups from the MedRecord instance.
* [`contains_group()`](medmodels.medrecord.medrecord.MedRecord.contains_group){target="_blank"} : Checks whether a specific group exists in the MedRecord.
:::

## Edges

### Getting edge indices

Edges are assigned a unique index when they are added to the MedRecord. To retrieve the indices for a specific edge, the corresponding source and target node have to be specified in [`edges_connecting()`](medmodels.medrecord.medrecord.MedRecord.edges_connecting){target="_blank"}. The same concept can also be used to get a list of all edge indices that are connecting a group of source nodes to a group of target nodes.

```python
patient_diagnosis_edges = medrecord.edges_connecting(
    source_node=medrecord.group("Patient"), target_node=medrecord.group("Diagnosis")
)
```

:::{dropdown} Methods used in the snippet

* [`edges_connecting()`](medmodels.medrecord.medrecord.MedRecord.edges_connecting){target="_blank"} : Retrieves the edges connecting the specified source and target nodes in the MedRecord.
:::

All outgoing or incoming edges of a node or a list of nodes can be retrieved with the functions [`outgoing_edges()`](medmodels.medrecord.medrecord.MedRecord.outgoing_edges){target="_blank"} or [`incoming_edges()`](medmodels.medrecord.medrecord.MedRecord.incoming_edges){target="_blank"} respectively. If the edges of a list of nodes is requested, the return will be a dictionary with the nodes as keys and their edges as values in lists. Otherwise for a single node, the return will be a simple list.

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

:::{dropdown} Methods used in the snippet

* [`outgoing_edges()`](medmodels.medrecord.medrecord.MedRecord.outgoing_edges){target="_blank"} : Lists the outgoing edges of the specified node(s) in the MedRecord.
* [`incoming_edges()`](medmodels.medrecord.medrecord.MedRecord.incoming_edges){target="_blank"} : Lists the incoming edges of the specified node(s) in the MedRecord.
* [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="_blank"} : Select nodes that match that query.
* [`node()`](medmodels.medrecord.querying.NodeOperand){target="_blank"} : Returns a [`NodeOperand`](medmodels.medrecord.querying.NodeOperand){target="_blank"} to query on the nodes of the MedRecord.
* [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`MultipleValuesOperand`](medmodels.medrecord.querying.MultipleValuesOperand){target="_blank"} to query on the values of the nodes for that attribute.
* [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="_blank"} : Query node indices containing that argument.
:::

From the edge indices, the source and target nodes can be retrieved with [`edge_endpoints()`](medmodels.medrecord.medrecord.MedRecord.edge_endpoints){target="_blank"}.

```python
medrecord.edge_endpoints(diabetes_incoming_edges)
```

    {25: ('pat_3', 'diagnosis_15777000'), 8: ('pat_1', 'diagnosis_15777000')}

:::{dropdown} Methods used in the snippet

* [`edge_endpoints()`](medmodels.medrecord.medrecord.MedRecord.edge_endpoints){target="_blank"} : Retrieves the source and target nodes of the specified edge(s) in the MedRecord.
:::

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

:::{dropdown} Methods used in the snippet

* [`edge[]`](medmodels.medrecord.medrecord.MedRecord.edge){target="_blank"} : Provides access to edge attributes within the MedRecord instance via an indexer.
:::

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

:::{dropdown} Methods used in the snippet

* [`edge[]`](medmodels.medrecord.medrecord.MedRecord.edge){target="_blank"} : Provides access to edge attributes within the MedRecord instance via an indexer.
:::

### Selecting edges

Edges can also be selected using the query engine. The logic operators and functions are similar to the ones used for `select_nodes()`.

```python
medrecord.select_edges(edge().attribute("cost").greater_than(500))
```

    [114, 117, 124]

:::{dropdown} Methods used in the snippet

* [`select_edges()`](medmodels.medrecord.medrecord.MedRecord.select_edges){target="_blank"} : Select edges that match that query.
* [`edge()`](medmodels.medrecord.querying.EdgeOperand){target="_blank"} : Returns a [`EdgeOperand`](medmodels.medrecord.querying.EdgeOperand){target="_blank"} to query on the edges of the MedRecord.
* [`attribute()`](medmodels.medrecord.querying.EdgeOperand.attribute){target="_blank"} : Returns a [`MultipleValuesOperand()`](medmodels.medrecord.querying.MultipleValuesOperand){target="_blank"} to query on the values of the nodes for that attribute.
* [`greater_than()`](medmodels.medrecord.querying.MultipleValuesOperand.greater_than){target="_blank"} : Query values that are greater than that value.
:::

## Saving the MedRecord

A MedRecord instance and all its data can be saved as a RON (Rusty Object Notation) file. From there, it can also be loaded and a new MedRecord instance can be created from an existing RON file.

```python
medrecord.to_ron("medrecord.ron")
medrecord_loaded = MedRecord.from_ron("medrecord.ron")
```

:::{dropdown} Methods used in the snippet

* [`to_ron()`](medmodels.medrecord.medrecord.MedRecord.to_ron){target="_blank"} : Writes the MedRecord instance to a RON file.
* [`from_ron()`](medmodels.medrecord.medrecord.MedRecord.from_ron){target="_blank"} : Creates a MedRecord instance from a RON file.
:::

## Clearing the MedRecord

All data can be removed from the MedRecord with the [`clear()`](medmodels.medrecord.medrecord.MedRecord.clear){target="_blank"} function.

```python
medrecord.clear()
medrecord.node_count()
```

    0

:::{dropdown} Methods used in the snippet

* [`clear()`](medmodels.medrecord.medrecord.MedRecord.clear){target="_blank"} : Clears all data from the MedRecord instance.
* [`node_count()`](medmodels.medrecord.medrecord.MedRecord.node_count){target="_blank"} : Returns the total number of nodes currently managed by the MedRecord.
:::
