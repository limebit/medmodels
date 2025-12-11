# MedRecord

```{toctree}
:maxdepth: 2
:caption: Contents:
:hidden:
02a_query_engine/index
02b_schema
```

## Preface

Every major library has a central object that constitutes its core. For [PyTorch](https://pytorch.org/), it is the `torch.Tensor`, whereas for [Numpy](https://numpy.org/), it is the `np.array`. In our case, MedModels centres around the `MedRecord` as its foundational structure.

MedModels delivers advanced data analytics methods out-of-the-box by utilizing a structured approach to data storage. This is enabled by the [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="\_blank"} class, which organizes data of any complexity within a graph structure. With its Rust backend implementation, MedRecord guarantees high performance, even when working with extremely large datasets.

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 5
---
```

## Adding Nodes to a MedRecord

Let's begin by introducing some sample medical data:

:::{list-table} Patients
:widths: 15 15 15 15
:header-rows: 1

- - ID
  - Age
  - Sex
  - Loc
- - Patient 01
  - 72
  - M
  - USA
- - Patient 02
  - 74
  - M
  - USA
- - Patient 03
    - 64
    - F
    - GER
      :::

This data, stored for example in a Pandas DataFrame, looks like this:

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 8-15
---
```

In the example below, we create a new MedRecord using the builder pattern. We instantiate a [`MedRecordBuilder`](medmodels.medrecord.builder.MedRecordBuilder){target="_blank"} and instruct it to add the Pandas DataFrame as nodes, using the _'ID'\_ column for indexing. Additionally, we assign these nodes to the group 'Patients'.
The Builder Pattern simplifies creating complex objects by constructing them step by step. It improves flexibility, readability, and consistency, making it easier to manage and configure objects in a controlled way.

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 32
---
```

:::{dropdown} Methods used in the snippet

- [`builder()`](medmodels.medrecord.medrecord.MedRecord.builder){target="\_blank"} : Creates a new [`MedRecordBuilder`](medmodels.medrecord.builder.MedRecordBuilder){target="\_blank"} instance to build a [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="\_blank"}.
- [`add_nodes()`](medmodels.medrecord.builder.MedRecordBuilder.add_nodes){target="\_blank"} : Adds nodes to the MedRecord from different data formats and optionally assigns them to a group.
- [`build()`](medmodels.medrecord.builder.MedRecordBuilder.build){target="\_blank"} : Constructs a MedRecord instance from the builder’s configuration.
  :::

The MedModels MedRecord object, `record`, now contains three patients. Each patient is identified by a unique index and has specific attributes, such as age, sex, and location. These patients serve as the initial nodes in the graph structure of our MedRecord, and are represented as follows:

```{image} https://raw.githubusercontent.com/limebit/medmodels-static/main/imgs/user_guide/02/02_medrecord_intro_01.png
:class: transparent-image
```

We can now proceed by adding additional data, such as the following medications.

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 18-20
---
```

Using the builder pattern to construct the MedRecord allows us to pass as many nodes and edges as needed. If nodes are not added during the initial graph construction, they can easily be added later to an existing MedRecord by calling [`add_nodes()`](medmodels.medrecord.builder.MedRecordBuilder.add_nodes){target="\_blank"}, where you provide the DataFrame and specify the column containing the node indices.

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 34
---
```

:::{dropdown} Methods used in the snippet

- [`add_nodes()`](medmodels.medrecord.medrecord.MedRecord.add_nodes){target="\_blank"} : Adds nodes to the MedRecord from different data formats and optionally assigns them to a group.
  :::

This will expand the MedRecord, adding several new nodes to the graph. However, these nodes are not yet connected, so let's establish relationships between them!

```{image} https://raw.githubusercontent.com/limebit/medmodels-static/main/imgs/user_guide/02/02_medrecord_intro_02.png
:class: transparent-image
```

:::{note}
Nodes can be added to the MedRecord in a lot of different formats, such as a Pandas DataFrame (as previously shown), but also from a Polars DataFrame:

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 36-40
---
```

Or from a [`NodeTuple`](medmodels.medrecord.types.NodeTuple){target="\_blank"}:

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 42-51
---
```

:::

## Adding Edges to a MedRecord

To capture meaningful relationships between nodes, such as linking patients to prescribed medications, we add edges to the MedRecord. These edges must be specified in a relation table, such as the one shown below:

:::{list-table} Patient-Medication Relation
:widths: 15 15 15
:header-rows: 1

- - Pat_ID
  - Med_ID
  - time
- - Patient 02
  - Med 01
  - 2020/06/07
- - Patient 02
  - Med 02
  - 2018/02/02
- - Patient 03
    - Med 02
    - 2019/03/02
      :::

We can add these edges then to our MedRecord Graph:

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 53
---
```

:::{dropdown} Methods used in the snippet

- [`add_edges()`](medmodels.medrecord.medrecord.MedRecord.add_edges){target="\_blank"} : Adds edges to the MedRecord from different data formats and optionally assigns them to a group.

:::

This results in an enlarged Graph with more information.

```{image} https://raw.githubusercontent.com/limebit/medmodels-static/main/imgs/user_guide/02/02_medrecord_intro_03b.png
:class: transparent-image
```

## Adding Groups to a MedRecord

For certain analyses, we may want to define specific subcohorts within our MedRecord for easier access. We can do this by defining named groups withing our MedRecord.

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 55
---
```

:::{dropdown} Methods used in the snippet

- [`add_group()`](medmodels.medrecord.medrecord.MedRecord.add_group){target="\_blank"} : Adds a group to the MedRecord instance with an optional list of node and/or edge indices.

:::

This group will include all the defined nodes, allowing for easier access during complex analyses. Both nodes and edges can be added to a group, with no limitations on group size. Additionally, nodes and edges can belong to multiple groups without restriction.

```{image} https://raw.githubusercontent.com/limebit/medmodels-static/main/imgs/user_guide/02/02_medrecord_intro_04.png
:class: transparent-image
```

## Saving and Loading MedRecords

When building a MedRecord, you may want to save it to create a persistent version. This can be done by storing it as a RON (Rusty Object Notation) file. The MedRecord can then be reloaded, allowing you to create a new instance from the saved RON file.

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 90-91
---
```

:::{dropdown} Methods used in the snippet

- [`to_ron()`](medmodels.medrecord.medrecord.MedRecord.to_ron){target="\_blank"} : Writes the MedRecord instance to a RON file.
- [`from_ron()`](medmodels.medrecord.medrecord.MedRecord.from_ron){target="\_blank"} : Creates a MedRecord instance from a RON file.
  :::

## Overview Tables

The MedRecord class is designed to efficiently handle large datasets while maintaining a standardized data structure that supports complex analysis methods. As a result, the structure within the MedRecord can become intricate and difficult to manage. To address this, MedModels offers tools to help keep track of the graph-based data. One such tool is the [`overview()`](medmodels.medrecord.medrecord.MedRecord.overview){target="\_blank"} method, which prints an overview over all nodes and edges in the MedRecord.

```{exec-literalinclude} scripts/02_medrecord_intro.py
---
language: python
setup-lines: 1-64
lines: 66
---
```

:::{dropdown} Methods used in the snippet

- [`overview()`](medmodels.medrecord.medrecord.MedRecord.overview){target="\_blank"} : Gets a summary for all nodes and edges in groups and their attributes.
  :::

## Accessing Elements in a MedRecord

Now that we have stored some structured data in our MedRecord, we might want to access certain elements of it. The main way to do this is by either selecting the data with their indices or via groups that they are in.

We can, for example, get all available nodes:

```{exec-literalinclude} scripts/02_medrecord_intro.py
---
language: python
setup-lines: 1-64
lines: 72
---
```

Or access the attributes of a specific node:

```{exec-literalinclude} scripts/02_medrecord_intro.py
---
language: python
setup-lines: 1-32
lines: 75
---
```

Or a specific edge:

```{exec-literalinclude} scripts/02_medrecord_intro.py
---
language: python
setup-lines: 1-54
lines: 78
---
```

Or get all available groups:

```{exec-literalinclude} scripts/02_medrecord_intro.py
---
language: python
setup-lines: 1-71
lines: 81
---
```

Or get all that nodes belong to a certain group:

```{exec-literalinclude} scripts/02_medrecord_intro.py
---
language: python
setup-lines: 1-34
lines: 84
---
```

:::{dropdown} Methods used in the snippet

- [`nodes`](medmodels.medrecord.medrecord.MedRecord.nodes){target="\_blank"} : Lists the node indices in the MedRecord instance.
- [`node[]`](medmodels.medrecord.medrecord.MedRecord.node){target="\_blank"} : Provides access to node information within the MedRecord instance via an indexer, returning a dictionary with node indices as keys and node attributes as values.
- [`edge[]`](medmodels.medrecord.medrecord.MedRecord.edge){target="\_blank"} : Provides access to edge attributes within the MedRecord via an indexer, returning a dictionary with edge indices and edge attributes as values.
- [`groups()`](medmodels.medrecord.medrecord.MedRecord.groups){target="\_blank"} : Lists the groups in the MedRecord instance.
- [`nodes_in_group()`](medmodels.medrecord.medrecord.MedRecord.nodes_in_group){target="\_blank"} : Retrieves the node indices associated with the specified group(s) in the MedRecord.
  :::

The MedRecord can be queried in very advanced ways in order to find very specific nodes based on time, relations, neighbors or other. These advanced querying methods are covered in one of the next sections of the user guide, [Query Engine](02a_query_engine/index.md).

## Full example Code

The full code examples for this chapter can be found here:

```{literalinclude} scripts/02_medrecord_intro.py
---
language: python
lines: 2-87
---
```
