# Query Engine

```{toctree}
:maxdepth: 1
:caption: Contents:
query_engine_attributes_and_values.md
query_engine_group_by.md
query_engine_or_not.md
query_function_arguments.md
```

## What is the Query Engine?

The **MedRecord Query Engine** enables users to find node and edges' indices stored in the graph structure efficiently. Thanks to an intuitive interface, complex queries can be performed, allowing you to filter nodes and edges by their properties and relationships. This section introduces the basic concepts of querying MedRecords and explores advanced techniques for working with complex datasets.

## Example dataset

An example dataset for the following demonstrations was generated with the method [`from_simple_example_dataset`](medmodels.medrecord.medrecord.MedRecord.from_simple_example_dataset){target="_blank"} from the [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="_blank"} class.

```{literalinclude} scripts/show_dataset.py
---
language: python
lines: 8
---
```

This example dataset includes a set of patients, drugs, diagnoses and procedures. For this section, we will only use the patients, drugs and the edges that connect these two groups.

```{exec-literalinclude} scripts/show_dataset.py
---
language: python
setup-lines: 1-37
lines: 39
---
```

```{exec-literalinclude} scripts/show_dataset.py
---
language: python
setup-lines: 1-37
lines: 40
---
```

```{exec-literalinclude} scripts/show_dataset.py
---
language: python
setup-lines: 1-37
lines: 41
---
```

## Node Queries

The [`NodeOperand`](medmodels.medrecord.querying.NodeOperand){target="_blank"} querying class allow you to define specific criteria for selecting nodes within a [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="_blank"}. These operands enable flexible and complex queries by combining multiple conditions, such as group membership, attributes' selection and querying, attribute values, and relationships to other nodes or edges. This section introduces the basic usage of node operands to create a powerful foundation for your data queries.

The function [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} and its counterpart [`query_edges()`](medmodels.medrecord.medrecord.MedRecord.query_edges){target="_blank"} are the main ways to use these queries. They can retrieve different types of data from the MedRecord, such as the indices of some nodes that fulfill some criteria (using [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"}), or even the mean `age` of those nodes ([`mean()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.mean){target="_blank"}).

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 1-11
lines: 14-20
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"} : Returns a [`NodeIndicesOperand`](medmodels.medrecord.querying.NodeIndicesOperand){target="_blank"} representing the indices of the nodes queried.
- [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.

:::

You can get to the same result via different approaches. That makes the query engine very versatile and adaptive to your specific needs. Let's complicate it a bit more involving more than one operand.

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 1-11
lines: 24-34
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"}: Returns a [`NodeIndicesOperand`](medmodels.medrecord.querying.NodeIndicesOperand){target="_blank"} representing the indices of the nodes queried.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="_blank"} : Query node indices containing that argument.
- [`has_attribute()`](medmodels.medrecord.querying.NodeOperand.has_attribute){target="_blank"} : Query nodes that have that attribute.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`NodeMultipleValuesWithIndexOperand`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand){target="_blank"} to query on the values of the nodes for that attribute.
- [`greater_than()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.greater_than){target="_blank"} : Query values that are greater than that value.
- [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.

:::

:::{note}
The [`has_attribute()`](medmodels.medrecord.querying.NodeOperand.has_attribute){target="_blank"} method is not needed in this example, since the [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} one already checks whether the nodes have the attribute. It is placed there merely for educational purposes. This will happen in different examples in this user guide to ensure the maximum amount of methods are portrayed.
:::

### Reusing Node Queries

As you can see, the query engine can prove to be highly useful for finding nodes that fulfill different criteria, these criteria being as specific and narrowing as we like. A key feature of the query engine is that it allows for re-using previous queries in new ones. For instance, the previous query can be written as follows:

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 1-18
lines: 38-48
---
```

:::{dropdown} Methods used in the snippet

- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"}: Returns a [`NodeIndicesOperand`](medmodels.medrecord.querying.NodeIndicesOperand){target="_blank"} representing the indices of the nodes queried.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="_blank"} : Query node indices containing that argument.
- [`has_attribute()`](medmodels.medrecord.querying.NodeOperand.has_attribute){target="_blank"} : Query nodes that have that attribute.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`NodeMultipleValuesWithIndexOperand`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand){target="_blank"} to query on the values of the nodes for that attribute.
- [`greater_than()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.greater_than){target="_blank"} : Query values that are greater than that value.
- [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.

:::

### Neighbors

Another very useful method is [`neighbors()`](medmodels.medrecord.querying.NodeOperand.neighbors){target="_blank"}, which can be used to query through the nodes that are neighbors to those nodes (they have edges connecting them).

In this following example we are selecting the nodes that fulfill the following criteria:

- Are in group `patient`.
- Their node index contains the string _"pat"_
- Their attribute `age` is greater than 30, and their attribute `gender` is equal to _"M"_.
- They are connected to nodes which attribute `description` contains the word _"fentanyl"_ in either upper or lowercase.

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 1-11, 24-34
lines: 52-62
---
```

:::{dropdown} Methods used in the snippet

- [`neighbors()`](medmodels.medrecord.querying.NodeOperand.neighbors){target="_blank"} : Returns a [`NodeOperand()`](medmodels.medrecord.querying.NodeOperand){target="_blank"} to query the neighbors of those nodes.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`NodeMultipleValuesWithIndexOperand()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand){target="_blank"} to query on the values of the nodes for that attribute.
- [`lowercase()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.lowercase){target="_blank"} : Converts the values that are strings to lowercase.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="_blank"} : Query node indices containing that argument.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"}: Returns a [`NodeIndicesOperand`](medmodels.medrecord.querying.NodeIndicesOperand){target="_blank"} representing the indices of the nodes queried.
- [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.

:::

## Edge Queries

The querying class [`EdgeOperand`](medmodels.medrecord.querying.EdgeOperand){target="_blank"} provides a way to query through the edges contained in a [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="_blank"}. Edge operands show the same functionalities as Node operands, creating a very powerful tandem to query throughout your data. In this section, we will portray different ways the edge operands can be employed.

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 1-11
lines: 66-72
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="_blank"} : Query edges that belong to that group.
- [`index()`](medmodels.medrecord.querying.EdgeOperand.index){target="_blank"}: Returns a [`EdgeIndicesOperand`](medmodels.medrecord.querying.EdgeIndicesOperand){target="_blank"} representing the indices of the edges queried.
- [`query_edges()`](medmodels.medrecord.medrecord.MedRecord.query_edges){target="_blank"} : Retrieves information on the edges from the MedRecord given the query.

:::

The edge operand follows the same principles as the node operand, with some extra queries applicable only to edges like [`source_node()`](medmodels.medrecord.querying.EdgeOperand.source_node){target="_blank"} or [`target_node()`](medmodels.medrecord.querying.EdgeOperand.target_node){target="_blank"} (instead of [`neighbors()`](medmodels.medrecord.querying.NodeOperand.neighbors){target="_blank"}).

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 1-11
lines: 76-85
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="_blank"} : Query edges that belong to that group.
- [`attribute()`](medmodels.medrecord.querying.EdgeOperand.attribute){target="_blank"} : Returns a [`EdgeMultipleValuesWithoutIndexOperand()`](medmodels.medrecord.querying.EdgeMultipleValuesWithoutIndexOperand){target="_blank"} to query on the values of the edges for that attribute.
- [`less_than()`](medmodels.medrecord.querying.EdgeMultipleValuesWithoutIndexOperand.less_than){target="_blank"} : Query values that are less than that value.
- [`source_node()`](medmodels.medrecord.querying.EdgeOperand.source_node){target="_blank"} : Returns a [`NodeOperand()`](medmodels.medrecord.querying.NodeOperand) to query on the source nodes for those edges.
- [`is_max()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.is_max){target="_blank"} : Query on the values that hold the maximum value. 
- [`target_node()`](medmodels.medrecord.querying.EdgeOperand.target_node){target="_blank"} : Returns a [`NodeOperand()`](medmodels.medrecord.querying.NodeOperand){target="_blank"} to query on the target nodes for those edges.
- [`contains()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.contains){target="_blank"} : Query values containing that argument.
- [`index()`](medmodels.medrecord.querying.EdgeOperand.index){target="_blank"}: Returns a [`EdgeIndicesOperand`](medmodels.medrecord.querying.EdgeIndicesOperand){target="_blank"} representing the indices of the edges queried.
- [`query_edges()`](medmodels.medrecord.medrecord.MedRecord.query_edges){target="_blank"} :  Retrieves information on the edges from the MedRecord given the query.

:::

## Combining Node & Edge Queries

The full power of the query engine appears once you combine both operands inside the queries. In the following query, we are able to query for nodes that:

- Are in group `patient`
- Their attribute `age` is greater than 30, and their attribute `gender` is equal to _"M"_.
- They have at least an edge that is in in the `patient_drug` group, which attribute `cost` is less than 200 and its attribute `quantity` is equal to 1.

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 1-11
lines: 89-108
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="_blank"} : Query edges that belong to that group.
- [`attribute()`](medmodels.medrecord.querying.EdgeOperand.attribute){target="_blank"} : Returns a [`EdgeMultipleValuesWithIndexOperand()`](medmodels.medrecord.querying.EdgeMultipleValuesWithIndexOperand){target="_blank"} to query on the values of the edges for that attribute.
- [`less_than()`](medmodels.medrecord.querying.EdgeMultipleValuesWithIndexOperand.less_than){target="_blank"} : Query values that are less than that value.
- [`equal_to()`](medmodels.medrecord.querying.EdgeMultipleValuesWithIndexOperand.equal_to){target="_blank"} : Query values that are equal to that value.
- [`index()`](medmodels.medrecord.querying.EdgeOperand.index){target="_blank"}: Returns a [`EdgeIndicesOperand`](medmodels.medrecord.querying.EdgeIndicesOperand){target="_blank"} representing the indices of the edges queried.
- [`is_int()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.is_int){target="_blank"} : Query on the values which format is `int`.
- [`greater_than()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.greater_than){target="_blank"} : Query values that are greater than that value.
- [`edges()`](medmodels.medrecord.querying.NodeOperand.edges){target="_blank"} : Returns a [`EdgeOperand()`](medmodels.medrecord.querying.EdgeOperand){target="_blank"} to query on the edges of those nodes.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"}: Returns a [`NodeIndicesOperand`](medmodels.medrecord.querying.NodeIndicesOperand){target="_blank"} representing the indices of the nodes queried.
- [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.

:::

## Clones

Since the statements in the query engine are additive, every operation modifies the state of the query. That means that it is not possible to revert to a previous state unless the entire query is rewritten from scratch for that intermediate step. This can become inefficient and redundant, particularly when multiple branches of a query or comparisons with intermediate results are required.

To address this limitation, the [`clone()`](medmodels.medrecord.querying.NodeSingleValueWithoutIndexOperand.clone){target="_blank"} method was introduced. This method allows users to create independent copies - or **clones** - of operands or computed values at any point in the query chain. Clones are completely decoupled from the original object, meaning that modifications of the clone do not affect the original, and vice versa. This functionality applies to all types of operands.

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 1-11
lines: 155-171
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"}: Returns a [`NodeIndexOperand`](medmodels.medrecord.querying.NodeIndexOperand){target="_blank"} to query on the indices.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="_blank"} : Query node indices containing that argument.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`NodeMultipleValuesWithIndexOperand()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand){target="_blank"} to query on the values of the edges for that attribute.
- [`mean()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.mean){target="_blank"}: Returns a [`NodeSingleValueWithoutIndexOperand`](medmodels.medrecord.querying.NodeSingleValueWithoutIndexOperand){target="_blank"} containing the mean of those values.
- [`clone()`](medmodels.medrecord.querying.NodeSingleValueWithoutIndexOperand.clone){target="_blank"} : Returns a clone of the operand.
- [`subtract()`](medmodels.medrecord.querying.NodeSingleValueWithoutIndexOperand.subtract){target="_blank"} : Subtract the argument from the single value operand.
- [`greater_than()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.greater_than){target="_blank"} : Query values that are greater than that value.
- [`less_than()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.less_than){target="_blank"} : Query values that are less than that value.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"}: Returns a [`NodeIndicesOperand`](medmodels.medrecord.querying.NodeIndicesOperand){target="_blank"} representing the indices of the edges queried.
- [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.

:::


## Full example Code

The full code examples for this chapter can be found here:

```{literalinclude} scripts/query_engine.py
---
language: python
lines: 2-180
---
```