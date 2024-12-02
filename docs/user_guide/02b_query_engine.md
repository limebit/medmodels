# Query Engine

## What is the Query Engine?

The **MedRecord Query Engine** enables users to find node and edges' indices stored in the graph structure efficiently. Thanks to an intuitive interface, complex queries can be performed, allowing you to filter nodes and edges by their properties and relationships. This section introduces the basic concepts of querying MedRecords and explores advanced techniques for working with complex datasets.

## Example dataset

An example dataset for the following demonstrations was generated with the method [`from_example_dataset`](medmodels.medrecord.medrecord.MedRecord.from_example_dataset){target="\_blank"} from the [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="\_blank"} class.

```{literalinclude} scripts/02b_show_dataset.py
---
language: python
lines: 8
---
```

This example dataset includes a set of patients, drugs, diagnoses and procedures. For this section, we will only use the patients, drugs and the edges that connect these two groups.

```{exec-literalinclude} scripts/02b_show_dataset.py
---
language: python
setup-lines: 1-37
lines: 39
---
```

```{exec-literalinclude} scripts/02b_show_dataset.py
---
language: python
setup-lines: 1-37
lines: 40
---
```

```{exec-literalinclude} scripts/02b_show_dataset.py
---
language: python
setup-lines: 1-37
lines: 41
---
```

## Node Queries

The [`NodeOperand`](medmodels.medrecord.querying.NodeOperand){target="\_blank"} querying class allow you to define specific criteria for selecting nodes within a [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="\_blank"}. These operands enable flexible and complex queries by combining multiple conditions, such as group membership, attributes' selection and querying, attribute values, and relationships to other nodes or edges. This section introduces the basic usage of node operands to create a powerful foundation for your data queries.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-5
lines: 8-13
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="\_blank"} : Query nodes that belong to that group.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="\_blank"} : Select nodes that match that query.

:::

You can get to the same result via different approaches. That makes the query engine very versatile and adaptive to your specific needs. Let's complicate it a bit more involving more than one operand.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-5
lines: 16-25
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="\_blank"} : Query nodes that belong to that group.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="\_blank"}: Returns a [`NodeIndexOperand`](medmodels.medrecord.querying.NodeIndexOperand){target="\_blank"}` to query on the indices.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="\_blank"} : Query node indices containing that argument.
- [`has_attribute()`](medmodels.medrecord.querying.NodeOperand.has_attribute){target="\_blank"} : Query nodes that have that attribute.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="\_blank"} : Returns a [`MultipleValuesOperand`](medmodels.medrecord.querying.MultipleValuesOperand){target="\_blank"} to query on the values of the nodes for that attribute.
- [`greater_than()`](medmodels.medrecord.querying.MultipleValuesOperand.greater_than){target="\_blank"}` : Query values that are greater than that value.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="\_blank"} : Select nodes that match that query.

:::

:::{note}
The [`has_attribute()`](medmodels.medrecord.querying.NodeOperand.has_attribute){target="\_blank"} method is not needed in this example, since the [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="\_blank"} one already checks whether the nodes have the attribute. It is placed there merely for educational purposes. This will happen in different examples in this user guide to ensure the maximum amount of methods are portrayed.
:::

### Reusing Node Queries

As you can see, the query engine can prove to be highly useful for finding nodes that fulfill different criteria, these criteria being as specific and narrowing as we like. A key feature of the query engine is that it allows for re-using previous queries in new ones. For instance, the previous query can be written as follows:

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-13
lines: 28-37
---
```

:::{dropdown} Methods used in the snippet

- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="\_blank"}: Returns a [`NodeIndexOperand`](medmodels.medrecord.querying.NodeIndexOperand){target="\_blank"}` to query on the indices.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="\_blank"} : Query node indices containing that argument.
- [`has_attribute()`](medmodels.medrecord.querying.NodeOperand.has_attribute){target="\_blank"} : Query nodes that have that attribute.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="\_blank"} : Returns a [`MultipleValuesOperand`](medmodels.medrecord.querying.MultipleValuesOperand){target="\_blank"} to query on the values of the nodes for that attribute.
- [`greater_than()`](medmodels.medrecord.querying.MultipleValuesOperand.greater_than){target="\_blank"}` : Query values that are greater than that value.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="\_blank"} : Select nodes that match that query.

:::

### Advanced Query Operations

In case, for instance, that you do not know whether there are different ways to assign the `gender` attribute across the [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="\_blank"} (with leading/trailing whitespaces or formatted in lower/uppercase), you can modify the value of the attributes of a node/edge inside the query.

:::{note}

It is important to note that modifying these values **does not** change the actual value of the attributes within the [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="\_blank"}: it just changes the value of those variables in the query.

:::

You can also perform mathematical calculations like [`mean()`](medmodels.medrecord.querying.MultipleValuesOperand.mean){target="_blank"}, [`median()`](medmodels.medrecord.querying.MultipleValuesOperand.median){target="\_blank"} or [`min()`](medmodels.medrecord.querying.MultipleValuesOperand.min){target="\_blank"} and assign them to a variable. Also, you can keep manipulating the operand, like in the following example, where we are subtracting \_5_ years from the `mean_age` to query on that value.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-5
lines: 40-56
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="\_blank"} : Query nodes that belong to that group.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="\_blank"}: Returns a [`NodeIndexOperand`](medmodels.medrecord.querying.NodeIndexOperand){target="\_blank"}` to query on the indices.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="\_blank"} : Query node indices containing that argument.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="\_blank"} : Returns a [`MultipleValuesOperand`](medmodels.medrecord.querying.MultipleValuesOperand){target="\_blank"} to query on the values of the nodes for that attribute.
- [`lowercase()`](medmodels.medrecord.querying.MultipleValuesOperand.lowercase){target="\_blank"} : Converts the values that are strings to lowercase.
- [`trim()`](medmodels.medrecord.querying.MultipleValuesOperand.trim){target="\_blank"} : Removes leading and trailing whitespacing from the values.
- [`equal_to()`](medmodels.medrecord.querying.MultipleValuesOperand.equal_to){target="\_blank"} : Query values equal to that value.

- [`has_attribute()`](medmodels.medrecord.querying.NodeOperand.has_attribute){target="\_blank"} : Query nodes that have that attribute.
- [`mean()`](medmodels.medrecord.querying.MultipleValuesOperand.mean){target="\_blank"}: Returns a [`SingleValueOperand`](medmodels.medrecord.querying.SingleValueOperand){target="\_blank"} containing the mean of those values.
- [`subtract()`](medmodels.medrecord.querying.SingleValueOperand.subtract){target="\_blank"} : Subtract the argument from the single value operand.

- [`less_than()`](medmodels.medrecord.querying.MultipleValuesOperand.less_than){target="\_blank"} : Query values that are less than that value.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="\_blank"} : Select nodes that match that query.

:::

:::{note}
Query methods used for changing the operands cannot be concatenated or assigned to variables, since their Return is None. That is, the following code snippet will set `gender_lowercase` as None, and as a result, an AttributeError will be thrown:

```python
# Wrong implementation
gender_lowercase = node.attribute("gender").lowercase()
gender_lowercase.equal_to("m")

AttributeError("'NoneType' object has no attribute 'equal_to'")

# Wrong implementation
gender = node.attribute("gender")
gender.lowercase().trim()
gender.equal_to("m")

AttributeError("'NoneType' object has no attribute 'trim'")

# Correct implementation
gender = node.attribute("gender")
gender.lowercase()
gender.trim()
gender.equal_to("m")
```

Nor do the ones that compare operands to other operands, since their Return value is also None.

```python
# Wrong implementation
gender = node.attribute("gender")
gender.equal_to("M").not_equal_to("F")

AttributeError("'NoneType' object has no attribute 'not_equal_to'")

# Correct implementation
gender = node.attribute("gender")
gender.equal_to("M")
gender.not_equal_to("F")
```

:::

Another very useful method is [`neighbors()`](medmodels.medrecord.querying.NodeOperand.neighbors){target="\_blank"}, which can be used to query through the nodes that are neighbors to those nodes (they have edges connecting them).

In this following example we are selecting the nodes that fulfill the following criteria:

- Are in group `patient`.
- Their node index contains the string _"pat"_
- Their attribute `age` is greater than 30, and their attribute `gender` is equal to _"M"_.
- They are connected to nodes which attribute `description` contains the word _"fentanyl"_ in either upper or lowercase.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-5, 16-25
lines: 59-68
---
```

:::{dropdown} Methods used in the snippet

- [`neighbors()`](medmodels.medrecord.querying.NodeOperand.neighbors){target="\_blank"} : Returns a [`NodeOperand()`](medmodels.medrecord.querying.NodeOperand){target="\_blank"} to query the neighbors of those nodes.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="\_blank"} : Returns a [`MultipleValuesOperand()`](medmodels.medrecord.querying.MultipleValuesOperand){target="\_blank"} to query on the values of the nodes for that attribute.
- [`lowercase()`](medmodels.medrecord.querying.MultipleValuesOperand.lowercase){target="\_blank"} : Converts the values that are strings to lowercase.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="\_blank"} : Query node indices containing that argument.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="\_blank"} : Select nodes that match that query.

:::

## Edge Queries

The querying class [`EdgeOperand`](medmodels.medrecord.querying.EdgeOperand){target="\_blank"} provides a way to query through the edgs contained in a [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="\_blank"}. Edge operands show the same functionalities as Node operands, creating a very powerful tandem to query throughout your data. In this section, we will portray different ways the edge operands can be employed.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-5
lines: 71-77
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="\_blank"} : Query nodes that belong to that group.
- [`select_edges()`](medmodels.medrecord.medrecord.MedRecord.select_edges){target="\_blank"} : Select edges that match that query.

:::

The edge operand follows the same principles as the node operand, with some extra queries applicable only to edges like [`source_node()`](medmodels.medrecord.querying.EdgeOperand.source_node){target="\_blank"} or [`target_node()`](medmodels.medrecord.querying.EdgeOperand.target_node){target="\_blank"} (instead of [`neighbors()`](medmodels.medrecord.querying.NodeOperand.neighbors){target="\_blank"}).

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-5
lines: 80-89
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="\_blank"} : Query nodes that belong to that group.
- [`attribute()`](medmodels.medrecord.querying.EdgeOperand.attribute){target="\_blank"} : Returns a [`MultipleValuesOperand()`](medmodels.medrecord.querying.MultipleValuesOperand){target="\_blank"} to query on the values of the nodes for that attribute.
- [`less_than()`](medmodels.medrecord.querying.MultipleValuesOperand.less_than){target="\_blank"} : Query values that are less than that value.
- [`source_node()`](medmodels.medrecord.querying.EdgeOperand.source_node){target="\_blank"} : Returns a [`NodeOperand()`](medmodels.medrecord.querying.NodeOperand) to query on the source nodes for those edges.
- [`is_max()`](medmodels.medrecord.querying.MultipleValuesOperand.is_max){target="\_blank"} : Query on the values that hold on the maximum value among all of the.
- [`target_node()`](medmodels.medrecord.querying.EdgeOperand.target_node){target="\_blank"} : Returns a [`NodeOperand()`](medmodels.medrecord.querying.NodeOperand){target="\_blank"} to query on the target nodes for those edges.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="\_blank"} : Query node indices containing that argument.
- [`select_edges()`](medmodels.medrecord.medrecord.MedRecord.select_edges){target="\_blank"} : Select edges that match that query.

:::

## Combining Node & Edge Queries

The full power of the query engine appears once you combine both operands inside the queries. In the following query, we are able to query for nodes that:

- Are in group `patient`
- Their attribute `age` is greater than 30, and their attribute `gender` is equal to _"M"_.
- They have at least an edge that is in in the `patient_drug` group, which attribute `cost` is less than 200 and its attribute `quantity` is equal to 1.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-5
lines: 92-108
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="\_blank"} : Query nodes that belong to that group.
- [`attribute()`](medmodels.medrecord.querying.EdgeOperand.attribute){target="\_blank"} : Returns a [`MultipleValuesOperand()`](medmodels.medrecord.querying.MultipleValuesOperand){target="\_blank"} to query on the values of the nodes for that attribute.
- [`less_than()`](medmodels.medrecord.querying.MultipleValuesOperand.less_than){target="\_blank"} : Query values that are less than that value.
- [`equal_to()`](medmodels.medrecord.querying.MultipleValuesOperand.equal_to){target="\_blank"} : Query values that are equal to that value.
- [`is_int()`](medmodels.medrecord.querying.MultipleValuesOperand.is_int){target="\_blank"} : Query on the values which format is `int`.
- [`greater_than()`](medmodels.medrecord.querying.MultipleValuesOperand.greater_than){target="\_blank"} : Query values that are greater than that value.
- [`edges()`](medmodels.medrecord.querying.NodeOperand.edges){target="\_blank"} : Returns a [`EdgeOperand()`](medmodels.medrecord.querying.EdgeOperand){target="\_blank"} to query on the edges of those nodes.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="\_blank"} : Select nodes that match that query.

:::

## OR & NOT operations

The inherent structure of the query engine works with logical **AND** operations. However, a complete query engine should also include **OR** and **NOT** operations to be able to address all scenarios. For that the methods [`exclude()`](medmodels.medrecord.querying.NodeOperand.exclude){target="\_blank"} and [`either_or()`](medmodels.medrecord.querying.NodeOperand.either_or){target="\_blank"}.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-5
lines: 111-131
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="\_blank"} : Query nodes that belong to that group.
- [`attribute()`](medmodels.medrecord.querying.EdgeOperand.attribute){target="\_blank"} : Returns a [`MultipleValuesOperand()`](medmodels.medrecord.querying.MultipleValuesOperand){target="\_blank"} to query on the values of the nodes for that attribute.
- [`less_than()`](medmodels.medrecord.querying.MultipleValuesOperand.less_than){target="\_blank"} : Query values that are less than that value.
- [`equal_to()`](medmodels.medrecord.querying.MultipleValuesOperand.equal_to){target="\_blank"} : Query values that are equal to that value.
- [`greater_than()`](medmodels.medrecord.querying.MultipleValuesOperand.greater_than){target="\_blank"} : Query values that are greater than that value.
- [`edges()`](medmodels.medrecord.querying.NodeOperand.edges){target="\_blank"} : Returns a [`EdgeOperand()`](medmodels.medrecord.querying.EdgeOperand){target="\_blank"} to query on the edges of those nodes.
- [`either_or()`](medmodels.medrecord.querying.NodeOperand.either_or){target="\_blank"} : Queries edges that match either one or the other given queries.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="\_blank"} : Select nodes that match that query.

:::

This includes also _"pat_3"_, that was not included in the previous section because none of its edges was included in the `query_edge_either()`, but it can be found in the `query_edge_or()` now.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-5, 110-130
lines: 134-140
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="\_blank"} : Query nodes that belong to that group.: Query edges that belong to that group.
- [`exclude()`](medmodels.medrecord.querying.NodeOperand.exclude){target="\_blank"} : Exclude the nodes that belong to the given query.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="\_blank"} : Select nodes that match that query.

:::

So this gives us all the patient nodes that were not selected with the previous query (logical **NOT** applied).

## Clones

Since the statements in the query engine are additive, every operation modifies the state of the query. That means that it is not possible to revert to a previous state unless the entire query is rewritten from scratch for that intermediate step. This can become inefficient and redundant, particularly when multiple branches of a query or comparisons with intermediate results are required.

To address this limitation, the [`clone()`](medmodels.medrecord.querying.SingleValueOperand.clone){target="\_blank"} method was introduced. This method allows users to create independent copies - or **clones** - of operands or computed values at any point in the query chain. Clones are completely decoupled from the original object, meaning that modifications of the clone do not affect the original, and vice versa. This functionality applies to all types of operands.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-5
lines: 143-158
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="\_blank"} : Query nodes that belong to that group.: Query edges that belong to that group.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="\_blank"}: Returns a [`NodeIndexOperand`](medmodels.medrecord.querying.NodeIndexOperand){target="\_blank"}` to query on the indices.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="\_blank"} : Query node indices containing that argument.
- [`contains()`](medmodels.medrecord.querying.EdgeOperand.attribute){target="\_blank"} : Returns a [`MultipleValuesOperand()`](medmodels.medrecord.querying.MultipleValuesOperand){target="\_blank"} to query on the values of the edges for that attribute.
- [`mean()`](medmodels.medrecord.querying.MultipleValuesOperand.mean){target="\_blank"}: Returns a [`SingleValueOperand`](medmodels.medrecord.querying.SingleValueOperand){target="\_blank"} containing the mean of those values.
- [`clone()`](medmodels.medrecord.querying.SingleValueOperand.clone){target="\_blank"} : Returns a clone of the operand.
- [`subtract()`](medmodels.medrecord.querying.SingleValueOperand.subtract){target="\_blank"} : Subtract the argument from the single value operand.
- [`greater_than()`](medmodels.medrecord.querying.MultipleValuesOperand.greater_than){target="\_blank"} : Query values that are greater than that value.
- [`less_than()`](medmodels.medrecord.querying.MultipleValuesOperand.less_than){target="\_blank"} : Query values that are less than that value.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="\_blank"} : Select nodes that match that query.

:::

## Queries as Function Arguments

In all previous snippets, we have used queries with the method [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="\_blank"} for representation purposes of its capacities. However, queries can also be used as function arguments to other methods or indexers from the [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="\_blank"} that take edge/node indices or the queries that result on those indices as arguments. Here are some examples of those functions:

- Using the [`node[]`](medmodels.medrecord.medrecord.MedRecord.node){target="\_blank"}, an indexer that retrieves the attributes for the given node indices.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-154
lines: 161
---
```

- Using [`groups_of_node()`](medmodels.medrecord.medrecord.MedRecord.groups_of_node){target="\_blank"}, a method that retrieves the groups to which a specific node index belongs to.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-154
lines: 162
---
```

- Using [`edge_endpoints()`](medmodels.medrecord.medrecord.MedRecord.edge_endpoints){target="\_blank"}, a method that retrieves the source and target nodes of the specified edge(s) in the MedRecord.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-155
lines: 163
---
```

:::{dropdown} Methods used in the snippet

- [`node[]`](medmodels.medrecord.medrecord.MedRecord.node){target="\_blank"} : Provides access to node information within the MedRecord instance via an indexer, returning a dictionary with node indices as keys and node attributes as values.
- [`groups_of_node()`](medmodels.medrecord.medrecord.MedRecord.groups_of_node){target="\_blank"} : Retrieves the groups associated with the specified node(s) in the MedRecord.
- [`edge_endpoints()`](medmodels.medrecord.medrecord.MedRecord.edge_endpoints){target="\_blank"} : Retrieves the source and target nodes of the specified edge(s) in the MedRecord.

:::

## Full example Code

The full code examples for this chapter can be found here:

```{literalinclude} scripts/02b_query_engine.py
---
language: python
---
```
