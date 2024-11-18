# Query Engine

## What is the Query Engine?

The **MedRecord Query Engine** enables users to retrieve data stored in the graph structure efficiently. Through its intuitive interface, it supports complex queries, allowing you to filter nodes and edges based on their properties and relationships. This section introduces the basic concepts of querying MedRecords, as well as more advanced use cases for handling intricate datasets.

## Node Queries

Node operands allow you to define specific criteria for selecting nodes within a [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="_blank"}. These operands enable flexible and complex queries by combining multiple conditions, such as group membership, attribute values, and relationships to other nodes. This section introduces the basic usage of node operands to streamline your data queries. For that, the [`NodeOperand`](medmodels.medrecord.querying.NodeOperand){target="_blank"} querying class is used.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 7-12
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="_blank"} : Select nodes that match that query.

:::

You can get to the same result via different approaches. That makes the query engine very versatile and adaptive to your specific needs. For instance, this produces the same result as [`medrecord.nodes_in_group("patient")`](medmodels.medrecord.medrecord.MedRecord.nodes_in_group){target="_blank"}. Let's complicate it a bit more involving more than one operand.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 15-24
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"}: Returns a [`NodeIndexOperand`](medmodels.medrecord.querying.NodeIndexOperand){target="_blank"}` to query on the indices.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="_blank"} : Query node indices containing that argument.
- [`has_attribute()`](medmodels.medrecord.querying.NodeOperand.has_attribute){target="_blank"} : Query nodes that have that attribute.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`MultipleValuesOperand`](medmodels.medrecord.querying.MultipleValuesOperand){target="_blank"} to query on the values of the nodes for that attribute.
- [`greater_than()`](medmodels.medrecord.querying.MultipleValuesOperand.greater_than){target="_blank"}` : Query values that are greater than that value.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="_blank"} : Select nodes that match that query.

:::

:::{note}
The [`has_attribute()`](medmodels.medrecord.querying.NodeOperand.has_attribute){target="_blank"} method is not needed in this example, since the [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} one already checks whether the nodes have the attribute. It is there merely for educational purposes. This will happen in different examples in this user guide to ensure the maximum amount of methods are portrayed.
:::

In case, for instance, that you do not know whether there are different ways to assign the `gender` attribute across the [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="_blank"} (with leading/trailing whitespaces or formatted in lower/uppercase), you can also modify the attributes of a node/edge. You can also perform mathematical calculations like [`mean()`](medmodels.medrecord.querying.MultipleValuesOperand.mean){target="_blank"}, [`median()`](medmodels.medrecord.querying.MultipleValuesOperand.median){target="_blank"} or [`min()`](medmodels.medrecord.querying.MultipleValuesOperand.min){target="_blank"} and assign them to a variable. Also, you can keep manipulating the operand, like in the following example, where we are subtracting _5_ years from the `mean_age` to query on that value.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 27-43
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"}: Returns a [`NodeIndexOperand`](medmodels.medrecord.querying.NodeIndexOperand){target="_blank"}` to query on the indices.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="_blank"} : Query node indices containing that argument.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`MultipleValuesOperand`](medmodels.medrecord.querying.MultipleValuesOperand){target="_blank"} to query on the values of the nodes for that attribute.
- [`lowercase()`](medmodels.medrecord.querying.MultipleValuesOperand.lowercase){target="_blank"} : Converts the values that are strings to lowercase.
- [`trim()`](medmodels.medrecord.querying.MultipleValuesOperand.trim){target="_blank"} : Removes leading and trailing whitespacing from the values.
- [`equal_to()`](medmodels.medrecord.querying.MultipleValuesOperand.equal_to){target="_blank"} : Query values equal to that value.

- [`has_attribute()`](medmodels.medrecord.querying.NodeOperand.has_attribute){target="_blank"} : Query nodes that have that attribute.
- [`mean()`](medmodels.medrecord.querying.MultipleValuesOperand.mean){target="_blank"}: Returns a [`SingleValueOperand`](medmodels.medrecord.querying.SingleValueOperand){target="_blank"} containing the mean of those values.
- [`subtract()`](medmodels.medrecord.querying.SingleValueOperand.subtract){target="_blank"} : Subtract the argument from the single value operand.

- [`less_than()`](medmodels.medrecord.querying.MultipleValuesOperand.less_than){target="_blank"} : Query values that are less than that value.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="_blank"} : Select nodes that match that query.

:::

:::{note}
Query methods used for changing the operands cannot be concatenated or assigned to variables, since their `Return` is null. That is, the following code snippet will set `gender_lowercase` as `None`, and thus the query will not be able to find any nodes.

```python
# Wrong implementation
gender_lowercase = node.attribute("gender").lowercase()
gender.equal_to("m")

# Wrong implementation
gender = node.attribute("gender")
gender.lowercase().trim()
gender.equal_to("m")

# Correct implementation
gender = node.attribute("gender")
gender.lowercase()
gender.trim()
gender.equal_to("m")
```

Nor do the ones that compare operands to other operands, since their `Return` value is also null.

```python
# Wrong implementation
gender = node.attribute("gender")
gender.equal_to("M").not_equal_to("F")

# Correct implementation
gender = node.attribute("gender")
gender.equal_to("M")
gender.not_equal_to("F")
```

:::

As you can see, the query engine is highly useful for finding nodes that fulfill different criteria in a highly optimized way, using descriptive programming as a precise tool. We can use previously defined queries too, and also use the [`neighbors()`](medmodels.medrecord.querying.NodeOperand.neighbors){target="_blank"} method to query also through the nodes that are neighbors to those nodes.

For instance, in the following example we are selecting the nodes that have the following characteristics

- Are in group `patient`.
- Their node index contains the string _"pat"_
- Their attribute `age` is greater than 30, and their attribute `gender` is equal to _"M"_.
- They are  connected to  nodes which attribute `description` contains the word _"fentanyl"_  in either upper or lowercase.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4, 15-24
lines: 46-55
---
```

:::{dropdown} Methods used in the snippet

- [`neighbors()`](medmodels.medrecord.querying.NodeOperand.neighbors){target="_blank"} : Returns a [`NodeOperand()`](medmodels.medrecord.querying.NodeOperand){target="_blank"} to query the neighbors of those nodes.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`MultipleValuesOperand()`](medmodels.medrecord.querying.MultipleValuesOperand){target="_blank"} to query on the values of the nodes for that attribute.
- [`lowercase()`](medmodels.medrecord.querying.MultipleValuesOperand.lowercase){target="_blank"} : Converts the values that are strings to lowercase.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="_blank"} : Query node indices containing that argument.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes){target="_blank"} : Select nodes that match that query.

:::

## Edge Queries

Edge operands provide a way to define and filter relationships between nodes in a [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="_blank"}. By using edge operands, you can create queries that target specific connections based on attributes like time, duration, or custom properties. This section explores how to leverage edge operands to gain insights from the relationships in your data. For that, the [`EdgeOperand`](medmodels.medrecord.querying.EdgeOperand){target="_blank"} querying class is used.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 58-64
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`select_edges()`](medmodels.medrecord.medrecord.MedRecord.select_edges){target="_blank"} : Select edges that match that query.

:::

The edge operand follows the same principles as the node operand, with some extra queries applicable only to edges like [`source_node()`](medmodels.medrecord.querying.EdgeOperand.source_node){target="_blank"} or [`target_node()`](medmodels.medrecord.querying.EdgeOperand.target_node){target="_blank"}  (instead of [`neighbors()`](medmodels.medrecord.querying.NodeOperand.neighbors){target="_blank"}).

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 67-76
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`attribute()`](medmodels.medrecord.querying.EdgeOperand.attribute){target="_blank"} : Returns a [`MultipleValuesOperand()`](medmodels.medrecord.querying.MultipleValuesOperand){target="_blank"} to query on the values of the nodes for that attribute.
- [`less_than()`](medmodels.medrecord.querying.MultipleValuesOperand.less_than){target="_blank"} : Query values that are less than that value.
- [`source_node()`](medmodels.medrecord.querying.EdgeOperand.source_node){target="_blank"} : Returns a [`NodeOperand()`](medmodels.medrecord.querying.NodeOperand) to query on the source nodes for those edges.
- [`is_max()`](medmodels.medrecord.querying.MultipleValuesOperand.is_max){target="_blank"} : Query on the values that hold on the maximum value among all of the.
- [`target_node()`](medmodels.medrecord.querying.EdgeOperand.target_node){target="_blank"} : Returns a [`NodeOperand()`](medmodels.medrecord.querying.NodeOperand){target="_blank"} to query on the target nodes for those edges.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="_blank"} : Query node indices containing that argument.
- [`select_edges()`](medmodels.medrecord.medrecord.MedRecord.select_edges){target="_blank"} : Select edges that match that query.

:::

## Combining Node & Edge Queries

The full power of the query engine appears once you combine both operands inside the queries. In the following query, we are able to query for nodes that:

- Are in group `patient`
- Their attribute `age` is greater than _30_, and their attribute `gender` is equal to _"M"_.
- They have at least an edge that is in in the `patient_drug` group, which attribute `cost` is less than _200_ and its attribute `quantity` is equal to _1_.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 79-95
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`attribute()`](medmodels.medrecord.querying.EdgeOperand.attribute){target="_blank"} : Returns a [`MultipleValuesOperand()`](medmodels.medrecord.querying.MultipleValuesOperand){target="_blank"} to query on the values of the nodes for that attribute.
- [`less_than()`](medmodels.medrecord.querying.MultipleValuesOperand.less_than){target="_blank"} : Query values that are less than that value.
- [`equal_to()`](medmodels.medrecord.querying.MultipleValuesOperand.equal_to){target="_blank"} : Query values that are equal to that value.
- [`is_int()`](medmodels.medrecord.querying.MultipleValuesOperand.is_int){target="_blank"} : Query on the values which format is `int`.
- [`greater_than()`](medmodels.medrecord.querying.MultipleValuesOperand.greater_than){target="_blank"} : Query values that are greater than that value.
- [`edges()`](medmodels.medrecord.querying.NodeOperand.edges){target="_blank"} : Returns a [`EdgeOperand()`](medmodels.medrecord.querying.EdgeOperand){target="_blank"} to query on the edges of those nodes.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes`){target="_blank"} : Select nodes that match that query.

:::

## OR & NOT operations

The inherent structure of the query engine works with logical **AND** operations. However, a complete query engine should also include **OR** and **NOT** operations to be able to address all scenarios. For that the methods [`exclude()`](medmodels.medrecord.querying.NodeOperand.exclude){target="_blank"} and [`either_or()`](medmodels.medrecord.querying.NodeOperand.either_or){target="_blank"}.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 98-118
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`attribute()`](medmodels.medrecord.querying.EdgeOperand.attribute){target="_blank"} : Returns a [`MultipleValuesOperand()`](medmodels.medrecord.querying.MultipleValuesOperand){target="_blank"} to query on the values of the nodes for that attribute.
- [`less_than()`](medmodels.medrecord.querying.MultipleValuesOperand.less_than){target="_blank"} : Query values that are less than that value.
- [`equal_to()`](medmodels.medrecord.querying.MultipleValuesOperand.equal_to){target="_blank"} : Query values that are equal to that value.
- [`greater_than()`](medmodels.medrecord.querying.MultipleValuesOperand.greater_than){target="_blank"} : Query values that are greater than that value.
- [`edges()`](medmodels.medrecord.querying.NodeOperand.edges){target="_blank"} : Returns a [`EdgeOperand()`](medmodels.medrecord.querying.EdgeOperand){target="_blank"} to query on the edges of those nodes.
- [`either_or()`](medmodels.medrecord.querying.NodeOperand.either_or){target="_blank"} : Queries edges that match either one or the other given queries.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes`){target="_blank"} : Select nodes that match that query.

:::

This includes also _"pat_3"_, that was not included in the previous section because none of its edges was included in the `query_edge_either()`, but it can be found in the `query_edge_or()` now.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4, 98-116
lines: 121-127
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="_blank"} : Query nodes that belong to that group.: Query edges that belong to that group.
- [`exclude()`](medmodels.medrecord.querying.NodeOperand.exclude){target="_blank"} : Exclude the nodes that belong to the given query.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes`){target="_blank"} : Select nodes that match that query.

:::

So this gives us all the patient nodes that were not selected with the previous query (logical **NOT** applied).

## Clones

Since the statements in the query engine are additive, we cannot go back to a previous state of the query unless we want to rewrite the whole query again for an intermemediate step. For that reason, [`clone()`](medmodels.medrecord.querying.SingleValueOperand.clone){target="_blank"} method was devised. Clones of all types of operands can be made.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 130-143
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="_blank"} : Query nodes that belong to that group.: Query edges that belong to that group.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"}: Returns a [`NodeIndexOperand`](medmodels.medrecord.querying.NodeIndexOperand){target="_blank"}` to query on the indices.
- [`contains()`](medmodels.medrecord.querying.NodeIndexOperand.contains){target="_blank"} : Query node indices containing that argument.
- [`contains()`](medmodels.medrecord.querying.EdgeOperand.attribute){target="_blank"} : Returns a [`MultipleValuesOperand()`](medmodels.medrecord.querying.MultipleValuesOperand){target="_blank"} to query on the values of the edges for that attribute.
- [`mean()`](medmodels.medrecord.querying.MultipleValuesOperand.mean){target="_blank"}: Returns a [`SingleValueOperand`](medmodels.medrecord.querying.SingleValueOperand){target="_blank"} containing the mean of those values.
- [`clone()`](medmodels.medrecord.querying.SingleValueOperand.clone){target="_blank"} : Returns a clone of the operand.
- [`subtract()`](medmodels.medrecord.querying.SingleValueOperand.subtract){target="_blank"} : Subtract the argument from the single value operand.
- [`greater_than()`](medmodels.medrecord.querying.MultipleValuesOperand.greater_than){target="_blank"} : Query values that are greater than that value.
- [`less_than()`](medmodels.medrecord.querying.MultipleValuesOperand.less_than){target="_blank"} : Query values that are less than that value.
- [`select_nodes()`](medmodels.medrecord.medrecord.MedRecord.select_nodes`){target="_blank"} : Select nodes that match that query.

:::

## Full example Code

The full code examples for this chapter can be found here:

```{literalinclude} scripts/02b_query_engine.py
---
language: python
---
```
