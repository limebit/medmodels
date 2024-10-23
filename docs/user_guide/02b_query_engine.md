# Query Engine

## What is the Query Engine?

The **MedRecord Query Engine** enables users to retrieve data stored in the graph structure efficiently. Through its intuitive interface, it supports complex queries, allowing you to filter nodes and edges based on their properties and relationships. This section introduces the basic concepts of querying MedRecords, as well as more advanced use cases for handling intricate datasets.

## Node Queries

Node operands allow you to define specific criteria for selecting nodes within a MedRecord. These operands enable flexible and complex queries by combining multiple conditions, such as group membership, attribute values, and relationships to other nodes. This section introduces the basic usage of node operands to streamline your data queries.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 7-12
---
```

You can get to the same result via different approaches. That makes the query engine very versatile and adaptative to your specific needs. For instance, this produces the same result as `medrecord.nodes_in_group("patient")`. Let's complicate it a bit more involving more than one operand.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 15-24
---
```

:::{note}
The `has attribute()` function is not needed in this example, since the `attribute()` one already checks whether the nodes have the attribute. It is there merely for educational purposes. This will happen in different examples in this user guide to ensure the maximum amount of functions are portrayed.
:::

In case, for instance, that you do not know whether there are different ways to assign the `gender` attribute across the MedRecord (with leading/trailing whitespaces or formatted in lower/uppercase), you can also modify the attribues of a node/edge. You can also perform mathematical calculations like `mean()`, `median()` or `min()` and assign them to a variable. Also, you can keep manipulating the operand, like in the following example, where we are adding _5_ years to the `"mean_age"` to query on that value.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 27-43
---
```

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

As you can see, the query engine is highly useful for finding nodes that fulfill different criteria in a highly optimized way, using descriptive programming as a precise tool. We can use previously defined queries too, and also use the `neighbors()` function to query also through the nodes that are neighbors to those nodes.

For instance, in the following example we are selecting the nodes that have the following characteristics

- Are in group `"patient"`.
- Their node index contains the string _"pat"_
- Their attribute `"age"` is greater than 30, and their attribute `"gender"` is equal to _"M"_.
- They are  connected to  nodes which attribute `"description"` contains the word _"fentanyl"_  in either upper or lowercase.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4, 15-24
lines: 46-55
---
```

## Edge Queries

Edge operands provide a way to define and filter relationships between nodes in a MedRecord. By using edge operands, you can create queries that target specific connections based on attributes like time, duration, or custom properties. This section explores how to leverage edge operands to gain insights from the relationships in your data.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 58-64
---
```

The edge operand follows the same principles as the node operand, with some extra queries applicable only to edges like `source_node()` or `target_node()` (instead of the `neighbors()` one).

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 67-76
---
```

## Combining Node & Edge Queries

The full power of the query engine appears once you combine both operands inside the queries. In the following query, we are able to query for nodes that:

- Are in group `"patient"`
- Their attribute `"age"` is greater than _30_, and their attribute `"gender"` is equal to _"M"_.
- They have at least an edge that is in in the `"patient_drug"` group, which attribute `"cost"` is less than _200_ and its attribute `"quantity"` is equal to _1_.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 79-95
---
```

## OR & NOT operations

The inherent structure of the query engine works with logical **AND** operations. However, a complete query engine should also include **OR** and **NOT** operations to be able to address all scenarios. For that the functions `exclude()` and `either_or()`.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 98-118
---
```

This includes also `"pat_3"`, that was not included because none of its edges was included in the `query_edge_either()`, but it is in the `query_edge_or()` now.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4, 98-116
lines: 121-127
---
```

So this gives us all the patient nodes that were not selected with the previous query (logical **NOT** applied).

## Clones

Since the statements in the query engine are additive, we cannot go back to a previous state of the query unless we want to rewrite the whole query again for an intermemediate step. For that reason, `clone()` function was devised.

```{exec-literalinclude} scripts/02b_query_engine.py
---
language: python
setup-lines: 1-4
lines: 130-143
---
```

## Full example Code

The full code examples for this chapter can be found here:

```{literalinclude} scripts/02b_query_engine.py
---
language: python
---
```
