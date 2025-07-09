# Groupby Operands

Sometimes, you'll want to look at the data in your MedRecord in groups, assessing the properties and attributes of each one of them individually. For that, the [`group_by()`](medmodels.medrecord.querying.NodeOperand.group_by){target="_blank"} method was devised.

## Grouping Nodes

You can group nodes based on a specific criterion. For example, you can group all patients by their `gender` and then inspect the `age` of the patients within each group. This is done by passing a `NodeOperandGroupDiscriminator` to the [`group_by()`](medmodels.medrecord.querying.NodeOperand.group_by){target="_blank"} method.

In the snippet below, we group the nodes by the `gender` attribute and then retrieve the `age` for the patients in each of these groups.

```{exec-literalinclude} scripts/group_by.py
---
language: python
setup-lines: 1-14
lines: 17-25
---
```

:::{dropdown} Methods used in the snippet

  - [`group_by()`](medmodels.medrecord.querying.NodeOperand.group_by){target="_blank"} : Groups the nodes based on the given discriminator, returning a [`NodeGroupOperand`](medmodels.medrecord.querying.NodeGroupOperand){target="_blank"}.
  - [`attribute()`](medmodels.medrecord.querying.NodeGroupOperand.attribute){target="_blank"} : Returns a [`NodeMultipleValuesWithIndexGroupOperand`](medmodels.medrecord.querying.NodeGroupOperand){target="_blank"} to query on the values of the nodes per group for that attribute.
  - [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.

:::

Furthermore, you can perform calculations on these newly formed groups. For instance, you can calculate the `mean` age for each gender group.

```{exec-literalinclude} scripts/group_by.py
---
language: python
setup-lines: 1-14
lines: 28-37
---
```

:::{dropdown} Methods used in the snippet

  - [`group_by()`](medmodels.medrecord.querying.NodeOperand.group_by){target="_blank"} : Groups the nodes based on the given discriminator, returning a [`NodeGroupOperand`](medmodels.medrecord.querying.NodeGroupOperand){target="_blank"}.
  - [`attribute()`](medmodels.medrecord.querying.NodeGroupOperand.attribute){target="_blank"} : Returns a [`NodeMultipleValuesWithIndexGroupOperand`](medmodels.medrecord.querying.NodeGroupOperand){target="_blank"} to query on the values of the nodes per group for that attribute.
  - [`mean()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexGroupOperand.mean){target="_blank"} : Calculates the mean of the values within each group.
  - [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.

:::

-----

## Grouping Edges

Similarly to nodes, you can also group edges. A common use case is grouping edges by their `SourceNode` or `TargetNode` (we could also group them per `Attribute`). In the following example, we group the edges based on their source node and retrieve their `time` attribute.

```{exec-literalinclude} scripts/group_by.py
---
language: python
setup-lines: 1-14
lines: 40-49
---
```

:::{dropdown} Methods used in the snippet

  - [`index()`](medmodels.medrecord.querying.EdgeOperand.index){target="_blank"}: Returns an [`EdgeIndicesOperand`](medmodels.medrecord.querying.EdgeIndicesOperand){target="_blank"} representing the indices of the edges queried.
  - [`less_than()`](medmodels.medrecord.querying.EdgeIndicesOperand.less_than){target="_blank"}: Query edge indices that are less than the specified value.
  - [`group_by()`](medmodels.medrecord.querying.EdgeOperand.group_by){target="_blank"} : Groups the edges based on the given discriminator.
  - [`attribute()`](medmodels.medrecord.querying.EdgeGroupOperand.attribute){target="_blank"} : Returns a [`EdgeMultipleValuesWithIndexGroupOperand`](medmodels.medrecord.querying.EdgeMultipleValuesWithIndexGroupOperand){target="_blank"} to query on the values of the edges per group for that attribute.
  - [`query_edges()`](medmodels.medrecord.medrecord.MedRecord.query_edges){target="_blank"} : Retrieves information on the edges from the MedRecord given the query.

:::

You can also perform aggregations on edge groups, such as counting how many edges are associated with each source node.

```{exec-literalinclude} scripts/group_by.py
---
language: python
setup-lines: 1-14
lines: 52-58
---
```

:::{dropdown} Methods used in the snippet

  - [`group_by()`](medmodels.medrecord.querying.EdgeOperand.group_by){target="_blank"} : Groups the edges based on the given discriminator.
  - [`index()`](medmodels.medrecord.querying.EdgeGroupOperand.index){target="_blank"}: Returns an [`EdgeIndicesGroupOperand`](medmodels.medrecord.querying.EdgeIndicesGroupOperand){target="_blank"} representing the indices of the edges queried within each group.
  - [`count()`](medmodels.medrecord.querying.EdgeIndicesGroupOperand.count){target="_blank"}: Counts the number of edges within each group.
  - [`query_edges()`](medmodels.medrecord.medrecord.MedRecord.query_edges){target="_blank"} : Retrieves information on the edges from the MedRecord given the query.

:::

-----

## Full example Code

The full code examples for this chapter can be found here:

```{literalinclude} scripts/group_by.py
---
language: python
lines: 2-58
---
```