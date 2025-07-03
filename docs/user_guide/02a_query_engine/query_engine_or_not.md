# OR & NOT operations

The inherent structure of the query engine works with logical **AND** operations. However, a complete query engine should also include **OR** and **NOT** operations to be able to address all scenarios. For that the methods [`exclude()`](medmodels.medrecord.querying.NodeOperand.exclude){target="_blank"} and [`either_or()`](medmodels.medrecord.querying.NodeOperand.either_or){target="_blank"} are included.

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 2-10
lines: 112-133
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="_blank"} : Query edges that belong to that group.
- [`attribute()`](medmodels.medrecord.querying.EdgeOperand.attribute){target="_blank"} : Returns a [`EdgeMultipleValuesWithIndexOperand()`](medmodels.medrecord.querying.EdgeMultipleValuesWithIndexOperand){target="_blank"} to query on the values of the edges for that attribute.
- [`less_than()`](medmodels.medrecord.querying.EdgeMultipleValuesWithIndexOperand.less_than){target="_blank"} : Query values that are less than that value.
- [`equal_to()`](medmodels.medrecord.querying.EdgeMultipleValuesWithIndexOperand.equal_to){target="_blank"} : Query values that are equal to that value.
- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`NodeMultipleValuesWithIndexOperand()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand){target="_blank"} to query on the values of the nodes for that attribute.
- [`greater_than()`](medmodels.medrecord.querying.NodeMultipleValuesWithIndexOperand.greater_than){target="_blank"} : Query values that are greater than that value.
- [`edges()`](medmodels.medrecord.querying.NodeOperand.edges){target="_blank"} : Returns a [`EdgeOperand()`](medmodels.medrecord.querying.EdgeOperand){target="_blank"} to query on the edges of those nodes.
- [`either_or()`](medmodels.medrecord.querying.NodeOperand.either_or){target="_blank"} : Queries edges that match either one or the other given queries.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"}: Returns a [`NodeIndicesOperand`](medmodels.medrecord.querying.NodeIndicesOperand){target="_blank"} representing the indices of the nodes queried.
- [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.

:::

This includes also _"pat_3"_, that was not included in the previous section because none of its edges was included in the `query_edge_either()`, but it can be found in the `query_edge_or()` now.

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 1-10, 112-130
lines: 136-151
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.EdgeOperand.in_group){target="_blank"} : Query edges that belong to that group.
- [`exclude()`](medmodels.medrecord.querying.NodeOperand.exclude){target="_blank"} : Exclude the nodes that belong to the given query.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"}: Returns a [`NodeIndicesOperand`](medmodels.medrecord.querying.NodeIndicesOperand){target="_blank"} representing the indices of the nodes queried.
- [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} : Retrieves information on the nodes from the MedRecord given the query.
:::

So this gives us all the patient nodes that were not selected with the previous query (logical **NOT** applied).

## Full example Code

The full code examples for this chapter can be found here:

```{literalinclude} scripts/query_engine.py
---
language: python
lines: 2-12, 112-151
---
```