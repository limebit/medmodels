# Queries as Function Arguments

In all other sections, we have used queries with the method [`query_nodes()`](medmodels.medrecord.medrecord.MedRecord.query_nodes){target="_blank"} for representation purposes of its capacities. However, queries can also be used as function arguments to other methods or indexers from the [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="_blank"} that take edge/node indices or the queries that result on those indices as arguments. Here are some examples of those functions:

- Using the [`add_group()`](medmodels.medrecord.medrecord.MedRecord.add_group){target="_blank"} to create groups in the MedRecord out of chosen subset of patients. We need to [`unfreeze_schema()`](medmodels.medrecord.medrecord.MedRecord.unfreeze_schema){target="_blank"} first, since this new group does not exist in the schema and we have a provided schema in the example dataset.

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 1-11, 24-31
lines: 174-176
---
```

- Using the [`node[]`](medmodels.medrecord.medrecord.MedRecord.node){target="_blank"} indexer, which retrieves the attributes for the given node indices.

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 1-11, 112-130
lines: 178
---
```

- Using [`groups_of_node()`](medmodels.medrecord.medrecord.MedRecord.groups_of_node){target="_blank"}, a method that retrieves the groups to which a specific node index belongs to.

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 1-31
lines: 179
---
```

- Using [`edge_endpoints()`](medmodels.medrecord.medrecord.MedRecord.edge_endpoints){target="_blank"}, a method that retrieves the source and target nodes of the specified edge(s) in the MedRecord.

```{exec-literalinclude} scripts/query_engine.py
---
language: python
setup-lines: 1-11, 76-85
lines: 180
---
```

:::{dropdown} Methods used in the snippet

- [`unfreeze_schema()`](medmodels.medrecord.medrecord.MedRecord.unfreeze_schema){target="_blank"} : Unfreezes the schema. Changes are automatically inferred.
- [`add_group()`](medmodels.medrecord.medrecord.MedRecord.add_group){target="_blank"} : Adds a group to the MedRecord, optionally with node and edge indices.
- [`groups`](medmodels.medrecord.medrecord.MedRecord.groups){target="_blank"} : Lists the groups in the MedRecord instance.
- [`node[]`](medmodels.medrecord.medrecord.MedRecord.node){target="_blank"} : Provides access to node information within the MedRecord instance via an indexer, returning a dictionary with node indices as keys and node attributes as values.
- [`groups_of_node()`](medmodels.medrecord.medrecord.MedRecord.groups_of_node){target="_blank"} : Retrieves the groups associated with the specified node(s) in the MedRecord.
- [`edge_endpoints()`](medmodels.medrecord.medrecord.MedRecord.edge_endpoints){target="_blank"} : Retrieves the source and target nodes of the specified edge(s) in the MedRecord.

:::



## Full example Code

The full code examples for this chapter can be found here:

```{literalinclude} scripts/query_engine.py
---
language: python
lines: 2-12, 24-33, 112-132, 76-84, 174-180
---
```