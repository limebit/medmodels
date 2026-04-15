# Query Engine with MIMIC-IV Demo Data

## What is the Query Engine?

The **MedRecord Query Engine** enables users to retrieve node and edge indices from the graph structure efficiently. Queries can be defined as Python functions and reused for cohort creation or downstream analyses.

This guide continues from the MedRecord built from the MIMIC-IV demo `patients` and `admissions` tables.

## Example dataset

The following setup recreates the MedRecord used in the examples below.

```{literalinclude} scripts/query_engine_mimic.py
---
language: python
lines: 2-30
---
```

## Node Queries

The simplest way to use the Query Engine is to restrict nodes to a group and filter them by an attribute.

The following query returns all patients with `anchor_age > 65`.

```{exec-literalinclude} scripts/query_engine_mimic.py
---
language: python
setup-lines: 2-30
lines: 34-40
---
```

We can also query admissions. In the next example, we select admissions with in-hospital mortality.

```{exec-literalinclude} scripts/query_engine_mimic.py
---
language: python
setup-lines: 2-30
lines: 44-50
---
```

## Reusing Node Queries

Queries can be reused when building new queries or defining groups.

```{exec-literalinclude} scripts/query_engine_mimic.py
---
language: python
setup-lines: 2-30
lines: 54-60
---
```

This query first reuses the patient selection and then restricts it further to patients with `anchor_age > 80`.

## Edge Queries

Edges can also be queried directly. Since patient-to-admission links were assigned to the `Patient→Admission` group, they can be retrieved as follows:

```{exec-literalinclude} scripts/query_engine_mimic.py
---
language: python
setup-lines: 2-30
lines: 64-69
---
```

## Queries as Function Arguments

Queries can be passed directly to methods such as [`add_group()`](medmodels.medrecord.medrecord.MedRecord.add_group){target="_blank"}.

```{exec-literalinclude} scripts/query_engine_mimic.py
---
language: python
setup-lines: 2-30
lines: 73-75
---
```

This creates a reusable group of elderly patients directly from the query.

## Full example Code

The full code examples for this chapter can be found here:

```{literalinclude} scripts/query_engine_mimic.py
---
language: python
lines: 2-75
---
```
