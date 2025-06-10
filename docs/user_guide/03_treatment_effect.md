# Treatment Effect Calculation

## What is Treatment Effect Calculation?

The **Treatment Effect** module's objective is finding the differences in outcomes between a treatment group and a control group. This can be assessed thanks to a variety of metrics, such as the [`odds_ratio()`](medmodels.treatment_effect.estimate.Estimate.odds_ratio){target="_blank"}, the [`absolute_risk_reduction()`](medmodels.treatment_effect.estimate.Estimate.absolute_risk_reduction){target="_blank"}, or the [`hedges_g()`](medmodels.treatment_effect.estimate.Estimate.hedges_g){target="_blank"}.

## Example dataset

An example dataset for the following demonstrations was generated with the method [`from_advanced_example_dataset()`](medmodels.medrecord.medrecord.MedRecord.from_advanced_example_dataset){target="_blank"} from the [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="_blank"} class.

```{literalinclude} scripts/03_show_dataset.py
---
language: python
lines: 8
---
```

This example dataset includes a set of patients, drugs, diagnoses and procedures. For this section, we will use the patients, diagnoses, drugs, and the edges that connect the patients' nodes with the other two groups.

```{exec-literalinclude} scripts/03_show_dataset.py
---
language: python
setup-lines: 1-65
lines: 67
---
```

```{exec-literalinclude} scripts/03_show_dataset.py
---
language: python
setup-lines: 1-65
lines: 68
---
```

```{exec-literalinclude} scripts/03_show_dataset.py
---
language: python
setup-lines: 1-65
lines: 69
---
```

```{exec-literalinclude} scripts/03_show_dataset.py
---
language: python
setup-lines: 1-65
lines: 70
---
```

```{exec-literalinclude} scripts/03_show_dataset.py
---
language: python
setup-lines: 1-65
lines: 71
---
```

## Building a Treatment Effect Instance

As with other modules in *MedModels*, the [`TreatmentEffect`](medmodels.treatment_effect.treatment_effect.TreatmentEffect){target="_blank"} class is meant to be instantiated using a builder pattern, thanks to its [`builder()`](medmodels.treatment_effect.treatment_effect.TreatmentEffect.builder){target="_blank"} method.

This instantiation requires a minimum of two arguments: a `treatment` and an `outcome`. These have to be the names of the MedRecord's `Groups` that contain the respective nodes. Here, we can see how we can create these groups by using the Query Engine. More information on how to use this powerful and efficient tool here: [Query Engine](02a_query_engine.md).

```{literalinclude} scripts/03_treatment_effect.py
---
language: python
lines: 12-32
---
```

:::{dropdown} Methods used in the snippet

- [`in_group()`](medmodels.medrecord.querying.NodeOperand.in_group){target="_blank"} : Query nodes that belong to that group.
- [`attribute()`](medmodels.medrecord.querying.NodeOperand.attribute){target="_blank"} : Returns a [`NodeMultipleValuesOperand`](medmodels.medrecord.querying.NodeMultipleValuesOperand){target="_blank"} representing the values of that attribute for the nodes.
- [`lowercase()`](medmodels.medrecord.querying.NodeMultipleValuesOperand.lowercase){target="_blank"} : Convert the multiple values to lowercase.
- [`contains()`](medmodels.medrecord.querying.NodeMultipleValuesOperand.contains){target="_blank"} : Query which multiple values contain a value.
- [`index()`](medmodels.medrecord.querying.NodeOperand.index){target="_blank"} : Returns a [`NodeIndicesOperand`](medmodels.medrecord.querying.NodeIndicesOperand){target="_blank"} representing the indices of the nodes queried.
- [`unfreeze_schema()`](medmodels.medrecord.medrecord.MedRecord.unfreeze_schema){target="_blank"}: Unfreezes the schema. Changes in the schema are automatically inferred.
- [`add_group()`](medmodels.medrecord.medrecord.MedRecord.add_group){target="_blank"}: Adds a group to the MedRecord, optionally with node and edge indices.

:::

:::{note}
Since the MedRecord we are using has a [`Provided`](medmodels.medrecord.schema.SchemaType.Provided){target="_blank"} schema, we need to use [`unfreeze_schema()`](medmodels.medrecord.medrecord.MedRecord.unfreeze_schema){target="_blank"} in order to add new groups to the MedRecord.
:::

Once we have the required `treatment` and `outcome` groups (_insulin_ and _diabetes_), we can go forward and create a treatment effect instance.


```{literalinclude} scripts/03_treatment_effect.py
---
language: python
lines: 34-36
---
```

:::{dropdown} Methods used in the snippet

- [`TreatmentEffect`](medmodels.treatment_effect.treatment_effect.TreatmentEffect){target="_blank"}: The TreatmentEffect class for analyzing treatment effects in medical records.
- [`builder()`](medmodels.treatment_effect.treatment_effect.TreatmentEffect.builder){target="_blank"}: Creates a TreatmentEffectBuilder instance for the TreatmentEffect class.
- [`with_treatment()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_treatment){target="_blank"}: Sets the treatment group for the treatment effect estimation.
- [`with_outcome()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_outcome){target="_blank"}: Sets the outcome group for the treatment effect estimation.
- [`build()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.build){target="_blank"}: Builds the treatment effect with all the provided configurations.

:::


### Customizing the Treatment Effect Properties

In this previous section we have seen how to instantiate a Treatment Effect class with default properties. However, using the aforementioned builder pattern, we can tinker with the different aspects to take into consideration for calculating the treatment effect metrics.