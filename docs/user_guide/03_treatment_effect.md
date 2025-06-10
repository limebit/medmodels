# Treatment Effect Calculation

## What is Treatment Effect Calculation?

The **Treatment Effect** module's objective is finding the differences in outcomes between a treatment group and a control group. This can be assessed thanks to a variety of metrics, such as the [`odds_ratio()`](medmodels.treatment_effect.estimate.Estimate.odds_ratio){target="_blank"}, the [`absolute_risk_reduction()`](medmodels.treatment_effect.estimate.Estimate.absolute_risk_reduction){target="_blank"}, or the [`hedges_g()`](medmodels.treatment_effect.estimate.Estimate.hedges_g){target="_blank"}.

The example dataset used in the following section is explained in this dropdown:

:::{dropdown} Example Dataset

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
:::

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

With this instance of a treatment effect class, we can test in which groups the patients of our MedRecord are divided into thanks to a [`ContingencyTable`](medmodels.treatment_effect.estimate.ContingencyTable){target="_blank"}. This Contingency Table, contains the counts of how many patients are divided into the four important subgroups the treatment effect cares about the most:
- *Treated with outcome*: Patients who received the treatment and experienced the outcome.
- *Treated with no outcome*: Patients who received the treatment but did not experience the outcome.
- *Control with outcome*: Patients who did not receive the treatment but experienced the outcome.
- *Control with outcome*: Patients who neither received the treatment nor experienced the outcome.

```{exec-literalinclude} scripts/03_treatment_effect.py
---
language: python
setup-lines: 1-36
lines: 38
---
```

:::{dropdown} Methods used in the snippet

- [`subject_counts`](medmodels.treatment_effect.estimate.Estimate.subject_counts){target="_blank"}: Overview of how many subjects are in which group from the contingency table.

:::

### Customizing the Treatment Effect Properties

In the previous section, we saw how to instantiate a TreatmentEffect class with default settings. Using the builder pattern, however, we can customize key properties that influence how treatment effect metrics are calculated.

One important property is *time*. By default, treatment and outcome groups are formed without considering the time at which each event occurred. But when a time attribute is provided, the logic changes: the *Treated* groups _(with or without outcome)_ are now determined based on whether the outcome happened after the treatment. This allows for a more causal interpretation by ensuring that only post-treatment outcomes are considered in the analysis for treated individuals.

```{exec-literalinclude} scripts/03_treatment_effect.py
---
language: python
setup-lines: 1-36
lines: 41-49
---
```

:::{dropdown} Methods used in the snippet

- [`TreatmentEffect`](medmodels.treatment_effect.treatment_effect.TreatmentEffect){target="_blank"}: The TreatmentEffect class for analyzing treatment effects in medical records.
- [`builder()`](medmodels.treatment_effect.treatment_effect.TreatmentEffect.builder){target="_blank"}: Creates a TreatmentEffectBuilder instance for the TreatmentEffect class.
- [`with_treatment()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_treatment){target="_blank"}: Sets the treatment group for the treatment effect estimation.
- [`with_outcome()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_outcome){target="_blank"}: Sets the outcome group for the treatment effect estimation.
- [`with_time_attribute()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_time_attribute){target="_blank"}: Sets the time attribute to be used in the treatment effect estimation.
- [`build()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.build){target="_blank"}: Builds the treatment effect with all the provided configurations.

:::

You can also further customize the time-related properties, such as the *grace period*, the *follow-up period*, or whether we should exclude the patients in which there was an outcome before the treatment:

```{literalinclude} scripts/03_treatment_effect.py
---
language: python
lines: 52-61
---
```

:::{dropdown} Methods used in the snippet

- [`TreatmentEffect`](medmodels.treatment_effect.treatment_effect.TreatmentEffect){target="_blank"}: The TreatmentEffect class for analyzing treatment effects in medical records.
- [`builder()`](medmodels.treatment_effect.treatment_effect.TreatmentEffect.builder){target="_blank"}: Creates a TreatmentEffectBuilder instance for the TreatmentEffect class.
- [`with_treatment()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_treatment){target="_blank"}: Sets the treatment group for the treatment effect estimation.
- [`with_outcome()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_outcome){target="_blank"}: Sets the outcome group for the treatment effect estimation.
- [`with_grace_period()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_grace_period){target="_blank"}: Sets the grace period for the treatment effect estimation.
- [`with_follow_up_period()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_follow_up_period){target="_blank"}: Sets the follow-up period for the treatment effect estimation.
- [`with_outcome_before_treatment_exclusion()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_outcome_before_treatment_exclusion){target="_blank"}: Define whether we allow the outcome to exist before the treatment or not.
- [`build()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.build){target="_blank"}: Builds the treatment effect with all the provided configurations.

:::


*Washout periods* for specific drugs that may impact the test results can also be included:

```{literalinclude} scripts/03_treatment_effect.py
---
language: python
lines: 65-82
---
```

:::{dropdown} Methods used in the snippet

- [`TreatmentEffect`](medmodels.treatment_effect.treatment_effect.TreatmentEffect){target="_blank"}: The TreatmentEffect class for analyzing treatment effects in medical records.
- [`builder()`](medmodels.treatment_effect.treatment_effect.TreatmentEffect.builder){target="_blank"}: Creates a TreatmentEffectBuilder instance for the TreatmentEffect class.
- [`with_treatment()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_treatment){target="_blank"}: Sets the treatment group for the treatment effect estimation.
- [`with_outcome()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_outcome){target="_blank"}: Sets the outcome group for the treatment effect estimation.
- [`with_washout_period()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_washout_period_attribute){target="_blank"}: Sets the washout period for the treatment effect estimation.
- [`build()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.build){target="_blank"}: Builds the treatment effect with all the provided configurations.

:::

And we have also integrated matching algorithms, like *nearest neighbors* or *propensity matching* to conform control groups that can clearly resemble the treated population.

```{literalinclude} scripts/03_treatment_effect.py
---
language: python
lines: 85-93
---
```

:::{dropdown} Methods used in the snippet

- [`TreatmentEffect`](medmodels.treatment_effect.treatment_effect.TreatmentEffect){target="_blank"}: The TreatmentEffect class for analyzing treatment effects in medical records.
- [`builder()`](medmodels.treatment_effect.treatment_effect.TreatmentEffect.builder){target="_blank"}: Creates a TreatmentEffectBuilder instance for the TreatmentEffect class.
- [`with_treatment()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_treatment){target="_blank"}: Sets the treatment group for the treatment effect estimation.
- [`with_outcome()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_outcome){target="_blank"}: Sets the outcome group for the treatment effect estimation.
- [`with_nearest_neighbors_matching()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.with_nearest_neighbors_matching){target="_blank"}: Adjust the treatment effect estimate using nearest neighbors matching.
- [`build()`](medmodels.treatment_effect.builder.TreatmentEffectBuilder.build){target="_blank"}: Builds the treatment effect with all the provided configurations.

:::


## Estimating metrics

Once we have instantiated the Treatment Effect class with the desired properties, we can go on and estimate a lot of different metrics, such as:

- *Odds ratio*.

```{exec-literalinclude} scripts/03_treatment_effect.py
---
language: python
setup-lines: 1-36
lines: 96
---
```

:::{dropdown} Methods used in the snippet

- [`odds_ratio`](medmodels.treatment_effect.estimate.Estimate.odds_ratio){target="_blank"}: Calculates the odds ratio (OR).

:::


- *Relative risk*.

```{exec-literalinclude} scripts/03_treatment_effect.py
---
language: python
setup-lines: 1-36
lines: 99
---
```

:::{dropdown} Methods used in the snippet

- [`relative_risk`](medmodels.treatment_effect.estimate.Estimate.relative_risk){target="_blank"}: Calculates the relative risk (RR).

:::


- *Average treament effect*, where we calculate the difference between the outcome means of the treated and control sets for an outcome variable (e.g., _duration_days_).

```{exec-literalinclude} scripts/03_treatment_effect.py
---
language: python
setup-lines: 1-36
lines: 102
---
```

:::{dropdown} Methods used in the snippet

- [`average_treatment_effect`](medmodels.treatment_effect.estimate.Estimate.average_treatment_effect){target="_blank"}: Calculates the Average Treatment Effect (ATE).

:::

## Generating a full metrics report

You can also create a report with all the possible metrics in the treatment effect class:

```{exec-literalinclude} scripts/03_treatment_effect.py
---
language: python
setup-lines: 1-36
lines: 105
---
```

And also another report with all the continuous estimators.

```{exec-literalinclude} scripts/03_treatment_effect.py
---
language: python
setup-lines: 1-36
lines: 108
---
```

## Full example Code

The full code examples for this chapter can be found here:

```{literalinclude}  scripts/03_treatment_effect.py
---
language: python
lines: 3-108
---
```
