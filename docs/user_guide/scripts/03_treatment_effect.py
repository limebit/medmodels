# ruff: noqa: D100, D103
from medmodels import MedRecord
from medmodels.medrecord.querying import (
    NodeIndicesOperand,
    NodeOperand,
)
from medmodels.treatment_effect import TreatmentEffect

medrecord = MedRecord().from_advanced_example_dataset()


def find_alendronic_drugs(node: NodeOperand) -> NodeIndicesOperand:
    node.in_group("drug")
    description = node.attribute("description")
    description.lowercase()
    description.contains("alendronic")

    return node.index()


def find_fracture_diagnoses(node: NodeOperand) -> NodeIndicesOperand:
    node.in_group("diagnosis")
    description = node.attribute("description")
    description.lowercase()
    description.contains("fracture")

    return node.index()


medrecord.unfreeze_schema()
medrecord.add_group("alendronic", find_alendronic_drugs)
medrecord.add_group("fracture", find_fracture_diagnoses)

treatment_effect_basic = (
    TreatmentEffect.builder()
    .with_treatment("alendronic")
    .with_outcome("fracture")
    .with_patients_group("patient")
    .build()
)

treatment_effect_basic.estimate.subject_counts(medrecord)

# Adding time attribute to the treatment effect
treatment_effect_with_time = (
    TreatmentEffect.builder()
    .with_treatment("alendronic")
    .with_outcome("fracture")
    .with_time_attribute("time")
    .build()
)

treatment_effect_with_time.estimate.subject_counts(medrecord)

# Highly customized treatment effect instance
treatment_effect_customized = (
    TreatmentEffect.builder()
    .with_treatment("alendronic")
    .with_outcome("fracture")
    .with_time_attribute("time")
    .with_grace_period(days=30)
    .with_follow_up_period(days=365)
    .with_outcome_before_treatment_exclusion(days=15)
    .build()
)


# Using washout drugs (drugs that should not be taken before treatment)
def find_corticosteroids(node: NodeOperand) -> NodeIndicesOperand:
    node.in_group("drug")
    description = node.attribute("description")
    description.lowercase()
    description.contains("sone")

    return node.index()


medrecord.add_group("corticosteroids", find_corticosteroids)

treatment_effect_with_washout = (
    TreatmentEffect.builder()
    .with_treatment("alendronic")
    .with_outcome("fracture")
    .with_washout_period({"corticosteroids": 15})
    .build()
)

# Using matching algorithms
treatment_effect_matching = (
    TreatmentEffect.builder()
    .with_treatment("alendronic")
    .with_outcome("fracture")
    .with_nearest_neighbors_matching(
        essential_covariates=["age", "gender"], number_of_neighbors=2
    )
    .build()
)

treatment_effect_matching.estimate.subject_counts(medrecord)


# Using queries to filter controls
def query_patients_over_70(node: NodeOperand) -> NodeIndicesOperand:
    node.in_group("patient")
    node.attribute("age").greater_than(70)

    return node.index()


treatment_effect_over_70 = (
    TreatmentEffect.builder()
    .with_treatment("alendronic")
    .with_outcome("fracture")
    .filter_controls(query_patients_over_70)
    .build()
)

treatment_effect_over_70.estimate.subject_counts(medrecord)

# Estimating odds ratio
treatment_effect_over_70.estimate.odds_ratio(medrecord)

# Relative risk estimation
treatment_effect_over_70.estimate.relative_risk(medrecord)

# Average Treatment Effect
treatment_effect_over_70.estimate.average_treatment_effect(medrecord, "duration_days")

# Full report of treatment effect estimation
treatment_effect_over_70.report.full_report(medrecord)

# Continuous treatment effect estimators report
treatment_effect_over_70.report.continuous_estimators_report(medrecord, "duration_days")
