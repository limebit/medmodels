from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import medmodels.treatment_effect.treatment_effect as tee
from medmodels.medrecord.querying import NodeQuery
from medmodels.treatment_effect.builder import TreatmentEffectBuilder


class TestTreatmentEffectBuilder:
    def test_with_treatment(self) -> None:
        builder = TreatmentEffectBuilder()

        assert builder._treatment is None

        builder.with_treatment("treatment_group")
        assert builder._treatment == "treatment_group"

    def test_with_outcome(self) -> None:
        builder = TreatmentEffectBuilder()

        assert builder._outcome is None

        builder.with_outcome("outcome_group")
        assert builder._outcome == "outcome_group"

    def test_with_patients_group(self) -> None:
        builder = TreatmentEffectBuilder()

        assert builder._patients_group is None

        builder.with_patients_group("all_patients")
        assert builder._patients_group == "all_patients"

    def test_with_time_attribute(self) -> None:
        builder = TreatmentEffectBuilder()

        assert builder._time_attribute is None

        time_attribute = "time"
        builder.with_time_attribute(time_attribute)
        assert builder._time_attribute == time_attribute

    def test_with_washout_period(self) -> None:
        builder = TreatmentEffectBuilder()

        assert builder._washout_period_days is None
        assert builder._washout_period_reference is None

        washout_days = {"treatment": 30}
        washout_reference = "first"

        builder.with_washout_period(days=washout_days, reference=washout_reference)
        assert builder._washout_period_days == washout_days
        assert builder._washout_period_reference == washout_reference

        # Test setting only days
        builder_days_only = TreatmentEffectBuilder()
        builder_days_only.with_washout_period(days=washout_days)
        assert builder_days_only._washout_period_days == washout_days
        assert builder_days_only._washout_period_reference is None

        # Test setting only reference
        builder_reference_only = TreatmentEffectBuilder()
        builder_reference_only.with_washout_period(reference=washout_reference)
        assert builder_reference_only._washout_period_reference == washout_reference
        assert builder_reference_only._washout_period_days is None

    def test_with_grace_period(self) -> None:
        builder = TreatmentEffectBuilder()

        assert builder._grace_period_days is None
        assert builder._grace_period_reference is None

        grace_days = 7
        grace_reference = "last"

        builder.with_grace_period(days=grace_days, reference=grace_reference)
        assert builder._grace_period_days == grace_days
        assert builder._grace_period_reference == grace_reference

        # Test setting only days
        builder_days_only = TreatmentEffectBuilder()
        builder_days_only.with_grace_period(days=grace_days)
        assert builder_days_only._grace_period_days == grace_days
        assert builder_days_only._grace_period_reference is None

        # Test setting only reference
        builder_ref_only = TreatmentEffectBuilder()
        builder_ref_only.with_grace_period(reference=grace_reference)
        assert builder_ref_only._grace_period_reference == grace_reference
        assert builder_ref_only._grace_period_days is None

    def test_with_follow_up_period(self) -> None:
        builder = TreatmentEffectBuilder()

        assert builder._follow_up_period_days is None
        assert builder._follow_up_period_reference is None

        follow_up_days = 365
        follow_up_reference = "first"

        builder.with_follow_up_period(
            days=follow_up_days, reference=follow_up_reference
        )
        assert builder._follow_up_period_days == follow_up_days
        assert builder._follow_up_period_reference == follow_up_reference

        # Test setting only days
        builder_days_only = TreatmentEffectBuilder()
        builder_days_only.with_follow_up_period(days=follow_up_days)
        assert builder_days_only._follow_up_period_days == follow_up_days
        assert builder_days_only._follow_up_period_reference is None

        # Test setting only reference
        builder_reference_only = TreatmentEffectBuilder()
        builder_reference_only.with_follow_up_period(reference=follow_up_reference)
        assert builder_reference_only._follow_up_period_reference == follow_up_reference
        assert builder_reference_only._follow_up_period_days is None

    def test_with_outcome_before_treatment_exclusion(self) -> None:
        builder = TreatmentEffectBuilder()

        assert builder._outcome_before_treatment_days is None

        exclusion_days = 90
        builder.with_outcome_before_treatment_exclusion(days=exclusion_days)
        assert builder._outcome_before_treatment_days == exclusion_days

    def test_filter_controls(self) -> None:
        builder = TreatmentEffectBuilder()

        assert builder._filter_controls_query is None

        mock_query = MagicMock(spec=NodeQuery)
        builder.filter_controls(query=mock_query)
        assert builder._filter_controls_query is mock_query

    def test_with_propensity_matching(self) -> None:
        builder = TreatmentEffectBuilder()

        # Check initial state
        assert builder._matching_method is None
        assert builder._matching_essential_covariates is None
        assert builder._matching_one_hot_covariates is None
        assert builder._matching_number_of_neighbors is None
        assert builder._matching_model is None
        assert builder._matching_hyperparameters is None

        # Call with defaults
        builder.with_propensity_matching()
        assert builder._matching_method == "propensity"
        assert builder._matching_essential_covariates is None
        assert builder._matching_one_hot_covariates is None
        assert builder._matching_model == "logit"
        assert builder._matching_number_of_neighbors == 1
        assert builder._matching_hyperparameters is None

        # Call with specific values
        builder2 = TreatmentEffectBuilder()
        essential_covariates = ["age", "gender"]
        one_hot_covariates = ["gender"]
        model = "dec_tree"
        neighbors = 5
        hyperparameters = {"n_estimators": 100}
        builder2.with_propensity_matching(
            essential_covariates=essential_covariates,
            one_hot_covariates=one_hot_covariates,
            model=model,
            number_of_neighbors=neighbors,
            hyperparameters=hyperparameters,
        )
        assert builder2._matching_method == "propensity"
        assert builder2._matching_essential_covariates == essential_covariates
        assert builder2._matching_one_hot_covariates == one_hot_covariates
        assert builder2._matching_model == model
        assert builder2._matching_number_of_neighbors == neighbors
        assert builder2._matching_hyperparameters == hyperparameters

    def test_with_nearest_neighbors_matching(self) -> None:
        builder = TreatmentEffectBuilder()

        # Check initial state
        assert builder._matching_method is None
        assert builder._matching_essential_covariates is None
        assert builder._matching_one_hot_covariates is None
        assert builder._matching_number_of_neighbors is None

        # Call with defaults
        builder.with_nearest_neighbors_matching()
        assert builder._matching_method == "nearest_neighbors"
        assert builder._matching_essential_covariates is None
        assert builder._matching_one_hot_covariates is None
        assert builder._matching_number_of_neighbors == 1
        assert builder._matching_model is None
        assert builder._matching_hyperparameters is None

        # Call with specific values
        builder2 = TreatmentEffectBuilder()
        essential_covariates = ["age", "gender"]
        one_hot_covariates = ["age"]
        neighbors = 3
        builder2.with_nearest_neighbors_matching(
            essential_covariates=essential_covariates,
            one_hot_covariates=one_hot_covariates,
            number_of_neighbors=neighbors,
        )
        assert builder2._matching_method == "nearest_neighbors"
        assert builder2._matching_essential_covariates == essential_covariates
        assert builder2._matching_one_hot_covariates == one_hot_covariates
        assert builder2._matching_number_of_neighbors == neighbors
        assert builder._matching_model is None
        assert builder._matching_hyperparameters is None

    def test_build(self) -> None:
        builder = TreatmentEffectBuilder()

        # Set mandatory fields
        treatment = "treatment"
        outcome = "outcome"
        builder.with_treatment(treatment).with_outcome(outcome)

        # Set optional fields
        patients_group = "patient"
        time_attribute = "time"
        washout_days = {"treatment": 10}
        washout_ref = "first"
        grace_days = 5
        grace_ref = "last"
        follow_up_days = 180
        follow_up_ref = "first"
        outcome_before_days = 30
        filter_query = MagicMock(spec=NodeQuery)
        matching_method = "propensity"
        essential_covariates = ["age"]
        one_hot_covariates = ["gender"]
        matching_model = "logit"
        neighbors = 2
        hyperparams = {"C": 1.0}

        builder.with_patients_group(patients_group).with_time_attribute(
            time_attribute
        ).with_washout_period(
            days=washout_days, reference=washout_ref
        ).with_grace_period(days=grace_days, reference=grace_ref).with_follow_up_period(
            days=follow_up_days, reference=follow_up_ref
        ).with_outcome_before_treatment_exclusion(
            days=outcome_before_days
        ).filter_controls(query=filter_query).with_propensity_matching(
            essential_covariates=essential_covariates,
            one_hot_covariates=one_hot_covariates,
            model=matching_model,
            number_of_neighbors=neighbors,
            hyperparameters=hyperparams,
        )

        # Build the object
        treatment_effect = builder.build()

        # Check the type
        assert isinstance(treatment_effect, tee.TreatmentEffect)

        # Check if attributes were transferred correctly
        assert treatment_effect._treatments_group == treatment
        assert treatment_effect._outcomes_group == outcome
        assert treatment_effect._patients_group == patients_group
        assert treatment_effect._time_attribute == time_attribute
        assert treatment_effect._washout_period_days == washout_days
        assert treatment_effect._washout_period_reference == washout_ref
        assert treatment_effect._grace_period_days == grace_days
        assert treatment_effect._grace_period_reference == grace_ref
        assert treatment_effect._follow_up_period_days == follow_up_days
        assert treatment_effect._follow_up_period_reference == follow_up_ref
        assert treatment_effect._outcome_before_treatment_days == outcome_before_days
        assert treatment_effect._filter_controls_query == filter_query
        assert treatment_effect._matching_method == matching_method
        assert treatment_effect._matching_essential_covariates == essential_covariates
        assert treatment_effect._matching_one_hot_covariates == one_hot_covariates
        assert treatment_effect._matching_model == matching_model
        assert treatment_effect._matching_number_of_neighbors == neighbors
        assert treatment_effect._matching_hyperparameters == hyperparams

    def test_build_minimal(self) -> None:
        builder = TreatmentEffectBuilder()
        treatment = "treatment"
        outcome = "outcome"

        # Set only mandatory fields
        builder.with_treatment(treatment).with_outcome(outcome)

        treatment_effect = builder.build()

        assert isinstance(treatment_effect, tee.TreatmentEffect)
        assert treatment_effect._treatments_group == treatment
        assert treatment_effect._outcomes_group == outcome

    def test_build_invalid(self) -> None:
        builder = TreatmentEffectBuilder()

        # Attempt to build without mandatory fields
        with pytest.raises(
            ValueError,
            match="Treatment and outcome groups must be set before "
            + "building the treatment effect.",
        ):
            builder.build()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
