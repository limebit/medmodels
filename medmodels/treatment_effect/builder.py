"""This module contains the TreatmentEffectBuilder class."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import medmodels.treatment_effect.treatment_effect as tee

if TYPE_CHECKING:
    from medmodels.medrecord.querying import NodeQuery
    from medmodels.medrecord.types import (
        Group,
        MedRecordAttribute,
        MedRecordAttributeInputList,
    )
    from medmodels.treatment_effect.matching.algorithms.propensity_score import Model
    from medmodels.treatment_effect.matching.matching import MatchingMethod


class TreatmentEffectBuilder:
    """Builder class for the TreatmentEffect object.

    The TreatmentEffectBuilder class is used to build a TreatmentEffect object with
    the desired configurations for the treatment effect estimation using a builder
    pattern.

    By default, it configures a static treatment effect estimation. To configure a
    time-dependent treatment effect estimation, the time_attribute must be set.
    """

    _treatment: Optional[Group]
    _outcome: Optional[Group]

    _patients_group: Optional[Group]
    _time_attribute: Optional[MedRecordAttribute]

    _washout_period_days: Optional[Dict[str, int]]
    _washout_period_reference: Optional[Literal["first", "last"]]

    _grace_period_days: Optional[int]
    _grace_period_reference: Optional[Literal["first", "last"]]

    _follow_up_period_days: Optional[int]
    _follow_up_period_reference: Optional[Literal["first", "last"]]

    _outcome_before_treatment_days: Optional[int]

    _filter_controls_query: Optional[NodeQuery]

    _matching_method: Optional[MatchingMethod]
    _matching_essential_covariates: Optional[MedRecordAttributeInputList]
    _matching_one_hot_covariates: Optional[MedRecordAttributeInputList]
    _matching_model: Optional[Model]
    _matching_number_of_neighbors: Optional[int]
    _matching_hyperparameters: Optional[Dict[str, Any]]

    def __init__(self) -> None:
        """Initializes the TreatmentEffectBuilder with all attributes set to None."""
        self._treatment = None
        self._outcome = None
        self._patients_group = None
        self._time_attribute = None
        self._washout_period_days = None
        self._washout_period_reference = None
        self._grace_period_days = None
        self._grace_period_reference = None
        self._follow_up_period_days = None
        self._follow_up_period_reference = None
        self._outcome_before_treatment_days = None
        self._filter_controls_query = None
        self._matching_method = None
        self._matching_essential_covariates = None
        self._matching_one_hot_covariates = None
        self._matching_model = None
        self._matching_number_of_neighbors = None
        self._matching_hyperparameters = None

    def with_treatment(self, treatment: Group) -> TreatmentEffectBuilder:
        """Sets the treatment group for the treatment effect estimation.

        Args:
            treatment (Group): The treatment group.

        Returns:
            TreatmentEffectBuilder: The current instance of the TreatmentEffectBuilder.
        """
        self._treatment = treatment

        return self

    def with_outcome(self, outcome: Group) -> TreatmentEffectBuilder:
        """Sets the outcome group for the treatment effect estimation.

        Args:
            outcome (Group): The group to be used as the outcome.

        Returns:
            TreatmentEffectBuilder: The current instance of the TreatmentEffectBuilder
                with updated outcome group.
        """
        self._outcome = outcome

        return self

    def with_patients_group(self, group: Group) -> TreatmentEffectBuilder:
        """Sets the group of patients to be used in the treatment effect estimation.

        Args:
            group (Group): The group of patients.

        Returns:
            TreatmentEffectBuilder: The current instance of the TreatmentEffectBuilder
                with updated patients group.
        """
        self._patients_group = group

        return self

    def with_time_attribute(
        self, attribute: MedRecordAttribute
    ) -> TreatmentEffectBuilder:
        """Sets the time attribute to be used in the treatment effect estimation.

        It turs the treatment effect estimation from a static to a time-dependent
        analysis.

        Args:
            attribute (MedRecordAttribute): The time attribute.

        Returns:
            TreatmentEffectBuilder: The current instance of the TreatmentEffectBuilder
                with updated time attribute.
        """
        self._time_attribute = attribute

        return self

    def with_washout_period(
        self,
        days: Optional[Dict[str, int]] = None,
        reference: Optional[Literal["first", "last"]] = None,
    ) -> TreatmentEffectBuilder:
        """Sets the washout period for the treatment effect estimation.

        The washout period is the period of time before the treatment that is not
        considered in the estimation.

        Args:
            days (Optional[Dict[str, int]], optional): The duration of the washout
                period in days. If None, the duration is left as it was. Defaults to
                None.
            reference (Optional[Literal['first', 'last']], optional): The reference
                point for the washout period. Must be either 'first' or 'last'.
                Defaults to None.

        Returns:
            TreatmentEffectBuilder: The current instance of the TreatmentEffectBuilder
                with updated time attribute.
        """
        if days is not None:
            self._washout_period_days = days
        if reference is not None:
            self._washout_period_reference = reference

        return self

    def with_grace_period(
        self,
        days: Optional[int] = None,
        reference: Optional[Literal["first", "last"]] = None,
    ) -> TreatmentEffectBuilder:
        """Sets the grace period for the treatment effect estimation.

        The grace period is the period of time after the treatment that is not
        considered in the estimation.

        Args:
            days (Optional[int], optional): The duration of the grace period in days.
                If None, the duration is left as it was. Defaults to 0.
            reference (Optional[Literal['first', 'last']], optional): The reference
                point for the grace period. Must be either 'first' or 'last'.
                Defaults to None.

        Returns:
            TreatmentEffectBuilder: The current instance of the TreatmentEffectBuilder
                with updated time attribute.
        """
        if days is not None:
            self._grace_period_days = days

        if reference is not None:
            self._grace_period_reference = reference

        return self

    def with_follow_up_period(
        self,
        days: Optional[int] = None,
        reference: Optional[Literal["first", "last"]] = None,
    ) -> TreatmentEffectBuilder:
        """Sets the follow-up period for the treatment effect estimation.

        Args:
            days (Optional[int], optional): The duration of the follow-up period
                in days. If None, the duration is left as it was. Defaults to 365.
            reference (Optional[Literal['first', 'last']], optional): The reference
                point for the follow-up period. Must be either 'first' or 'last'.
                Defaults to None.

        Returns:
            TreatmentEffectBuilder: The current instance of the TreatmentEffectBuilder
                with updated time attribute.
        """
        if days is not None:
            self._follow_up_period_days = days

        if reference is not None:
            self._follow_up_period_reference = reference

        return self

    def with_outcome_before_treatment_exclusion(
        self, days: int
    ) -> TreatmentEffectBuilder:
        """Define whether we allow the outcome to exist before the treatment or not.

        The outcome_before_treatment_days parameter is used to set the number of days
        before the treatment that the outcome should not exist. If not set, the outcome
        is allowed to exist before the treatment.

        Args:
            days (int): The number of days before the treatment that the outcome should
                not exist.

        Returns:
            TreatmentEffectBuilder: The current instance of the TreatmentEffectBuilder
                with updated time attribute.
        """
        self._outcome_before_treatment_days = days

        return self

    def filter_controls(self, query: NodeQuery) -> TreatmentEffectBuilder:
        """Filter the control group based on the provided query.

        Args:
            query (NodeQuery): The query to be applied to the control group.

        Returns:
            TreatmentEffectBuilder: The current instance of the TreatmentEffectBuilder
                with updated time attribute.
        """
        self._filter_controls_query = query

        return self

    def with_propensity_matching(
        self,
        essential_covariates: Optional[MedRecordAttributeInputList] = None,
        one_hot_covariates: Optional[MedRecordAttributeInputList] = None,
        model: Model = "logit",
        number_of_neighbors: int = 1,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> TreatmentEffectBuilder:
        """Adjust the treatment effect estimate using propensity score matching.

        Args:
            essential_covariates (Optional[MedRecordAttributeInputList], optional):
                Covariates that are essential for matching. Defaults to None.
            one_hot_covariates (Optional[MedRecordAttributeInputList], optional):
                Covariates that are one-hot encoded for matching. Defaults to None.
            model (Model, optional): Model to choose for the matching. Defaults to
                "logit".
            number_of_neighbors (int, optional): Number of neighbors to consider
                for the matching. Defaults to 1.
            hyperparameters (Optional[Dict[str, Any]], optional): Hyperparameters for
                the matching model. Defaults to None.

        Returns:
            TreatmentEffectBuilder: The current instance of the TreatmentEffectBuilder
                with updated matching configurations.
        """
        self._matching_method = "propensity"
        self._matching_essential_covariates = essential_covariates
        self._matching_one_hot_covariates = one_hot_covariates
        self._matching_model = model
        self._matching_number_of_neighbors = number_of_neighbors
        self._matching_hyperparameters = hyperparameters

        return self

    def with_nearest_neighbors_matching(
        self,
        essential_covariates: Optional[MedRecordAttributeInputList] = None,
        one_hot_covariates: Optional[MedRecordAttributeInputList] = None,
        number_of_neighbors: int = 1,
    ) -> TreatmentEffectBuilder:
        """Adjust the treatment effect estimate using nearest neighbors matching.

        Args:
            essential_covariates (Optional[MedRecordAttributeInputList], optional):
                Covariates that are essential for matching. Defaults to None.
            one_hot_covariates (Optional[MedRecordAttributeInputList], optional):
                Covariates that are one-hot encoded for matching. Defaults to None.
            number_of_neighbors (int, optional): Number of neighbors to consider for the
                matching. Defaults to 1.

        Returns:
            TreatmentEffectBuilder: The current instance of the TreatmentEffectBuilder
                with updated matching configurations.
        """
        self._matching_method = "nearest_neighbors"
        self._matching_essential_covariates = essential_covariates
        self._matching_one_hot_covariates = one_hot_covariates
        self._matching_number_of_neighbors = number_of_neighbors

        return self

    def build(self) -> tee.TreatmentEffect:
        """Builds the treatment effect with all the provided configurations.

        Returns:
            tee.TreatmentEffect: treatment effect object

        Raises:
            ValueError: If the treatment and outcome groups are not set before
                building the treatment effect.
        """
        if self._treatment is None or self._outcome is None:
            msg = (
                "Treatment and outcome groups must be set before "
                + "building the treatment effect."
            )
            raise ValueError(msg)

        # Only pass attributes that are not None
        config = {k.lstrip("_"): v for k, v in vars(self).items() if v is not None}

        treatment_effect = tee.TreatmentEffect.__new__(tee.TreatmentEffect)
        tee.TreatmentEffect._set_configuration(treatment_effect, **config)

        return treatment_effect
