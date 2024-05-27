from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

from medmodels.medrecord.querying import NodeOperation

if TYPE_CHECKING:
    from medmodels.treatment_effect_estimation.treatment_effect import TreatmentEffect


class Configure:
    _treatment_effect: TreatmentEffect

    def __init__(self, treatment_effect: TreatmentEffect) -> None:
        self._treatment_effect = treatment_effect

    def washout_period(
        self,
        washout_period_days: Optional[Dict[str, int]] = None,
        washout_period_reference: str = "first",
    ) -> None:
        if washout_period_reference not in ["first", "last"]:
            # TODO: add "both"
            raise ValueError(
                "The washout_period_reference parameter must be either 'first' or 'last'"
            )
        if washout_period_days is not None:
            self._treatment_effect._washout_period_days = washout_period_days
        self._treatment_effect._washout_period_reference = washout_period_reference

        self._treatment_effect.groups_sorted = False

    def grace_period(
        self,
        grace_period_days: Optional[int] = None,
        grace_period_reference: str = "last",
    ) -> None:
        """Sets the grace period for the treatment effect estimation. The grace period
        is the period of time after the treatment that is not considered in the
        estimation.

        Args:
            grace_period_days (int, optional): The duration of the grace period in days.
                If None, the duration is left as it was. Defaults to None.
            grace_period_reference (str, optional): The reference point for the grace
                period. Must be either 'first' or 'last'. Defaults to 'last'.

        Raises:
            ValueError: If the grace_period_reference parameter is not 'first' or 'last'.
        """
        if grace_period_reference not in ["first", "last"]:
            raise ValueError(
                "The grace_period_reference parameter must be either 'first' or 'last'."
            )
        if grace_period_days is not None:
            self._treatment_effect._grace_period_days = grace_period_days
        self._treatment_effect._grace_period_reference = grace_period_reference

        self._treatment_effect.groups_sorted = False

    def followup_duration(
        self,
        follow_up_period_days: Optional[int] = None,
        follow_up_reference: str = "last",
    ) -> None:
        """Sets the follow-up period for the treatment effect estimation.

        Args:
            follow_up_period_days (int, optional): The duration of the follow-up period
                in days. If None, the duration is left as it was. Defaults to None.
            follow_up_from (str, optional): The reference point for the follow-up
                period. Must be either 'first' or 'last'. Defaults to 'last'.

        Raises:
            ValueError: If the follow_up_from parameter is not 'first' or 'last'.
        """
        if follow_up_reference not in ["first", "last"]:
            raise ValueError(
                "The follow_up_from parameter must be either 'first' or 'last'."
            )
        if follow_up_period_days is not None:
            self._treatment_effect._follow_up_period_days = follow_up_period_days
        self._treatment_effect._follow_up_reference = follow_up_reference

        self._treatment_effect.groups_sorted = False

    def outcome_before_treatment(
        self, outcome_before_treatment_days: Optional[int] = None
    ) -> None:
        """Define whether we allow the outcome to exist before the treatment or not.
        The outcome_before_treatment_days parameter is used to set the number of days
        before the treatment that the outcome should not exist.

        Args:
            outcome_before_treatment_days (int, optional): The number of days before the
                treatment that the outcome should not exist. If None, the outcome is
                allowed to exist before the treatment. Defaults to None.
        """
        self._treatment_effect._outcome_before_treatment_days = (
            outcome_before_treatment_days
        )

        self._treatment_effect.groups_sorted = False

    def filter_controls(self, operation: Optional[NodeOperation] = None) -> None:
        """Filter the control group based on the provided operation.

        Args:
            operation (NodeOperation): The operation to be applied to the control group.
        """
        self._treatment_effect._filter_controls = operation

        self._treatment_effect.groups_sorted = False

    def default_variables(self) -> None:
        """Sets all configuration parameters to their default values."""
        self._treatment_effect._initialize_attributes()

        self._treatment_effect.groups_sorted = False
