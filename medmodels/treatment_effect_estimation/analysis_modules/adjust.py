from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Set, Tuple

from medmodels import MedRecord
from medmodels.medrecord.types import NodeIndex
from medmodels.treatment_effect_estimation.analysis_modules.adjust_modules.neighbors_class import (
    NeighborsMatching,
)
from medmodels.treatment_effect_estimation.analysis_modules.adjust_modules.propensity_class import (
    PropensityMatching,
)

if TYPE_CHECKING:
    import sys

    from medmodels.treatment_effect_estimation.treatment_effect import TreatmentEffect

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
    else:
        from typing_extensions import TypeAlias


MatchingMethod: TypeAlias = Literal["propensity", "nearest_neighbors"]


class Adjust:
    _treatment_effect: TreatmentEffect

    def __init__(self, treatment_effect: TreatmentEffect) -> None:
        self._treatment_effect = treatment_effect

    def _apply_matching(
        self,
        medrecord: MedRecord,
        treated_group: Set[NodeIndex],
        control_true: Set[NodeIndex],
        control_false: Set[NodeIndex],
        method: Optional[MatchingMethod] = None,
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex]]:
        """
        Update the treatment effect object with the matched controls.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.
            treated_group (Set[NodeIndex]): The set of all patients in the treatment
                group.
            control_true (Set[NodeIndex]): The set of patients in the control group with
                the outcome of interest.
            control_false (Set[NodeIndex]): The set of patients in the control group
                without the outcome of interest.
            method (Optional[MatchingMethod]): The method to use for matching. Defaults
                to None.

        Returns:
            Tuple[Set[NodeIndex], Set[NodeIndex]]: The updated control_true and
                control_false sets after matching.
        """
        if method is None:
            return control_true, control_false

        # If it is not None, apply the matching method
        method_function = getattr(self, method)
        control_group = control_true | control_false
        matched_controls = method_function(medrecord, treated_group, control_group)
        control_true, control_false = self._treatment_effect._find_controls(
            medrecord=medrecord,
            control_group=matched_controls,
            treated_group=treated_group,
        )

        return control_true, control_false

    def propensity(
        self,
        medrecord: MedRecord,
        treated_group: Set[NodeIndex],
        control_group: Set[NodeIndex],
    ) -> Set[NodeIndex]:
        """
        Adjust the treatment effect estimate using propensity score matching.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.
            treated_group (Set[NodeIndex]): The set of all patients in the treatment
                group.
            control_group (Set[NodeIndex]): The set of all patients in the control
                group.

        Returns:
            Set[NodeIndex]: The set of matched control subjects.
        """
        propensity_class = PropensityMatching(
            medrecord,
            treated_group,
            control_group,
            essential_covariates=self._treatment_effect._matching_essential_covariates,
            one_hot_covariates=self._treatment_effect._matching_one_hot_covariates,
            model=self._treatment_effect._matching_model,
            distance_metric=self._treatment_effect._matching_distance_metric,
            number_of_neighbors=self._treatment_effect._matching_number_of_neighbors,
            hyperparam=self._treatment_effect._matching_hyperparam,
        )

        return propensity_class.matched_controls

    def nearest_neighbors(
        self,
        medrecord: MedRecord,
        treated_group: Set[NodeIndex],
        control_group: Set[NodeIndex],
    ) -> Set[NodeIndex]:
        """
        Adjust the treatment effect estimate using nearest neighbor matching.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.
            treated_group (Set[NodeIndex]): The set of all patients in the treatment
                group.
            control_group (Set[NodeIndex]): The set of all patients in the control
                group.

        Returns:
            Set[NodeIndex]: The set of matched control subjects.
        """
        propensity_class = NeighborsMatching(
            medrecord,
            treated_group,
            control_group,
            essential_covariates=self._treatment_effect._matching_essential_covariates,
            one_hot_covariates=self._treatment_effect._matching_one_hot_covariates,
            distance_metric=self._treatment_effect._matching_distance_metric,
            number_of_neighbors=self._treatment_effect._matching_number_of_neighbors,
        )
        return propensity_class.matched_controls
