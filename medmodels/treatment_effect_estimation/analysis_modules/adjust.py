from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Set, Literal, Tuple
from typing_extensions import TypeAlias

from medmodels import MedRecord
from medmodels.medrecord.types import (
    NodeIndex,
)

if TYPE_CHECKING:
    from medmodels.treatment_effect_estimation.treatment_effect import TreatmentEffect


MatchingMethod: TypeAlias = Literal["propensity", "nearest_neighbors"]


class Adjust:
    _treatment_effect: TreatmentEffect

    def __init__(self, treatment_effect: TreatmentEffect) -> None:
        self._treatment_effect = treatment_effect

    def _apply_matching(
        self,
        method: Optional[MatchingMethod],
        medrecord: MedRecord,
        treatment_all: Set[NodeIndex],
        control_true: Set[NodeIndex],
        control_false: Set[NodeIndex],
    ) -> Tuple[Set[NodeIndex], Set[NodeIndex]]:
        """
        Update the treatment effect object with the matched controls.

        Args:
            medrecord (MedRecord): The MedRecord object containing the data.
            treatment_all (Set[NodeIndex]): The set of all patients in the treatment
                group.
            control_true (Set[NodeIndex]): The set of patients in the control group with
                the outcome of interest.
            control_false (Set[NodeIndex]): The set of patients in the control group
                without the outcome of interest.
            control_false (Set[NodeIndex]): The set of patients in the control group
                without the outcome of interest.

        Returns:
            Tuple[Set[NodeIndex], Set[NodeIndex]]: The updated control_true and
                control_false sets after matching.
        """
        if method is None:
            return control_true, control_false

        # If it is not None, apply the matching method
        method_function = getattr(self, method)
        control_true, control_false = method_function(
            medrecord, treatment_all, control_true, control_false
        )

        return control_true, control_false
