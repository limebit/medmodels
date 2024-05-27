from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medmodels.treatment_effect_estimation.treatment_effect import TreatmentEffect


class Adjust:
    _treatment_effect: TreatmentEffect

    def __init__(self, treatment_effect: TreatmentEffect) -> None:
        self._treatment_effect = treatment_effect
