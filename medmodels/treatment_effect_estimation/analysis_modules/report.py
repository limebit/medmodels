from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Dict

from medmodels.medrecord.medrecord import MedRecord

if TYPE_CHECKING:
    from medmodels.treatment_effect_estimation.treatment_effect import TreatmentEffect


class Report:
    _treatment_effect: TreatmentEffect

    def __init__(self, treatment_effect: TreatmentEffect) -> None:
        self._treatment_effect = treatment_effect

    def full_report(self, medrecord: MedRecord) -> Dict[str, Any]:
        """Generates a full report of the treatment effect estimation.

        Returns:
            Dict[str, Any]: A dictionary containing the results of all available
                estimation methods.
        """
        results = {}
        methods = inspect.getmembers(
            self._treatment_effect.estimate, predicate=inspect.ismethod
        )
        for name, method in methods:
            if not name.startswith("_"):
                try:
                    results[name] = method(medrecord=medrecord)
                except Exception as e:
                    results[name] = f"Error: {e}"
        return results
