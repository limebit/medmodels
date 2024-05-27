"""Matching class template for patient matching algorithms"""

from typing import Set

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import NodeIndex


class Matching:
    """Base class for Patient Matching algorithms"""

    def __init__(
        self,
        medrecord: MedRecord,
        treated_group: Set[NodeIndex],
        control_group: Set[NodeIndex],
    ) -> None:
        """
        Initializes the Matching object with the MedRecord object and the treated and
        control groups.

        Args:
            medrecord (MedRecord): An instance of the MedRecord class containing patient
                medical data.
            treated_group (Set[NodeIndex]): A set containing the IDs of patients in the
                treated group.
            control_group (Set[NodeIndex]): A set containing the IDs of patients in the
                control group.

        Returns:
            None
        """
        self.medrecord = medrecord
        self.treated_group = treated_group
        assert len(self.treated_group) > 0, "Treated group cannot be empty"
        self.control_group = control_group
        assert len(self.control_group) > 0, "Control group cannot be empty"

        self.matched_control = set()
