from typing import Set

from medmodels.dataclass.dataclass import MedRecord


class Matching:
    def __init__(
        self,
        medrecord: MedRecord,
        treated_group: Set[str],
        control_group: Set[str],
    ) -> None:
        """Base class for Patient Matching algorithms

        :param medrecord: MedRecord object
        :type medrecord: MedRecord
        :param treated_group: set with the IDs of the treated patients
        :type treated_group: Set[str]
        :param control_group: set with the IDs of the control patients
        :type control_group: Set[str]
        """
        self.medrecord = medrecord
        assert isinstance(medrecord, MedRecord), "medrecord must be a MedRecord object"
        self.treated_group = treated_group
        assert len(self.treated_group) > 0, "Treated group cannot be empty"
        self.control_group = control_group
        assert len(self.control_group) > 0, "Control group cannot be empty"

        self.matched_control = set()
