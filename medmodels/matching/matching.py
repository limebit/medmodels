from typing import Set

from medmodels.dataclass.dataclass import MedRecord


class Matching:
    def __init__(
        self,
        medrecord: MedRecord,
        treated_group: Set[str],
        control_group: Set[str],
    ) -> None:
        """
        Initializes the base class for patient matching algorithms, ensuring that both
        the treated and control groups are non-empty and that a valid `MedRecord` object
        is provided.

        This class sets the foundation for matching algorithms by organizing medical
        records and patient groups for further processing.

        Args:
            medrecord (MedRecord): An instance of `MedRecord` containing patient
                records.
            treated_group (Set[str]): A set of IDs for the treated patients.
            control_group (Set[str]): A set of IDs for the control patients.

        Raises:
            AssertionError: If `medrecord` is not an instance of `MedRecord`, or if
                either the treated or control groups are empty.
        """
        self.medrecord = medrecord
        assert isinstance(medrecord, MedRecord), "medrecord must be a MedRecord object"
        self.treated_group = treated_group
        assert len(self.treated_group) > 0, "Treated group cannot be empty"
        self.control_group = control_group
        assert len(self.control_group) > 0, "Control group cannot be empty"

        self.matched_control = set()
