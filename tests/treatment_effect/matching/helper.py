"""Helper functions for creating a MedRecord object for the matching modules."""

from typing import List, Optional

import pandas as pd

from medmodels import MedRecord
from medmodels.medrecord.types import NodeIndex


def create_patients(patients_list: List[NodeIndex]) -> pd.DataFrame:
    """Creates a patients dataframe.

    Args:
        patients_list (List[NodeIndex]): List of patients to include in the dataframe.

    Returns:
        pd.DataFrame: A patients dataframe.
    """
    patients = pd.DataFrame(
        {
            "index": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
            "age": [20, 30, 40, 30, 40, 50, 60, 70, 80],
            "gender": [
                "male",
                "male",
                "female",
                "female",
                "male",
                "male",
                "female",
                "female",
                "male",
            ],
        }
    )

    return patients.loc[patients["index"].isin(patients_list)]


def create_medrecord(patients_list: Optional[List[NodeIndex]] = None) -> MedRecord:
    """Creates a MedRecord object.

    Args:
        patients_list (Optional[List[NodeIndex]], optional): List of patients to include
            in the MedRecord. Defaults to None.

    Returns:
        MedRecord: A MedRecord object.
    """
    if patients_list is None:
        patients_list = [
            "P1",
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
            "P7",
            "P8",
            "P9",
        ]
    patients = create_patients(patients_list=patients_list)
    medrecord = MedRecord.from_pandas(nodes=[(patients, "index")])
    medrecord.add_group(group="patients", nodes=patients["index"].to_list())
    medrecord.add_nodes(("P10", {}), "patients")

    return medrecord
