import unittest
from datetime import datetime
from typing import List

import pandas as pd

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import NodeIndex
from medmodels.treatment_effect.temporal_analysis import (
    find_reference_edge,
)


def create_patients(patient_list: List[NodeIndex]) -> pd.DataFrame:
    """Creates a patients dataframe.

    Returns:
        pd.DataFrame: A patients dataframe.
    """
    patients = pd.DataFrame(
        {
            "index": ["P1", "P2", "P3"],
            "age": [20, 30, 40],
            "gender": [
                "male",
                "female",
                "male",
            ],
        }
    )

    patients = patients.loc[patients["index"].isin(patient_list)]
    return patients


def create_diagnoses() -> pd.DataFrame:
    """Creates a diagnoses dataframe.

    Returns:
        pd.DataFrame: A diagnoses dataframe.
    """
    diagnoses = pd.DataFrame(
        {
            "index": ["D1"],
            "name": ["Stroke"],
        }
    )
    return diagnoses


def create_prescriptions() -> pd.DataFrame:
    """Creates a prescriptions dataframe.

    Returns:
        pd.DataFrame: A prescriptions dataframe.
    """
    prescriptions = pd.DataFrame(
        {
            "index": ["M1", "M2"],
            "name": ["Rivaroxaban", "Warfarin"],
        }
    )
    return prescriptions


def create_edges(patient_list: List[NodeIndex]) -> pd.DataFrame:
    """Creates an edges dataframe.

    Returns:
        pd.DataFrame: An edges dataframe.
    """
    edges = pd.DataFrame(
        {
            "source": [
                "M1",
                "M2",
                "M1",
                "M2",
                "D1",
            ],
            "target": [
                "P1",
                "P2",
                "P3",
                "P3",
                "P3",
            ],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(1999, 12, 15),
                datetime(2000, 7, 1),
            ],
        }
    )
    edges = edges.loc[edges["target"].isin(patient_list)]
    return edges


def create_medrecord(
    patient_list: List[NodeIndex] = [
        "P1",
        "P2",
        "P3",
    ],
) -> MedRecord:
    """Creates a MedRecord object.

    Returns:
        MedRecord: A MedRecord object.
    """
    patients = create_patients(patient_list=patient_list)
    diagnoses = create_diagnoses()
    prescriptions = create_prescriptions()
    edges = create_edges(patient_list=patient_list)
    medrecord = MedRecord.from_pandas(
        nodes=[(patients, "index"), (diagnoses, "index"), (prescriptions, "index")],
        edges=[(edges, "source", "target")],
    )
    medrecord.add_group(group="patients", nodes=patients["index"].to_list())
    medrecord.add_group(
        "Stroke",
        ["D1"],
    )
    medrecord.add_group(
        "Rivaroxaban",
        ["M1"],
    )
    medrecord.add_group(
        "Warfarin",
        ["M2"],
    )
    return medrecord


class TestTemporalAnalysis(unittest.TestCase):
    """"""

    def setUp(self):
        self.medrecord = create_medrecord()

    def test_find_reference_time(self):
        edge = find_reference_edge(
            self.medrecord,
            node_index="P1",
            reference="last",
            connected_group="Rivaroxaban",
        )
        self.assertEqual(0, edge)

        # adding medication time
        self.medrecord.add_edges(("M1", "P1", {"time": datetime(2000, 1, 15)}))

        edge = find_reference_edge(
            self.medrecord,
            node_index="P1",
            reference="last",
            connected_group="Rivaroxaban",
        )
        self.assertEqual(5, edge)

        edge = find_reference_edge(
            self.medrecord,
            node_index="P1",
            reference="first",
            connected_group="Rivaroxaban",
        )
        self.assertEqual(0, edge)

    def test_invalid_find_reference_time(self):
        with self.assertRaisesRegex(
            ValueError,
            "No edge with that time attribute or with a datetime data type was found for node P1 in this MedRecord",
        ):
            find_reference_edge(
                self.medrecord,
                node_index="P1",
                reference="last",
                connected_group="Rivaroxaban",
                time_attribute="not_time",
            )

        node_index = "P2"
        with self.assertRaisesRegex(
            ValueError,
            "No edge with that time attribute or with a datetime data type was found for node P2 in this MedRecord",
        ):
            find_reference_edge(
                self.medrecord,
                node_index=node_index,
                reference="last",
                connected_group="Rivaroxaban",
                time_attribute="time",
            )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestTemporalAnalysis)
    unittest.TextTestRunner(verbosity=2).run(run_test)
