import unittest
from typing import List, Optional

import pandas as pd
import pytest

from medmodels.medrecord.medrecord import MedRecord
from medmodels.medrecord.types import NodeIndex
from medmodels.treatment_effect.temporal_analysis import (
    find_node_in_time_window,
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

    return patients.loc[patients["index"].isin(patient_list)]


def create_diagnoses() -> pd.DataFrame:
    """Creates a diagnoses dataframe.

    Returns:
        pd.DataFrame: A diagnoses dataframe.
    """
    return pd.DataFrame(
        {
            "index": ["D1"],
            "name": ["Stroke"],
        }
    )


def create_prescriptions() -> pd.DataFrame:
    """Creates a prescriptions dataframe.

    Returns:
        pd.DataFrame: A prescriptions dataframe.
    """
    return pd.DataFrame(
        {
            "index": ["M1", "M2"],
            "name": ["Rivaroxaban", "Warfarin"],
        }
    )


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
                "2000-01-01",
                "2000-01-01",
                "2000-01-01",
                "1999-12-15",
                "2000-07-01",
            ],
        }
    )
    return edges.loc[edges["target"].isin(patient_list)]


def create_medrecord(
    patient_list: Optional[List[NodeIndex]] = None,
) -> MedRecord:
    """Creates a MedRecord object.

    Returns:
        MedRecord: A MedRecord object.
    """
    if patient_list is None:
        patient_list = ["P1", "P2", "P3"]
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


class TestTreatmentEffect(unittest.TestCase):
    """"""

    def setUp(self) -> None:
        self.medrecord = create_medrecord()

    def test_find_reference_time(self) -> None:
        edge = find_reference_edge(
            self.medrecord,
            node_index="P1",
            reference="last",
            connected_group="Rivaroxaban",
        )
        assert edge == 0

        # adding medication time
        self.medrecord.add_edges(("M1", "P1", {"time": "2000-01-15"}))

        edge = find_reference_edge(
            self.medrecord,
            node_index="P1",
            reference="last",
            connected_group="Rivaroxaban",
        )
        assert edge == 5

        edge = find_reference_edge(
            self.medrecord,
            node_index="P1",
            reference="first",
            connected_group="Rivaroxaban",
        )
        assert edge == 0

    def test_invalid_find_reference_time(self) -> None:
        with pytest.raises(ValueError, match="Time attribute not found in the edge attributes"):
            find_reference_edge(
                self.medrecord,
                node_index="P1",
                reference="last",
                connected_group="Rivaroxaban",
                time_attribute="not_time",
            )

        node_index = "P2"
        with pytest.raises(ValueError, match=f"No edge found for node {node_index} in this MedRecord"):
            find_reference_edge(
                self.medrecord,
                node_index=node_index,
                reference="last",
                connected_group="Rivaroxaban",
                time_attribute="time",
            )

    def test_node_in_time_window(self) -> None:
        # check if patient has outcome a year after treatment
        node_found = find_node_in_time_window(
            self.medrecord,
            subject_index="P3",
            event_node="D1",
            connected_group="Rivaroxaban",
            start_days=0,
            end_days=365,
            reference="last",
            time_attribute="time",
        )
        assert node_found

        # check if patient has outcome 30 days after treatment
        node_found2 = find_node_in_time_window(
            self.medrecord,
            subject_index="P3",
            connected_group="Rivaroxaban",
            event_node="D1",
            start_days=0,
            end_days=30,
            reference="last",
            time_attribute="time",
        )
        assert not node_found2

    def test_invalid_node_in_time_window(self) -> None:
        with pytest.raises(ValueError, match="Time attribute not found in the edge attributes"):
            find_node_in_time_window(
                self.medrecord,
                subject_index="P3",
                connected_group="Rivaroxaban",
                event_node="D1",
                start_days=0,
                end_days=30,
                reference="last",
                time_attribute="no_time",
            )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestTreatmentEffect)
    unittest.TextTestRunner(verbosity=2).run(run_test)
