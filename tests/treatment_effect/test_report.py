from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

import pandas as pd
import pytest

from medmodels import MedRecord
from medmodels.treatment_effect.treatment_effect import TreatmentEffect

if TYPE_CHECKING:
    from medmodels.medrecord.types import NodeIndex


def create_patients(patient_list: List[NodeIndex]) -> pd.DataFrame:
    """Creates a patients dataframe.

    Returns:
        pd.DataFrame: A patients dataframe.
    """
    patients = pd.DataFrame(
        {
            "index": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
            "age": [20, 30, 40, 30, 40, 50, 60, 70, 80],
            "gender": [
                "male",
                "female",
                "male",
                "female",
                "male",
                "female",
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


def create_edges1(patient_list: List[NodeIndex]) -> pd.DataFrame:
    """Creates an edges dataframe.

    Returns:
        pd.DataFrame: An edges dataframe.
    """
    edges = pd.DataFrame(
        {
            "source": [
                "M2",
                "M1",
                "M2",
                "M1",
                "M2",
                "M1",
                "M2",
            ],
            "target": [
                "P1",
                "P2",
                "P2",
                "P3",
                "P5",
                "P6",
                "P9",
            ],
            "time": [
                datetime(1999, 10, 15),
                datetime(2000, 1, 1),
                datetime(1999, 12, 15),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
            ],
        }
    )
    return edges.loc[edges["target"].isin(patient_list)]


def create_edges2(patient_list: List[NodeIndex]) -> pd.DataFrame:
    """Creates an edges dataframe with attribute "intensity".

    Returns:
        pd.DataFrame: An edges dataframe.
    """
    edges = pd.DataFrame(
        {
            "source": [
                "D1",
                "D1",
                "D1",
                "D1",
                "D1",
                "D1",
            ],
            "target": [
                "P1",
                "P2",
                "P3",
                "P3",
                "P4",
                "P7",
            ],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 7, 1),
                datetime(1999, 12, 15),
                datetime(2000, 1, 5),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
            ],
            "intensity": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
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
        patient_list = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    patients = create_patients(patient_list=patient_list)
    diagnoses = create_diagnoses()
    prescriptions = create_prescriptions()
    edges1 = create_edges1(patient_list=patient_list)
    edges2 = create_edges2(patient_list=patient_list)
    medrecord = MedRecord.from_pandas(
        nodes=[(patients, "index"), (diagnoses, "index"), (prescriptions, "index")],
        edges=[(edges1, "source", "target")],
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
    medrecord.add_edges((edges2, "source", "target"))
    return medrecord


@pytest.fixture
def medrecord() -> MedRecord:
    return create_medrecord()


class TestTreatmentEffectReport:
    def test_full_report(self, medrecord: MedRecord) -> None:
        """Test the full reporting of the TreatmentEffect class."""
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        # Calculate metrics
        full_report = tee.report.full_report(medrecord)

        report_test = {
            "absolute_risk_reduction": tee.estimate.absolute_risk_reduction(medrecord),
            "relative_risk": tee.estimate.relative_risk(medrecord),
            "odds_ratio": tee.estimate.odds_ratio(medrecord),
            "confounding_bias": tee.estimate.confounding_bias(medrecord),
            "hazard_ratio": tee.estimate.hazard_ratio(medrecord),
            "number_needed_to_treat": tee.estimate.number_needed_to_treat(medrecord),
        }
        assert full_report == report_test

    def test_continuous_estimators_report(self, medrecord: MedRecord) -> None:
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .build()
        )

        report_test = {
            "average_treatment_effect": tee.estimate.average_treatment_effect(
                medrecord,
                outcome_variable="intensity",
            ),
            "cohens_d": tee.estimate.cohens_d(medrecord, outcome_variable="intensity"),
            "hedges_g": tee.estimate.hedges_g(medrecord, outcome_variable="intensity"),
        }

        assert report_test == tee.report.continuous_estimators_report(
            medrecord, outcome_variable="intensity"
        )

    def test_continuous_estimators_report_with_time(self, medrecord: MedRecord) -> None:
        """Test the continuous report of the TreatmentEffect with time attribute."""
        tee = (
            TreatmentEffect.builder()
            .with_treatment("Rivaroxaban")
            .with_outcome("Stroke")
            .with_time_attribute("time")
            .build()
        )

        report_test = {
            "average_treatment_effect": tee.estimate.average_treatment_effect(
                medrecord,
                outcome_variable="intensity",
            ),
            "cohens_d": tee.estimate.cohens_d(medrecord, outcome_variable="intensity"),
            "hedges_g": tee.estimate.hedges_g(medrecord, outcome_variable="intensity"),
        }

        assert report_test == tee.report.continuous_estimators_report(
            medrecord, outcome_variable="intensity"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
