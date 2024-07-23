from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

from medmodels import MedRecord
from medmodels.data_synthesis.mtgan.modules.postprocessor import (
    AttributeType,
)
from medmodels.data_synthesis.mtgan.mtgan import MTGAN
from medmodels.data_synthesis.synthesizer import Synthesizer
from medmodels.medrecord.types import MedRecordAttribute, NodeIndex


def create_patients(patient_list: List[NodeIndex]) -> pd.DataFrame:
    """
    Create a patients dataframe.

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

    patients = patients.loc[patients["index"].isin(patient_list)]
    return patients


def create_diagnoses() -> pd.DataFrame:
    """
    Create a diagnoses dataframe.

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
    """
    Create a prescriptions dataframe.

    Returns:
        pd.DataFrame: A prescriptions dataframe.
    """
    prescriptions = pd.DataFrame(
        {
            "index": ["M1", "M2", "M3"],
            "name": ["Rivaroxaban", "Warfarin", "Ibuprofen"],
        }
    )
    return prescriptions


def create_edges1(patient_list: List[NodeIndex]) -> pd.DataFrame:
    """
    Create an edges dataframe.

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
                "M3",
                "M3",
                "M3",
                "M3",
                "M3",
                "M3",
            ],
            "target": [
                "P1",
                "P2",
                "P2",
                "P3",
                "P5",
                "P6",
                "P9",
                "P2",
                "P3",
                "P2",
                "P6",
                "P6",
                "P7",
            ],
            "time": [
                datetime(1999, 10, 15),
                datetime(2000, 1, 1),
                datetime(1999, 12, 15),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(1999, 12, 15),
                datetime(1999, 12, 15),
                datetime(1999, 12, 15),
            ],
        }
    )
    edges = edges.loc[edges["target"].isin(patient_list)]
    return edges


def create_edges2(patient_list: List[NodeIndex]) -> pd.DataFrame:
    """
    Create an edges dataframe with attribute "intensity".

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
                "D1",
            ],
            "target": ["P1", "P2", "P3", "P3", "P4", "P7", "P7"],
            "time": [
                datetime(2000, 1, 1),
                datetime(2000, 7, 1),
                datetime(1999, 12, 15),
                datetime(2000, 1, 5),
                datetime(2000, 1, 1),
                datetime(2000, 1, 1),
                datetime(2000, 1, 3),
            ],
            "intensity": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
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
        "P4",
        "P5",
        "P6",
        "P7",
        "P8",
        "P9",
    ],
) -> MedRecord:
    """
    Create a MedRecord object.

    Returns:
        MedRecord: A MedRecord object.
    """
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
    medrecord.add_group("concepts", ["D1", "M1", "M2", "M3"])
    return medrecord


medrecord = create_medrecord()
attributes_postprocessing: Dict[MedRecordAttribute, AttributeType] = {
    "gender": "categorical",
    "age": "regression",
}
hyperparameters_path = Path(
    "/Users/martin.iniguez/Desktop/projects/medmodels/data/mtgan/hyperparameters/test_hyperparameters.json"
)
mtgan = (
    Synthesizer.builder()
    .from_mtgan()
    .load_hyperparameters_from(hyperparameters_path)
    .with_preprocessor_hyperparameters()
    .with_postprocessor_hyperparameters()
    .with_postprocessor_attributes(attributes_postprocessing)
    .build()
)
mtgan_trained = mtgan.fit(medrecord=medrecord)
synthetic_medrecord = mtgan_trained.generate_synthetic_data()

medrecord2 = create_medrecord()
mtgan2 = MTGAN()
mtgan_trained2 = mtgan2.fit(medrecord=medrecord2)
synthetic_medrecord2 = mtgan_trained2.generate_synthetic_data()
