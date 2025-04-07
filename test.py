import pandas as pd
from datetime import datetime
from medmodels import MedRecord
from medmodels.medrecord.querying import NodeOperand
from typing import Set
from medmodels.medrecord.querying import NodeIndex

patients = pd.DataFrame({"index": ["P1", "P2"]})
treatments = pd.DataFrame({"index": ["T1"]})

treatment_edges = pd.DataFrame(
    {
        "source": ["T1"],
        "target": ["P1"],
        "time": [datetime(2000, 1, 1)],
    }
)
medrecord = MedRecord.from_pandas(
    nodes=[(patients, "index"), (treatments, "index")],
    edges=[(treatment_edges, "source", "target")],
)
medrecord.add_group("patients", ["P1", "P2"])


def query(
    node: NodeOperand,
):
    node.index().equal_to("P1")
    edges = node.edges()
    max_value = edges.attribute("time").max()
    edges.attribute("time").greater_than_or_equal_to(max_value)


subjects = medrecord.select_nodes(query)
print(subjects)

medrecord.add_edges(
    (
        pd.DataFrame(
            {
                "source": ["T1"],
                "target": ["P2"],
                "time": [datetime(2000, 1, 3)],
            }
        ),
        "source",
        "target",
    )
)

subjects_after = medrecord.select_nodes(query)
print(subjects_after)
