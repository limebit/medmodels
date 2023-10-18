from medmodels.new_dataclass import Medrecord as NewMedRecord
from medmodels.dataclass.dataclass import MedRecord as OldMedRecord
from medmodels.dataclass.utils import df_to_edges, df_to_nodes
from timeit import default_timer as timer
import pandas as pd
import polars as pl

node_size = 30

nodes = pd.DataFrame(
    [{"id": str(x), "foo": "bar", "bar": "foo"} for x in range(node_size)]
)
edges = pd.DataFrame(
    [
        {"id1": str(x), "id2": str(y), "foo": "bar", "bar": "foo"}
        for x in range(node_size)
        for y in range(30)
        if x != y
    ]
)

nodes_new = pl.from_pandas(nodes)
edges_new = pl.from_pandas(edges)

print("Computed Dataframes")

start = timer()

new_medrecord = NewMedRecord.from_dataframes(
    nodes_new, edges_new, "id", ["foo", "bar"], "id1", "id2", ["foo", "bar"]
)

print(new_medrecord.edges_between("1", "2"))

end = timer()
new_time = end - start

start = timer()

nodes_old = df_to_nodes(nodes, "id", ["foo", "bar"], drop_duplicates=False)
edges_old = df_to_edges(edges, "id1", "id2", ["foo", "bar"])

old_medrecord = OldMedRecord()
old_medrecord.add_nodes(nodes_old, "dimension")
old_medrecord.add_edges(edges_old)

end = timer()
old_time = end - start

print(
    f"New time: {new_time} - New nodes: {new_medrecord.node_count()} - New edges: {new_medrecord.edge_count()} - Old time: {old_time} - Speedup: {old_time / new_time}"  # noqa
)
