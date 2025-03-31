from medmodels import MedRecord

record = MedRecord()
record.add_nodes(
    [
        (0, {"foo": "bar"}),
        (1, {"baz": "qux"}),
        (2, {"quux": "corge"}),
        (3, {"grault": "garply"}),
        (4, {"waldo": "fred"}),
    ],
    group="group1",
)
record.add_nodes(
    [
        (5, {"plugh": "xyzzy"}),
        (6, {"thud": "wibble"}),
        (7, {"wobble": "wubble"}),
        (8, {"flob": "blub"}),
        (9, {"bloop": "blop"}),
    ],
    group="group2",
)
record.add_nodes(
    [
        (10, {"foo": "bar"}),
        (11, {"baz": "qux"}),
        (12, {"quux": "corge"}),
        (13, {"grault": "garply"}),
        (14, {"waldo": "fred"}),
    ],
)
# record = MedRecord.from_advanced_example_dataset()
record = MedRecord()
print(record)
print(record.overview_edges())
