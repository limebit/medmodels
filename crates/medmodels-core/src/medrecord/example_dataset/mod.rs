use super::{
    datatypes::DataType,
    schema::{
        provided::{ProvidedGroupSchema, ProvidedSchema},
        Schema,
    },
    AttributeType, MedRecordAttribute,
};
use crate::MedRecord;
use polars::{
    io::SerReader,
    prelude::{CsvReadOptions, DataType as PolasrDataType, Schema as PolarsSchema, TimeUnit},
};
use std::{collections::HashMap, io::Cursor, sync::Arc};

macro_rules! simple_dataset_schema {
    () => {
        ProvidedSchema {
            groups: HashMap::from([
                (
                    "diagnosis".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        edges: HashMap::new(),
                        strict: true,
                    },
                ),
                (
                    "drug".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        edges: HashMap::new(),
                        strict: true,
                    },
                ),
                (
                    "patient".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::from([
                            (
                                "gender".into(),
                                (DataType::String, AttributeType::Categorical).into(),
                            ),
                            (
                                "age".into(),
                                (DataType::Int, AttributeType::Continuous).into(),
                            ),
                        ]),
                        edges: HashMap::new(),
                        strict: true,
                    },
                ),
                (
                    "procedure".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        edges: HashMap::new(),
                        strict: true,
                    },
                ),
                (
                    "patient_diagnosis".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::new(),
                        edges: HashMap::from([
                            (
                                "time".into(),
                                (DataType::DateTime, AttributeType::Temporal).into(),
                            ),
                            (
                                "duration_days".into(),
                                (
                                    DataType::Option(Box::new(DataType::Float)),
                                    AttributeType::Continuous,
                                )
                                    .into(),
                            ),
                        ]),
                        strict: true,
                    },
                ),
                (
                    "patient_drug".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::new(),
                        edges: HashMap::from([
                            (
                                "time".into(),
                                (DataType::DateTime, AttributeType::Temporal).into(),
                            ),
                            (
                                "quantity".into(),
                                (DataType::Int, AttributeType::Continuous).into(),
                            ),
                            (
                                "cost".into(),
                                (DataType::Float, AttributeType::Continuous).into(),
                            ),
                        ]),
                        strict: true,
                    },
                ),
                (
                    "patient_procedure".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::new(),
                        edges: HashMap::from([
                            (
                                "time".into(),
                                (DataType::DateTime, AttributeType::Temporal).into(),
                            ),
                            (
                                "duration_minutes".into(),
                                (DataType::Float, AttributeType::Continuous).into(),
                            ),
                        ]),
                        strict: true,
                    },
                ),
            ]),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: Default::default(),
                strict: true,
            },
        }
    };
}

macro_rules! advanced_dataset_schema {
    () => {
        ProvidedSchema {
            groups: HashMap::from([
                (
                    "diagnosis".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        edges: HashMap::new(),
                        strict: true,
                    },
                ),
                (
                    "drug".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        edges: HashMap::new(),
                        strict: true,
                    },
                ),
                (
                    "patient".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::from([
                            (
                                "gender".into(),
                                (DataType::String, AttributeType::Categorical).into(),
                            ),
                            (
                                "age".into(),
                                (DataType::Int, AttributeType::Continuous).into(),
                            ),
                        ]),
                        edges: HashMap::new(),
                        strict: true,
                    },
                ),
                (
                    "procedure".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        edges: HashMap::new(),
                        strict: true,
                    },
                ),
                (
                    "event".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::new(),
                        edges: HashMap::new(),
                        strict: true,
                    },
                ),
                (
                    "patient_diagnosis".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::new(),
                        edges: HashMap::from([
                            (
                                "time".into(),
                                (DataType::DateTime, AttributeType::Temporal).into(),
                            ),
                            (
                                "duration_days".into(),
                                (
                                    DataType::Option(Box::new(DataType::Float)),
                                    AttributeType::Continuous,
                                )
                                    .into(),
                            ),
                        ]),
                        strict: true,
                    },
                ),
                (
                    "patient_drug".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::new(),
                        edges: HashMap::from([
                            (
                                "time".into(),
                                (DataType::DateTime, AttributeType::Temporal).into(),
                            ),
                            (
                                "quantity".into(),
                                (DataType::Int, AttributeType::Continuous).into(),
                            ),
                            (
                                "cost".into(),
                                (DataType::Float, AttributeType::Continuous).into(),
                            ),
                        ]),
                        strict: true,
                    },
                ),
                (
                    "patient_procedure".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::new(),
                        edges: HashMap::from([
                            (
                                "time".into(),
                                (DataType::DateTime, AttributeType::Temporal).into(),
                            ),
                            (
                                "duration_minutes".into(),
                                (DataType::Float, AttributeType::Continuous).into(),
                            ),
                        ]),
                        strict: true,
                    },
                ),
                (
                    "patient_event".into(),
                    ProvidedGroupSchema {
                        nodes: HashMap::new(),
                        edges: HashMap::from([(
                            "time".into(),
                            (DataType::DateTime, AttributeType::Temporal).into(),
                        )]),
                        strict: true,
                    },
                ),
            ]),
            default: ProvidedGroupSchema {
                nodes: Default::default(),
                edges: Default::default(),
                strict: true,
            },
        }
    };
}

const SIMPLE_DIAGNOSIS_DATA: &[u8] =
    include_bytes!("./synthetic_data/simple_dataset/diagnosis.csv");
const SIMPLE_DRUG_DATA: &[u8] = include_bytes!("./synthetic_data/simple_dataset/drug.csv");
const SIMPLE_PATIENT_DATA: &[u8] = include_bytes!("./synthetic_data/simple_dataset/patient.csv");
const SIMPLE_PROCEDURE_DATA: &[u8] =
    include_bytes!("./synthetic_data/simple_dataset/procedure.csv");
const SIMPLE_PATIENT_DIAGNOSIS: &[u8] =
    include_bytes!("./synthetic_data/simple_dataset/patient_diagnosis.csv");
const SIMPLE_PATIENT_DRUG: &[u8] =
    include_bytes!("./synthetic_data/simple_dataset/patient_drug.csv");
const SIMPLE_PATIENT_PROCEDURE: &[u8] =
    include_bytes!("./synthetic_data/simple_dataset/patient_procedure.csv");

const ADVANCED_DIAGNOSIS_DATA: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/diagnosis.csv");
const ADVANCED_DRUG_DATA: &[u8] = include_bytes!("./synthetic_data/advanced_dataset/drug.csv");
const ADVANCED_PATIENT_DATA: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/patient.csv");
const ADVANCED_PROCEDURE_DATA: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/procedure.csv");
const ADVANCED_EVENT_DATA: &[u8] = include_bytes!("./synthetic_data/advanced_dataset/event.csv");
const ADVANCED_PATIENT_DIAGNOSIS: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/patient_diagnosis.csv");
const ADVANCED_PATIENT_DRUG: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/patient_drug.csv");
const ADVANCED_PATIENT_PROCEDURE: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/patient_procedure.csv");
const ADVANCED_PATIENT_EVENT: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/patient_event.csv");

impl MedRecord {
    pub fn from_simple_example_dataset() -> Self {
        let cursor = Cursor::new(SIMPLE_DIAGNOSIS_DATA);
        let mut diagnosis_schema = PolarsSchema::with_capacity(2);
        diagnosis_schema.insert("diagnosis_code".into(), PolasrDataType::String);
        diagnosis_schema.insert("description".into(), PolasrDataType::String);
        let mut diagnosis = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(diagnosis_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        diagnosis.rechunk_mut();
        let diagnosis_ids = diagnosis
            .column("diagnosis_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect::<Vec<_>>();

        let cursor = Cursor::new(SIMPLE_DRUG_DATA);
        let mut drug_schema = PolarsSchema::with_capacity(2);
        drug_schema.insert("drug_code".into(), PolasrDataType::String);
        drug_schema.insert("description".into(), PolasrDataType::String);
        let mut drug = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(drug_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        drug.rechunk_mut();
        let drug_ids = drug
            .column("drug_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect::<Vec<_>>();

        let cursor = Cursor::new(SIMPLE_PATIENT_DATA);
        let mut patient_schema = PolarsSchema::with_capacity(3);
        patient_schema.insert("patient_id".into(), PolasrDataType::String);
        patient_schema.insert("gender".into(), PolasrDataType::String);
        patient_schema.insert("age".into(), PolasrDataType::Int64);
        let mut patient = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(patient_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        patient.rechunk_mut();
        let patient_ids = patient
            .column("patient_id")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect::<Vec<_>>();

        let cursor = Cursor::new(SIMPLE_PROCEDURE_DATA);
        let mut procedure_schema = PolarsSchema::with_capacity(2);
        procedure_schema.insert("procedure_code".into(), PolasrDataType::String);
        procedure_schema.insert("description".into(), PolasrDataType::String);
        let mut procedure = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(procedure_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        procedure.rechunk_mut();
        let procedure_ids = procedure
            .column("procedure_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect::<Vec<_>>();

        let cursor = Cursor::new(SIMPLE_PATIENT_DIAGNOSIS);
        let mut patient_diagnosis_schema = PolarsSchema::with_capacity(4);
        patient_diagnosis_schema.insert("patient_id".into(), PolasrDataType::String);
        patient_diagnosis_schema.insert("diagnosis_code".into(), PolasrDataType::String);
        patient_diagnosis_schema.insert(
            "time".into(),
            PolasrDataType::Datetime(TimeUnit::Microseconds, None),
        );
        patient_diagnosis_schema.insert("duration_days".into(), PolasrDataType::Float64);
        let patient_diagnosis = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(patient_diagnosis_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        let patient_diagnosis_ids = (0..patient_diagnosis.height() as u32).collect::<Vec<_>>();

        let cursor = Cursor::new(SIMPLE_PATIENT_DRUG);
        let mut patient_drug_schema = PolarsSchema::with_capacity(5);
        patient_drug_schema.insert("patient_id".into(), PolasrDataType::String);
        patient_drug_schema.insert("drug_code".into(), PolasrDataType::String);
        patient_drug_schema.insert(
            "time".into(),
            PolasrDataType::Datetime(polars::prelude::TimeUnit::Microseconds, None),
        );
        patient_drug_schema.insert("quantity".into(), PolasrDataType::Int64);
        patient_drug_schema.insert("cost".into(), PolasrDataType::Float64);
        let patient_drug = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(patient_drug_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        let patient_drug_ids = (patient_diagnosis.height() as u32
            ..(patient_diagnosis.height() + patient_drug.height()) as u32)
            .collect::<Vec<_>>();

        let cursor = Cursor::new(SIMPLE_PATIENT_PROCEDURE);
        let mut patient_procedure_schema = PolarsSchema::with_capacity(4);
        patient_procedure_schema.insert("patient_id".into(), PolasrDataType::String);
        patient_procedure_schema.insert("procedure_code".into(), PolasrDataType::String);
        patient_procedure_schema.insert(
            "time".into(),
            PolasrDataType::Datetime(polars::prelude::TimeUnit::Microseconds, None),
        );
        patient_procedure_schema.insert("duration_minutes".into(), PolasrDataType::Float64);
        let patient_procedure = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(patient_procedure_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        let patient_procedure_ids = ((patient_diagnosis.height() + patient_drug.height()) as u32
            ..(patient_diagnosis.height() + patient_drug.height() + patient_procedure.height())
                as u32)
            .collect::<Vec<_>>();

        let mut medrecord = Self::from_dataframes(
            vec![
                (diagnosis, "diagnosis_code"),
                (drug, "drug_code"),
                (patient, "patient_id"),
                (procedure, "procedure_code"),
            ],
            vec![
                (patient_diagnosis, "patient_id", "diagnosis_code"),
                (patient_drug, "patient_id", "drug_code"),
                (patient_procedure, "patient_id", "procedure_code"),
            ],
            None,
        )
        .expect("MedRecord can be built");

        medrecord
            .add_group("diagnosis".into(), Some(diagnosis_ids), None)
            .expect("Group can be added");
        medrecord
            .add_group("drug".into(), Some(drug_ids), None)
            .expect("Group can be added");
        medrecord
            .add_group("patient".into(), Some(patient_ids), None)
            .expect("Group can be added");
        medrecord
            .add_group("procedure".into(), Some(procedure_ids), None)
            .expect("Group can be added");

        medrecord
            .add_group(
                "patient_diagnosis".into(),
                None,
                Some(patient_diagnosis_ids),
            )
            .expect("Group can be added");
        medrecord
            .add_group("patient_drug".into(), None, Some(patient_drug_ids))
            .expect("Group can be added");
        medrecord
            .add_group(
                "patient_procedure".into(),
                None,
                Some(patient_procedure_ids),
            )
            .expect("Group can be added");

        medrecord.schema = Schema::Provided(simple_dataset_schema!());

        medrecord
    }
    pub fn from_advanced_example_dataset() -> Self {
        let cursor = Cursor::new(ADVANCED_DIAGNOSIS_DATA);
        let mut diagnosis_schema = PolarsSchema::with_capacity(2);
        diagnosis_schema.insert("diagnosis_code".into(), PolasrDataType::String);
        diagnosis_schema.insert("description".into(), PolasrDataType::String);
        let mut diagnosis = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(diagnosis_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        diagnosis.rechunk_mut();
        let diagnosis_ids = diagnosis
            .column("diagnosis_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect::<Vec<_>>();

        let cursor = Cursor::new(ADVANCED_DRUG_DATA);
        let mut drug_schema = PolarsSchema::with_capacity(2);
        drug_schema.insert("drug_code".into(), PolasrDataType::String);
        drug_schema.insert("description".into(), PolasrDataType::String);
        let mut drug = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(drug_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        drug.rechunk_mut();
        let drug_ids = drug
            .column("drug_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect::<Vec<_>>();

        let cursor = Cursor::new(ADVANCED_PATIENT_DATA);
        let mut patient_schema = PolarsSchema::with_capacity(3);
        patient_schema.insert("patient_id".into(), PolasrDataType::String);
        patient_schema.insert("gender".into(), PolasrDataType::String);
        patient_schema.insert("age".into(), PolasrDataType::Int64);
        let mut patient = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(patient_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        patient.rechunk_mut();
        let patient_ids = patient
            .column("patient_id")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect::<Vec<_>>();

        let cursor = Cursor::new(ADVANCED_PROCEDURE_DATA);
        let mut procedure_schema = PolarsSchema::with_capacity(2);
        procedure_schema.insert("procedure_code".into(), PolasrDataType::String);
        procedure_schema.insert("description".into(), PolasrDataType::String);
        let mut procedure = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(procedure_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        procedure.rechunk_mut();
        let procedure_ids = procedure
            .column("procedure_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect::<Vec<_>>();

        let cursor = Cursor::new(ADVANCED_EVENT_DATA);
        let mut event_schema = PolarsSchema::with_capacity(1);
        event_schema.insert("event".into(), PolasrDataType::String);
        let mut event = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(event_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        event.rechunk_mut();
        let event_ids = event
            .column("event")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect::<Vec<_>>();

        let cursor = Cursor::new(ADVANCED_PATIENT_DIAGNOSIS);
        let mut patient_diagnosis_schema = PolarsSchema::with_capacity(4);
        patient_diagnosis_schema.insert("patient_id".into(), PolasrDataType::String);
        patient_diagnosis_schema.insert("diagnosis_code".into(), PolasrDataType::String);
        patient_diagnosis_schema.insert(
            "time".into(),
            PolasrDataType::Datetime(TimeUnit::Microseconds, None),
        );
        patient_diagnosis_schema.insert("duration_days".into(), PolasrDataType::Float64);
        let patient_diagnosis = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(patient_diagnosis_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        let patient_diagnosis_ids = (0..patient_diagnosis.height() as u32).collect::<Vec<_>>();

        let cursor = Cursor::new(ADVANCED_PATIENT_DRUG);
        let mut patient_drug_schema = PolarsSchema::with_capacity(5);
        patient_drug_schema.insert("patient_id".into(), PolasrDataType::String);
        patient_drug_schema.insert("drug_code".into(), PolasrDataType::String);
        patient_drug_schema.insert(
            "time".into(),
            PolasrDataType::Datetime(polars::prelude::TimeUnit::Microseconds, None),
        );
        patient_drug_schema.insert("quantity".into(), PolasrDataType::Int64);
        patient_drug_schema.insert("cost".into(), PolasrDataType::Float64);
        let patient_drug = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(patient_drug_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        let patient_drug_ids = (patient_diagnosis.height() as u32
            ..(patient_diagnosis.height() + patient_drug.height()) as u32)
            .collect::<Vec<_>>();

        let cursor = Cursor::new(ADVANCED_PATIENT_PROCEDURE);
        let mut patient_procedure_schema = PolarsSchema::with_capacity(4);
        patient_procedure_schema.insert("patient_id".into(), PolasrDataType::String);
        patient_procedure_schema.insert("procedure_code".into(), PolasrDataType::String);
        patient_procedure_schema.insert(
            "time".into(),
            PolasrDataType::Datetime(polars::prelude::TimeUnit::Microseconds, None),
        );
        patient_procedure_schema.insert("duration_minutes".into(), PolasrDataType::Float64);
        let patient_procedure = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(patient_procedure_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        let patient_procedure_ids = ((patient_diagnosis.height() + patient_drug.height()) as u32
            ..(patient_diagnosis.height() + patient_drug.height() + patient_procedure.height())
                as u32)
            .collect::<Vec<_>>();

        let cursor = Cursor::new(ADVANCED_PATIENT_EVENT);
        let mut patient_event_schema = PolarsSchema::with_capacity(3);
        patient_event_schema.insert("patient_id".into(), PolasrDataType::String);
        patient_event_schema.insert("event".into(), PolasrDataType::String);
        patient_event_schema.insert(
            "time".into(),
            PolasrDataType::Datetime(polars::prelude::TimeUnit::Microseconds, None),
        );
        let patient_event = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(patient_event_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        let patient_event_ids = ((patient_diagnosis.height()
            + patient_drug.height()
            + patient_procedure.height()) as u32
            ..(patient_diagnosis.height()
                + patient_drug.height()
                + patient_procedure.height()
                + patient_event.height()) as u32)
            .collect::<Vec<_>>();

        let mut medrecord = Self::from_dataframes(
            vec![
                (diagnosis, "diagnosis_code"),
                (drug, "drug_code"),
                (patient, "patient_id"),
                (procedure, "procedure_code"),
                (event, "event"),
            ],
            vec![
                (patient_diagnosis, "patient_id", "diagnosis_code"),
                (patient_drug, "patient_id", "drug_code"),
                (patient_procedure, "patient_id", "procedure_code"),
                (patient_event, "patient_id", "event"),
            ],
            None,
        )
        .expect("MedRecord can be built");

        medrecord
            .add_group("diagnosis".into(), Some(diagnosis_ids), None)
            .expect("Group can be added");
        medrecord
            .add_group("drug".into(), Some(drug_ids), None)
            .expect("Group can be added");
        medrecord
            .add_group("patient".into(), Some(patient_ids), None)
            .expect("Group can be added");
        medrecord
            .add_group("procedure".into(), Some(procedure_ids), None)
            .expect("Group can be added");
        medrecord
            .add_group("event".into(), Some(event_ids), None)
            .expect("Group can be added");

        medrecord
            .add_group(
                "patient_diagnosis".into(),
                None,
                Some(patient_diagnosis_ids),
            )
            .expect("Group can be added");
        medrecord
            .add_group("patient_drug".into(), None, Some(patient_drug_ids))
            .expect("Group can be added");
        medrecord
            .add_group(
                "patient_procedure".into(),
                None,
                Some(patient_procedure_ids),
            )
            .expect("Group can be added");
        medrecord
            .add_group("patient_event".into(), None, Some(patient_event_ids))
            .expect("Group can be added");

        medrecord.schema = Schema::Provided(advanced_dataset_schema!());

        medrecord
    }
}

#[cfg(test)]
mod test {
    use super::{AttributeType, DataType};
    use crate::{
        medrecord::schema::provided::{ProvidedGroupSchema, ProvidedSchema},
        MedRecord,
    };
    use std::collections::HashMap;

    #[test]
    fn test_from_simple_example_dataset() {
        let mut medrecord = MedRecord::from_simple_example_dataset();

        assert!(medrecord.update_schema(simple_dataset_schema!()).is_ok());

        assert_eq!(73, medrecord.node_count());
        assert_eq!(160, medrecord.edge_count());

        assert_eq!(
            25,
            medrecord
                .nodes_in_group(&"diagnosis".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            19,
            medrecord.nodes_in_group(&"drug".into()).unwrap().count()
        );
        assert_eq!(
            5,
            medrecord.nodes_in_group(&"patient".into()).unwrap().count()
        );
        assert_eq!(
            24,
            medrecord
                .nodes_in_group(&"procedure".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            60,
            medrecord
                .edges_in_group(&"patient_diagnosis".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            50,
            medrecord
                .edges_in_group(&"patient_drug".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            50,
            medrecord
                .edges_in_group(&"patient_procedure".into())
                .unwrap()
                .count()
        );
    }

    #[test]
    fn test_from_advanced_example_dataset() {
        let mut medrecord = MedRecord::from_advanced_example_dataset();

        assert!(medrecord.update_schema(advanced_dataset_schema!()).is_ok());

        assert_eq!(1088, medrecord.node_count());
        assert_eq!(16883, medrecord.edge_count());

        assert_eq!(
            206,
            medrecord
                .nodes_in_group(&"diagnosis".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            185,
            medrecord.nodes_in_group(&"drug".into()).unwrap().count()
        );
        assert_eq!(
            600,
            medrecord.nodes_in_group(&"patient".into()).unwrap().count()
        );
        assert_eq!(
            1,
            medrecord.nodes_in_group(&"event".into()).unwrap().count()
        );
        assert_eq!(
            96,
            medrecord
                .nodes_in_group(&"procedure".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            5741,
            medrecord
                .edges_in_group(&"patient_diagnosis".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            10373,
            medrecord
                .edges_in_group(&"patient_drug".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            677,
            medrecord
                .edges_in_group(&"patient_procedure".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            92,
            medrecord
                .edges_in_group(&"patient_event".into())
                .unwrap()
                .count()
        );
    }
}
