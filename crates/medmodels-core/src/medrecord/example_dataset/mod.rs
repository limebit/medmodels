use super::{
    datatypes::DataType,
    schema::{AttributeSchema, AttributeType, GroupSchema, Schema},
    MedRecordAttribute,
};
use crate::MedRecord;
use polars::{
    io::SerReader,
    prelude::{CsvReadOptions, DataType as PolasrDataType, Schema as PolarsSchema, TimeUnit},
};
use std::{collections::HashMap, io::Cursor, sync::Arc};

macro_rules! simple_dataset_schema {
    () => {
        Schema::new_provided(
            HashMap::from([
                (
                    "diagnosis".into(),
                    GroupSchema::new(
                        AttributeSchema::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        AttributeSchema::default(),
                    ),
                ),
                (
                    "drug".into(),
                    GroupSchema::new(
                        AttributeSchema::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        AttributeSchema::default(),
                    ),
                ),
                (
                    "patient".into(),
                    GroupSchema::new(
                        AttributeSchema::from([
                            (
                                "gender".into(),
                                (DataType::String, AttributeType::Categorical).into(),
                            ),
                            (
                                "age".into(),
                                (DataType::Int, AttributeType::Continuous).into(),
                            ),
                        ]),
                        AttributeSchema::default(),
                    ),
                ),
                (
                    "procedure".into(),
                    GroupSchema::new(
                        AttributeSchema::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        AttributeSchema::default(),
                    ),
                ),
                (
                    "patient_diagnosis".into(),
                    GroupSchema::new(
                        AttributeSchema::default(),
                        AttributeSchema::from([
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
                    ),
                ),
                (
                    "patient_drug".into(),
                    GroupSchema::new(
                        AttributeSchema::default(),
                        AttributeSchema::from([
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
                    ),
                ),
                (
                    "patient_procedure".into(),
                    GroupSchema::new(
                        AttributeSchema::default(),
                        AttributeSchema::from([
                            (
                                "time".into(),
                                (DataType::DateTime, AttributeType::Temporal).into(),
                            ),
                            (
                                "duration_minutes".into(),
                                (DataType::Float, AttributeType::Continuous).into(),
                            ),
                        ]),
                    ),
                ),
            ]),
            GroupSchema::new(Default::default(), Default::default()),
        )
    };
}

macro_rules! advanced_dataset_schema {
    () => {
        Schema::new_provided(
            HashMap::from([
                (
                    "diagnosis".into(),
                    GroupSchema::new(
                        AttributeSchema::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        AttributeSchema::default(),
                    ),
                ),
                (
                    "drug".into(),
                    GroupSchema::new(
                        AttributeSchema::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        AttributeSchema::default(),
                    ),
                ),
                (
                    "patient".into(),
                    GroupSchema::new(
                        AttributeSchema::from([
                            (
                                "gender".into(),
                                (DataType::String, AttributeType::Categorical).into(),
                            ),
                            (
                                "age".into(),
                                (DataType::Int, AttributeType::Continuous).into(),
                            ),
                        ]),
                        AttributeSchema::default(),
                    ),
                ),
                (
                    "procedure".into(),
                    GroupSchema::new(
                        AttributeSchema::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        AttributeSchema::default(),
                    ),
                ),
                (
                    "event".into(),
                    GroupSchema::new(AttributeSchema::default(), AttributeSchema::default()),
                ),
                (
                    "patient_diagnosis".into(),
                    GroupSchema::new(
                        AttributeSchema::default(),
                        AttributeSchema::from([
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
                    ),
                ),
                (
                    "patient_drug".into(),
                    GroupSchema::new(
                        AttributeSchema::default(),
                        AttributeSchema::from([
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
                    ),
                ),
                (
                    "patient_procedure".into(),
                    GroupSchema::new(
                        AttributeSchema::default(),
                        AttributeSchema::from([
                            (
                                "time".into(),
                                (DataType::DateTime, AttributeType::Temporal).into(),
                            ),
                            (
                                "duration_minutes".into(),
                                (DataType::Float, AttributeType::Continuous).into(),
                            ),
                        ]),
                    ),
                ),
                (
                    "patient_event".into(),
                    GroupSchema::new(
                        AttributeSchema::default(),
                        AttributeSchema::from([(
                            "time".into(),
                            (DataType::DateTime, AttributeType::Temporal).into(),
                        )]),
                    ),
                ),
            ]),
            GroupSchema::new(Default::default(), Default::default()),
        )
    };
}

macro_rules! admissions_dataset_schema {
    () => {
        Schema::new_provided(
            HashMap::from([
                (
                    "admission".into(),
                    GroupSchema::new(
                        AttributeSchema::from([
                            (
                                "admission_type".into(),
                                (DataType::String, AttributeType::Categorical).into(),
                            ),
                            (
                                "start_time".into(),
                                (DataType::DateTime, AttributeType::Temporal).into(),
                            ),
                            (
                                "end_time".into(),
                                (DataType::DateTime, AttributeType::Temporal).into(),
                            ),
                        ]),
                        AttributeSchema::default(),
                    ),
                ),
                (
                    "diagnosis".into(),
                    GroupSchema::new(
                        AttributeSchema::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        AttributeSchema::default(),
                    ),
                ),
                (
                    "drug".into(),
                    GroupSchema::new(
                        AttributeSchema::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        AttributeSchema::default(),
                    ),
                ),
                (
                    "event".into(),
                    GroupSchema::new(AttributeSchema::default(), AttributeSchema::default()),
                ),
                (
                    "patient".into(),
                    GroupSchema::new(
                        AttributeSchema::from([
                            (
                                "gender".into(),
                                (DataType::String, AttributeType::Categorical).into(),
                            ),
                            (
                                "age".into(),
                                (DataType::Int, AttributeType::Continuous).into(),
                            ),
                        ]),
                        AttributeSchema::default(),
                    ),
                ),
                (
                    "procedure".into(),
                    GroupSchema::new(
                        AttributeSchema::from([(
                            "description".into(),
                            (DataType::String, AttributeType::Unstructured).into(),
                        )]),
                        AttributeSchema::default(),
                    ),
                ),
                (
                    "admission_diagnosis".into(),
                    GroupSchema::new(
                        AttributeSchema::default(),
                        AttributeSchema::from([
                            (
                                "time".into(),
                                (DataType::DateTime, AttributeType::Temporal).into(),
                            ),
                            (
                                "duration_days".into(),
                                (DataType::Float, AttributeType::Continuous).into(),
                            ),
                        ]),
                    ),
                ),
                (
                    "admission_drug".into(),
                    GroupSchema::new(
                        AttributeSchema::default(),
                        AttributeSchema::from([
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
                    ),
                ),
                (
                    "admission_procedure".into(),
                    GroupSchema::new(
                        AttributeSchema::default(),
                        AttributeSchema::from([
                            (
                                "time".into(),
                                (DataType::DateTime, AttributeType::Temporal).into(),
                            ),
                            (
                                "duration_minutes".into(),
                                (DataType::Float, AttributeType::Continuous).into(),
                            ),
                        ]),
                    ),
                ),
                (
                    "patient_admission".into(),
                    GroupSchema::new(AttributeSchema::default(), AttributeSchema::default()),
                ),
                (
                    "patient_event".into(),
                    GroupSchema::new(
                        AttributeSchema::default(),
                        AttributeSchema::from([(
                            "time".into(),
                            (DataType::DateTime, AttributeType::Temporal).into(),
                        )]),
                    ),
                ),
            ]),
            GroupSchema::new(Default::default(), Default::default()),
        )
    };
}

const SIMPLE_DIAGNOSIS_DATA: &[u8] =
    include_bytes!("./synthetic_data/simple_dataset/diagnosis.csv");
const SIMPLE_DRUG_DATA: &[u8] = include_bytes!("./synthetic_data/simple_dataset/drug.csv");
const SIMPLE_PATIENT_DATA: &[u8] = include_bytes!("./synthetic_data/simple_dataset/patient.csv");
const SIMPLE_PROCEDURE_DATA: &[u8] =
    include_bytes!("./synthetic_data/simple_dataset/procedure.csv");
const SIMPLE_PATIENT_DIAGNOSIS_DATA: &[u8] =
    include_bytes!("./synthetic_data/simple_dataset/patient_diagnosis.csv");
const SIMPLE_PATIENT_DRUG_DATA: &[u8] =
    include_bytes!("./synthetic_data/simple_dataset/patient_drug.csv");
const SIMPLE_PATIENT_PROCEDURE_DATA: &[u8] =
    include_bytes!("./synthetic_data/simple_dataset/patient_procedure.csv");

const ADVANCED_DIAGNOSIS_DATA: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/diagnosis.csv");
const ADVANCED_DRUG_DATA: &[u8] = include_bytes!("./synthetic_data/advanced_dataset/drug.csv");
const ADVANCED_PATIENT_DATA: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/patient.csv");
const ADVANCED_PROCEDURE_DATA: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/procedure.csv");
const ADVANCED_EVENT_DATA: &[u8] = include_bytes!("./synthetic_data/advanced_dataset/event.csv");
const ADVANCED_PATIENT_DIAGNOSIS_DATA: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/patient_diagnosis.csv");
const ADVANCED_PATIENT_DRUG_DATA: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/patient_drug.csv");
const ADVANCED_PATIENT_PROCEDURE_DATA: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/patient_procedure.csv");
const ADVANCED_PATIENT_EVENT_DATA: &[u8] =
    include_bytes!("./synthetic_data/advanced_dataset/patient_event.csv");

const ADMISSIONS_ADMISSION_DATA: &[u8] =
    include_bytes!("./synthetic_data/admissions_dataset/admission.csv");
const ADMISSIONS_DIAGNOSIS_DATA: &[u8] =
    include_bytes!("./synthetic_data/admissions_dataset/diagnosis.csv");
const ADMISSIONS_DRUG_DATA: &[u8] = include_bytes!("./synthetic_data/admissions_dataset/drug.csv");
const ADMISSIONS_EVENT_DATA: &[u8] =
    include_bytes!("./synthetic_data/admissions_dataset/event.csv");
const ADMISSIONS_PATIENT_DATA: &[u8] =
    include_bytes!("./synthetic_data/admissions_dataset/patient.csv");
const ADMISSIONS_PROCEDURE_DATA: &[u8] =
    include_bytes!("./synthetic_data/admissions_dataset/procedure.csv");
const ADMISSIONS_ADMISSION_DIAGNOSIS_DATA: &[u8] =
    include_bytes!("./synthetic_data/admissions_dataset/admission_diagnosis.csv");
const ADMISSIONS_ADMISSION_DRUG_DATA: &[u8] =
    include_bytes!("./synthetic_data/admissions_dataset/admission_drug.csv");
const ADMISSIONS_ADMISSION_PROCEDURE_DATA: &[u8] =
    include_bytes!("./synthetic_data/admissions_dataset/admission_procedure.csv");
const ADMISSIONS_PATIENT_ADMISSION_DATA: &[u8] =
    include_bytes!("./synthetic_data/admissions_dataset/patient_admission.csv");
const ADMISSIONS_PATIENT_EVENT_DATA: &[u8] =
    include_bytes!("./synthetic_data/admissions_dataset/patient_event.csv");

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
        let diagnosis_ids: Vec<_> = diagnosis
            .column("diagnosis_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

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
        let drug_ids: Vec<_> = drug
            .column("drug_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

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
        let patient_ids: Vec<_> = patient
            .column("patient_id")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

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
        let procedure_ids: Vec<_> = procedure
            .column("procedure_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

        let cursor = Cursor::new(SIMPLE_PATIENT_DIAGNOSIS_DATA);
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
        let patient_diagnosis_ids: Vec<_> = (0..patient_diagnosis.height() as u32).collect();

        let cursor = Cursor::new(SIMPLE_PATIENT_DRUG_DATA);
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
        let patient_drug_ids: Vec<_> = (patient_diagnosis.height() as u32
            ..(patient_diagnosis.height() + patient_drug.height()) as u32)
            .collect();

        let cursor = Cursor::new(SIMPLE_PATIENT_PROCEDURE_DATA);
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
        let patient_procedure_ids: Vec<_> = ((patient_diagnosis.height() + patient_drug.height())
            as u32
            ..(patient_diagnosis.height() + patient_drug.height() + patient_procedure.height())
                as u32)
            .collect();

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

        unsafe { medrecord.set_schema_unchecked(&mut simple_dataset_schema!()) };

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
        let diagnosis_ids: Vec<_> = diagnosis
            .column("diagnosis_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

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
        let drug_ids: Vec<_> = drug
            .column("drug_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

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
        let patient_ids: Vec<_> = patient
            .column("patient_id")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

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
        let procedure_ids: Vec<_> = procedure
            .column("procedure_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

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
        let event_ids: Vec<_> = event
            .column("event")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

        let cursor = Cursor::new(ADVANCED_PATIENT_DIAGNOSIS_DATA);
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
        let patient_diagnosis_ids: Vec<_> = (0..patient_diagnosis.height() as u32).collect();

        let cursor = Cursor::new(ADVANCED_PATIENT_DRUG_DATA);
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
        let patient_drug_ids: Vec<_> = (patient_diagnosis.height() as u32
            ..(patient_diagnosis.height() + patient_drug.height()) as u32)
            .collect();

        let cursor = Cursor::new(ADVANCED_PATIENT_PROCEDURE_DATA);
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
        let patient_procedure_ids: Vec<_> = ((patient_diagnosis.height() + patient_drug.height())
            as u32
            ..(patient_diagnosis.height() + patient_drug.height() + patient_procedure.height())
                as u32)
            .collect();

        let cursor = Cursor::new(ADVANCED_PATIENT_EVENT_DATA);
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
        let patient_event_ids: Vec<_> = ((patient_diagnosis.height()
            + patient_drug.height()
            + patient_procedure.height()) as u32
            ..(patient_diagnosis.height()
                + patient_drug.height()
                + patient_procedure.height()
                + patient_event.height()) as u32)
            .collect();

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

        unsafe { medrecord.set_schema_unchecked(&mut advanced_dataset_schema!()) };

        medrecord
    }

    pub fn from_admissions_example_dataset() -> Self {
        let cursor = Cursor::new(ADMISSIONS_ADMISSION_DATA);
        let mut admission_schema = PolarsSchema::with_capacity(4);
        admission_schema.insert("admission_id".into(), PolasrDataType::String);
        admission_schema.insert("admission_type".into(), PolasrDataType::String);
        admission_schema.insert(
            "start_time".into(),
            PolasrDataType::Datetime(polars::prelude::TimeUnit::Microseconds, None),
        );
        admission_schema.insert(
            "end_time".into(),
            PolasrDataType::Datetime(polars::prelude::TimeUnit::Microseconds, None),
        );
        let mut admission = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(admission_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        admission.rechunk_mut();
        let admission_ids: Vec<_> = admission
            .column("admission_id")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

        let cursor = Cursor::new(ADMISSIONS_DIAGNOSIS_DATA);
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
        let diagnosis_ids: Vec<_> = diagnosis
            .column("diagnosis_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

        let cursor = Cursor::new(ADMISSIONS_DRUG_DATA);
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
        let drug_ids: Vec<_> = drug
            .column("drug_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

        let cursor = Cursor::new(ADMISSIONS_EVENT_DATA);
        let mut event_schema = PolarsSchema::with_capacity(1);
        event_schema.insert("event".into(), PolasrDataType::String);
        let mut event = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(event_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        event.rechunk_mut();
        let event_ids: Vec<_> = event
            .column("event")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

        let cursor = Cursor::new(ADMISSIONS_PATIENT_DATA);
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
        let patient_ids: Vec<_> = patient
            .column("patient_id")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

        let cursor = Cursor::new(ADMISSIONS_PROCEDURE_DATA);
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
        let procedure_ids: Vec<_> = procedure
            .column("procedure_code")
            .expect("Column must exist")
            .as_materialized_series()
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect();

        let cursor = Cursor::new(ADMISSIONS_ADMISSION_DIAGNOSIS_DATA);
        let mut admission_diagnosis_schema = PolarsSchema::with_capacity(4);
        admission_diagnosis_schema.insert("admission_id".into(), PolasrDataType::String);
        admission_diagnosis_schema.insert("diagnosis_code".into(), PolasrDataType::String);
        admission_diagnosis_schema.insert(
            "time".into(),
            PolasrDataType::Datetime(polars::prelude::TimeUnit::Microseconds, None),
        );
        admission_diagnosis_schema.insert("duration_days".into(), PolasrDataType::Float64);
        let admission_diagnosis = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(admission_diagnosis_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        let admission_diagnosis_ids: Vec<_> = (0..admission_diagnosis.height() as u32).collect();

        let cursor = Cursor::new(ADMISSIONS_ADMISSION_DRUG_DATA);
        let mut admission_drug_schema = PolarsSchema::with_capacity(5);
        admission_drug_schema.insert("admission_id".into(), PolasrDataType::String);
        admission_drug_schema.insert("drug_code".into(), PolasrDataType::String);
        admission_drug_schema.insert(
            "time".into(),
            PolasrDataType::Datetime(polars::prelude::TimeUnit::Microseconds, None),
        );
        admission_drug_schema.insert("quantity".into(), PolasrDataType::Int64);
        admission_drug_schema.insert("cost".into(), PolasrDataType::Float64);
        let admission_drug = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(admission_drug_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        let admission_drug_ids: Vec<_> = (admission_diagnosis.height() as u32
            ..(admission_diagnosis.height() + admission_drug.height()) as u32)
            .collect();

        let cursor = Cursor::new(ADMISSIONS_ADMISSION_PROCEDURE_DATA);
        let mut admission_procedure_schema = PolarsSchema::with_capacity(4);
        admission_procedure_schema.insert("admission_id".into(), PolasrDataType::String);
        admission_procedure_schema.insert("procedure_code".into(), PolasrDataType::String);
        admission_procedure_schema.insert(
            "time".into(),
            PolasrDataType::Datetime(polars::prelude::TimeUnit::Microseconds, None),
        );
        admission_procedure_schema.insert("duration_minutes".into(), PolasrDataType::Float64);
        let admission_procedure = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(admission_procedure_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        let admission_procedure_ids: Vec<_> = ((admission_diagnosis.height()
            + admission_drug.height()) as u32
            ..(admission_diagnosis.height()
                + admission_drug.height()
                + admission_procedure.height()) as u32)
            .collect();

        let cursor = Cursor::new(ADMISSIONS_PATIENT_ADMISSION_DATA);
        let mut patient_admission_schema = PolarsSchema::with_capacity(2);
        patient_admission_schema.insert("patient_id".into(), PolasrDataType::String);
        patient_admission_schema.insert("admission_id".into(), PolasrDataType::String);
        let patient_admission = CsvReadOptions::default()
            .with_has_header(true)
            .with_schema_overwrite(Some(Arc::new(patient_admission_schema)))
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        let patient_admission_ids: Vec<_> = ((admission_diagnosis.height()
            + admission_drug.height()
            + admission_procedure.height()) as u32
            ..(admission_diagnosis.height()
                + admission_drug.height()
                + admission_procedure.height()
                + patient_admission.height()) as u32)
            .collect();

        let cursor = Cursor::new(ADMISSIONS_PATIENT_EVENT_DATA);
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
        let patient_event_ids: Vec<_> = ((admission_diagnosis.height()
            + admission_drug.height()
            + admission_procedure.height()
            + patient_admission.height()) as u32
            ..(admission_diagnosis.height()
                + admission_drug.height()
                + admission_procedure.height()
                + patient_admission.height()
                + patient_event.height()) as u32)
            .collect();

        let mut medrecord = Self::from_dataframes(
            vec![
                (admission, "admission_id"),
                (diagnosis, "diagnosis_code"),
                (drug, "drug_code"),
                (event, "event"),
                (patient, "patient_id"),
                (procedure, "procedure_code"),
            ],
            vec![
                (admission_diagnosis, "admission_id", "diagnosis_code"),
                (admission_drug, "admission_id", "drug_code"),
                (admission_procedure, "admission_id", "procedure_code"),
                (patient_admission, "patient_id", "admission_id"),
                (patient_event, "patient_id", "event"),
            ],
            None,
        )
        .expect("MedRecord can be built");

        medrecord
            .add_group("admission".into(), Some(admission_ids), None)
            .expect("Group can be added");
        medrecord
            .add_group("diagnosis".into(), Some(diagnosis_ids), None)
            .expect("Group can be added");
        medrecord
            .add_group("drug".into(), Some(drug_ids), None)
            .expect("Group can be added");
        medrecord
            .add_group("event".into(), Some(event_ids), None)
            .expect("Group can be added");
        medrecord
            .add_group("patient".into(), Some(patient_ids), None)
            .expect("Group can be added");
        medrecord
            .add_group("procedure".into(), Some(procedure_ids), None)
            .expect("Group can be added");

        medrecord
            .add_group(
                "admission_diagnosis".into(),
                None,
                Some(admission_diagnosis_ids),
            )
            .expect("Group can be added");
        medrecord
            .add_group("admission_drug".into(), None, Some(admission_drug_ids))
            .expect("Group can be added");
        medrecord
            .add_group(
                "admission_procedure".into(),
                None,
                Some(admission_procedure_ids),
            )
            .expect("Group can be added");
        medrecord
            .add_group(
                "patient_admission".into(),
                None,
                Some(patient_admission_ids),
            )
            .expect("Group can be added");
        medrecord
            .add_group("patient_event".into(), None, Some(patient_event_ids))
            .expect("Group can be added");

        unsafe { medrecord.set_schema_unchecked(&mut admissions_dataset_schema!()) };

        medrecord
    }
}

#[cfg(test)]
mod test {
    use super::{AttributeType, DataType};
    use crate::{
        medrecord::schema::{AttributeSchema, GroupSchema, Schema},
        MedRecord,
    };
    use std::collections::HashMap;

    #[test]
    fn test_from_simple_example_dataset() {
        let mut medrecord = MedRecord::from_simple_example_dataset();

        assert!(medrecord.set_schema(simple_dataset_schema!()).is_ok());

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

        assert!(medrecord.set_schema(advanced_dataset_schema!()).is_ok());

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
            1,
            medrecord.nodes_in_group(&"event".into()).unwrap().count()
        );
        assert_eq!(
            600,
            medrecord.nodes_in_group(&"patient".into()).unwrap().count()
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

    #[test]
    fn test_from_admissions_example_dataset() {
        let mut medrecord = MedRecord::from_admissions_example_dataset();

        assert!(medrecord.set_schema(admissions_dataset_schema!()).is_ok());

        assert_eq!(2518, medrecord.node_count());
        assert_eq!(10476, medrecord.edge_count());

        assert_eq!(
            132,
            medrecord
                .nodes_in_group(&"diagnosis".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            91,
            medrecord.nodes_in_group(&"drug".into()).unwrap().count()
        );
        assert_eq!(
            1,
            medrecord.nodes_in_group(&"event".into()).unwrap().count()
        );
        assert_eq!(
            100,
            medrecord.nodes_in_group(&"patient".into()).unwrap().count()
        );
        assert_eq!(
            194,
            medrecord
                .nodes_in_group(&"procedure".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            1413,
            medrecord
                .edges_in_group(&"admission_diagnosis".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            1329,
            medrecord
                .edges_in_group(&"admission_drug".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            5726,
            medrecord
                .edges_in_group(&"admission_procedure".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            2000,
            medrecord
                .edges_in_group(&"patient_admission".into())
                .unwrap()
                .count()
        );
        assert_eq!(
            8,
            medrecord
                .edges_in_group(&"patient_event".into())
                .unwrap()
                .count()
        );
    }
}
