use super::{datatypes::DataType, AttributeType, GroupSchema, MedRecordAttribute, Schema};
use crate::MedRecord;
use polars::{io::SerReader, prelude::CsvReadOptions};
use std::{collections::HashMap, io::Cursor};

const DIAGNOSIS_DATA: &[u8] = include_bytes!("./synthetic_data/diagnosis.csv");
const DRUG_DATA: &[u8] = include_bytes!("./synthetic_data/drug.csv");
const PATIENT_DATA: &[u8] = include_bytes!("./synthetic_data/patient.csv");
const PROCEDURE_DATA: &[u8] = include_bytes!("./synthetic_data/procedure.csv");
const PATIENT_DIAGNOSIS: &[u8] = include_bytes!("./synthetic_data/patient_diagnosis.csv");
const PATIENT_DRUG: &[u8] = include_bytes!("./synthetic_data/patient_drug.csv");
const PATIENT_PROCEDURE: &[u8] = include_bytes!("./synthetic_data/patient_procedure.csv");

impl MedRecord {
    pub fn from_example_dataset() -> Self {
        let cursor = Cursor::new(DIAGNOSIS_DATA);
        let mut diagnosis = CsvReadOptions::default()
            .with_has_header(true)
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

        let cursor = Cursor::new(DRUG_DATA);
        let mut drug = CsvReadOptions::default()
            .with_has_header(true)
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

        let cursor = Cursor::new(PATIENT_DATA);
        let mut patient = CsvReadOptions::default()
            .with_has_header(true)
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

        let cursor = Cursor::new(PROCEDURE_DATA);
        let mut procedure = CsvReadOptions::default()
            .with_has_header(true)
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

        let cursor = Cursor::new(PATIENT_DIAGNOSIS);
        let patient_diagnosis = CsvReadOptions::default()
            .with_has_header(true)
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        let patient_diagnosis_ids = (0..patient_diagnosis.height() as u32).collect::<Vec<_>>();

        let cursor = Cursor::new(PATIENT_DRUG);
        let patient_drug = CsvReadOptions::default()
            .with_has_header(true)
            .into_reader_with_file_handle(cursor)
            .finish()
            .expect("DataFrame can be built");
        let patient_drug_ids = (patient_diagnosis.height() as u32
            ..(patient_diagnosis.height() + patient_drug.height()) as u32)
            .collect::<Vec<_>>();

        let cursor = Cursor::new(PATIENT_PROCEDURE);
        let patient_procedure = CsvReadOptions::default()
            .with_has_header(true)
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

        medrecord.schema = Schema {
            groups: HashMap::from([
                (
                    "diagnosis".into(),
                    GroupSchema {
                        nodes: HashMap::from([("description".into(), DataType::String.into())]),
                        edges: HashMap::new(),
                        strict: Some(true),
                    },
                ),
                (
                    "drug".into(),
                    GroupSchema {
                        nodes: HashMap::from([("description".into(), DataType::String.into())]),
                        edges: HashMap::new(),
                        strict: Some(true),
                    },
                ),
                (
                    "patient".into(),
                    GroupSchema {
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
                        strict: Some(true),
                    },
                ),
                (
                    "procedure".into(),
                    GroupSchema {
                        nodes: HashMap::from([("description".into(), DataType::String.into())]),
                        edges: HashMap::new(),
                        strict: Some(true),
                    },
                ),
                (
                    "patient_diagnosis".into(),
                    GroupSchema {
                        nodes: HashMap::new(),
                        edges: HashMap::from([
                            (
                                "diagnosis_time".into(),
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
                        strict: Some(true),
                    },
                ),
                (
                    "patient_drug".into(),
                    GroupSchema {
                        nodes: HashMap::new(),
                        edges: HashMap::from([
                            (
                                "start_time".into(),
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
                        strict: Some(true),
                    },
                ),
                (
                    "patient_procedure".into(),
                    GroupSchema {
                        nodes: HashMap::new(),
                        edges: HashMap::from([
                            (
                                "procedure_time".into(),
                                (DataType::DateTime, AttributeType::Temporal).into(),
                            ),
                            (
                                "duration_minutes".into(),
                                (DataType::Int, AttributeType::Continuous).into(),
                            ),
                        ]),
                        strict: Some(true),
                    },
                ),
            ]),
            default: None,
            strict: Some(true),
        };

        medrecord
    }
}

#[cfg(test)]
mod test {
    use crate::MedRecord;

    #[test]
    fn test_from_exmaple_dataset() {
        let medrecord = MedRecord::from_example_dataset();

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
    }
}
