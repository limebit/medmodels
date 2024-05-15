use super::MedRecordAttribute;
use crate::MedRecord;
use polars::io::{csv::CsvReader, SerReader};
use std::io::Cursor;

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
        let diagnosis = CsvReader::new(cursor)
            .has_header(true)
            .finish()
            .expect("DataFrame can be built");
        let diagnosis_ids = diagnosis
            .column("diagnosis_code")
            .expect("Column must exist")
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect::<Vec<_>>();

        let cursor = Cursor::new(DRUG_DATA);
        let drug = CsvReader::new(cursor)
            .has_header(true)
            .finish()
            .expect("DataFrame can be built");
        let drug_ids = drug
            .column("drug_code")
            .expect("Column must exist")
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect::<Vec<_>>();

        let cursor = Cursor::new(PATIENT_DATA);
        let patient = CsvReader::new(cursor)
            .has_header(true)
            .finish()
            .expect("DataFrame can be built");
        let patient_ids = patient
            .column("patient_id")
            .expect("Column must exist")
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect::<Vec<_>>();

        let cursor = Cursor::new(PROCEDURE_DATA);
        let procedure = CsvReader::new(cursor)
            .has_header(true)
            .finish()
            .expect("DataFrame can be built");
        let procedure_ids = procedure
            .column("procedure_code")
            .expect("Column must exist")
            .iter()
            .map(|value| MedRecordAttribute::try_from(value).expect("AnyValue can be converted"))
            .collect::<Vec<_>>();

        let cursor = Cursor::new(PATIENT_DIAGNOSIS);
        let patient_diagnosis = CsvReader::new(cursor)
            .has_header(true)
            .finish()
            .expect("DataFrame can be built");

        let cursor = Cursor::new(PATIENT_DRUG);
        let patient_drug = CsvReader::new(cursor)
            .has_header(true)
            .finish()
            .expect("DataFrame can be built");

        let cursor = Cursor::new(PATIENT_PROCEDURE);
        let patient_procedure = CsvReader::new(cursor)
            .has_header(true)
            .finish()
            .expect("DataFrame can be built");

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
        )
        .expect("MedRecord can be built");

        medrecord
            .add_group("diagnosis".into(), Some(diagnosis_ids))
            .expect("Group can be added");
        medrecord
            .add_group("drug".into(), Some(drug_ids))
            .expect("Group can be added");
        medrecord
            .add_group("patient".into(), Some(patient_ids))
            .expect("Group can be added");
        medrecord
            .add_group("procedure".into(), Some(procedure_ids))
            .expect("Group can be added");

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
