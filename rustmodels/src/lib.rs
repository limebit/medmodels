mod medrecord;

use medrecord::PyMedRecord;
use pyo3::prelude::*;

#[pymodule]
fn _medmodels(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMedRecord>()?;
    Ok(())
}
