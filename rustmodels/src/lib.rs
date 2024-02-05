mod errors;
mod medrecord;
mod py_any_value;

use medrecord::PyMedRecord;
use pyo3::prelude::*;

#[pymodule]
fn _medmodels(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyMedRecord>()?;
    Ok(())
}
