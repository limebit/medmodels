mod index_mapping;
mod medrecord;
mod py_any_value;

use medrecord::Medrecord;
use pyo3::prelude::*;

#[pymodule]
fn medmodels(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Medrecord>()?;
    Ok(())
}
