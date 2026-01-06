# medmodels-pyo3-interop

Interoperability layer for passing MedModels types between independent PyO3 extension modules.

## Overview

PyO3's `#[pyclass]` stores type objects in static storage, meaning each compiled extension module gets its own copy. When two separate PyO3 extensions try to share a type like `MedRecord`, Python sees them as distinct types, breaking compatibility ([PyO3#1444](https://github.com/PyO3/pyo3/issues/1444)).

This crate provides a workaround by serializing `MedRecord` to bytes (via bincode) when crossing extension boundaries, allowing external Rust-based Python extensions to accept and return `MedRecord` objects from the main `medmodels` package.

## Key Components

- **`PyMedRecord`** - A wrapper around `MedRecord` implementing PyO3's `FromPyObject` and `IntoPyObject` traits via binary serialization
- **Type Re-exports** - Common Python binding types (`PyNodeIndex`, `PyEdgeIndex`, `PyGroup`, `PyAttributes`, etc.)

## Usage

```rust
use medmodels_pyo3_interop::PyMedRecord;
use pyo3::prelude::*;

#[pyfunction]
fn process_medrecord(record: PyMedRecord) -> PyResult<PyMedRecord> {
    // Access the inner MedRecord
    let inner = record.0;

    // Process and return
    Ok(inner.into())
}
```
