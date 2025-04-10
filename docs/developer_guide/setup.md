# Setting up MedModels

MedModels leverages a combination of Python and Rust code. To contribute effectively, you'll need to set up a development environment that supports both languages.

**Requirements:**

- Python (3.10, 3.11, 3.12 or 3.13)
- Rust compiler (follow instructions from [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install))

**Using the Makefile:**

MedModels utilizes a `Makefile` to manage development tasks. Here's a breakdown of the available commands and their functionalities:

- **install:** Sets up the virtual environment and installs the project in editable mode (meaning changes to the code are reflected without needing to reinstall).
- **install-dev:** Similar to `install`, but additionally installs development dependencies needed for running tests, linting, code formatting and the documentation.
- **install-tests:** Similar to install, but additionally installs development dependencies needed for running tests.
- **install-docs:** Similar to install, but additionally installs development dependencies needed for building the docs.
- **build-dev:** Builds the Rust crate and installs it as a python module (using `maturin develop`).
- **test:** Runs both Python (using `pytest`) and Rust unit tests (using `cargo test`).
- **test-python-coverage:** Runs Python tests and shows the line numbers of statements in each module that weren't executed.
- **docs** Builds the docs using `sphinx`.
- **docs-serve** Builds docs and live serves them to the localhost.
- **docs-clean** Removes all locally generated documentation files.
- **lint:** Runs code linters for both Python (using `ruff`) and Rust (using `cargo clippy`).
- **format:** Formats Python code (using `ruff`) and Rust code (using `rustfmt`).
- **clean:** Removes the virtual environment, cache directories, build artifacts, and other temporary files.
