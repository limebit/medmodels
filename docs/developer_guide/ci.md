# Continuous Integration (CI) Pipelines

This section outlines the automated CI pipelines that ensure code quality and maintainability in this repository. These pipelines are triggered by various events on GitHub and use GitHub Actions to execute specific tasks.

**Benefits of CI Pipelines**

- **Early detection of issues:** CI pipelines catch formatting inconsistencies, linting errors, and failing tests early on in the development process, preventing them from being merged into the main codebase.
- **Improved code quality:** Consistent formatting and linting enforce coding standards, promoting readability and maintainability.
- **Faster bug detection:** Automated testing helps identify regressions and bugs quickly, reducing time spent on manual testing.
- **Increased developer confidence:** With CI pipelines in place, developers can be more confident that their code changes won't break existing functionality.

**Types of CI Pipelines**

This repository utilizes three primary CI pipelines:

1. **Formatting (formatting.yml):**

   - **Trigger:** Runs on pull request creation or modification targeting the `main` branch.
   - **Purpose:** Ensures consistent code formatting across Python and Rust files.
   - **Actions:**
     - Checks out the code from the repository.
     - Sets up Python 3.9 environment.
     - Sets up Rust toolchain with `rustfmt` formatter.
     - Installs development dependencies using `make install-dev`.
     - Formats Python files using `black` with the `--check` and `--verbose` flags to check for formatting errors and provide detailed output within the `medmodels` and `examples` directories.
     - Formats Rust files using `cargo fmt -- --check` to check for formatting errors without making changes.

2. **Linting (linting.yml):**

   - **Trigger:** Runs on pull request creation or modification targeting the `main` branch.
   - **Purpose:** Detects potential coding errors and style inconsistencies in Python and Rust code.
   - **Actions:**
     - Checks out the code from the repository.
     - Sets up Python 3.9 environment.
     - Sets up Rust toolchain with `clippy` linter.
     - Installs development dependencies using `make install-dev`.
     - Runs Python linter `ruff` with the `--check` and `--output-format=github` flags to check for linting errors and display results in a GitHub-friendly format for the entire project (represented by `.`).
     - Runs Rust linter `cargo clippy` with the `--all-targets`, `--all-features`, and `-D warnings` flags to check all targets and features, while suppressing informational warnings.

3. **Testing (testing.yml):**
   - **Triggers:**
     - Runs on push events to the `main` branch.
     - Runs on pull request creation or modification targeting the `main` branch.
     - Runs on a scheduled basis at 10:22 PM UTC on every Friday evening (cron: "22 14 \* \* 6").
   - **Purpose:** Executes test suites to verify code functionality across multiple Python versions.
   - **Actions:**
     - Checks out the code from the repository.
     - Sets up Python environments for each version specified in the `matrix` configuration (currently `3.9`, `3.10`, `3.11`, `3.12`).
     - Sets up Rust toolchain.
     - Runs tests using `make test`. This command likely invokes a test runner (e.g., `pytest`, `unittest`) defined elsewhere in your project.

**Understanding CI Pipeline Results**

Each pipeline run will display a status (success, failure, or pending) in the GitHub Actions tab for the corresponding pull request or commit. Clicking on the pipeline will provide detailed logs, allowing you to investigate any errors or warnings that may have occurred.

**Contributing to the Project**

To ensure your code adheres to the project's standards, it's highly recommended that you run these pipelines locally before creating a pull request. You can achieve this by running a command like `make format` or `make lint` within your local development environment.

By working with these CI pipelines, you can contribute to a high-quality codebase and maintain a smooth development workflow for everyone involved.
