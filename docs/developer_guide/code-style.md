# Style Guide

This style guide outlines the coding conventions expected for contributions to the MedModels codebase. Maintaining consistent style helps ensure code readability and maintainability for everyone involved.

**Python**

- **Formatting:** We use `black` for automatic code formatting. Commit all changes made by `black` using `black .` or `make format`.
- **Linting:** We use `ruff` for static type checking. Ensure your code passes linting before submitting a pull request.

**ruff Configuration:**

We provide a ruff configuration in `pyproject.toml` configuration file that excludes certain directories from linting. You can find the configuration details below:

```toml
[tool.ruff]
exclude = [
    "__pycache__",
    ".git",
    ".github",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "hooks",
]
line-length = 88
```

- **Line Length:** Maintain a maximum line length of 88 characters.

**Rust**

- **Formatting:** We use `rustfmt` for code formatting. Ensure your code is formatted correctly before submitting a pull request. Tools like `cargo fmt` can be used to achieve this.
- **Linting:** We use `clippy` for static code analysis and catching potential errors or inefficiencies. Ensure your code passes linting before submitting a pull request. Tools like `cargo clippy` can be used to run linting.

**Additional Notes**

- Consider using IDE plugins or extensions for both Python and Rust that integrate with `black` and `rustfmt` respectively. This allows for automatic formatting within your development environment.
- Similarly, look for extensions that integrate with `clippy` to provide linting feedback directly in your IDE.
- If you encounter any specific style-related questions or conflicts, feel free to reach out to project maintainers for clarification.

By following these guidelines, you'll contribute to a well-formatted and maintainable codebase for MedModels!
