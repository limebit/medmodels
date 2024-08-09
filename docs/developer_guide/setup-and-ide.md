# Setting up your IDE and the local environment

MedModels leverages a combination of Python and Rust code. To contribute effectively, you'll need to set up a development environment that supports both languages. This section guides you through the process.

## Setup

**Requirements:**

- Python (3.9, 3.10, or 3.11)
- Rust compiler (follow instructions from [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install))

**Using the Makefile:**

MedModels utilizes a `Makefile` to manage development tasks. Here's a breakdown of the available commands and their functionalities:

- **install:** Sets up the virtual environment and installs the project in editable mode (meaning changes to the code are reflected without needing to reinstall).
- **install-dev:** Similar to `install`, but additionally installs development dependencies needed for running tests, linting, and code formatting.
- **install-tests:** Creates the virtual environment and installs the project along with its testing dependencies.
- **build-dev:** Installs Rust dependencies (using `maturin develop`).
- **test:** Runs both Python (using `pytest`) and Rust unit tests (using `cargo test`).
- **docs** Builds the docs using `sphinx`.
- **docs-serve** Builds docs and live serves them to the localhost.
- **docs-clean** Removes all locally generated documentation files.
- **lint:** Runs code linters for both Python (using `ruff`) and Rust (using `cargo clippy`).
- **format:** Formats Python code using `black`.
- **clean:** Removes the virtual environment, cache directories, build artifacts, and other temporary files.

**Using the Makefile with your IDE:**

The provided `Makefile` can be integrated with most IDEs. Consult your IDE's documentation for instructions on adding custom build tasks based on Makefiles. This allows you to easily run commands like `make install-dev` or `make test` directly from within your IDE.

By following these steps and leveraging the `Makefile`, you'll have a development environment ready to contribute to the MedModels codebase!

## IDE Config: VSCode

For a smooth development experience in VS Code, we recommend installing the following extensions. You can install them directly within VS Code using the provided `extensions.json` file. Here's how to do it:

1. Open the VS Code extensions view (**Go** > **Extensions** or use the shortcut <kbd>Ctrl+Shift+X</kbd> on Windows/Linux or <kbd>Cmd+Shift+X</kbd> on macOS).
2. Click the **Import Extensions** icon (three dots stacked diagonally) in the extensions view.
3. Select the provided `extensions.json` file from your local project directory.

This will import the list of recommended extensions and prompt you to install them.

**Recommended Extensions:**

The `.vscode/extensions.json` file includes the following extensions to enhance your development experience:

```json
{
  "recommendations": [
    "ms-python.python",
    "charliermarsh.ruff",
    "tamasfe.even-better-toml",
    "ms-vscode-remote.remote-containers",
    "EditorConfig.EditorConfig",
    "eamodio.gitlens",
    "ms-vsliveshare.vsliveshare",
    "ms-vscode.makefile-tools",
    "rust-lang.rust-analyzer",
    "njpwerner.autodocstring",
    "usernamehw.errorlens"
  ],
}
```

**VS Code Settings:**

Once you've installed the recommended extensions, consider adding the following settings to your VS Code settings file (`.vscode/settings.json`) to optimize your development workflow. These settings are provided below for your reference:

```json
{
  "editor.formatOnSave": true,
  "editor.formatOnPaste": true,
  "editor.autoIndent": "advanced",
  "files.associations": {
    "setup.cfg": "ini",
    "rust-toolchain": "toml"
  },
  "editor.rulers": [
    88
  ],
  "[rust]": {
    "editor.rulers": [
      100
    ]
  },
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "python.terminal.activateEnvironment": true,
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvInCurrentTerminal": true,
  "autoDocstring.docstringFormat": "google",
  "python.analysis.typeCheckingMode": "strict",
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```
