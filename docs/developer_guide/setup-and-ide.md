# Setting up your IDE and the local environment

MedModels leverages a combination of Python and Rust code. To contribute effectively, you'll need to set up a development environment that supports both languages. This section guides you through the process.

## Setup

**Requirements:**

- Python (3.9, 3.10, or 3.11)
- Rust compiler (follow instructions from [https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install))

**Using the Makefile:**

MedModels utilizes a `Makefile` to manage development tasks. Here's a breakdown of the available commands and their functionalities:

- **prepare-venv:** Creates a virtual environment named `.venv` if it doesn't already exist.
- **install:** Sets up the virtual environment and installs the project in editable mode (meaning changes to the code are reflected without needing to reinstall).
- **install-dev:** Similar to `install`, but additionally installs development dependencies needed for running tests, linting, and code formatting.
- **install-tests:** Creates the virtual environment and installs the project along with its testing dependencies.
- **build-dev:** Installs Rust dependencies (using `maturin develop`).
- **test:** Runs both Python (using `pytest`) and Rust unit tests (using `cargo test`).
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
    "ms-python.black-formatter",
    "usernamehw.errorlens"
  ]
}
```

**VS Code Settings:**

Once you've installed the recommended extensions, consider adding the following settings to your VS Code settings file (`.vscode/settings.json`) to optimize your development workflow. These settings are provided below for your reference:

```json
{
  "editor.formatOnSave": true, // Format code automatically on save
  "editor.formatOnPaste": true, // Format pasted content
  "editor.autoIndent": "advanced", // Enable smart auto-indentation
  "files.associations": {
    "setup.cfg": "ini", // Associate .setup.cfg files with INI syntax highlighting
    "rust-toolchain": "toml" // Associate rust-toolchain files with TOML syntax highlighting
  },
  "evenBetterToml.schema.enabled": false, // Disable validation schema for TOML files (if preferred)
  "editor.rulers": [
    // Set a visual guide at column 88 for most files
    88
  ],
  "[rust]": {
    "editor.rulers": [
      // Set a visual guide at column 100 for Rust files
      100
    ]
  },
  "files.trimTrailingWhitespace": true, // Remove trailing whitespace on save
  "files.insertFinalNewline": true, // Ensure a newline at the end of files
  "python.terminal.activateEnvironment": true, // Activate virtual environment in terminal automatically
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python", // Set the default Python interpreter to the virtual environment
  "python.terminal.activateEnvInCurrentTerminal": true, // Activate virtual environment in the current terminal window
  "autoDocstring.docstringFormat": "google", // Use Google-style docstrings
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter" // Set Black as the default Python code formatter
  }
}
```

These settings will provide a streamlined development experience for working on the MedModels codebase within VS Code.
