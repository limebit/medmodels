# Internal Workflow at MedModels

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
  ]
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
  "editor.rulers": [88],
  "[rust]": {
    "editor.rulers": [100]
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

## Issue Management

We use GitHub Issues for project tracking. Issues are categorized (e.g., `feat`, `docs`), prioritized (e.g., `p-high`), and assigned to components (`python` or `rust`) using labels.

### Issue Types

1. Standard Issues
2. Epic Issues

### Guidelines

- Project management team creates and updates issues
- Weekly refinement meetings for issue discussion
- Notify project management for minor updates or additions

### Standard Issues

Standard issues are small, easily reviewable tasks. They should include a clear description and follow the established [pull request guidelines](./pull-request.md).

### Epic Issues

Epic issues are large tasks divided into smaller sub-tasks, marked with the `epic` label. They should include:

1. Task description
2. Checklist of sub-tasks

Example:

```markdown
- [ ] Sub-task 1
- [ ] Sub-task 2
- [ ] Sub-task 3
```

The project management team uses this checklist to generate sub-tasks automatically. Sub-tasks require descriptions and appropriate labels.

An epic branch is created for each epic issue. Completed sub-tasks are merged into this branch via pull requests.

#### Epic Task Pull Requests

- May include failing tests (with explanations)
- Must still adhere to linting and formatting rules and follow the [pull request guidelines](./pull-request.md)

#### Epic Pull Request

The final pull request merges the epic branch into the main branch after all sub-tasks have been merged and must include links to all sub-task pull requests.

It must also still adhere to the [pull request guidelines](./pull-request.md).
