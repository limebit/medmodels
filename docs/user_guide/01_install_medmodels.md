# Install MedModels Locally

## Prerequisites

Before installing MedModels, ensure you have the following installed on your system:

- **Python:** Compatible with Python versions 3.10, 3.11, 3.12, or 3.13.
- **Rust Compiler:** Install Rust by following the official [Rust Installation Guide](https://www.rust-lang.org/tools/install).

## Local Installation

To install MedModels locally, you have two main options:

### Option 1: Using pip (recommended for general use)

```bash
pip install medmodels
```
This command installs the latest stable version of MedModels directly from PyPI.

### Option 2: From source (recommended for development)

Clone the repository and use the provided `Makefile`:

```bash
git clone https://github.com/limebit/medmodels.git
cd medmodels
make install
```

*Available Makefile Commands*:

- `make install`: Sets up a virtual environment and installs the project in editable mode, enabling live changes without reinstallation.

- `make install-dev`: Does everything that make install does, plus installs development dependencies for testing, linting, and formatting (useful for contributors)​

- `make install-tests`: Similar to make install, but additionally installs all testing-related dependencies so you can run the test suite​

- `make install-docs`: Sets up the project and installs documentation generation dependencies (e.g., Sphinx and any doc-specific packages) to build the documentation site. Use this if you plan to build/read the docs locally.

(All the above make commands will prepare a .venv virtual environment and install the necessary packages inside it.)

## OS-Specific Instructions

### MacOS

Ensure you have [Homebrew](https://brew.sh/) installed.

1. Install dependencies:

```bash
brew install python rust
```

2. Install MedModels:

Either via pip:

```bash
pip install medmodels
```

or using the Makefile (from the cloned repo):

```bash
make install
```

### Linux

1. Install Python and Rust via your distribution's package manager:

- Debian/Ubuntu
```bash
sudo apt update
sudo apt install python3 python3-pip
curl https://sh.rustup.rs -sSf | sh
```

- Fedora
```bash
sudo dnf install python3 python3-pip rustc cargo
```

2. Install MedModels:

Using pip:

```bash
pip install medmodels
```

Or using the Makefile (if building from source from the cloned repo):

```bash
make install
```

### Windows

1. Install Python from the official [Python](https://www.python.org/downloads/) downloads page. Choose Python 3.10 or newer.

2. Install Rust By using the official [Rust installer](https://www.rust-lang.org/tools/install).

3. Ensure both Python and Cargo (Rust package manager) are added to your PATH.

#### Option 1:

Using pip (recommended for standard usage):

```bash
pip install medmodels
```

Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```


#### Option 2:

From source (for development, using the provided Makefile). It will configure that virtual environment directly.

```bash
make install
```

If you encounter build issues on Windows, double-check the Rust installation via [Rustup](https://www.rust-lang.org/tools/install) and confirm your Rust toolchain (cargo) is properly configured.