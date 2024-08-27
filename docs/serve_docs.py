#!/usr/bin/env python3

import logging
import os
import subprocess
from pathlib import Path

from livereload import Server, shell

logging.basicConfig(level=logging.INFO)


def setup_live_docs_server() -> None:
    """Set up and run a live documentation server.

    This function initializes a server that automatically rebuilds and refreshes
    the documentation when source files are modified.
    """
    # Initialize server
    server = Server()

    # Set working directory to script location
    script_path = Path(__file__).resolve().parent
    os.chdir(script_path)

    # Define documentation rebuild command
    rebuild_cmd = shell("make html", cwd=str(script_path))

    # Initially build the documentation
    logging.info("Building initial documentation...")
    subprocess.run(["make", "html"], cwd=str(script_path), check=True)

    # File patterns to watch for changes
    watch_patterns = [
        "*.rst",
        "*.md",
        "conf.py",
        "api/*",
        "user_guide/*",
        "developer_guide/*",
        "../medmodels/**/*",
    ]

    # Set up file watchers
    for pattern in watch_patterns:
        server.watch(pattern, rebuild_cmd, delay=1)

    # Serve documentation
    server.serve(root=str(script_path / "_build" / "html"), host="0.0.0.0")


if __name__ == "__main__":
    logging.info("Starting live documentation server...")
    logging.info("Access the docs at http://localhost:5500")
    logging.info("Press Ctrl+C to stop the server.")
    setup_live_docs_server()
