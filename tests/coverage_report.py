"""Per-test isolated coverage runner.

Usage:
    python tests/report.py table  # Run each test in isolation, print coverage table
    python tests/report.py xml    # Run each test in islotation, generate coverage.xml
"""

import argparse
import fnmatch
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import coverage

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEDMODELS_ROOT = PROJECT_ROOT / "medmodels"
TESTS_ROOT = PROJECT_ROOT / "tests"
COVERAGE_FOLDER = PROJECT_ROOT / "coverage"
COVERAGE_XML = COVERAGE_FOLDER / "coverage.xml"

IGNORE_PATTERNS = [
    "**/__init__.py",
]


def is_ignored(path: Path) -> bool:
    """Check if a path should be ignored based on ignore patterns.

    Args:
        path (Path): Path to check.

    Returns:
        bool: True if the path should be ignored, False otherwise.
    """
    return any(fnmatch.fnmatch(str(path), pattern) for pattern in IGNORE_PATTERNS)


def map_to_test(module_path: Path) -> Path:
    """Map a module path to its corresponding test file path.

    Args:
        module_path (Path): Module file path.

    Returns:
        Path: full test path
    """
    rel = module_path.relative_to(MEDMODELS_ROOT)

    test_name = f"test_{rel.name}"

    if rel.stem.startswith("_"):
        return TESTS_ROOT / rel.with_name(f"test_{rel.stem.removeprefix('_')}.py")

    return TESTS_ROOT / rel.with_name(test_name)


def get_coverage_path(module_path: Path) -> Path:
    """Get the coverage file path for a given module.

    Args:
        module_path (Path): Module file path.

    Returns:
        Path: Coverage file path.
    """
    rel = module_path.relative_to(MEDMODELS_ROOT)
    return COVERAGE_FOLDER / rel.parent / f"{rel.stem}.coverage"


def main() -> None:
    """Main function to parse arguments and run the coverage report generation."""
    parser = argparse.ArgumentParser(
        description="Generate coverage reports for Python tests."
    )
    parser.add_argument(
        "mode",
        choices=["table", "xml"],
        help="Mode of operation: 'table' for isolated coverage, 'xml' for unified coverage.",
    )

    args = parser.parse_args()

    if COVERAGE_FOLDER.exists():
        logger.info("Removing existing coverage folder: %s", COVERAGE_FOLDER.resolve())
        shutil.rmtree(COVERAGE_FOLDER)

    COVERAGE_FOLDER.mkdir()

    modules = {
        module_path: (map_to_test(module_path), get_coverage_path(module_path))
        for module_path in MEDMODELS_ROOT.rglob("*.py")
        if not is_ignored(module_path)
    }

    for module_path, (test_path, coverage_path) in modules.items():
        if test_path.exists():
            logger.info(
                "Isolated run: %s testing %s",
                test_path,
                module_path.relative_to(MEDMODELS_ROOT),
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--data-file",
                    str(coverage_path),
                    "--include",
                    str(module_path),
                    "-m",
                    "pytest",
                    str(test_path),
                    "-q",
                    "--disable-warnings",
                    "-W error",
                ],
                check=True,
            )
        else:
            logger.warning("Test file does not exist for: %s", module_path)

            cov = coverage.Coverage(
                data_file=coverage_path,
                include=[str(module_path)],
            )
            cov.start()
            cov.stop()
            data = cov.get_data()
            data.touch_file(str(module_path))
            cov.save()

    combined_cov = coverage.Coverage()
    combined_cov.combine(
        [str(coverage_path) for (_, coverage_path) in modules.values()],
        keep=True,
    )

    combined_cov.save()

    if args.mode == "table":
        combined_cov.report(
            show_missing=True,
        )
    elif args.mode == "xml":
        combined_cov.xml_report(
            outfile=str(COVERAGE_XML),
        )
        logger.info("Coverage report saved to: %s", COVERAGE_XML.resolve())


if __name__ == "__main__":
    main()
