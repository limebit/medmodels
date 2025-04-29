# Creating a Pull Request

Once you've made your changes to the MedModels codebase and are ready to share them, you'll submit a pull request (PR). Here's a guide to walk you through the process:

**1. Push your Local Branch:**

- Commit your changes locally using clear and concise commit messages following the Conventional Commits standard guideline ([https://www.conventionalcommits.org/en/v1.0.0/](https://www.conventionalcommits.org/en/v1.0.0/)).
- This standard encourages using a specific format for commit messages that categorizes your changes. Here's the breakdown:

  - **Type:** Start your message stating the PR's type like `feat`, `fix`, `refactor`, `docs`, etc. We use the following types in this project:
    - `build`: Changes related to build infrastructure or tools.
    - `chore`: Changes that don't directly affect functionality, like updating dependencies or formatting.
    - `ci`: Changes related to the continuous integration pipeline.
    - `docs`: Changes to documentation or comments.
    - `feat`: Introduction of a new feature or functionality.
    - `fix`: Bug fixes or resolving issues.
    - `test`: Adding or improving tests.
    - `refactor`: Code improvements that don't introduce new features or fix bugs (e.g., improving code structure or readability).

- Push your local branch to your remote repository on GitHub.

**2. Check for Consistency:**

- Before creating a pull request, ensure that if you've changed or refactored any functionality, all instances where this functionality is used (including in docstrings and other modules) are correctly adjusted to reflect the new style or implementation.

**3. Ensure Test Coverage:**

- MedModels requires 100% Python isolated test coverage. That means that each module should be tested on its own (isolated coverage).
- For every Python file in the main `medmodels/` directory, there has to be a matching test file in the `tests/` directory (e.g., `medmodels/utils.py` → `tests/test_utils.py`).
- Use the Makefile command `make test-python-coverage` to run the tests with isolated coverage reporting. Also use the Makefile command `make test-python-coverage-non-isolated` to make sure all the tests are also 100% covered.
- If your code is not fully covered, add or revise tests accordingly before submitting your PR.

**4. Create a Pull Request:**

- Navigate to the [MedModels pull requests](https://github.com/limebit/medmodels/pulls) and create a new PR.
- Choose the branch you want your changes merged into from the "base" branch dropdown (usually the `main` branch).
- Provide a clear and descriptive title for your pull request that reflects the changes you made.
  - The title should follow the Conventional Commits type (e.g., "feat: Add support for new image format").
- In the body of the pull request, elaborate on your changes and provide any additional context or testing instructions if needed.

**5. Address Feedback and Merge:**

- Once submitted, reviewers may provide feedback or request changes. Address these comments and make any necessary adjustments to your code. Review comments should only be resolved by the comment author, not the pull request author.
- After receiving approval from reviewers, admins can merge your pull request into the main branch.
