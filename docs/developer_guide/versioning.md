# Versioning in MedModels

MedModels follows the principles of semantic versioning: [https://semver.org/](https://semver.org/) to ensure predictable changes based on version numbers.

The semantic versioning of MedModels is automatically derived from [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) used in the development process. More details on how to contribute to the codebase can be found in the [Create a Pull Reuest](./pull-request.md) section.

**Note:** Until MedModels reaches version `1.0.0`, breaking changes will result in a minor version bump (e.g., from `0.0.16` to `0.1.0`). All other changes will increment the patch version (e.g., from `0.0.16` to `0.0.17`).

## Breaking Changes: New Features and Improving Performance

MedModels prioritizes backwards compatibility. However, for significant improvements, we may introduce breaking changes with clear communication and migration guides.

## Continuous Improvement with User Focus

- We actively incorporate valuable user feedback to refine and enhance MedModels.
- To deliver the latest advancements sooner, features might occasionally be released before all possibilities are fully explored.
- When this happens, we introduce any necessary changes thoughtfully, with clear deprecation warnings and comprehensive migration guides to ensure a smooth transition for your code.

Most breaking changes typically involve minor adjustments to your code. While updates require some effort, we believe the improved functionality and overall experience of MedModels will be well worth it.

## Understanding Breaking Changes in MedModels

MedModels strives to deliver the latest advancements in medical modeling while ensuring a smooth user experience. However, occasionally, significant improvements necessitate changes to the way MedModels works. Here's a breakdown of what constitutes a "breaking change" and how we approach them:

**What are Breaking Changes?**

Breaking changes occur when modifications are made to core functionalities within MedModels. These functionalities are typically well-documented features like functions, classes, or methods that you rely on in your code.

**Examples of Breaking Changes:**

- **Retiring Outdated Features:** Sometimes, functionalities become outdated or redundant due to advancements. In such cases, we might remove these features after a clear deprecation period and provide alternative approaches.
- **Improved Algorithms, Altered Outputs:** As our algorithms evolve, the format or nature of model predictions might change. We'll do our best to document these changes and provide migration guidance.
- **Parameter Tweaks with Impact:** If modifying a function's default parameter value significantly affects existing code, it might be considered a breaking change.

**What's Not a Breaking Change?**

Here are some changes you can expect without worrying about breaking your code:

- **Bug Fixes:** We continuously fix bugs to improve MedModels' performance and stability. These fixes won't require code modifications.
- **Internal Restructuring:** Sometimes, we might reorganize internal code for better efficiency. This won't affect how you use MedModels functionalities.
- **Optional Enhancements:** We might introduce new optional parameters to existing methods, providing more flexibility without breaking existing code.

**Keeping You Informed**

We prioritize clear communication. Whenever possible, we'll introduce deprecation warnings for features planned for removal. We'll also provide detailed changelogs and migration guides to help you adapt your code seamlessly to any breaking changes.

By understanding these guidelines, you can stay informed about changes in MedModels and ensure your code continues to function effectively with the latest advancements.

## Deprecation Warnings: Saying Goodbye (But Not Without Notice)

Whenever possible, we give functionalities a heads-up before retirement! This means you'll get a deprecation warning if we plan to remove a feature. For example, imagine a function gets a promotion and a new name. We'll keep the old function around for a while, but it will politely warn you to use the new, improved version instead.

**Some Changes Require Bigger Announcements:**

However, not all improvements are as straightforward. Occasionally, major advancements to our internal algorithms might necessitate broader changes to how you interact with MedModels. In these cases, individual functionalities might not receive specific deprecation warnings. But don't worry, we'll still keep you informed! We'll provide detailed explanations in the changelog and a handy migration guide to help you smoothly transition your code to the new and improved MedModels.

## Giving You Time to Adjust: The Deprecation Window

Once a functionality is marked for retirement (deprecation), we typically give it a two-version grace period before it's officially removed. Here's how it works:

- **Before Version 1.0.0:** Imagine a function gets marked for retirement in version 0.18.3. It will stick around for two minor version bumps, finally saying goodbye in version 0.20.0.
- **After Version 1.0.0:** Once we reach version 1.0.0, things change slightly. A function deprecated in version 1.2.3 will be removed two major versions later, in version 3.0.0.

This system provides you with ample time to adjust your code. As minor versions typically release every three months, you'll generally have three to six months to prepare for any upcoming breaking changes. That means you can update your code at your own pace, ensuring a smooth transition with the latest MedModels advancements.
