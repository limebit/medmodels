---
sd_hide_title: true
---

# Home

```{toctree}
:maxdepth: 2
:caption: Contents:
:hidden:

user_guide/index
api/index
developer_guide/index
```

::::{grid}
:reverse:
:gutter: 3 4 4 4
:margin: 1 2 1 2

:::{grid-item}
:columns: 12 4 4 4

```{image} https://raw.githubusercontent.com/limebit/medmodels-static/main/logos/icon_white.svg
:class: only-dark
```

```{image} https://raw.githubusercontent.com/limebit/medmodels-static/main/logos/icon_black.svg
:width: 200px
:class: only-light
```

:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-fs-5

```{rubric} MedModels Documentation

```

The MedModels documentation is your go-to resource for exploring the package. It offers complete API descriptions and a detailed user guide, giving you everything you need to effectively utilize its features.


```{only} html
[![black](https://img.shields.io/badge/code_style-black-black.svg)](https://black.readthedocs.io/en/stable/)
![python versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
[![license](https://img.shields.io/github/license/limebit/medmodels.svg)](https://github.com/limebit/medmodels/blob/main/LICENSE)
[![license](https://github.com/limebit/medmodels/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/limebit/medmodels/actions/workflows/testing.yml)
```

:::

::::

---

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {material-outlined}`hub;1.5em;sd-mr-1` User Guide
:link-type: doc

The User Guide is your go-to resource for mastering MedModels and quickly learning the essentials for analyzing complex medical data with ease.

+++
[Learn more »](./user_guide/index.md)
:::

:::{grid-item-card} {material-outlined}`troubleshoot;1.5em;sd-mr-1` API Reference
:link-type: doc

The API reference provides detailed information on all MedModels functionalities and technical interfaces, serving as a quick lookup resource.

+++
[Learn more »](./api/index.md)
:::

:::{grid-item-card} {material-outlined}`school;1.5em;sd-mr-1` Development Guide
:link-type: doc

The Developer Guide provides comprehensive guidelines on how to contribute to MedModels.

+++
[Learn more »](developer_guide/index.md)
:::

::::
