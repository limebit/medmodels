---
sd_hide_title: true
---

# Home

```{toctree}
:maxdepth: 2
:caption: Contents:
:hidden:

api/index
user_guide/index
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

````{div} sd-d-flex-row

```{button-ref} user_guide/index
:ref-type: doc
:color: primary
:class: sd-rounded-pill sd-mr-3

User Guide
```
```{button-ref} api/index
:ref-type: doc
:color: secondary
:class: sd-rounded-pill

API Reference
```

````

```{only} html
[![black](https://img.shields.io/badge/code_style-black-black.svg)](https://black.readthedocs.io/en/stable/)
![python versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![license](https://img.shields.io/github/license/limebit/medmodels.svg)](https://github.com/limebit/medmodels/blob/main/LICENSE)
[![license](https://github.com/limebit/medmodels/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/limebit/medmodels/actions/workflows/testing.yml)
```

:::

::::

---

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {material-outlined}`hub;1.5em;sd-mr-1` MedRecord
:link-type: doc

The {py:class}`MedRecord() <medmodels.medrecord.medrecord.MedRecord>` class is providing a straight forward and easy to use dataclass that enables you to keep your medical data in a flexible graph format implemented in Rust to make data handling as fast and efficient as possible.

+++
[Learn more »](./api/medrecord.md)
:::

:::{grid-item-card} {material-outlined}`troubleshoot;1.5em;sd-mr-1` Treatment Effect
:link-type: doc

Use MedModels to caluclate {py:class}`TreatmentEffect() <medmodels.treatment_effect.treatment_effect.TreatmentEffect>` with various options of patient matching and controlling for confounding

+++
[Learn more »](./api/treatment_effect.md)
:::

:::{grid-item-card} {material-outlined}`school;1.5em;sd-mr-1` User Guide
:link-type: doc

Checkout our user guide to get strated!

+++
[Learn more »](user_guide/index.md)
:::

::::
