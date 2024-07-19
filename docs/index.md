---
sd_hide_title: true
---

# 🏠 Home

```{toctree}
:maxdepth: 3
:caption: Contents:
:hidden:

index.md
api/index
user_guide/getstarted
```

::::{grid}
:reverse:
:gutter: 3 4 4 4
:margin: 1 2 1 2

:::{grid-item}
:columns: 12 4 4 4

```{image} ../images/medmodels_logo.svg
:width: 200px
:class: sd-m-auto
:name: landing-page-logo
```

:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-fs-5

```{rubric} MedModels Documentation

```

Yadda yadda

````{div} sd-d-flex-row

```{button-ref} user_guide/getstarted
:ref-type: doc
:color: primary
:class: sd-rounded-pill sd-mr-3

Getting Started
```
```{button-ref} api/index
:ref-type: doc
:color: secondary
:class: sd-rounded-pill

API Reference
```

````

:::

::::

---

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`markdown;1.5em;sd-mr-1` MedRecord
:link: syntax/core
:link-type: ref

The MedModels MedRecord is providing a straight forward and easy to use dataclass that
enables you to keep your medical data in a flexible graph format implemented in Rust to
make data handling as fast and efficient as possible.

+++
[Learn more »](medmodels/medrecord)
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Treatment Effect
:link: roles-directives
:link-type: ref

Use MedModels to caluclate Treatment Effects with various options of patient matching and controlling for confounding

+++
[Learn more »](roles-directives)
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` User Guide
:link: configuration
:link-type: doc

Checkout our user guide to get strated!

+++
[Learn more »](configuration)
:::

::::

---
