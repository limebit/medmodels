# User Guide

```{toctree}
:maxdepth: 1
:hidden:

self
```

```{toctree}
:maxdepth: 2
:caption: Contents:
:hidden:

01_install_medmodels
02_medrecord
03_treatment_effect
04_medrecord_comparer
```

<h2 class="no-number">Welcome to the MedModels User Guide!</h2>

MedModels is a powerful and versatile open-source Python package designed to streamline the analysis of real-world evidence data within the healthcare domain. This user guide is your one-stop shop for everything you need to know about leveraging MedModels' functionalities to unlock valuable insights from your medical data.

MedModels' central object is the [`MedRecord`](medmodels.medrecord.medrecord.MedRecord){target="_blank"} class. Thanks to its Rust backend implementation, MedRecord provides a high efficiency and performance, even when working with large datasets. Learn more about how to use it here: [MedRecord Guide](02_medrecord.md).

Once you have your data stored in a MedRecord object, you will be able to unlock the full power of MedModels, using different modules such as [`TreatmentEffect`](medmodels.treatment_effect.treatment_effect){target="_blank"}.