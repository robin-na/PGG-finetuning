---
contributors:
- Fabian Dvorak
- Regina Stumpf
- Sebastian Fehrler
- Urs Fischbacher
output:
  pdf_document: default
  html_document:
    df_print: paged
---

## Replication Package for "Adverse Reactions to the Use of Large Language Models in Social Interactions"\
Authors: F. Dvorak, R. Strumpf, S. Fehrler & U. Fischbacher\
Journal: *PNAS Nexus*

## Overview

The package contains the following items.

**Data**

Data from the online experiments.

* *MainDataRawClean.csv* (the raw data of the main experiment)
* *TuringDataRawClean.csv* (the raw data of the Turing experiment)
* *AIData.csv* (the AI decisions used in both experiments)
* *TuringStatements...csv* (the statements for the Turing experiment)

Note: The original raw data of the online experiments contain confidential information (IP Addresses). These variables have been removed before publication.

The *Data.R* script loads and processes the relevant data files from the online experiments, and writes the following csv files, which are used in the analysis:

* *MainData.csv* (processed data of main experiment)
* *TuringData.csv* (processed data of Turing tests)

See [DICTIONARY.txt](DICTIONARY.txt) for an explanation of the most important variables in these files. All data files are in the folder */data*.

**Scripts**

* *Main.R* (reproduces the results reported in the paper)
* *Config.R* (configuration script, installs the required R packages)
* *Data.R* (compiles the data of the online experiments)
* *Analysis.R* (reproduces the analysis)

To replicate the analysis, download the replication package and run the file *Main.R*. This reproduces all figures and tables reported in the paper and the supplementary meatrial. The replicator should expect the code to run for about 5 hours on a standard notebook.

## Data Availability and Provenance Statements

The authors generated the data by conducting two online experiments and hold the rights to distribute the data. The data is in the public domain, and can be accessed here: [https://osf.io/fvk2c](https://osf.io/fvk2c). The data not subject to usage and redistribution restrictions except those mentioned by the license file [LICENSE.txt](LICENSE.txt).

### Statement about Rights

We certify that the authors of the manuscript have legitimate access to and permission to use the data used in this manuscript. We have documented permission to redistribute/publish the data contained within this replication package in the [LICENSE.txt](LICENSE.txt) file.

### License for Data

The data are licensed under a Creative Commons Attribution 4.0 International Public License license. See [LICENSE.txt](LICENSE.txt) for details.

### Summary of Availability

The data are publicly available [here](https://osf.io/fvk2c).

### Details on each Data Source

| Data.Name          | Data.Files            | License            | Provided | Citation                             |
|:---------------|:----------------------------------|:-------------------|:---------|:-----------------------------|
| Main Experiment    | *MainDataRawClean.csv*       | CC Attribution 4.0 | TRUE     | Dvorak, Stumpf & Schmelz (2024) |
| Turing Experiment| *TuringRawDataClean.csv*       | CC Attribution 4.0 | TRUE     | Dvorak, Stumpf, Fehrler & Fischbacher (2025) |
| AI decisions   | *AIData.csv* | CC Attribution 4.0 | TRUE     | Dvorak, Stumpf, Fehrler & Fischbacher (2025) |

### Example for public use data collected by the authors

All data used in this study are publicly available in the [OSF repository](https://osf.io/fvk2c)(DOI: 10.17605/OSF.IO/FVK2C). The data were collected by the authors, and are available under a Creative Commons non-commercial license.

## Computational requirements

The code should run under Microsoft Windows, Linux and MacOS on any standard desktop machine or laptop.

### Software Requirements

Successful replication requires the R version >= 4.4.1 with the following R packages installed:

gld_2.6.6, sandwich_3.1-0, rlang_1.1.4, magrittr_2.0.3, furrr_0.3.1, e1071_1.7-16, compiler_4.4.1, systemfonts_1.1.0, vctrs_0.6.5, shape_1.4.6.1, pkgconfig_2.0.3, crayon_1.5.3, fastmap_1.2.0, backports_1.5.0, labeling_0.4.3, pander_0.6.5, utf8_1.2.4, rmarkdown_2.28, tzdb_0.4.0, haven_2.5.4, ragg_1.3.3, bit_4.5.0, xfun_0.47, cachem_1.1.0, jsonlite_1.8.9, highr_0.11, broom_1.0.6, parallel_4.4.1, R6_2.5.1, bslib_0.8.0, stringi_1.8.4, parallelly_1.39.0, car_3.1-2, boot_1.3-30, jquerylib_0.1.4, cellranger_1.1.0, iterators_1.0.14, Rcpp_1.0.13, knitr_1.48, zoo_1.8-12, splines_4.4.1, timechange_0.3.0, tidyselect_1.2.1, rstudioapi_0.16.0, abind_1.4-5, yaml_2.3.10, codetools_0.2-20, listenv_0.9.1, lattice_0.22-6, plyr_1.8.9, withr_3.0.1, evaluate_1.0.0, survival_3.6-4, future_1.34.0, proxy_0.4-27, zip_2.3.1, xml2_1.3.6, pillar_1.9.0, carData_3.0-5, foreach_1.5.2, generics_0.1.3, vroom_1.6.5, hms_1.1.3, munsell_0.5.1, scales_1.3.0, rootSolve_1.8.2.4, globals_0.16.3, class_7.3-22, glue_1.7.0, lmom_3.2, tools_4.4.1, data.table_1.16.0, Exact_3.3, mvtnorm_1.3-1, grid_4.4.1, colorspace_2.1-1, nlme_3.1-164, cli_3.6.3, textshaping_0.4.0, fansi_1.0.6, expm_1.0-0, viridisLite_0.4.2, svglite_2.1.3, gtable_0.3.5, broom.mixed_0.2.9.6, sass_0.4.9, digest_0.6.37, farver_2.1.2, htmltools_0.5.8.1, lifecycle_1.0.4, httr_1.4.7, bit64_4.5.2, MASS_7.3-60.2

The file *Config.R* will install all dependencies (latest version), and should be run once prior to running other scripts.

### Controlled Randomness

The random seed is set to 1 in line 27 of the analysis script *Analysis.R*.

### Memory and Runtime Requirements

No specific memory and compute-time requirements. Replicators should be able to reproduce the results using a standard 2025 desktop machine or laptop.

#### Summary

Approximate time needed to reproduce the analyses on a standard 2025 desktop machine is 5 hours.

#### Details

The code was last run on a Intel-Core i7-7500U CPU (2.70GHz), with 16GB RAM, running windows 10.

## Description of programs/code

The script *Main.R* reproduces all results.

The following R scripts are triggered by *Main.R* and each conduct one step in the analysis.

* *Config.R* installs and loads R packages.
* *Data.R* loads and the data of the experiments and processes variables.
* *Analysis.R* reproduces the results reported in the paper and supplementary materials.

### License for Code

The code is licensed under the GPL-3 license. See [LICENSE.txt](LICENSE.txt) for details.

## Instructions to Replicators

### Statistical replication

- Download the replication package to your local machine.
- Run the *Main.R* script.

### Details

The script *Main.R* installs all necessary R packages (triggered by *Config.R*) and sources several other R scripts that generate the results.

## List of tables and programs

The provided code reproduces:

- All statistics provided in text of the paper. The descriptive statistics provided in the methods section are in */tables/Descriptive_statistics.txt*. The statistics reported in the main text can be found in the file */tables/Tests_reported_in_text.txt* in the order they appear in the text.
- All tables and figures in the paper. These can be found in the folders */tables* and */figures*.

| Figure/Table      | Program         | Line Number | Output object           |
|:------------------|:----------------|:------------|:------------------------|
| Table 1           | Analysis.R      | 323         | tables/Table_1.txt      |
| Figure 1          | Analysis.R      | 1110        | figures/Figure_1.pdf    |
| Figure 2          | Analysis.R      | 1677        | figures/Figure_2.pdf    |
| Figure 3          | Analysis.R      | 1164        | figures/Figure_3.pdf    |
| Figure SM1        | Analysis.R      | 1279        | figures/Figure_SM1.pdf  |
| Figure SM2        | Analysis.R      | 1376        | figures/Figure_SM2.pdf  |
| Figure SM3        | Analysis.R      | 1678        | figures/Figure_SM3.pdf  |
| Table SM1         | Analysis.R      | 469         | tables/Table_SM1.txt    |

## Acknowledgements

The content of this page follows the template provided by Vilhuber et al., (2022). A template README for social science replication packages (v1.1). Zenodo. https://doi.org/10.5281/zenodo.7293838.

## References

Dvorak, F., Stumpf, R., Fehrler S. & Fischbacher, U. (2025). Adverse Reactions to the Use of Large Language Models in Social Interactions. PNAS Nexus

Vilhuber, L., Connolly, M., Koren, M., Llull, J., & Morrow, P. (2022). A template README for social science replication packages (v1.1). Zenodo. https://doi.org/10.5281/zenodo.7293838

---