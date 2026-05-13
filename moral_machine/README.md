# Moral Machine + World Values Survey Data

This folder is for experiments that predict Moral Machine choices using personas generated from World Values Survey (WVS) responses.

## Layout

- `moral_machine/raw/`: compressed Moral Machine files downloaded from the OSF project.
- `moral_machine/metadata/`: OSF API manifests and local checksums.
- `world_values_survey/raw/`: WVS longitudinal data archives downloaded from the WVS site.
- `world_values_survey/documentation/`: WVS codebooks and citation/reference files.
- `world_values_survey/metadata/`: WVS source notes and download headers.
- `scripts/`: reproducible download helpers.

## Source Notes

Moral Machine source: https://osf.io/3hvt2/

WVS source: https://www.worldvaluessurvey.org/AJDocumentationSmpl.jsp?CndWAVE=-1&INID=&SAID=-1

For the planned wave comparison, the WVS longitudinal time-series file is the most convenient starting point: the WVS documentation says version 3.0 combines waves 1 through 7 for 1981-2022. Keep the WVS files local only; WVS allows non-commercial research use with citation and prohibits redistribution of the original datasets.
