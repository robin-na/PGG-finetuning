# World Values Survey Data

Target source:

https://www.worldvaluessurvey.org/AJDocumentationSmpl.jsp?CndWAVE=-1&INID=&SAID=-1

For comparing personas across periods, use the WVS time-series CSV archive:

`WVS TimeSeries 1981 2022 Csv v5 0.zip`

The WVS documentation describes this as a longitudinal time-series file combining waves 1-7 for 1981-2022. WVS requires a registration/license form before downloading the dataset. Run the helper after setting the name, institution, and email to submit to WVS:

```bash
WVS_FULL_NAME="..." \
WVS_INSTITUTION="..." \
WVS_EMAIL="..." \
moral_machine/scripts/download_wvs_timeseries.sh
```

The script uses the default project title and purpose for this Moral Machine persona-prediction study. Override `WVS_TITLE`, `WVS_PROJECT`, or `WVS_PURPOSE` if needed.

WVS use constraints to keep in mind: non-commercial use, cite WVS in publications, and do not redistribute the original data files.
