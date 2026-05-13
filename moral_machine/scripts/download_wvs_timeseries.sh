#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/world_values_survey/raw"
META_DIR="$ROOT_DIR/world_values_survey/metadata"
OUT_FILE="$OUT_DIR/WVS_TimeSeries_1981_2022_Csv_v5_0.zip"

: "${WVS_FULL_NAME:?Set WVS_FULL_NAME to the name to submit to WVS.}"
: "${WVS_INSTITUTION:?Set WVS_INSTITUTION to the institution to submit to WVS.}"
: "${WVS_EMAIL:?Set WVS_EMAIL to the email to submit to WVS.}"

WVS_TITLE="${WVS_TITLE:-Researcher}"
WVS_PROJECT="${WVS_PROJECT:-LLM persona prediction of Moral Machine choices}"
WVS_PURPOSE="${WVS_PURPOSE:-Academic research comparing LLM predictions under personas generated from WVS waves 1-7.}"

mkdir -p "$OUT_DIR" "$META_DIR"

COOKIE_JAR="$META_DIR/wvs_cookies.txt"
HEADERS="$META_DIR/wvs_download_headers.txt"

curl -k --fail --location --silent --show-error \
  --cookie-jar "$COOKIE_JAR" \
  --output "$META_DIR/wvs_timeseries_documentation_page.html" \
  "https://www.worldvaluessurvey.org/AJDocumentationSmpl.jsp?CndWAVE=-1&INID=&SAID=-1"

curl -k --fail --location --silent --show-error \
  --cookie "$COOKIE_JAR" \
  --cookie-jar "$COOKIE_JAR" \
  --output "$META_DIR/wvs_timeseries_license_page.html" \
  --data-urlencode "DOID=11931" \
  --data-urlencode "CndWAVE=-1" \
  --data-urlencode "SAID=-1" \
  "https://www.worldvaluessurvey.org/AJDownloadLicense.jsp"

curl -k --fail --location --show-error \
  --cookie "$COOKIE_JAR" \
  --cookie-jar "$COOKIE_JAR" \
  --dump-header "$HEADERS" \
  --output "$OUT_FILE" \
  --data-urlencode "DOID=11931" \
  --data-urlencode "CndWAVE=0" \
  --data-urlencode "SAID=0" \
  --data-urlencode "LITITLE=$WVS_TITLE" \
  --data-urlencode "LINOMBRE=$WVS_FULL_NAME" \
  --data-urlencode "LIEMPRESA=$WVS_INSTITUTION" \
  --data-urlencode "LIEMAIL=$WVS_EMAIL" \
  --data-urlencode "LIPROJECT=$WVS_PROJECT" \
  --data-urlencode "LIUSE=2" \
  --data-urlencode "LIPURPOSE=$WVS_PURPOSE" \
  --data-urlencode "LIAGREE=1" \
  "https://www.worldvaluessurvey.org/AJDownload.jsp"

if ! unzip -tq "$OUT_FILE" >/dev/null 2>&1; then
  echo "WVS download did not produce a ZIP file. Inspect $OUT_FILE and $HEADERS." >&2
  exit 1
fi

shasum -a 256 "$OUT_FILE" > "$META_DIR/local_sha256.txt"
ls -lh "$OUT_FILE" > "$META_DIR/local_files.txt"
