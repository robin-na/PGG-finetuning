#!/usr/bin/env node
import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { pipeline } from "node:stream/promises";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const rootDir = resolve(dirname(new URL(import.meta.url).pathname), "..");
const outDir = resolve(rootDir, "world_values_survey/raw");
const metaDir = resolve(rootDir, "world_values_survey/metadata");
const outFile = resolve(outDir, "WVS_TimeSeries_1981_2022_Csv_v5_0.zip");

const fullName = process.env.WVS_FULL_NAME;
const institution = process.env.WVS_INSTITUTION;
const email = process.env.WVS_EMAIL;

if (!fullName || !institution || !email) {
  throw new Error("Set WVS_FULL_NAME, WVS_INSTITUTION, and WVS_EMAIL.");
}

const title = process.env.WVS_TITLE ?? "Researcher";
const project = process.env.WVS_PROJECT ?? "LLM persona prediction of Moral Machine choices";
const purpose =
  process.env.WVS_PURPOSE ??
  "Academic research comparing LLM predictions under personas generated from WVS waves 1-7.";

await mkdir(outDir, { recursive: true });
await mkdir(metaDir, { recursive: true });

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ acceptDownloads: true });
const page = await context.newPage();

const responseLog = [];
page.on("response", async (response) => {
  const url = response.url();
  if (url.includes("AJDownload")) {
    responseLog.push({
      url,
      status: response.status(),
      headers: response.headers(),
    });
  }
});
page.on("dialog", async (dialog) => {
  await dialog.accept();
});

try {
  await page.goto("https://www.worldvaluessurvey.org/AJDocumentationSmpl.jsp?CndWAVE=-1&INID=&SAID=-1", {
    waitUntil: "load",
    timeout: 120000,
  });
  await page.getByText("WVS TimeSeries 1981 2022 Csv v5 0.zip", { exact: true }).click();
  await page.waitForLoadState("load", { timeout: 30000 });

  await page.locator('input[name="LITITLE"]').fill(title);
  await page.locator('input[name="LINOMBRE"]').fill(fullName);
  await page.locator('input[name="LIEMPRESA"]').fill(institution);
  await page.locator('input[name="LIEMAIL"]').fill(email);
  await page.locator('input[name="LIPROJECT"]').fill(project);
  await page.locator('select[name="LIUSE"]').selectOption("2");
  await page.locator('textarea[name="LIPURPOSE"]').fill(purpose);
  await page.locator('input[name="LIAGREE"]').check();

  const [download] = await Promise.all([
    page.waitForEvent("download", { timeout: 120000 }),
    page.locator("#botonDwn").click(),
  ]);

  await download.saveAs(outFile);
} finally {
  await writeFile(resolve(metaDir, "wvs_playwright_response_log.json"), JSON.stringify(responseLog, null, 2));
  await browser.close();
}

const outStat = await stat(outFile);
if (outStat.size < 1000) {
  throw new Error(`Downloaded WVS file is unexpectedly small: ${outStat.size} bytes`);
}

const hash = createHash("sha256");
await pipeline(
  (await import("node:fs")).createReadStream(outFile),
  async function* (source) {
    for await (const chunk of source) {
      hash.update(chunk);
      yield chunk;
    }
  },
  createWriteStream("/dev/null"),
);

await writeFile(resolve(metaDir, "local_sha256.txt"), `${hash.digest("hex")}  ${outFile}\n`);
await writeFile(
  resolve(metaDir, "local_files.txt"),
  `${outStat.size} bytes  ${outFile}\n`,
);

console.log(`Downloaded ${outFile}`);
