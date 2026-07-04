/* QA screenshots: scroll to each section, wait for reveals, capture. */
import puppeteer from "puppeteer-core";

const OUT = process.argv[2] ?? "/tmp";
const URL = "http://localhost:3199/";
const SECTIONS = ["game", "scientists", "engine", "climb", "lab", "rl", "theater", "lessons", "finale"];

const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: "new",
  args: ["--no-first-run", "--hide-scrollbars"],
});
const page = await browser.newPage();
await page.setViewport({ width: 1440, height: 1150, deviceScaleFactor: 1 });
await page.goto(URL, { waitUntil: "networkidle0", timeout: 60000 });
await new Promise((r) => setTimeout(r, 1500));
await page.screenshot({ path: `${OUT}/q_hero.png` });

for (const id of SECTIONS) {
  await page.evaluate((sid) => {
    document.getElementById(sid)?.scrollIntoView({ behavior: "instant", block: "start" });
  }, id);
  await new Promise((r) => setTimeout(r, 1200));
  await page.screenshot({ path: `${OUT}/q_${id}.png` });
}

// mobile pass on hero + theater
await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 1 });
await page.goto(URL, { waitUntil: "networkidle0" });
await new Promise((r) => setTimeout(r, 1200));
await page.screenshot({ path: `${OUT}/q_mobile_hero.png` });
await page.evaluate(() => {
  document.getElementById("theater")?.scrollIntoView({ behavior: "instant" });
});
await new Promise((r) => setTimeout(r, 1200));
await page.screenshot({ path: `${OUT}/q_mobile_theater.png` });

await browser.close();
console.log("shots done");
