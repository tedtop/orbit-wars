/* Focused QA: theater (desktop+mobile), lessons, console errors. */
import puppeteer from "puppeteer-core";

const OUT = process.argv[2] ?? "/tmp";
const browser = await puppeteer.launch({
  executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  headless: "new",
  args: ["--no-first-run", "--hide-scrollbars"],
});
const page = await browser.newPage();
const errors = [];
page.on("console", (m) => {
  if (m.type() === "error") errors.push(m.text());
});
page.on("pageerror", (e) => errors.push(String(e)));

await page.setViewport({ width: 1440, height: 1150 });
await page.goto("http://localhost:3199/", { waitUntil: "networkidle0", timeout: 60000 });
await page.evaluate(() => document.getElementById("theater")?.scrollIntoView({ behavior: "instant" }));
await new Promise((r) => setTimeout(r, 2500));
await page.screenshot({ path: `${OUT}/t_desktop.png` });

// scrub + switch replay to exercise controls
await page.evaluate(() => document.getElementById("lessons")?.scrollIntoView({ behavior: "instant" }));
await new Promise((r) => setTimeout(r, 900));
await page.screenshot({ path: `${OUT}/t_lessons.png` });

await page.setViewport({ width: 390, height: 844 });
await page.goto("http://localhost:3199/", { waitUntil: "networkidle0" });
await new Promise((r) => setTimeout(r, 1000));
await page.screenshot({ path: `${OUT}/t_mobile_hero.png` });
await page.evaluate(() => document.getElementById("theater")?.scrollIntoView({ behavior: "instant" }));
await new Promise((r) => setTimeout(r, 2000));
await page.screenshot({ path: `${OUT}/t_mobile_theater.png` });

console.log("console errors:", errors.length ? errors.join(" | ") : "none");
await browser.close();
