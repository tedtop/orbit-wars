# website_opus — first draft of the presentation site

This was the first take on the project presentation website, built with Claude
Opus during the final days of the competition. It was superseded by
[`website_fable/`](../website_fable/) — the version that shipped to
**https://kaggle-orbit-wars.vercel.app** — but is kept as part of the record
(and because its `scripts/build_data.py` pipeline and data files were the
foundation the final site's data assets were curated from).

Its raw replay cache (`public/data/replays/`, ~1 GB of full episode JSONs) is
not committed; the compacted, curated episodes live in
`website_fable/public/data/replays/`.

```bash
npm install
npm run dev
```
