# FunASR product website and documentation

This is the source for www.funasr.com. It uses Jinja templates and a static
Python build; no browser framework or runtime application server is required.

## Content ownership

- `data/documentation.json`: task groups, bilingual titles and Markdown sources.
- Repository `docs/*.md`: documentation body, shared with GitHub readers.
- `data/deployments.json`: versions, hardware, evidence and operational limits.
- `templates/`: generated homepage, deployment manuals and documentation shell.
- `assets/css/experience.css`: shared product and legacy reading experience.
- `legacy/`: preserved public routes, content and demo assets. Do not delete or
  replace this snapshot with a partial output tree.
- Repository `gh-pages-output/index.html` and `zh/index.html`: GitHub Pages entry
  points. Their existing API/tutorial routes remain available.

The build emits search indexes, maps source-relative document links, preserves
Unicode heading anchors and fingerprints all product assets. Historical
benchmark evidence must retain its exact tested model/runtime/hardware; new
upstream support is not evidence that old tests covered a new checkpoint.

## Validate

```sh
python -m pip install -r web-pages/product-site/requirements-site.txt
python -m pytest web-pages/product-site/tests -q
python web-pages/product-site/build.py --output /tmp/funasr-site
python web-pages/product-site/validate.py /tmp/funasr-site
```

Browser tests live in `tests/browser/`. Install their pinned npm dependencies
and Chromium, then run `npm test` from that directory. The suite owns its local
HTTP server and rejects reuse of an occupied port.

Production releases use `web-pages/scripts/deploy-product-site.sh`; see
`docs/operations/funasr-com-site-release.md` for backup, atomic switching and
rollback. Only publish a tested artifact, never a partially built output.
