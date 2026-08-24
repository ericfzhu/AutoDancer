# AutoDancer agent atlas

This directory contains the explorable map of AutoDancer’s live-game RL system.
It covers the system’s building blocks, the historical versions of each block,
the current measured baseline, and proposed Architecture A7.

## Files

- `atlas/data.mjs` is the single authored source of truth.
- `atlas.html` is the generated, self-contained interactive page.
- `SYSTEM.md` is the generated text twin.
- `CONTEXT.md` defines repository-specific terms used by both artifacts.
- `atlas/template.html` and `atlas/build.mjs` are vendored from the
  `system-atlas` skill so the output is reproducible.
- `atlas/data.example.mjs` is the original data-shape reference.

Rebuild after editing `data.mjs`:

```powershell
node docs/agent-system/atlas/build.mjs
```

Validate the generated inline JavaScript:

```powershell
node docs/agent-system/atlas/validate.mjs
```

Serve it locally from the repository root:

```powershell
python -m http.server 8780 --directory docs/agent-system
```

Then open <http://localhost:8780/atlas.html>. The generated page is also safe
to open directly, but a local server preserves all interactive behavior across
browsers.

## Maintenance rule

Update the version lineage and any affected flows whenever the observation
schema, policy architecture, reward profile, action contract, collection
strategy, evaluation gate, or live-worker lifecycle changes. Do not hand-edit
`SYSTEM.md` or `atlas.html`; rebuild them from `atlas/data.mjs`.
