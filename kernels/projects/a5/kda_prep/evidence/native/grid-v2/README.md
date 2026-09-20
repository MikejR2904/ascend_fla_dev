# Complete ordinary public-grid receipts and byte-identical block dimensions

Chunk bd1/2/3/4: 1620 cases each, all pass. Decode bd1/2/4/8/16/28: 3240 cases each,
all pass. Every output, cache and prep hash matches across the corresponding
retrieved block dimensions. See `cross-bd-retrieved.json` and the top-level
`evidence/summary.json` for unresolved acceptance stages.

These ten runs contain 51,870 original JSON receipts, including case results,
actual dispatch audits, complete 50-entry compilation receipts, environments and
summaries. Each JSONL line keeps the entire original receipt with its own raw
environment. Per-run `manifest.json` maps each filename to shard and line and
records its original byte count and SHA256. No receipt fields are omitted.

`restore-verification.json` and `restore-verification-high.json` record a fresh restoration and exact comparison of
every original file. From the kda_prep unit directory:

```bash
python evidence_archive.py verify evidence/native/grid-v2/grid-v2-chunk-bd1 --restore tmp/restored-grid
```

The source/evidence delivery archive and remaining native boundary acceptance
are separate unfinished stages. Special-value failures are not part of these
passing ordinary grids.
