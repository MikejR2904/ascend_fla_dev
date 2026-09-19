# Native evidence index

Archive SHA256: `25cfee4b9e0b556f5fc7079e5239bf6f1f143220919438cb5abfd2d442053269`. 447 payload files plus `archive-index.json`; every SHA256 verified after extraction. Complete test/performance logs are retained; only private machine prefixes are redacted. Device-ownership process records remain private.

Selected locations below refer to original line numbers inside the archive. These excerpts do not replace the complete logs.

`logs/suite-bd1.log`:

```text
    35  FP32_FULL_PASS
   740  NATIVE_PASS 89 1
```

`logs/suite-bd2.log`:

```text
    35  FP32_FULL_PASS
   740  NATIVE_PASS 89 2
```

`logs/suite-bd3.log`:

```text
    35  FP32_FULL_PASS
   740  NATIVE_PASS 89 3
```

`logs/suite-bd4.log`:

```text
    35  FP32_FULL_PASS
   740  NATIVE_PASS 89 4
```

`logs/closeout-bd4.log`:

```text
    35  FP32_FULL_PASS
    36  NATIVE_PASS 1 4
```

`logs/host-closeout.log`:

```text
     3  83 passed in 5.64s
```

`logs/host-closeout-full.log`:

```text
    11  708 passed, 5 skipped in 139.96s (0:02:19)
```

`logs/restore-source-tests.log`:

```text
     3  83 passed in 9.35s
```

Complete numeric receipts: `receipts/suite-bd{1,2,3,4}/summary.json`, `boundaries.json`, `environment.json`, and `compile.json`. Raw performance samples: `receipts/perf-bd4/performance.json` and `logs/perf-bd4.log`.

Recompute all 356 full-chain cases, 172 boundary checks, per-case fixed budgets, and input/stage/output byte equality with `compare_runs.py`; see the unit README.
