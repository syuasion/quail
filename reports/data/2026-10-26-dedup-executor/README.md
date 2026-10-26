# Deduplication runs

Summaries of the five quail-b runs cited in the deduplication pull request
(`claude/dedup-executor`). Each folder holds the run's `manifest.json` and its
`measurements.parquet` converted to `measurements.csv`, one row per query and
engine. The full runs, with per-item records, stay on the `quail-results`
Modal volume under `/results/benchmarks/quailb/<run id>`.

| Folder | What it is |
| --- | --- |
| `results-quail-main-*` | Quail on main; SUPPORT and RUNS, scale factor 0.5 |
| `results-quail-this-pr-*` | Quail at commit `00b7b3f4`; same queries |
| `results-stock-vllm-*` | Stock vLLM, operator-at-a-time and pipelining; same queries |
| `validation-13-queries-this-pr-*` | Quail at commit `a1514c29`; the 13-query check, scale factor 0.1 |
| `validation-13-queries-before-*` | Quail at commit `5be6855`, before the change; same check |

All runs: Qwen3 4B FP8 on one H100.
