# Benchmark Results

Total queries: 35
Errors: 4
Abstentions: 10

## Total Latency

| Percentile | Time (ms) |
|-----------|-----------|
| P50 | 90.6 |
| P70 | 99.3 |
| P100 | 4188166.2 |

## Per-Stage Latency

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|-------|----------|----------|-----------|
| generation | 35532.1 | 53245.5 | 59374.6 |
| guardrail | 9.6 | 10.2 | 176.7 |
| retrieval | 80.3 | 87.1 | 4188037.4 |