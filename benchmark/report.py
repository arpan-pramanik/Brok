import json
import sys
from pathlib import Path


def generate_report(results: list[dict]) -> str:
    stage_times: dict[str, list[float]] = {}
    total_times = []
    errors = 0
    abstentions = 0

    for r in results:
        if r.get("error"):
            errors += 1
            continue
        total_times.append(r.get("total_time_ms", r.get("benchmark_time_ms", 0)))
        if r.get("abstained"):
            abstentions += 1
        for s in r.get("stages", []):
            stage_times.setdefault(s["stage"], []).append(s["duration_ms"])

    def pct(vals, p):
        if not vals:
            return 0
        vals = sorted(vals)
        idx = min(int(len(vals) * p), len(vals) - 1)
        return round(vals[idx], 1)

    lines = ["# Benchmark Results\n"]
    lines.append(f"Total queries: {len(results)}")
    lines.append(f"Errors: {errors}")
    lines.append(f"Abstentions: {abstentions}\n")

    lines.append("## Total Latency\n")
    lines.append("| Percentile | Time (ms) |")
    lines.append("|-----------|-----------|")
    lines.append(f"| P50 | {pct(total_times, 0.5)} |")
    lines.append(f"| P70 | {pct(total_times, 0.7)} |")
    lines.append(f"| P100 | {pct(total_times, 1.0)} |\n")

    lines.append("## Per-Stage Latency\n")
    lines.append("| Stage | P50 (ms) | P70 (ms) | P100 (ms) |")
    lines.append("|-------|----------|----------|-----------|")
    for stage, times in sorted(stage_times.items()):
        lines.append(f"| {stage} | {pct(times, 0.5)} | {pct(times, 0.7)} | {pct(times, 1.0)} |")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            results = json.load(f)
    else:
        results = json.load(sys.stdin)

    report = generate_report(results)
    output = Path(__file__).parent.parent / "docs" / "BENCHMARK_RESULTS.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report)
    print(f"report written to {output}")
    print(report)
