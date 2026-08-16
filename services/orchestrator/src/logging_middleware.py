import time
from dataclasses import dataclass, field


@dataclass
class TimingLogger:
    stages: list[dict] = field(default_factory=list)
    _current_stage: str = ""
    _stage_start: float = 0

    def start(self, stage: str):
        self._current_stage = stage
        self._stage_start = time.time()

    def end(self) -> dict:
        elapsed = (time.time() - self._stage_start) * 1000
        entry = {"stage": self._current_stage, "duration_ms": round(elapsed, 2)}
        self.stages.append(entry)
        return entry

    def total_ms(self) -> float:
        return sum(s["duration_ms"] for s in self.stages)

    def reset(self):
        self.stages = []
        self._current_stage = ""
        self._stage_start = 0
