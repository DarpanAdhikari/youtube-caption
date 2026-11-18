from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable


@dataclass
class ProgressEvent:
    stage: str
    message: str = ""
    current: int | None = None
    total: int | None = None
    detail: str = ""
    timestamp: str = ""

    @property
    def percent(self) -> float | None:
        if self.current is None or not self.total:
            return None
        return min(100.0, max(0.0, 100.0 * self.current / self.total))

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "message": self.message,
            "current": self.current,
            "total": self.total,
            "detail": self.detail,
            "percent": self.percent,
            "timestamp": self.timestamp,
        }


class ProgressReporter:
    def __init__(
        self,
        on_event: Callable[[ProgressEvent], None] | None = None,
        quiet: bool = False,
    ):
        self.on_event = on_event
        self.quiet = quiet

    def emit(
        self,
        stage: str,
        message: str = "",
        current: int | None = None,
        total: int | None = None,
        detail: str = "",
    ) -> ProgressEvent:
        event = ProgressEvent(
            stage=stage,
            message=message,
            current=current,
            total=total,
            detail=detail,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        if self.on_event is not None:
            self.on_event(event)
        elif not self.quiet:
            print(self.format_event(event), flush=True)
        return event

    def stage(self, stage: str, message: str = "") -> ProgressEvent:
        return self.emit(stage, message=message)

    def update(
        self,
        stage: str,
        current: int,
        total: int,
        message: str = "",
        detail: str = "",
    ) -> ProgressEvent:
        return self.emit(stage, message=message, current=current, total=total, detail=detail)

    def log(self, message: str) -> ProgressEvent:
        return self.emit("log", message=message)

    @staticmethod
    def format_event(event: ProgressEvent) -> str:
        prefix = f"[{event.stage}]"
        text = f"{prefix} {event.message}".rstrip()
        if event.current is not None and event.total:
            text += f" ({event.current}/{event.total}, {event.percent:.0f}%)"
        if event.detail:
            text += f" - {event.detail}"
        return text


class NullProgress(ProgressReporter):
    def __init__(self):
        super().__init__(quiet=True)
