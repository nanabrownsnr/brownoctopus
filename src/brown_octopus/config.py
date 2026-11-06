"""Runtime configuration loaded from environment variables."""

import os
import math
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


@dataclass(frozen=True)
class OctopusConfig:
    index_path: Path = Path("data/indexes/default")
    catalog_path: Path = Path("data/mcps.json")
    ttl: int = 8
    active_cap: int = 30
    min_tools: int = 4
    max_tools: int = 16
    max_returned_tools: int | None = None
    min_gap_percent: float = 2.0
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "OctopusConfig":
        return cls(
            index_path=Path(os.getenv("OCTOPUS_INDEX_PATH", "data/indexes/default")),
            catalog_path=Path(os.getenv("OCTOPUS_CATALOG_PATH", "data/mcps.json")),
            ttl=_int("OCTOPUS_CAPABILITY_TTL", 8),
            active_cap=_int("OCTOPUS_ACTIVE_CAP", 30),
            min_tools=_int("OCTOPUS_MIN_TOOLS", 4),
            max_tools=_int("OCTOPUS_MAX_TOOLS", 16),
            max_returned_tools=(None if os.getenv("OCTOPUS_MAX_RETURNED_TOOLS") is None else _int("OCTOPUS_MAX_RETURNED_TOOLS", 1)),
            min_gap_percent=_float("OCTOPUS_MIN_GAP_PERCENT", 2.0),
            log_level=os.getenv("OCTOPUS_LOG_LEVEL", "INFO"),
        )

    def validate(self) -> None:
        for name in ("min_tools", "max_tools", "max_returned_tools"):
            value = getattr(self, name)
            if value is None and name == "max_returned_tools":
                continue
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_tools > self.max_tools:
            raise ValueError("min_tools must not exceed max_tools")
        if not isinstance(self.min_gap_percent, (int, float)) or isinstance(self.min_gap_percent, bool) or not math.isfinite(self.min_gap_percent) or self.min_gap_percent < 0:
            raise ValueError("min_gap_percent must be finite and nonnegative")
        if self.ttl < 0 or self.active_cap < 1:
            raise ValueError("TTL and active capability cap must be valid")
