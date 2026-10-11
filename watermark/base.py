"""Watermark detector interface.

A detector may be registered only when its real, documented algorithm and its
required configuration (keys, tokenizer, scheme definition) are available.
Detectors must never infer a watermark from punctuation, generic token
frequencies, hidden characters, or similar proxies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

UNAVAILABLE_MESSAGE = "Watermark testing unavailable for this scheme."


@dataclass
class WatermarkResult:
    scheme: str
    implementation: str
    version: str
    config_status: str                 # "ok" | "missing" | "invalid" | "n/a"
    ran: bool
    message: str = ""
    eligible_tokens: Optional[int] = None
    statistic_name: str = ""
    statistic: Optional[float] = None
    null_hypothesis: str = ""
    p_value: Optional[float] = None
    adjusted_p_value: Optional[float] = None
    multiple_testing: str = ""
    limitations: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WatermarkDetector(ABC):
    scheme: str = ""
    implementation: str = ""
    version: str = "0"
    requires: Tuple[str, ...] = ()
    description: str = ""

    @abstractmethod
    def check_config(self, config: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
        """Return (is_runnable, human-readable status)."""

    @abstractmethod
    def detect(self, text: str, config: Dict[str, Any], n_tests: int = 1) -> WatermarkResult:
        """Run the actual statistical test."""

    def not_run(self, status: str, reason: str) -> WatermarkResult:
        return WatermarkResult(self.scheme, self.implementation, self.version, status, False,
                               message=f"{UNAVAILABLE_MESSAGE} {reason}".strip(), limitations=reason)
