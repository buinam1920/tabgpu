"""Task protocol and v0.1 single-entry task mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from tabgpu.assets import ModelSpec
from tabgpu.tasks.cutout import CutoutTask


class Task(Protocol):
    """Extensibility seam for future tasks (e.g. v0.2 upscaling) without a plugin registry."""

    name: str
    default_model: str
    js_module: str
    models: dict[str, ModelSpec]

    def job_meta(self, model_spec: ModelSpec) -> dict[str, Any]: ...

    def prepare(self, src: Path, model_spec: ModelSpec) -> tuple[bytes, dict[str, Any]]: ...

    def infer_local(self, payload_bytes: bytes, meta: dict[str, Any], session: Any) -> bytes: ...

    def validate_result(self, result_bytes: bytes, meta: dict[str, Any]) -> None: ...

    def finalize(
        self,
        src: Path,
        result_bytes: bytes,
        meta: dict[str, Any],
        out: Path,
        options: dict[str, Any],
    ) -> None: ...


TASKS: dict[str, Task] = {
    "cutout": CutoutTask(),
}

__all__ = ["CutoutTask", "TASKS", "Task"]
