"""Pinned asset manifest and atomic SHA-256 download/verification cache."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Callable

from platformdirs import user_cache_dir

_CHUNK_SIZE = 1024 * 1024  # 1 MiB

# Files actively fetched by ort.webgpu.min.js@1.31.1 at runtime in Chrome.
REQUIRED_ORT_WEBGPU_FILES: tuple[str, ...] = (
    "ort.webgpu.min.js",
    "ort-wasm-simd-threaded.asyncify.mjs",
    "ort-wasm-simd-threaded.asyncify.wasm",
)


class AssetIntegrityError(RuntimeError):
    """Raised when an asset fails SHA-256 verification."""


class UnknownAssetError(KeyError):
    """Raised when a requested model or runtime asset is not in the pinned manifest."""


@dataclass(frozen=True)
class OrtAssetSpec:
    filename: str
    url: str
    sha256: str
    size: int
    content_type: str


@dataclass(frozen=True)
class ModelSpec:
    id: str
    task: str
    display_name: str
    filename: str
    url: str
    revision: str
    sha256: str
    size: int
    license: str
    upstream: str
    status: str
    input_name: str
    output_name: str
    input_shape: tuple[int, int, int, int]
    input_size: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    postprocess: str


@dataclass(frozen=True)
class AssetManifest:
    schema_version: int
    ort_version: str
    ort_license: str
    ort_files: dict[str, OrtAssetSpec]
    models: dict[str, ModelSpec]


def load_manifest(manifest_path: Path | None = None) -> AssetManifest:
    """Load the pinned asset manifest bundled with tabgpu (or from a custom path)."""
    if manifest_path is not None:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        raw = json.loads(
            resources.files("tabgpu")
            .joinpath("assets_manifest.json")
            .read_text(encoding="utf-8")
        )

    ort_raw = raw["ort_web"]
    ort_files: dict[str, OrtAssetSpec] = {}
    for fname, item in ort_raw["files"].items():
        ort_files[fname] = OrtAssetSpec(
            filename=fname,
            url=str(item["url"]),
            sha256=str(item["sha256"]).lower(),
            size=int(item["size"]),
            content_type=str(item["content_type"]),
        )

    models: dict[str, ModelSpec] = {}
    for model_id, item in raw["models"].items():
        models[model_id] = ModelSpec(
            id=str(item["id"]),
            task=str(item["task"]),
            display_name=str(item["display_name"]),
            filename=str(item["filename"]),
            url=str(item["url"]),
            revision=str(item["revision"]),
            sha256=str(item["sha256"]).lower(),
            size=int(item["size"]),
            license=str(item["license"]),
            upstream=str(item["upstream"]),
            status=str(item["status"]),
            input_name=str(item["input_name"]),
            output_name=str(item["output_name"]),
            input_shape=tuple(int(x) for x in item["input_shape"]),  # type: ignore[arg-type]
            input_size=(int(item["input_size"][0]), int(item["input_size"][1])),
            mean=(
                float(item["mean"][0]),
                float(item["mean"][1]),
                float(item["mean"][2]),
            ),
            std=(
                float(item["std"][0]),
                float(item["std"][1]),
                float(item["std"][2]),
            ),
            postprocess=str(item["postprocess"]),
        )

    return AssetManifest(
        schema_version=int(raw["schema_version"]),
        ort_version=str(ort_raw["version"]),
        ort_license=str(ort_raw["license"]),
        ort_files=ort_files,
        models=models,
    )


def default_cache_dir() -> Path:
    """Return the host OS cache directory for tabgpu (overridable via TABGPU_CACHE_DIR)."""
    override = os.environ.get("TABGPU_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(user_cache_dir("tabgpu", appauthor=False)).resolve()


def compute_sha256(path: Path) -> str:
    """Compute lowercase hex SHA-256 digest of a file in streaming chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest().lower()


def verify_sha256(path: Path, expected_sha256: str) -> bool:
    """Return True iff `path` exists and its SHA-256 matches `expected_sha256`."""
    if not path.is_file():
        return False
    return compute_sha256(path) == expected_sha256.lower()


def _default_downloader(url: str, dest_part: Path, timeout: float = 120.0) -> str:
    """Stream `url` into `dest_part` and return its hex SHA-256 digest."""
    req = urllib.request.Request(url, headers={"User-Agent": "tabgpu/0.1"})
    digest = hashlib.sha256()
    with urllib.request.urlopen(req, timeout=timeout) as resp, dest_part.open("wb") as fh:
        while True:
            chunk = resp.read(_CHUNK_SIZE)
            if not chunk:
                break
            fh.write(chunk)
            digest.update(chunk)
        fh.flush()
        os.fsync(fh.fileno())
    return digest.hexdigest().lower()


class AssetManager:
    """Downloads, verifies, and caches pinned ONNX models and onnxruntime-web files."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        manifest: AssetManifest | None = None,
        downloader: Callable[[str, Path], str] | None = None,
    ) -> None:
        self.cache_dir = (cache_dir or default_cache_dir()).resolve()
        self.manifest = manifest or load_manifest()
        self._downloader = downloader or _default_downloader
        self._locks_guard = threading.Lock()
        self._file_locks: dict[Path, threading.Lock] = {}
        # Process-local stat cache avoiding redundant 29MB re-hashing on every HTTP GET.
        self._verified_stats: dict[Path, tuple[int, int, str]] = {}

    @property
    def models_dir(self) -> Path:
        return self.cache_dir / "models"

    @property
    def ort_dir(self) -> Path:
        return self.cache_dir / "ort_web" / self.manifest.ort_version

    @property
    def certs_dir(self) -> Path:
        return self.cache_dir / "certs"

    def _lock_for(self, path: Path) -> threading.Lock:
        with self._locks_guard:
            lock = self._file_locks.get(path)
            if lock is None:
                lock = threading.Lock()
                self._file_locks[path] = lock
            return lock

    def _is_verified_stat(self, path: Path, expected_sha256: str) -> bool:
        try:
            st = path.stat()
        except OSError:
            return False
        cached = self._verified_stats.get(path)
        return cached == (st.st_mtime_ns, st.st_size, expected_sha256.lower())

    def _record_verified_stat(self, path: Path, expected_sha256: str) -> None:
        st = path.stat()
        self._verified_stats[path] = (st.st_mtime_ns, st.st_size, expected_sha256.lower())

    def get_model_spec(self, model_id: str) -> ModelSpec:
        if model_id not in self.manifest.models:
            valid = ", ".join(sorted(self.manifest.models))
            raise UnknownAssetError(f"Unknown model {model_id!r}. Available: {valid}")
        return self.manifest.models[model_id]

    def get_ort_spec(self, filename: str) -> OrtAssetSpec:
        if filename not in self.manifest.ort_files:
            raise UnknownAssetError(f"Unknown onnxruntime-web file {filename!r}")
        return self.manifest.ort_files[filename]

    def model_path(self, model_id: str) -> Path:
        spec = self.get_model_spec(model_id)
        return self.models_dir / spec.filename

    def ort_path(self, filename: str) -> Path:
        spec = self.get_ort_spec(filename)
        return self.ort_dir / spec.filename

    def ensure_model(self, model_id: str) -> Path:
        spec = self.get_model_spec(model_id)
        dest = self.models_dir / spec.filename
        return self._ensure_file(
            url=spec.url,
            dest=dest,
            expected_sha256=spec.sha256,
            label=f"model {spec.id} ({spec.filename})",
        )

    def ensure_ort_file(self, filename: str) -> tuple[Path, OrtAssetSpec]:
        spec = self.get_ort_spec(filename)
        dest = self.ort_dir / spec.filename
        path = self._ensure_file(
            url=spec.url,
            dest=dest,
            expected_sha256=spec.sha256,
            label=f"onnxruntime-web {spec.filename}",
        )
        return path, spec

    def ensure_ort_runtime(self) -> list[Path]:
        """Prefetch and verify all onnxruntime-web files required for WebGPU inference."""
        paths: list[Path] = []
        for fname in REQUIRED_ORT_WEBGPU_FILES:
            if fname in self.manifest.ort_files:
                p, _ = self.ensure_ort_file(fname)
                paths.append(p)
        return paths

    def ensure_ort_entry(self) -> Path:
        path, _ = self.ensure_ort_file("ort.webgpu.min.js")
        return path

    def get_cached_ort_file(self, filename: str) -> tuple[Path, OrtAssetSpec]:
        """Return a verified, already-cached ORT file without triggering outbound downloads."""
        spec = self.get_ort_spec(filename)
        dest = self.ort_dir / spec.filename
        if self._is_verified_stat(dest, spec.sha256):
            return dest, spec
        lock = self._lock_for(dest)
        with lock:
            if self._is_verified_stat(dest, spec.sha256):
                return dest, spec
            if not dest.is_file():
                raise FileNotFoundError(f"onnxruntime-web file {filename!r} is not cached")
            actual = compute_sha256(dest)
            if actual != spec.sha256.lower():
                raise AssetIntegrityError(
                    f"Cached onnxruntime-web {filename} failed SHA-256 verification."
                )
            self._record_verified_stat(dest, spec.sha256)
            return dest, spec

    def cache_status(self) -> dict[str, Any]:
        """Inspect current cache state without downloading anything."""
        models_status: dict[str, str] = {}
        for mid, spec in self.manifest.models.items():
            p = self.models_dir / spec.filename
            if not p.exists():
                models_status[mid] = "not_cached"
            elif verify_sha256(p, spec.sha256):
                models_status[mid] = "cached_ok"
            else:
                models_status[mid] = "checksum_mismatch"

        ort_status: dict[str, str] = {}
        for fname, spec in self.manifest.ort_files.items():
            p = self.ort_dir / fname
            if not p.exists():
                ort_status[fname] = "not_cached"
            elif verify_sha256(p, spec.sha256):
                ort_status[fname] = "cached_ok"
            else:
                ort_status[fname] = "checksum_mismatch"

        req_cached = sum(
            1 for f in REQUIRED_ORT_WEBGPU_FILES if ort_status.get(f) == "cached_ok"
        )
        return {
            "cache_dir": str(self.cache_dir),
            "ort_version": self.manifest.ort_version,
            "models": models_status,
            "ort_files": ort_status,
            "ort_required_cached": req_cached,
            "ort_required_total": len(REQUIRED_ORT_WEBGPU_FILES),
        }

    def _ensure_file(
        self,
        *,
        url: str,
        dest: Path,
        expected_sha256: str,
        label: str,
    ) -> Path:
        expected = expected_sha256.lower()
        if self._is_verified_stat(dest, expected):
            return dest

        dest.parent.mkdir(parents=True, exist_ok=True)
        lock = self._lock_for(dest)

        with lock:
            if self._is_verified_stat(dest, expected):
                return dest

            if dest.is_file():
                actual = compute_sha256(dest)
                if actual == expected:
                    self._record_verified_stat(dest, expected)
                    return dest
                raise AssetIntegrityError(
                    f"Cached {label} at {dest} failed SHA-256 check "
                    f"(expected {expected}, got {actual}). "
                    f"Delete the corrupted file to re-download."
                )

            part_path = dest.with_name(
                f".{dest.name}.{os.getpid()}.{secrets.token_hex(6)}.part"
            )
            try:
                actual = self._downloader(url, part_path)
                if actual.lower() != expected:
                    raise AssetIntegrityError(
                        f"Downloaded {label} from {url} failed SHA-256 check "
                        f"(expected {expected}, got {actual.lower()})."
                    )
                os.replace(part_path, dest)
                self._record_verified_stat(dest, expected)
                return dest
            except Exception:
                try:
                    part_path.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
