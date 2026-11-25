"""Image pipeline, local CPU fallback, job lease coordinator, and reporting."""

from __future__ import annotations

import csv
import io
import math
import os
import re
import secrets
import struct
import threading
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np
from PIL import Image, ImageCms, ImageFilter, ImageOps

SUPPORTED_INPUT_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}

_HEX_COLOR_RE = re.compile(r"^#([0-9a-fA-F]{6})$")
_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_CSV_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


class OutputSafetyError(ValueError):
    """Raised when an output path would overwrite a source/existing file or collide."""


class InvalidMaskError(ValueError):
    """Raised when an uploaded mask PNG is malformed or has unexpected format/dimensions."""


class WorkerConflictError(RuntimeError):
    """Raised when a second worker tries to register while an active worker holds the session."""


class WorkerClosedError(RuntimeError):
    """Raised when the host is no longer accepting LAN workers (CPU fallback or shutdown)."""


class StaleLeaseError(RuntimeError):
    """Raised when a worker references an expired, unknown, or superseded lease_id."""


class PrepareFailedError(RuntimeError):
    """Raised when a source image cannot be decoded or prepared (job marked failed)."""


class UnknownJobError(KeyError):
    """Raised when a worker references an unknown job_id."""


def _collision_key(path: Path) -> str:
    """Case- and Unicode-normalization-insensitive key (safe across Windows, macOS APFS, Linux)."""
    return unicodedata.normalize("NFC", str(path.resolve())).casefold()


def _sanitize_csv_cell(value: str) -> str:
    """Prevent spreadsheet formula injection in untrusted/path CSV cells."""
    if value and value[0] in _CSV_FORMULA_PREFIXES:
        return "'" + value
    return value


def parse_bg_color(bg: str) -> tuple[int, int, int] | None:
    """Parse --bg option ('transparent', 'white', or '#RRGGBB') into RGB tuple or None."""
    val = bg.strip().lower()
    if val == "transparent":
        return None
    if val == "white":
        return (255, 255, 255)
    match = _HEX_COLOR_RE.match(bg.strip())
    if not match:
        raise ValueError(
            f"Invalid --bg value {bg!r}. Expected 'transparent', 'white', or '#RRGGBB'."
        )
    hex_str = match.group(1)
    return (int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))


def resolve_output_format(fmt: str, bg: str, src: Path) -> Literal["png", "jpg", "webp"]:
    """Resolve output format and validate compatibility with background transparency."""
    f = fmt.strip().lower()
    if f == "jpeg":
        f = "jpg"
    if f not in ("auto", "png", "jpg", "webp"):
        raise ValueError(
            f"Invalid --format value {fmt!r}. Expected 'auto', 'png', 'jpg', or 'webp'."
        )
    bg_rgb = parse_bg_color(bg)
    if f == "jpg" and bg_rgb is None:
        raise ValueError(
            "Incompatible options: --format jpg cannot preserve --bg transparent. "
            "Use --format png/webp or specify a solid --bg (e.g. --bg white)."
        )
    if f == "auto":
        if bg_rgb is None:
            return "png"
        ext = src.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            return "jpg"
        if ext == ".webp":
            return "webp"
        return "png"
    return f  # type: ignore[return-value]


def discover_inputs(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand files and directories into a deterministic sorted list of supported image files."""
    found: list[Path] = []
    seen: set[str] = set()
    for raw in inputs:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"Input path does not exist: {p}")
        if p.is_dir():
            candidates = sorted(
                (c for c in p.iterdir() if c.is_file() and c.suffix.lower() in SUPPORTED_INPUT_EXTS),
                key=lambda x: x.name.lower(),
            )
            for c in candidates:
                key = os.path.normcase(str(c.resolve()))
                if key not in seen:
                    seen.add(key)
                    found.append(c)
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED_INPUT_EXTS:
                raise ValueError(f"Unsupported image extension for input file: {p}")
            key = os.path.normcase(str(p.resolve()))
            if key not in seen:
                seen.add(key)
                found.append(p)
        else:
            raise ValueError(f"Input path is neither a file nor directory: {p}")
    return found


def plan_outputs(
    sources: list[Path],
    out_dir: Path,
    *,
    fmt: str = "auto",
    bg: str = "transparent",
    overwrite: bool = False,
) -> list[tuple[Path, Path, Literal["png", "jpg", "webp"]]]:
    """Map source files to output paths, rejecting source overwrites, collisions, and existing files."""
    if not sources:
        raise ValueError("No input images found.")

    out_dir_resolved = out_dir.resolve()
    src_keys = {_collision_key(s) for s in sources}
    seen_out: dict[str, Path] = {}
    planned: list[tuple[Path, Path, Literal["png", "jpg", "webp"]]] = []

    for src in sources:
        resolved_fmt = resolve_output_format(fmt, bg, src)
        out_path = out_dir_resolved / f"{src.stem}.{resolved_fmt}"
        out_key = _collision_key(out_path)

        # Always refuse to overwrite a source file (handles case-insensitive APFS, NFC/NFD, symlinks).
        same_as_source = out_key in src_keys
        if not same_as_source and out_path.exists():
            for existing_src in sources:
                try:
                    if existing_src.exists() and os.path.samefile(out_path, existing_src):
                        same_as_source = True
                        break
                except OSError:
                    continue
        if same_as_source:
            raise OutputSafetyError(
                f"Refusing to overwrite source file: output {out_path} matches an input file."
            )

        if out_key in seen_out:
            raise OutputSafetyError(
                f"Output path collision: {seen_out[out_key]} and {src} both map to {out_path}."
            )

        if not overwrite and out_path.exists():
            raise OutputSafetyError(
                f"Refusing to overwrite existing output file {out_path}. "
                "Pass --overwrite to replace existing files."
            )

        seen_out[out_key] = src
        planned.append((src, out_path, resolved_fmt))

    return planned


def load_image_srgb(path: Path) -> Image.Image:
    """Load an image, apply EXIF orientation, normalize 16-bit modes, and convert ICC to sRGB.

    Returns an RGBA image if the source had transparency, otherwise RGB.
    """
    with Image.open(path) as raw_img:
        icc_bytes = raw_img.info.get("icc_profile")
        img = ImageOps.exif_transpose(raw_img)
        if img is None:
            img = raw_img.copy()
        else:
            img.load()

        # Normalize 16-bit integer grayscale modes without clipping to 255.
        if img.mode.startswith("I;16") or img.mode == "I":
            arr = np.asarray(img, dtype=np.float32)
            if img.mode.startswith("I;16") or float(np.max(arr)) > 255.0:
                scaled = np.clip(np.rint(arr / 257.0), 0.0, 255.0).astype(np.uint8)
            else:
                scaled = np.clip(np.rint(arr), 0.0, 255.0).astype(np.uint8)
            img = Image.fromarray(scaled, mode="L")

        has_alpha = img.mode in ("RGBA", "LA", "PA") or (
            img.mode == "P" and "transparency" in img.info
        )

        if img.mode == "CMYK":
            if icc_bytes:
                try:
                    src_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
                    return ImageCms.profileToProfile(
                        img, src_prof, _SRGB_PROFILE, outputMode="RGB"
                    )
                except Exception:
                    pass
            return img.convert("RGB")

        if has_alpha:
            rgba = img.convert("RGBA")
            alpha = rgba.getchannel("A")
            rgb = rgba.convert("RGB")
            rgb = _convert_rgb_icc_to_srgb(rgb, icc_bytes)
            rgb.putalpha(alpha)
            return rgb

        rgb = img.convert("RGB")
        return _convert_rgb_icc_to_srgb(rgb, icc_bytes)


def _convert_rgb_icc_to_srgb(rgb: Image.Image, icc_bytes: bytes | None) -> Image.Image:
    if not icc_bytes:
        return rgb
    try:
        src_prof = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        converted = ImageCms.profileToProfile(
            rgb, src_prof, _SRGB_PROFILE, outputMode="RGB"
        )
        return converted if converted is not None else rgb
    except Exception:
        return rgb


def encode_clean_rgb_png(img: Image.Image, size: tuple[int, int]) -> bytes:
    """Resize an sRGB image and encode an EXIF- and ICC-free RGB PNG for worker transfer."""
    rgb = img.convert("RGB")
    if rgb.size != size:
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)
    # Build a clean image from raw pixels so Pillow drops any carried iCCP/EXIF metadata.
    clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    buf = io.BytesIO()
    clean.save(buf, format="PNG", optimize=False, icc_profile=None)
    return buf.getvalue()


def validate_grayscale_png_mask(
    data: bytes,
    expected_width: int,
    expected_height: int,
) -> Image.Image:
    """Strictly validate PNG header, dimensions, 1-channel grayscale mode, and full decode."""
    if not isinstance(data, (bytes, bytearray)) or len(data) < 33:
        raise InvalidMaskError("Mask payload is too small to be a valid PNG.")
    if data[:8] != _PNG_SIG:
        raise InvalidMaskError("Mask payload does not have a valid PNG signature.")

    ihdr_len = struct.unpack(">I", data[8:12])[0]
    ihdr_type = data[12:16]
    if ihdr_type != b"IHDR" or ihdr_len != 13:
        raise InvalidMaskError("PNG missing valid IHDR chunk.")

    width, height, bit_depth, color_type, compression, filter_method, interlace = (
        struct.unpack(">IIBBBBB", data[16:29])
    )
    if width != expected_width or height != expected_height:
        raise InvalidMaskError(
            f"Mask dimensions {width}x{height} do not match expected "
            f"{expected_width}x{expected_height}."
        )
    if bit_depth != 8 or color_type != 0:
        raise InvalidMaskError(
            f"Mask must be an 8-bit 1-channel grayscale PNG "
            f"(got bit_depth={bit_depth}, color_type={color_type})."
        )
    if compression != 0 or filter_method != 0 or interlace not in (0, 1):
        raise InvalidMaskError("Unsupported PNG IHDR parameters.")

    try:
        with Image.open(io.BytesIO(data)) as decoded:
            if decoded.format != "PNG":
                raise InvalidMaskError(f"Unexpected image format: {decoded.format!r}")
            if decoded.size != (expected_width, expected_height):
                raise InvalidMaskError(
                    f"Decoded size {decoded.size} does not match "
                    f"({expected_width}, {expected_height})."
                )
            if decoded.mode != "L":
                raise InvalidMaskError(f"Expected grayscale mode 'L', got {decoded.mode!r}.")
            decoded.load()
            return decoded.copy()
    except InvalidMaskError:
        raise
    except Exception as exc:
        raise InvalidMaskError(f"Malformed PNG mask payload: {exc}") from exc


def apply_cutout_postprocess(raw_output: np.ndarray, mode: str) -> np.ndarray:
    """Convert raw model output tensor [1, 1, H, W] into uint8 [H, W] mask."""
    arr = np.asarray(raw_output, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]

    if mode == "sigmoid_min_max":
        clipped = np.clip(arr, -60.0, 60.0)
        arr = 1.0 / (1.0 + np.exp(-clipped))
        arr = _min_max_norm(arr)
    elif mode == "sigmoid":
        clipped = np.clip(arr, -60.0, 60.0)
        arr = 1.0 / (1.0 + np.exp(-clipped))
    elif mode == "min_max":
        arr = _min_max_norm(arr)
    else:
        raise ValueError(f"Unsupported postprocess mode: {mode!r}")

    arr = np.clip(arr, 0.0, 1.0)
    return np.rint(arr * 255.0).astype(np.uint8)


def _min_max_norm(arr: np.ndarray) -> np.ndarray:
    mi = float(np.min(arr))
    ma = float(np.max(arr))
    span = ma - mi
    if span < 1e-6:
        return np.clip(arr, 0.0, 1.0)
    return (arr - mi) / span


def composite_mask(
    original_srgb: Image.Image,
    mask_l: Image.Image,
    *,
    bg: str = "transparent",
    feather: float = 0.0,
) -> Image.Image:
    """Resize mask onto full-res sRGB image, apply optional feather, and composite background."""
    if feather < 0:
        raise ValueError("--feather must be >= 0")

    mask = mask_l.convert("L")
    if mask.size != original_srgb.size:
        mask = mask.resize(original_srgb.size, Image.Resampling.LANCZOS)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=float(feather)))

    if original_srgb.mode == "RGBA":
        orig_alpha = np.asarray(original_srgb.getchannel("A"), dtype=np.uint16)
        mask_arr = np.asarray(mask, dtype=np.uint16)
        combined = ((orig_alpha * mask_arr + 127) // 255).astype(np.uint8)
        alpha = Image.fromarray(combined, mode="L")
        fg_rgb = original_srgb.convert("RGB")
    else:
        alpha = mask
        fg_rgb = original_srgb.convert("RGB")

    bg_rgb = parse_bg_color(bg)
    if bg_rgb is None:
        out = fg_rgb.copy()
        out.putalpha(alpha)
        return out

    fg_rgba = fg_rgb.copy()
    fg_rgba.putalpha(alpha)
    canvas = Image.new("RGBA", original_srgb.size, (*bg_rgb, 255))
    composited = Image.alpha_composite(canvas, fg_rgba)
    return composited.convert("RGB")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes atomically to `path` via temp file in same directory + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    try:
        with tmp.open("wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def atomic_save_image(
    img: Image.Image,
    out_path: Path,
    fmt: Literal["png", "jpg", "webp"],
) -> None:
    """Encode an image with stripped EXIF/metadata and write it atomically."""
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG", icc_profile=None)
    elif fmt == "jpg":
        rgb = img.convert("RGB")
        rgb.save(buf, format="JPEG", quality=92, subsampling=0)
    elif fmt == "webp":
        img.save(buf, format="WEBP", quality=90)
    else:
        raise ValueError(f"Unsupported output format: {fmt!r}")
    atomic_write_bytes(out_path, buf.getvalue())


class LocalOrtRunner:
    """Lazy host-CPU ONNX Runtime session wrapper."""

    def __init__(self, model_path_provider: Callable[[], Path]) -> None:
        self._model_path_provider = model_path_provider
        self._session: Any = None
        self._lock = threading.Lock()

    def get_session(self) -> Any:
        with self._lock:
            if self._session is None:
                import onnxruntime as ort

                model_path = self._model_path_provider()
                opts = ort.SessionOptions()
                self._session = ort.InferenceSession(
                    str(model_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
            return self._session


@dataclass
class ColdStartMetrics:
    provider: str = ""
    adapter_name: str = ""
    model_fetch_ms: float | None = None
    session_compile_ms: float | None = None


@dataclass
class WorkerState:
    worker_id: str
    adapter_name: str
    provider: str
    webgpu_available: bool
    user_agent: str
    connected_at: float
    last_seen: float
    disconnected: bool = False
    fatal_error: str | None = None


@dataclass
class JobRecord:
    job_id: str
    src_path: Path
    out_path: Path
    output_format: Literal["png", "jpg", "webp"]
    status: Literal["pending", "leased", "done", "failed"] = "pending"
    attempts: int = 0
    worker_type: str = ""
    lease_id: str | None = None
    leased_to: str | None = None
    leased_at: float | None = None
    lease_expires_at: float | None = None
    committing: bool = False
    committed_lease_id: str | None = None
    prepare_ms: float | None = None
    inference_ms: float | None = None
    round_trip_ms: float | None = None
    finalize_ms: float | None = None
    error: str = ""
    _prepared_bytes: bytes | None = field(default=None, repr=False)
    _prepared_meta: dict[str, Any] | None = field(default=None, repr=False)


class JobCoordinator:
    """Single-worker lease coordinator with atomic commits and host CPU fallback."""

    def __init__(
        self,
        *,
        task: Any,
        model_spec: Any,
        planned_jobs: list[tuple[Path, Path, Literal["png", "jpg", "webp"]]],
        finalize_options: dict[str, Any],
        lease_seconds: float = 45.0,
        max_lease_age_seconds: float | None = None,
        worker_timeout_seconds: float = 20.0,
        max_retries: int = 2,
        max_in_flight_per_worker: int = 2,
        clock: Callable[[], float] = time.monotonic,
        on_progress: Callable[[JobRecord], None] | None = None,
    ) -> None:
        self.task = task
        self.model_spec = model_spec
        self.finalize_options = dict(finalize_options)
        self.lease_seconds = float(lease_seconds)
        self.max_lease_age_seconds = (
            float(max_lease_age_seconds)
            if max_lease_age_seconds is not None
            else max(self.lease_seconds * 4.0, 180.0)
        )
        self.worker_timeout_seconds = float(worker_timeout_seconds)
        self.max_retries = int(max_retries)
        self.max_in_flight_per_worker = int(max_in_flight_per_worker)
        self._clock = clock
        self._on_progress = on_progress

        self._lock = threading.RLock()
        self._accepting_workers = True
        self.jobs: list[JobRecord] = []
        self._jobs_by_id: dict[str, JobRecord] = {}
        for idx, (src, out, fmt) in enumerate(planned_jobs, start=1):
            job_id = f"job-{idx:04d}-{secrets.token_hex(4)}"
            rec = JobRecord(
                job_id=job_id,
                src_path=src,
                out_path=out,
                output_format=fmt,
            )
            self.jobs.append(rec)
            self._jobs_by_id[job_id] = rec

        self.active_worker: WorkerState | None = None
        self.cold_start = ColdStartMetrics()
        self.ever_connected = False
        self.worker_disconnected_event = threading.Event()
        self.worker_connected_event = threading.Event()
        self.all_done_event = threading.Event()

    def total_jobs(self) -> int:
        return len(self.jobs)

    def counts(self) -> dict[str, int]:
        with self._lock:
            out = {"pending": 0, "leased": 0, "done": 0, "failed": 0}
            for job in self.jobs:
                out[job.status] += 1
            return out

    def is_finished(self) -> bool:
        with self._lock:
            return all(j.status in ("done", "failed") for j in self.jobs)

    def has_failures(self) -> bool:
        """Return True if any job failed OR failed to reach terminal 'done' state."""
        with self._lock:
            return any(j.status != "done" for j in self.jobs)

    def _check_all_done_locked(self) -> None:
        if all(j.status in ("done", "failed") for j in self.jobs):
            self.all_done_event.set()

    def _effective_worker_timeout_locked(self, worker: WorkerState) -> float:
        if worker.provider.lower() == "wasm":
            return max(self.worker_timeout_seconds, 90.0)
        return self.worker_timeout_seconds

    def is_worker_alive(self) -> bool:
        with self._lock:
            w = self.active_worker
            if w is None or w.disconnected or w.fatal_error is not None:
                return False
            timeout = self._effective_worker_timeout_locked(w)
            return (self._clock() - w.last_seen) <= timeout

    def close_to_workers(self, wait_committing_seconds: float = 10.0) -> None:
        """Stop accepting LAN workers, wait for in-flight commits, and reclaim unfinished leases."""
        with self._lock:
            self._accepting_workers = False
            if self.active_worker is not None:
                self.active_worker.disconnected = True
                self.worker_disconnected_event.set()

        # Wait briefly for any mid-finalize commit to finish its atomic disk replace.
        deadline = time.monotonic() + max(0.0, wait_committing_seconds)
        while time.monotonic() < deadline:
            with self._lock:
                if not any(j.committing for j in self.jobs):
                    break
            time.sleep(0.05)

        with self._lock:
            if self.active_worker is not None:
                self._release_worker_leases_locked(
                    self.active_worker.worker_id, count_attempt=False
                )
            for job in self.jobs:
                if job.status == "leased" and not job.committing:
                    self._return_job_to_pending_or_fail_locked(
                        job,
                        error="released for host CPU fallback",
                        count_attempt=False,
                    )

    def register_worker(
        self,
        *,
        adapter_name: str = "unknown",
        provider: str = "webgpu",
        webgpu_available: bool = True,
        user_agent: str = "",
    ) -> WorkerState:
        with self._lock:
            if not self._accepting_workers:
                raise WorkerClosedError(
                    "Host is no longer accepting LAN workers (CPU fallback or shutdown in progress)."
                )
            now = self._clock()
            self.reclaim_expired_leases()
            if self.active_worker is not None:
                timeout = self._effective_worker_timeout_locked(self.active_worker)
                alive = (
                    not self.active_worker.disconnected
                    and self.active_worker.fatal_error is None
                    and (now - self.active_worker.last_seen) <= timeout
                )
                if alive:
                    raise WorkerConflictError("Another GPU worker session is already active.")
                self._release_worker_leases_locked(
                    self.active_worker.worker_id, count_attempt=False
                )

            worker = WorkerState(
                worker_id=secrets.token_urlsafe(12),
                adapter_name=adapter_name or "unknown",
                provider=provider or "unknown",
                webgpu_available=bool(webgpu_available),
                user_agent=user_agent[:256],
                connected_at=now,
                last_seen=now,
            )
            self.active_worker = worker
            self.ever_connected = True
            self.cold_start.adapter_name = worker.adapter_name
            self.cold_start.provider = worker.provider
            self.worker_disconnected_event.clear()
            self.worker_connected_event.set()
            return worker

    def record_metrics(
        self,
        worker_id: str,
        *,
        model_fetch_ms: float | None = None,
        session_compile_ms: float | None = None,
        provider: str | None = None,
        adapter_name: str | None = None,
    ) -> None:
        with self._lock:
            self._require_active_worker_locked(worker_id)
            if model_fetch_ms is not None:
                self.cold_start.model_fetch_ms = float(model_fetch_ms)
            if session_compile_ms is not None:
                self.cold_start.session_compile_ms = float(session_compile_ms)
            if provider:
                self.cold_start.provider = str(provider)
                if self.active_worker:
                    self.active_worker.provider = str(provider)
            if adapter_name:
                self.cold_start.adapter_name = str(adapter_name)
                if self.active_worker:
                    self.active_worker.adapter_name = str(adapter_name)

    def heartbeat(self, worker_id: str) -> None:
        with self._lock:
            worker = self._require_active_worker_locked(worker_id)
            now = self._clock()
            worker.last_seen = now
            for job in self.jobs:
                if job.status == "leased" and job.leased_to == worker_id and not job.committing:
                    leased_at = job.leased_at if job.leased_at is not None else now
                    ceiling = leased_at + self.max_lease_age_seconds
                    job.lease_expires_at = min(now + self.lease_seconds, ceiling)

    def disconnect_worker(self, worker_id: str, reason: str = "client_disconnect") -> None:
        with self._lock:
            if self.active_worker is None or self.active_worker.worker_id != worker_id:
                return
            self.active_worker.disconnected = True
            # Always release any unfinished non-committing leases back to pending.
            self._release_worker_leases_locked(worker_id, count_attempt=False)
            self.worker_disconnected_event.set()

    def _require_active_worker_locked(self, worker_id: str) -> WorkerState:
        if (
            self.active_worker is None
            or self.active_worker.worker_id != worker_id
            or self.active_worker.disconnected
            or self.active_worker.fatal_error is not None
        ):
            raise StaleLeaseError("Worker session is not active.")
        return self.active_worker

    def _release_worker_leases_locked(self, worker_id: str, *, count_attempt: bool) -> int:
        released = 0
        for job in self.jobs:
            if job.status == "leased" and job.leased_to == worker_id and not job.committing:
                self._return_job_to_pending_or_fail_locked(
                    job,
                    error="worker disconnected before commit",
                    count_attempt=count_attempt,
                )
                released += 1
        return released

    def _return_job_to_pending_or_fail_locked(
        self,
        job: JobRecord,
        *,
        error: str,
        count_attempt: bool,
    ) -> None:
        if count_attempt:
            job.attempts += 1
        job.lease_id = None
        job.leased_to = None
        job.leased_at = None
        job.lease_expires_at = None
        if job.attempts > self.max_retries:
            job.status = "failed"
            job.error = error
            job._prepared_bytes = None
            job._prepared_meta = None
            if self._on_progress:
                self._on_progress(job)
        else:
            job.status = "pending"
            job.error = error
        self._check_all_done_locked()

    def reclaim_expired_leases(self) -> int:
        """Reclaim any expired worker leases (or dead worker leases)."""
        with self._lock:
            now = self._clock()
            reclaimed = 0
            if (
                self.active_worker is not None
                and not self.active_worker.disconnected
                and (now - self.active_worker.last_seen)
                > self._effective_worker_timeout_locked(self.active_worker)
            ):
                self.active_worker.disconnected = True
                self.worker_disconnected_event.set()
                reclaimed += self._release_worker_leases_locked(
                    self.active_worker.worker_id, count_attempt=False
                )

            for job in self.jobs:
                if (
                    job.status == "leased"
                    and not job.committing
                    and job.lease_expires_at is not None
                    and now > job.lease_expires_at
                ):
                    self._return_job_to_pending_or_fail_locked(
                        job,
                        error="lease expired before result commit",
                        count_attempt=True,
                    )
                    reclaimed += 1
            return reclaimed

    def lease_next(self, worker_id: str) -> tuple[str, dict[str, Any] | None]:
        """Lease the next pending job for `worker_id` (supporting prefetch up to max_in_flight)."""
        with self._lock:
            if not self._accepting_workers:
                return "done", None
            worker = self._require_active_worker_locked(worker_id)
            now = self._clock()
            worker.last_seen = now
            self.reclaim_expired_leases()

            in_flight = sum(
                1 for j in self.jobs if j.status == "leased" and j.leased_to == worker_id
            )
            if in_flight >= self.max_in_flight_per_worker:
                return "wait", None

            for job in self.jobs:
                if job.status == "pending":
                    lease_id = secrets.token_urlsafe(12)
                    job.status = "leased"
                    job.lease_id = lease_id
                    job.leased_to = worker_id
                    job.leased_at = now
                    job.lease_expires_at = now + self.lease_seconds
                    meta = self.task.job_meta(self.model_spec)
                    return "job", {
                        "job_id": job.job_id,
                        "lease_id": lease_id,
                        "meta": meta,
                        "input_url": f"/api/jobs/{job.job_id}/input",
                        "result_url": f"/api/jobs/{job.job_id}/result",
                    }

            if any(j.status == "leased" for j in self.jobs):
                return "wait", None
            self._check_all_done_locked()
            return "done", None

    def get_job_input(self, job_id: str, worker_id: str, lease_id: str) -> bytes:
        """Prepare (lazily) and return EXIF-stripped resized input bytes for a leased job."""
        with self._lock:
            worker = self._require_active_worker_locked(worker_id)
            worker.last_seen = self._clock()
            job = self._jobs_by_id.get(job_id)
            if job is None:
                raise UnknownJobError(job_id)
            if job.status != "leased" or job.lease_id != lease_id or job.leased_to != worker_id:
                raise StaleLeaseError("Invalid or stale lease_id for input fetch.")
            if job._prepared_bytes is not None:
                return job._prepared_bytes
            src_path = job.src_path

        t0 = self._clock()
        try:
            payload, meta = self.task.prepare(src_path, self.model_spec)
        except Exception as exc:
            # Mark unreadable input permanently failed so one bad image never kills the worker.
            with self._lock:
                if job.status == "leased" and job.lease_id == lease_id:
                    job.attempts += 1
                    job.status = "failed"
                    job.worker_type = (
                        self.active_worker.provider if self.active_worker else "host"
                    )
                    job.lease_id = None
                    job.leased_to = None
                    job.lease_expires_at = None
                    job.error = f"prepare failed: {exc}"
                    job._prepared_bytes = None
                    job._prepared_meta = None
                    self._check_all_done_locked()
                    if self._on_progress:
                        self._on_progress(job)
            raise PrepareFailedError(f"Failed to prepare input image: {exc}") from exc

        prep_ms = (self._clock() - t0) * 1000.0
        with self._lock:
            if job.status == "leased" and job.lease_id == lease_id:
                job._prepared_bytes = payload
                job._prepared_meta = meta
                job.prepare_ms = prep_ms
            return payload

    def commit_result(
        self,
        job_id: str,
        worker_id: str,
        lease_id: str,
        result_bytes: bytes,
        *,
        inference_ms: float | None = None,
    ) -> Literal["committed", "already_committed"]:
        """Validate mask, atomically finalize output, and commit lease idempotently."""
        with self._lock:
            job = self._jobs_by_id.get(job_id)
            if job is None:
                raise UnknownJobError(job_id)

            if job.status == "done":
                if job.committed_lease_id == lease_id:
                    return "already_committed"
                raise StaleLeaseError("Job was already completed by a different lease.")

            if (
                job.status != "leased"
                or job.lease_id != lease_id
                or job.leased_to != worker_id
            ):
                raise StaleLeaseError("Stale or mismatched lease_id.")

            now = self._clock()
            if job.lease_expires_at is not None and now > job.lease_expires_at:
                self._return_job_to_pending_or_fail_locked(
                    job,
                    error="lease expired before result commit",
                    count_attempt=True,
                )
                raise StaleLeaseError("Lease expired before result commit.")

            if job.committing:
                return "already_committed"

            meta = job._prepared_meta or self.task.job_meta(self.model_spec)
            src_path = job.src_path
            out_path = job.out_path
            out_fmt = job.output_format
            leased_at = job.leased_at or now
            provider = self.active_worker.provider if self.active_worker else "webgpu"
            if self.active_worker and self.active_worker.worker_id == worker_id:
                self.active_worker.last_seen = now
            job.committing = True

        try:
            self.task.validate_result(result_bytes, meta)

            t_fin = self._clock()
            opts = dict(self.finalize_options)
            opts["format"] = out_fmt
            self.task.finalize(src_path, result_bytes, meta, out_path, opts)
            fin_ms = (self._clock() - t_fin) * 1000.0
            rt_ms = (self._clock() - leased_at) * 1000.0
        except Exception:
            with self._lock:
                job.committing = False
            raise

        with self._lock:
            job.committing = False
            job.status = "done"
            job.committed_lease_id = lease_id
            job.lease_id = None
            job.leased_to = None
            job.lease_expires_at = None
            job.worker_type = provider
            job.inference_ms = inference_ms
            job.round_trip_ms = rt_ms
            job.finalize_ms = fin_ms
            job.error = ""
            job._prepared_bytes = None
            job._prepared_meta = None
            self._check_all_done_locked()
            if self._on_progress:
                self._on_progress(job)
            return "committed"

    def fail_job(
        self,
        job_id: str,
        worker_id: str,
        lease_id: str,
        *,
        error: str,
        fatal: bool = False,
    ) -> None:
        """Record a worker-reported job failure; if fatal, trigger disconnect for CPU fallback."""
        with self._lock:
            job = self._jobs_by_id.get(job_id)
            if job is None:
                raise UnknownJobError(job_id)
            if job.status == "done":
                return
            if job.status != "leased" or job.lease_id != lease_id or job.leased_to != worker_id:
                raise StaleLeaseError("Stale or mismatched lease_id.")

            if fatal and self.active_worker and self.active_worker.worker_id == worker_id:
                self.active_worker.fatal_error = error
                self.active_worker.disconnected = True
                self._return_job_to_pending_or_fail_locked(
                    job, error=error, count_attempt=False
                )
                self._release_worker_leases_locked(worker_id, count_attempt=False)
                self.worker_disconnected_event.set()
            else:
                self._return_job_to_pending_or_fail_locked(
                    job, error=error, count_attempt=True
                )

    def run_remaining_on_cpu(self, cpu_runner: LocalOrtRunner) -> None:
        """Close LAN worker registration, wait for in-flight commits, and finish all jobs on CPU."""
        self.close_to_workers()

        session: Any = None
        while True:
            with self._lock:
                next_job = next((j for j in self.jobs if j.status == "pending"), None)
                if next_job is None:
                    self._check_all_done_locked()
                    return
                next_job.status = "leased"
                next_job.leased_to = "host-cpu"
                next_job.lease_id = "host-cpu"
                next_job.leased_at = self._clock()
                next_job.committing = True

            try:
                if session is None:
                    session = cpu_runner.get_session()

                t0 = self._clock()
                payload, meta = self.task.prepare(next_job.src_path, self.model_spec)
                prep_ms = (self._clock() - t0) * 1000.0

                t_inf = self._clock()
                result_bytes = self.task.infer_local(payload, meta, session)
                inf_ms = (self._clock() - t_inf) * 1000.0

                self.task.validate_result(result_bytes, meta)

                t_fin = self._clock()
                opts = dict(self.finalize_options)
                opts["format"] = next_job.output_format
                self.task.finalize(next_job.src_path, result_bytes, meta, next_job.out_path, opts)
                fin_ms = (self._clock() - t_fin) * 1000.0
                total_ms = (self._clock() - (next_job.leased_at or t0)) * 1000.0

                with self._lock:
                    next_job.committing = False
                    next_job.status = "done"
                    next_job.committed_lease_id = "host-cpu"
                    next_job.lease_id = None
                    next_job.leased_to = None
                    next_job.worker_type = "cpu"
                    next_job.prepare_ms = prep_ms
                    next_job.inference_ms = inf_ms
                    next_job.round_trip_ms = total_ms
                    next_job.finalize_ms = fin_ms
                    next_job.error = ""
                    next_job._prepared_bytes = None
                    next_job._prepared_meta = None
                    self._check_all_done_locked()
                    if self._on_progress:
                        self._on_progress(next_job)
            except Exception as exc:
                with self._lock:
                    next_job.committing = False
                    next_job.attempts += 1
                    next_job.status = "failed"
                    next_job.worker_type = "cpu"
                    next_job.lease_id = None
                    next_job.leased_to = None
                    next_job.error = str(exc)
                    next_job._prepared_bytes = None
                    next_job._prepared_meta = None
                    self._check_all_done_locked()
                    if self._on_progress:
                        self._on_progress(next_job)

    def mark_remaining_failed(self, reason: str) -> None:
        """Close worker acceptance, wait for active commits, and mark unfinished jobs failed."""
        self.close_to_workers()
        with self._lock:
            for job in self.jobs:
                if job.status in ("pending", "leased"):
                    job.status = "failed"
                    job.error = reason
                    job.lease_id = None
                    job.leased_to = None
                    job._prepared_bytes = None
                    job._prepared_meta = None
            self._check_all_done_locked()

    def write_csv_report(self, report_path: Path) -> None:
        """Write a deterministic per-file CSV report atomically (with formula escaping)."""
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow(
            [
                "job_id",
                "source_path",
                "output_path",
                "status",
                "worker",
                "attempts",
                "prepare_ms",
                "inference_ms",
                "round_trip_ms",
                "finalize_ms",
                "error",
            ]
        )
        with self._lock:
            for job in self.jobs:
                writer.writerow(
                    [
                        job.job_id,
                        _sanitize_csv_cell(str(job.src_path)),
                        _sanitize_csv_cell(str(job.out_path)),
                        job.status,
                        _sanitize_csv_cell(job.worker_type),
                        job.attempts,
                        f"{job.prepare_ms:.1f}" if job.prepare_ms is not None else "",
                        f"{job.inference_ms:.1f}" if job.inference_ms is not None else "",
                        f"{job.round_trip_ms:.1f}" if job.round_trip_ms is not None else "",
                        f"{job.finalize_ms:.1f}" if job.finalize_ms is not None else "",
                        _sanitize_csv_cell(job.error),
                    ]
                )
        atomic_write_bytes(report_path, buf.getvalue().encode("utf-8"))

    def benchmark_summary(self) -> dict[str, Any]:
        """Compute separated cold-start and warm per-image p50/p95 timing statistics."""
        with self._lock:
            done_jobs = [j for j in self.jobs if j.status == "done"]
            lan_jobs = [j for j in done_jobs if j.worker_type != "cpu"]
            warm_pool = lan_jobs[1:] if len(lan_jobs) >= 2 else done_jobs
            rt_vals = sorted(
                j.round_trip_ms for j in warm_pool if j.round_trip_ms is not None
            )
            inf_vals = sorted(
                j.inference_ms for j in warm_pool if j.inference_ms is not None
            )
            return {
                "cold_start": {
                    "provider": self.cold_start.provider or ("cpu" if done_jobs else ""),
                    "adapter_name": self.cold_start.adapter_name,
                    "model_fetch_ms": self.cold_start.model_fetch_ms,
                    "session_compile_ms": self.cold_start.session_compile_ms,
                },
                "warm": {
                    "sample_count": len(warm_pool),
                    "round_trip_p50_ms": _percentile(rt_vals, 50),
                    "round_trip_p95_ms": _percentile(rt_vals, 95),
                    "inference_p50_ms": _percentile(inf_vals, 50),
                    "inference_p95_ms": _percentile(inf_vals, 95),
                },
                "counts": self.counts(),
            }


def _percentile(sorted_vals: list[float], pct: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)
