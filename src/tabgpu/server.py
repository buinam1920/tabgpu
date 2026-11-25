"""Temporary LAN HTTPS server with Bearer auth, Host/Origin checks, rate limiting, and lease API."""

from __future__ import annotations

import hmac
import ipaddress
import json
import secrets
import socket
import ssl
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from tabgpu.assets import AssetManager, ModelSpec, UnknownAssetError
from tabgpu.pipeline import (
    InvalidMaskError,
    JobCoordinator,
    StaleLeaseError,
    UnknownJobError,
    WorkerConflictError,
)
from tabgpu.tls import create_ssl_context

MAX_JSON_BODY_BYTES = 64 * 1024  # 64 KiB
DEFAULT_MAX_RESULT_BODY_BYTES = 16 * 1024 * 1024  # 16 MiB
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
_FORBIDDEN_QUERY_PARAMS = {"token", "auth", "access_token", "bearer", "key"}


def detect_lan_ip() -> str:
    """Detect the primary LAN IPv4 address without sending network packets."""
    for target in (("192.0.2.1", 80), ("8.8.8.8", 80), ("10.255.255.255", 1)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(target)
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127.") and ip != "0.0.0.0":
                return ip
        except OSError:
            continue
        finally:
            sock.close()

    for ip in list_lan_ipv4_addresses():
        return ip
    return "127.0.0.1"


def list_lan_ipv4_addresses() -> list[str]:
    """Return unique non-loopback local IPv4 addresses."""
    ips: list[str] = []
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            try:
                addr = ipaddress.IPv4Address(ip)
            except ValueError:
                continue
            if not addr.is_loopback and not addr.is_unspecified and ip not in ips:
                ips.append(ip)
    except OSError:
        pass
    return ips


def check_port_available(bind_ip: str, port: int) -> tuple[bool, str]:
    """Check whether binding TCP `(bind_ip, port)` succeeds right now."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_ip, port))
        return True, "available"
    except OSError as exc:
        return False, str(exc)
    finally:
        sock.close()


class AuthRateLimiter:
    """Sliding-window per-IP rate limiter for invalid authentication attempts."""

    def __init__(
        self,
        *,
        max_failures: int = 5,
        window_seconds: float = 60.0,
        clock=time.monotonic,
    ) -> None:
        self.max_failures = int(max_failures)
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._lock = threading.Lock()
        self._failures: dict[str, deque[float]] = {}

    def _prune_locked(self, ip: str, now: float) -> deque[float]:
        dq = self._failures.get(ip)
        if dq is None:
            dq = deque()
            self._failures[ip] = dq
        cutoff = now - self.window_seconds
        while dq and dq[0] <= cutoff:
            dq.popleft()
        return dq

    def is_blocked(self, ip: str) -> tuple[bool, int]:
        with self._lock:
            now = self._clock()
            dq = self._prune_locked(ip, now)
            if len(dq) >= self.max_failures:
                retry_after = max(1, int(self.window_seconds - (now - dq[0])) + 1)
                return True, retry_after
            return False, 0

    def record_failure(self, ip: str) -> tuple[bool, int]:
        with self._lock:
            now = self._clock()
            dq = self._prune_locked(ip, now)
            dq.append(now)
            if len(dq) >= self.max_failures:
                retry_after = max(1, int(self.window_seconds - (now - dq[0])) + 1)
                return True, retry_after
            return False, 0


def _load_web_text(rel_path: str) -> bytes:
    parts = [p for p in rel_path.split("/") if p]
    node = resources.files("tabgpu").joinpath("web")
    for part in parts:
        node = node.joinpath(part)
    return node.read_bytes()


class TabGpuHTTPSServer(ThreadingHTTPServer):
    """Hardened temporary HTTPS server backing the Chrome WebGPU worker."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        *,
        bind_host: str,
        port: int,
        advertised_host: str,
        cert_path: Path,
        key_path: Path,
        token: str,
        coordinator: JobCoordinator,
        asset_manager: AssetManager,
        model_spec: ModelSpec,
        model_path: Path,
        max_result_body_bytes: int = DEFAULT_MAX_RESULT_BODY_BYTES,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        rate_limit_max_failures: int = 5,
        rate_limit_window_seconds: float = 60.0,
    ) -> None:
        self.bind_host = bind_host
        self.advertised_host = advertised_host
        self.token = token
        self.coordinator = coordinator
        self.asset_manager = asset_manager
        self.model_spec = model_spec
        self.model_path = model_path
        self.max_result_body_bytes = int(max_result_body_bytes)
        self.request_timeout = float(request_timeout)
        self.rate_limiter = AuthRateLimiter(
            max_failures=rate_limit_max_failures,
            window_seconds=rate_limit_window_seconds,
        )
        self._web_cache: dict[str, tuple[bytes, str]] = {
            "/": (_load_web_text("index.html"), "text/html; charset=utf-8"),
            "/worker.js": (_load_web_text("worker.js"), "application/javascript; charset=utf-8"),
            "/tasks/cutout.js": (
                _load_web_text("tasks/cutout.js"),
                "application/javascript; charset=utf-8",
            ),
        }

        super().__init__((bind_host, port), _TabGpuRequestHandler)
        self.bound_port: int = int(self.server_address[1])
        self.allowed_hosts, self.allowed_origins = self._compute_allowed_origins()
        self._ssl_context = create_ssl_context(cert_path, key_path)
        self.socket = self._ssl_context.wrap_socket(self.socket, server_side=True)

    def _compute_allowed_origins(self) -> tuple[set[str], set[str]]:
        raw_hosts = {self.advertised_host.lower()}
        if self.bind_host not in ("0.0.0.0", "::"):
            raw_hosts.add(self.bind_host.lower())

        hosts: set[str] = set()
        origins: set[str] = set()
        for h in raw_hosts:
            host_port = f"{h}:{self.bound_port}"
            hosts.add(host_port)
            origins.add(f"https://{host_port}")
            if self.bound_port == 443:
                hosts.add(h)
                origins.add(f"https://{h}")
        return hosts, origins

    @property
    def worker_url(self) -> str:
        return f"https://{self.advertised_host}:{self.bound_port}/#{self.token}"

    @property
    def base_url(self) -> str:
        return f"https://{self.advertised_host}:{self.bound_port}"

    def handle_error(self, request: Any, client_address: Any) -> None:
        # Suppress noisy tracebacks from self-signed TLS probe resets or client aborts.
        import sys

        _, exc, _ = sys.exc_info()
        if isinstance(exc, (ssl.SSLError, ConnectionResetError, BrokenPipeError, TimeoutError, OSError)):
            return
        super().handle_error(request, client_address)


class _TabGpuRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: TabGpuHTTPSServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.request_timeout)

    def log_message(self, format: str, *args: Any) -> None:
        # Quiet by default; host CLI prints structured progress.
        return

    # ------------------------------------------------------------------
    # Response helpers & security headers
    # ------------------------------------------------------------------
    def _send_security_headers(
        self,
        *,
        is_html: bool = False,
        clear_site_data: bool = False,
    ) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        if is_html:
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; "
                "script-src 'self' 'wasm-unsafe-eval'; "
                "connect-src 'self'; "
                "img-src 'self' blob: data:; "
                "worker-src 'self' blob:; "
                "style-src 'self' 'unsafe-inline'; "
                "object-src 'none'; "
                "base-uri 'none'; "
                "frame-ancestors 'none'",
            )
        if clear_site_data:
            self.send_header("Clear-Site-Data", '"cache", "storage"')

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        *,
        is_html: bool = False,
        clear_site_data: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self._send_security_headers(is_html=is_html, clear_site_data=clear_site_data)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        *,
        clear_site_data: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._send_bytes(
            status,
            data,
            "application/json; charset=utf-8",
            clear_site_data=clear_site_data,
            extra_headers=extra_headers,
        )

    def _stream_file(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self._send_security_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    # ------------------------------------------------------------------
    # Validation & auth helpers
    # ------------------------------------------------------------------
    def _validate_host_and_origin(self) -> bool:
        host = (self.headers.get("Host") or "").strip().lower()
        if not host or host not in self.server.allowed_hosts:
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"error": "invalid_host", "detail": "Host header not allowed."},
            )
            return False

        origin = (self.headers.get("Origin") or "").strip().lower()
        if origin and origin not in self.server.allowed_origins:
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"error": "invalid_origin", "detail": "Cross-origin requests are not allowed."},
            )
            return False

        parsed = urlsplit(self.path)
        if parsed.query:
            qs = parse_qs(parsed.query, keep_blank_values=True)
            if any(k.lower() in _FORBIDDEN_QUERY_PARAMS for k in qs):
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {
                        "error": "token_in_query",
                        "detail": "Tokens must never be sent in query strings; use Authorization: Bearer.",
                    },
                )
                return False

        return True

    def _require_auth(self) -> bool:
        client_ip = str(self.client_address[0])
        blocked, retry_after = self.server.rate_limiter.is_blocked(client_ip)
        if blocked:
            self._send_json(
                HTTPStatus.TOO_MANY_REQUESTS,
                {"error": "rate_limited", "detail": "Too many failed authentication attempts."},
                extra_headers={"Retry-After": str(retry_after)},
            )
            return False

        auth = self.headers.get("Authorization")
        if not auth:
            blocked, retry_after = self.server.rate_limiter.record_failure(client_ip)
            if blocked:
                self._send_json(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    {"error": "rate_limited", "detail": "Too many failed authentication attempts."},
                    extra_headers={"Retry-After": str(retry_after)},
                )
            else:
                self._send_json(
                    HTTPStatus.UNAUTHORIZED,
                    {"error": "unauthorized", "detail": "Missing Authorization: Bearer token."},
                    extra_headers={"WWW-Authenticate": "Bearer"},
                )
            return False

        parts = auth.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            blocked, retry_after = self.server.rate_limiter.record_failure(client_ip)
            status = HTTPStatus.TOO_MANY_REQUESTS if blocked else HTTPStatus.UNAUTHORIZED
            extra = {"Retry-After": str(retry_after)} if blocked else {"WWW-Authenticate": "Bearer"}
            self._send_json(
                status,
                {"error": "invalid_auth_scheme", "detail": "Expected Authorization: Bearer <token>."},
                extra_headers=extra,
            )
            return False

        supplied = parts[1].strip()
        if not hmac.compare_digest(supplied, self.server.token):
            blocked, retry_after = self.server.rate_limiter.record_failure(client_ip)
            if blocked:
                self._send_json(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    {"error": "rate_limited", "detail": "Too many failed authentication attempts."},
                    extra_headers={"Retry-After": str(retry_after)},
                )
            else:
                self._send_json(
                    HTTPStatus.FORBIDDEN,
                    {"error": "forbidden", "detail": "Invalid bearer token."},
                )
            return False

        return True

    def _read_bounded_body(self, max_bytes: int) -> bytes | None:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            self._send_json(
                HTTPStatus.LENGTH_REQUIRED,
                {"error": "length_required", "detail": "Content-Length header is required."},
            )
            return None
        try:
            length = int(raw_len.strip())
        except ValueError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_content_length", "detail": "Malformed Content-Length."},
            )
            return None
        if length < 0:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_content_length", "detail": "Negative Content-Length."},
            )
            return None
        if length > max_bytes:
            self._send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {
                    "error": "payload_too_large",
                    "detail": f"Request body exceeds limit of {max_bytes} bytes.",
                },
            )
            self.close_connection = True
            return None

        if length == 0:
            return b""

        chunks: list[bytes] = []
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 65536))
            if not chunk:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "incomplete_body", "detail": "Connection closed before body finished."},
                )
                return None
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_json_body(self) -> dict[str, Any] | None:
        raw = self._read_bounded_body(MAX_JSON_BODY_BYTES)
        if raw is None:
            return None
        if not raw:
            return {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_json", "detail": "Request body must be valid JSON."},
            )
            return None
        if not isinstance(parsed, dict):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_json", "detail": "JSON body must be an object."},
            )
            return None
        return parsed

    # ------------------------------------------------------------------
    # HTTP verbs
    # ------------------------------------------------------------------
    def do_GET(self) -> None:
        if not self._validate_host_and_origin():
            return

        path = urlsplit(self.path).path

        # Static web worker shell (unauthenticated; token arrives in #fragment client-side).
        if path in self.server._web_cache:
            content, ctype = self.server._web_cache[path]
            self._send_bytes(HTTPStatus.OK, content, ctype, is_html=(path == "/"))
            return

        # Pinned onnxruntime-web vendor files (strictly allowlisted by manifest).
        if path.startswith("/vendor/ort/"):
            fname = unquote(path[len("/vendor/ort/") :])
            if "/" in fname or "\\" in fname or ".." in fname:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            try:
                file_path, spec = self.server.asset_manager.ensure_ort_file(fname)
            except UnknownAssetError:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            except Exception as exc:
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": "asset_unavailable", "detail": str(exc)},
                )
                return
            self._stream_file(file_path, spec.content_type)
            return

        # Authenticated endpoints below.
        if not self._require_auth():
            return

        if path == "/api/config":
            spec = self.server.model_spec
            self._send_json(
                HTTPStatus.OK,
                {
                    "task": self.server.coordinator.task.name,
                    "task_module": self.server.coordinator.task.js_module,
                    "model_id": spec.id,
                    "model_display_name": spec.display_name,
                    "model_url": f"/model/{spec.id}.onnx",
                    "model_size_bytes": self.server.model_path.stat().st_size,
                    "ort_version": self.server.asset_manager.manifest.ort_version,
                    "total_jobs": self.server.coordinator.total_jobs(),
                    "lease_seconds": self.server.coordinator.lease_seconds,
                },
            )
            return

        if path == f"/model/{self.server.model_spec.id}.onnx":
            self._stream_file(self.server.model_path, "application/octet-stream")
            return

        if path.startswith("/api/jobs/") and path.endswith("/input"):
            job_id = unquote(path[len("/api/jobs/") : -len("/input")])
            worker_id = (self.headers.get("X-Worker-Id") or "").strip()
            lease_id = (self.headers.get("X-Lease-Id") or "").strip()
            if not worker_id or not lease_id:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "missing_lease_headers", "detail": "X-Worker-Id and X-Lease-Id are required."},
                )
                return
            try:
                data = self.server.coordinator.get_job_input(job_id, worker_id, lease_id)
            except UnknownJobError:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown_job"})
                return
            except StaleLeaseError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": "stale_lease", "detail": str(exc)})
                return
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "prepare_failed", "detail": str(exc)},
                )
                return
            self._send_bytes(HTTPStatus.OK, data, "image/png")
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:
        if not self._validate_host_and_origin():
            return
        if not self._require_auth():
            return

        path = urlsplit(self.path).path

        if path == "/api/register":
            body = self._read_json_body()
            if body is None:
                return
            try:
                worker = self.server.coordinator.register_worker(
                    adapter_name=str(body.get("adapter_name", "unknown")),
                    provider=str(body.get("provider", "webgpu")),
                    webgpu_available=bool(body.get("webgpu_available", False)),
                    user_agent=str(self.headers.get("User-Agent", "")),
                )
            except WorkerConflictError as exc:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"error": "worker_conflict", "detail": str(exc)},
                )
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "worker_id": worker.worker_id,
                    "lease_seconds": self.server.coordinator.lease_seconds,
                    "total_jobs": self.server.coordinator.total_jobs(),
                },
            )
            return

        if path == "/api/metrics":
            body = self._read_json_body()
            if body is None:
                return
            worker_id = str(body.get("worker_id") or self.headers.get("X-Worker-Id") or "").strip()
            try:
                self.server.coordinator.record_metrics(
                    worker_id,
                    model_fetch_ms=body.get("model_fetch_ms"),
                    session_compile_ms=body.get("session_compile_ms"),
                    provider=body.get("provider"),
                    adapter_name=body.get("adapter_name"),
                )
            except StaleLeaseError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": "stale_worker", "detail": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        if path == "/api/heartbeat":
            body = self._read_json_body()
            if body is None:
                return
            worker_id = str(body.get("worker_id") or self.headers.get("X-Worker-Id") or "").strip()
            try:
                self.server.coordinator.heartbeat(worker_id)
            except StaleLeaseError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": "stale_worker", "detail": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"status": "ok", "counts": self.server.coordinator.counts()})
            return

        if path == "/api/jobs/next":
            body = self._read_json_body()
            if body is None:
                return
            worker_id = str(body.get("worker_id") or self.headers.get("X-Worker-Id") or "").strip()
            try:
                status, job_info = self.server.coordinator.lease_next(worker_id)
            except StaleLeaseError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": "stale_worker", "detail": str(exc)})
                return
            resp: dict[str, Any] = {"status": status, "counts": self.server.coordinator.counts()}
            if job_info is not None:
                resp.update(job_info)
            self._send_json(HTTPStatus.OK, resp)
            return

        if path.startswith("/api/jobs/") and path.endswith("/result"):
            job_id = unquote(path[len("/api/jobs/") : -len("/result")])
            worker_id = (self.headers.get("X-Worker-Id") or "").strip()
            lease_id = (self.headers.get("X-Lease-Id") or "").strip()
            if not worker_id or not lease_id:
                self._send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"error": "missing_lease_headers", "detail": "X-Worker-Id and X-Lease-Id are required."},
                )
                return

            body_bytes = self._read_bounded_body(self.server.max_result_body_bytes)
            if body_bytes is None:
                return

            inf_ms: float | None = None
            raw_inf = self.headers.get("X-Inference-Ms")
            if raw_inf is not None:
                try:
                    inf_ms = float(raw_inf)
                except ValueError:
                    inf_ms = None

            try:
                outcome = self.server.coordinator.commit_result(
                    job_id,
                    worker_id,
                    lease_id,
                    body_bytes,
                    inference_ms=inf_ms,
                )
            except UnknownJobError:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown_job"})
                return
            except StaleLeaseError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": "stale_lease", "detail": str(exc)})
                return
            except InvalidMaskError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_mask", "detail": str(exc)})
                return
            except Exception as exc:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": "finalize_failed", "detail": str(exc)},
                )
                return

            self._send_json(
                HTTPStatus.OK,
                {"status": outcome, "counts": self.server.coordinator.counts()},
            )
            return

        if path.startswith("/api/jobs/") and path.endswith("/fail"):
            job_id = unquote(path[len("/api/jobs/") : -len("/fail")])
            body = self._read_json_body()
            if body is None:
                return
            worker_id = str(body.get("worker_id") or self.headers.get("X-Worker-Id") or "").strip()
            lease_id = str(body.get("lease_id") or self.headers.get("X-Lease-Id") or "").strip()
            try:
                self.server.coordinator.fail_job(
                    job_id,
                    worker_id,
                    lease_id,
                    error=str(body.get("error", "worker error"))[:512],
                    fatal=bool(body.get("fatal", False)),
                )
            except UnknownJobError:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown_job"})
                return
            except StaleLeaseError as exc:
                self._send_json(HTTPStatus.CONFLICT, {"error": "stale_lease", "detail": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"status": "recorded"})
            return

        if path == "/api/disconnect":
            body = self._read_json_body()
            if body is None:
                return
            worker_id = str(body.get("worker_id") or self.headers.get("X-Worker-Id") or "").strip()
            reason = str(body.get("reason", "client_disconnect"))
            self.server.coordinator.disconnect_worker(worker_id, reason=reason)
            self._send_json(
                HTTPStatus.OK,
                {"status": "disconnected"},
                clear_site_data=True,
            )
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})


def generate_session_token() -> str:
    """Generate a compact, high-entropy (96-bit) URL-safe per-run token."""
    return secrets.token_urlsafe(12)
