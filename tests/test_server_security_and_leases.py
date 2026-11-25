"""Tests for HTTPS server auth, security headers, rate limiting, leases, and CPU fallback."""

from __future__ import annotations

import http.client
import io
import json
import ssl
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tabgpu.assets import AssetManager, ModelSpec
from tabgpu.pipeline import JobCoordinator, LocalOrtRunner, plan_outputs
from tabgpu.server import TabGpuHTTPSServer
from tabgpu.tasks.cutout import CutoutTask
from tabgpu.tls import ensure_tls_cert


def _fake_model_spec() -> ModelSpec:
    return ModelSpec(
        id="fast",
        task="cutout",
        display_name="Fake Fast",
        filename="fake.onnx",
        url="https://example.invalid/fake.onnx",
        revision="test",
        sha256="0" * 64,
        size=12,
        license="MIT",
        upstream="https://example.invalid",
        status="stable",
        input_name="input_image",
        output_name="output_image",
        input_shape=(1, 3, 16, 16),
        input_size=(16, 16),
        mean=(0.5, 0.5, 0.5),
        std=(1.0, 1.0, 1.0),
        postprocess="min_max",
    )


def _grayscale_png(width: int = 16, height: int = 16, value: int = 200) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), value).save(buf, format="PNG")
    return buf.getvalue()


class _MutableClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _start_test_server(
    tmp_path: Path,
    *,
    num_images: int = 2,
    max_result_body_bytes: int = 64 * 1024,
    rate_limit_max_failures: int = 3,
    lease_seconds: float = 30.0,
    clock: _MutableClock | None = None,
):
    in_dir = tmp_path / "inputs"
    out_dir = tmp_path / "outputs"
    in_dir.mkdir(exist_ok=True)
    srcs: list[Path] = []
    for i in range(num_images):
        p = in_dir / f"img_{i}.png"
        Image.new("RGB", (24, 24), (200, 50, 50)).save(p)
        srcs.append(p)

    model_path = tmp_path / "fake.onnx"
    model_path.write_bytes(b"FAKEONNXMODEL")

    spec = _fake_model_spec()
    task = CutoutTask(models={"fast": spec})
    planned = plan_outputs(srcs, out_dir, fmt="png", bg="transparent")
    clk = clock or _MutableClock()
    coord = JobCoordinator(
        task=task,
        model_spec=spec,
        planned_jobs=planned,
        finalize_options={"bg": "transparent", "feather": 0.0},
        lease_seconds=lease_seconds,
        worker_timeout_seconds=20.0,
        clock=clk,
    )

    tls_mat = ensure_tls_cert(tmp_path / "certs", ["127.0.0.1"])
    token = "test-secret-token-xyz"
    server = TabGpuHTTPSServer(
        bind_host="127.0.0.1",
        port=0,
        advertised_host="127.0.0.1",
        cert_path=tls_mat.cert_path,
        key_path=tls_mat.key_path,
        token=token,
        coordinator=coord,
        asset_manager=AssetManager(cache_dir=tmp_path / "cache"),
        model_spec=spec,
        model_path=model_path,
        max_result_body_bytes=max_result_body_bytes,
        rate_limit_max_failures=rate_limit_max_failures,
        rate_limit_window_seconds=60.0,
    )
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
    )
    thread.start()
    return server, thread, coord, clk, out_dir, token


def _https_req(
    port: int,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    ctx = ssl._create_unverified_context()
    conn = http.client.HTTPSConnection("127.0.0.1", port, context=ctx, timeout=10)
    try:
        hdrs = dict(headers or {})
        if body is not None and "Content-Length" not in hdrs:
            hdrs["Content-Length"] = str(len(body))
        conn.request(method, path, body=body, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        resp_headers = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, resp_headers, data
    finally:
        conn.close()


def test_auth_headers_origin_host_and_rate_limiting(tmp_path: Path) -> None:
    server, thread, _coord, _clk, _out, token = _start_test_server(
        tmp_path, rate_limit_max_failures=3
    )
    port = server.bound_port
    try:
        # Static UI loads with strict no-store + COOP/COEP headers
        status, hdrs, body = _https_req(port, "GET", "/")
        assert status == 200
        assert "no-store" in hdrs["cache-control"]
        assert hdrs["pragma"] == "no-cache"
        assert hdrs["cross-origin-opener-policy"] == "same-origin"
        assert hdrs["cross-origin-embedder-policy"] == "require-corp"
        assert b"tabgpu" in body

        # Token in query string rejected
        status, _, _ = _https_req(port, "GET", f"/api/config?token={token}")
        assert status == 400

        # Bad Host header rejected with 403
        status, _, _ = _https_req(
            port,
            "GET",
            "/api/config",
            headers={"Host": "evil.example:8787", "Authorization": f"Bearer {token}"},
        )
        assert status == 403

        # Bad Origin header rejected with 403
        status, _, _ = _https_req(
            port,
            "GET",
            "/api/config",
            headers={"Origin": "https://evil.example", "Authorization": f"Bearer {token}"},
        )
        assert status == 403

        # Valid same-origin + Bearer token succeeds
        status, _, data = _https_req(
            port,
            "GET",
            "/api/config",
            headers={
                "Origin": f"https://127.0.0.1:{port}",
                "Authorization": f"Bearer {token}",
            },
        )
        assert status == 200
        assert json.loads(data)["model_id"] == "fast"

        # Invalid auth attempts trigger 401 -> 403 -> 429 rate limiting
        s1, _, _ = _https_req(port, "GET", "/api/config")
        assert s1 == 401
        s2, _, _ = _https_req(
            port, "GET", "/api/config", headers={"Authorization": "Bearer wrong-token"}
        )
        assert s2 == 403
        s3, hdrs3, _ = _https_req(
            port, "GET", "/api/config", headers={"Authorization": "Bearer still-wrong"}
        )
        assert s3 == 429
        assert "retry-after" in hdrs3

        # Subsequent request even with valid token stays blocked during window
        s4, _, _ = _https_req(
            port, "GET", "/api/config", headers={"Authorization": f"Bearer {token}"}
        )
        assert s4 == 429
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_oversized_body_malformed_png_and_wrong_dimensions_rejected(tmp_path: Path) -> None:
    server, thread, _coord, _clk, _out, token = _start_test_server(
        tmp_path, num_images=1, max_result_body_bytes=4096
    )
    port = server.bound_port
    auth = {"Authorization": f"Bearer {token}"}
    try:
        # Register worker and lease job
        st, _, data = _https_req(
            port,
            "POST",
            "/api/register",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"adapter_name": "TestGPU", "provider": "webgpu"}).encode(),
        )
        assert st == 200
        wid = json.loads(data)["worker_id"]

        st, _, data = _https_req(
            port,
            "POST",
            "/api/jobs/next",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"worker_id": wid}).encode(),
        )
        assert st == 200
        lease = json.loads(data)
        result_url = lease["result_url"]
        lease_headers = {**auth, "X-Worker-Id": wid, "X-Lease-Id": lease["lease_id"]}

        # Oversized body -> 413
        st, _, _ = _https_req(
            port,
            "POST",
            result_url,
            headers=lease_headers,
            body=b"\x00" * 5000,
        )
        assert st == 413

        # Malformed PNG -> 400
        st, _, _ = _https_req(
            port,
            "POST",
            result_url,
            headers=lease_headers,
            body=b"this is definitely not a png image",
        )
        assert st == 400

        # Wrong dimensions PNG -> 400
        st, _, _ = _https_req(
            port,
            "POST",
            result_url,
            headers=lease_headers,
            body=_grayscale_png(8, 8),
        )
        assert st == 400

        # RGB PNG (non-grayscale) -> 400
        rgb_buf = io.BytesIO()
        Image.new("RGB", (16, 16), (255, 255, 255)).save(rgb_buf, format="PNG")
        st, _, _ = _https_req(
            port,
            "POST",
            result_url,
            headers=lease_headers,
            body=rgb_buf.getvalue(),
        )
        assert st == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_leases_idempotency_stale_rejection_and_cpu_fallback_after_disconnect(tmp_path: Path) -> None:
    clk = _MutableClock(1000.0)
    server, thread, coord, _, out_dir, token = _start_test_server(
        tmp_path, num_images=2, lease_seconds=10.0, clock=clk
    )
    port = server.bound_port
    auth = {"Authorization": f"Bearer {token}"}
    try:
        # First worker registers
        st, _, data = _https_req(
            port,
            "POST",
            "/api/register",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"adapter_name": "RTX 4070", "provider": "webgpu"}).encode(),
        )
        assert st == 200
        wid = json.loads(data)["worker_id"]

        # Second concurrent worker rejected with 409
        st, _, _ = _https_req(
            port,
            "POST",
            "/api/register",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"adapter_name": "SecondGPU", "provider": "webgpu"}).encode(),
        )
        assert st == 409

        # Lease job 1
        st, _, data = _https_req(
            port,
            "POST",
            "/api/jobs/next",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"worker_id": wid}).encode(),
        )
        job1 = json.loads(data)
        assert job1["status"] == "job"

        # Fetch resized input PNG
        st, hdrs, inp_bytes = _https_req(
            port,
            "GET",
            job1["input_url"],
            headers={**auth, "X-Worker-Id": wid, "X-Lease-Id": job1["lease_id"]},
        )
        assert st == 200
        assert hdrs["content-type"] == "image/png"
        with Image.open(io.BytesIO(inp_bytes)) as im:
            assert im.size == (16, 16)

        # Stale lease_id rejected with 409
        st, _, _ = _https_req(
            port,
            "POST",
            job1["result_url"],
            headers={**auth, "X-Worker-Id": wid, "X-Lease-Id": "stale-lease-id"},
            body=_grayscale_png(16, 16),
        )
        assert st == 409

        # Valid commit succeeds and writes atomic output
        mask_bytes = _grayscale_png(16, 16, 255)
        st, _, data = _https_req(
            port,
            "POST",
            job1["result_url"],
            headers={
                **auth,
                "X-Worker-Id": wid,
                "X-Lease-Id": job1["lease_id"],
                "X-Inference-Ms": "12.5",
            },
            body=mask_bytes,
        )
        assert st == 200
        assert json.loads(data)["status"] == "committed"
        assert (out_dir / "img_0.png").is_file()

        # Duplicate commit with SAME lease_id is idempotent (200 already_committed)
        st, _, data = _https_req(
            port,
            "POST",
            job1["result_url"],
            headers={**auth, "X-Worker-Id": wid, "X-Lease-Id": job1["lease_id"]},
            body=mask_bytes,
        )
        assert st == 200
        assert json.loads(data)["status"] == "already_committed"

        # Lease job 2, then let worker disconnect before completing job 2
        st, _, data = _https_req(
            port,
            "POST",
            "/api/jobs/next",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"worker_id": wid}).encode(),
        )
        job2 = json.loads(data)
        assert job2["status"] == "job"

        st, hdrs, _ = _https_req(
            port,
            "POST",
            "/api/disconnect",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"worker_id": wid, "reason": "user_stopped"}).encode(),
        )
        assert st == 200
        assert hdrs["clear-site-data"] == '"cache", "storage"'
        assert not coord.is_worker_alive()

        # Host CPU fallback finishes remaining job 2
        class _CpuStub(LocalOrtRunner):
            def __init__(self) -> None:
                pass

            def get_session(self):
                return SimpleNamespace(
                    get_inputs=lambda: [SimpleNamespace(name="input_image")],
                    run=lambda _o, _f: [np.ones((1, 1, 16, 16), dtype=np.float32)],
                )

        coord.run_remaining_on_cpu(_CpuStub())
        assert coord.is_finished()
        assert not coord.has_failures()
        assert (out_dir / "img_1.png").is_file()
        assert coord.jobs[0].worker_type == "webgpu"
        assert coord.jobs[1].worker_type == "cpu"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_lease_expiry_recovery_and_retry_bound(tmp_path: Path) -> None:
    clk = _MutableClock(2000.0)
    server, thread, coord, _, _out_dir, token = _start_test_server(
        tmp_path, num_images=1, lease_seconds=5.0, clock=clk
    )
    port = server.bound_port
    auth = {"Authorization": f"Bearer {token}"}
    try:
        st, _, data = _https_req(
            port,
            "POST",
            "/api/register",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"adapter_name": "GPU", "provider": "webgpu"}).encode(),
        )
        assert st == 200
        wid = json.loads(data)["worker_id"]

        # Lease job 1
        st, _, data = _https_req(
            port,
            "POST",
            "/api/jobs/next",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"worker_id": wid}).encode(),
        )
        first_lease = json.loads(data)
        assert first_lease["status"] == "job"

        # Advance clock past lease_seconds (5s), keep worker alive (<20s)
        clk.advance(6.0)
        reclaimed = coord.reclaim_expired_leases()
        assert reclaimed == 1
        assert coord.jobs[0].status == "pending"
        assert coord.jobs[0].attempts == 1

        # Old lease_id commit rejected with 409
        st, _, _ = _https_req(
            port,
            "POST",
            first_lease["result_url"],
            headers={**auth, "X-Worker-Id": wid, "X-Lease-Id": first_lease["lease_id"]},
            body=_grayscale_png(16, 16),
        )
        assert st == 409

        # Re-leasing issues a fresh lease_id; exhausting max_retries marks job failed
        for _ in range(2):
            st, _, data = _https_req(
                port,
                "POST",
                "/api/jobs/next",
                headers={**auth, "Content-Type": "application/json"},
                body=json.dumps({"worker_id": wid}).encode(),
            )
            leased = json.loads(data)
            assert leased["status"] == "job"
            assert leased["lease_id"] != first_lease["lease_id"]
            st, _, _ = _https_req(
                port,
                "POST",
                f"/api/jobs/{leased['job_id']}/fail",
                headers={**auth, "Content-Type": "application/json"},
                body=json.dumps(
                    {"worker_id": wid, "lease_id": leased["lease_id"], "error": "boom"}
                ).encode(),
            )
            assert st == 200

        assert coord.is_finished()
        assert coord.has_failures()
        assert coord.jobs[0].status == "failed"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_idle_tcp_client_does_not_freeze_server_or_shutdown(tmp_path: Path) -> None:
    import socket
    import time

    server, thread, _coord, _clk, _out, token = _start_test_server(tmp_path, num_images=1)
    server.handshake_timeout = 1.0
    port = server.bound_port
    idle_sock = None
    try:
        # Open a raw TCP socket that sends zero bytes (never starts TLS handshake)
        idle_sock = socket.create_connection(("127.0.0.1", port), timeout=2.0)

        # Real HTTPS request MUST still complete promptly while idle_sock is connected
        t0 = time.monotonic()
        status, _, data = _https_req(
            port,
            "GET",
            "/api/config",
            headers={"Authorization": f"Bearer {token}"},
        )
        elapsed = time.monotonic() - t0
        assert status == 200
        assert json.loads(data)["model_id"] == "fast"
        assert elapsed < 3.0

        # Server shutdown MUST return promptly even while idle_sock is still open
        t_sd = time.monotonic()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3.0)
        assert (time.monotonic() - t_sd) < 3.0
        assert not thread.is_alive()
    finally:
        if idle_sock is not None:
            idle_sock.close()


def test_late_worker_rejected_after_cpu_fallback_and_non_done_fails(tmp_path: Path) -> None:
    server, thread, coord, _clk, out_dir, token = _start_test_server(tmp_path, num_images=1)
    port = server.bound_port
    auth = {"Authorization": f"Bearer {token}"}
    try:
        # Before finishing, has_failures() must be True (pending != done)
        assert coord.has_failures() is True

        class _CpuStub(LocalOrtRunner):
            def __init__(self) -> None:
                pass

            def get_session(self):
                return SimpleNamespace(
                    get_inputs=lambda: [SimpleNamespace(name="input_image")],
                    run=lambda _o, _f: [np.ones((1, 1, 16, 16), dtype=np.float32)],
                )

        coord.run_remaining_on_cpu(_CpuStub())
        assert coord.is_finished()
        assert coord.has_failures() is False
        assert (out_dir / "img_0.png").is_file()

        # Late worker attempting to register after fallback starts receives 410 Gone
        st, _, _ = _https_req(
            port,
            "POST",
            "/api/register",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"adapter_name": "LateGPU", "provider": "webgpu"}).encode(),
        )
        assert st == 410
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_unreadable_input_returns_422_and_worker_finishes_remaining_jobs(tmp_path: Path) -> None:
    in_dir = tmp_path / "inputs"
    out_dir = tmp_path / "outputs"
    in_dir.mkdir()
    bad_src = in_dir / "01_corrupt.png"
    good_src = in_dir / "02_good.png"
    bad_src.write_bytes(b"definitely not an image")
    Image.new("RGB", (24, 24), (20, 120, 220)).save(good_src)

    model_path = tmp_path / "fake.onnx"
    model_path.write_bytes(b"FAKEONNX")
    spec = _fake_model_spec()
    task = CutoutTask(models={"fast": spec})
    planned = plan_outputs([bad_src, good_src], out_dir, fmt="png", bg="transparent")
    coord = JobCoordinator(
        task=task,
        model_spec=spec,
        planned_jobs=planned,
        finalize_options={"bg": "transparent", "feather": 0.0},
    )
    tls_mat = ensure_tls_cert(tmp_path / "certs", ["127.0.0.1"])
    token = "secret-422-token"
    server = TabGpuHTTPSServer(
        bind_host="127.0.0.1",
        port=0,
        advertised_host="127.0.0.1",
        cert_path=tls_mat.cert_path,
        key_path=tls_mat.key_path,
        token=token,
        coordinator=coord,
        asset_manager=AssetManager(cache_dir=tmp_path / "cache"),
        model_spec=spec,
        model_path=model_path,
    )
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    port = server.bound_port
    auth = {"Authorization": f"Bearer {token}"}
    try:
        st, _, data = _https_req(
            port, "POST", "/api/register",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"adapter_name": "GPU", "provider": "webgpu"}).encode(),
        )
        assert st == 200
        wid = json.loads(data)["worker_id"]

        # Lease job 1 (corrupt) -> GET input returns 422 with skip=True and marks only job 1 failed
        st, _, data = _https_req(
            port, "POST", "/api/jobs/next",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"worker_id": wid}).encode(),
        )
        j1 = json.loads(data)
        st, _, err_data = _https_req(
            port, "GET", j1["input_url"],
            headers={**auth, "X-Worker-Id": wid, "X-Lease-Id": j1["lease_id"]},
        )
        assert st == 422
        err_json = json.loads(err_data)
        assert err_json["skip"] is True
        assert str(bad_src) not in err_data.decode()

        # Worker continues to lease and complete job 2 (good)
        st, _, data = _https_req(
            port, "POST", "/api/jobs/next",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"worker_id": wid}).encode(),
        )
        j2 = json.loads(data)
        assert j2["status"] == "job"
        st, _, _ = _https_req(
            port, "GET", j2["input_url"],
            headers={**auth, "X-Worker-Id": wid, "X-Lease-Id": j2["lease_id"]},
        )
        assert st == 200
        st, _, _ = _https_req(
            port, "POST", j2["result_url"],
            headers={**auth, "X-Worker-Id": wid, "X-Lease-Id": j2["lease_id"]},
            body=_grayscale_png(16, 16, 255),
        )
        assert st == 200
        assert (out_dir / "02_good.png").is_file()
        assert coord.counts() == {"pending": 0, "leased": 0, "done": 1, "failed": 1}
        assert coord.has_failures() is True
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_heartbeat_lease_ceiling_non_ascii_auth_desync_and_no_unauth_ort_download(tmp_path: Path) -> None:
    clk = _MutableClock(3000.0)
    server, thread, coord, _, _out, token = _start_test_server(
        tmp_path, num_images=1, lease_seconds=10.0, clock=clk
    )
    coord.max_lease_age_seconds = 25.0
    port = server.bound_port
    auth = {"Authorization": f"Bearer {token}"}
    try:
        # Uncached ORT vendor file returns 404 without triggering outbound downloads
        st, _, _ = _https_req(port, "GET", "/vendor/ort/ort-wasm-simd-threaded.asyncify.wasm")
        assert st == 404

        # Non-ASCII Bearer header returns 403 cleanly (no TypeError crash) and sets Connection: close
        st, hdrs, _ = _https_req(
            port, "GET", "/api/config", headers={"Authorization": "Bearer caf\u00e9"}
        )
        assert st == 403
        assert hdrs.get("connection", "").lower() == "close"

        # Rejected POST with unconsumed body closes connection (prevents keep-alive desync)
        st, hdrs, _ = _https_req(
            port,
            "POST",
            "/api/jobs/job-0001/result",
            headers={**auth},  # missing X-Worker-Id / X-Lease-Id -> 400 before reading body
            body=b"GET /api/config HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n",
        )
        assert st == 400
        assert hdrs.get("connection", "").lower() == "close"

        # Heartbeat cannot extend a hung lease past max_lease_age_seconds (25s)
        st, _, data = _https_req(
            port, "POST", "/api/register",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"adapter_name": "GPU", "provider": "webgpu"}).encode(),
        )
        assert st == 200
        wid = json.loads(data)["worker_id"]
        st, _, _ = _https_req(
            port, "POST", "/api/jobs/next",
            headers={**auth, "Content-Type": "application/json"},
            body=json.dumps({"worker_id": wid}).encode(),
        )
        assert st == 200

        # Heartbeat repeatedly past 25s ceiling
        for _ in range(4):
            clk.advance(8.0)
            st, _, _ = _https_req(
                port, "POST", "/api/heartbeat",
                headers={**auth, "Content-Type": "application/json"},
                body=json.dumps({"worker_id": wid}).encode(),
            )
            assert st == 200

        # At t=3032 (> 3000 + 25), lease must be reclaimed despite active heartbeats
        assert coord.reclaim_expired_leases() == 1
        assert coord.jobs[0].status == "pending"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


