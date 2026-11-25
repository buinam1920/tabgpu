"""Command-line interface for tabgpu (`cutout` and `doctor`)."""

from __future__ import annotations

import argparse
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

from tabgpu import __version__
from tabgpu.assets import AssetIntegrityError, AssetManager, UnknownAssetError
from tabgpu.pipeline import (
    JobCoordinator,
    JobRecord,
    LocalOrtRunner,
    OutputSafetyError,
    discover_inputs,
    parse_bg_color,
    plan_outputs,
    resolve_output_format,
)
from tabgpu.server import (
    TabGpuHTTPSServer,
    check_port_available,
    detect_lan_ip,
    generate_session_token,
    list_lan_ipv4_addresses,
    validate_ipv4_bind_address,
)
from tabgpu.tasks import TASKS
from tabgpu.tls import TLSError, ensure_tls_cert


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tabgpu",
        description=(
            "Borrow a LAN laptop's GPU via a Chrome WebGPU tab (zero install, "
            "no app-managed persistence) with automatic host CPU fallback."
        ),
    )
    parser.add_argument("--version", action="version", version=f"tabgpu {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    # --- doctor ---
    doc = sub.add_parser(
        "doctor",
        help="Inspect LAN interfaces, port availability, local onnxruntime, cache, and firewall hints.",
    )
    doc.add_argument("--port", type=int, default=8787, help="Port to check (default: 8787).")
    doc.add_argument("--bind", type=str, default=None, help="Specific IPv4 bind address to test.")
    doc.add_argument("--cache-dir", type=Path, default=None, help="Override host cache directory.")

    # --- cutout ---
    cut = sub.add_parser(
        "cutout",
        help="Batch AI image background removal via Chrome WebGPU on LAN or host CPU.",
    )
    cut.add_argument(
        "inputs",
        nargs="+",
        help="Input image files and/or directories.",
    )
    cut.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output directory for composited images.",
    )
    cut.add_argument(
        "--model",
        choices=("fast", "balanced", "best"),
        default="balanced",
        help="Model profile: fast (u2netp 320px), balanced (isnet 1024px, default), best (birefnet-lite 1024px).",
    )
    cut.add_argument(
        "--mode",
        choices=("auto", "lan", "cpu"),
        default="auto",
        help="Execution mode: auto (wait for LAN worker then CPU fallback), lan (LAN only), cpu (local CPU only).",
    )
    cut.add_argument(
        "--bg",
        default="transparent",
        help="Output background: transparent (default), white, or #RRGGBB.",
    )
    cut.add_argument(
        "--format",
        dest="output_format",
        choices=("auto", "png", "jpg", "webp"),
        default="auto",
        help="Output image format (default: auto).",
    )
    cut.add_argument(
        "--feather",
        type=float,
        default=0.0,
        help="Optional Gaussian feather radius in pixels for mask edges (default: 0).",
    )
    cut.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing non-source files in the output directory.",
    )
    cut.add_argument(
        "--bind",
        default=None,
        help="IPv4 network interface address to bind (defaults to detected LAN IP, not 0.0.0.0).",
    )
    cut.add_argument(
        "--host",
        default=None,
        help="Advertised IPv4 address or hostname for pairing URL and TLS SAN.",
    )
    cut.add_argument(
        "--port",
        type=int,
        default=8787,
        help="HTTPS port for the temporary LAN server (default: 8787).",
    )
    cut.add_argument(
        "--cert",
        type=Path,
        default=None,
        help="Optional path to user-supplied PEM TLS certificate.",
    )
    cut.add_argument(
        "--key",
        type=Path,
        default=None,
        help="Optional path to user-supplied PEM TLS private key.",
    )
    cut.add_argument(
        "--wait-seconds",
        type=float,
        default=60.0,
        help="Seconds to wait for a LAN GPU worker before CPU fallback / timeout (default: 60).",
    )
    cut.add_argument(
        "--lease-seconds",
        type=float,
        default=45.0,
        help="Per-job lease timeout in seconds (default: 45).",
    )
    cut.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to write CSV job report (default: <output>/tabgpu_report.csv).",
    )
    cut.add_argument(
        "--bench",
        action="store_true",
        help="Print separated cold-start and warm p50/p95 timing summary.",
    )
    cut.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Override host asset/cert cache directory.",
    )

    return parser


def _print_firewall_hints(port: int) -> None:
    system = platform.system().lower()
    print("\nFirewall hints (allow inbound LAN TCP only on trusted private networks):")
    if system == "windows":
        print(
            f"  Windows: Ensure your Wi-Fi/Ethernet network profile is set to 'Private', and allow\n"
            f"           TCP port {port} in Windows Defender Firewall (inbound rule for Private profile),\n"
            f"           or click 'Allow access (Private networks)' if Windows prompts on first run."
        )
    elif system == "darwin":
        print(
            f"  macOS:   System Settings -> Network -> Firewall: allow Python incoming connections\n"
            f"           and ensure Stealth Mode does not block local LAN TCP port {port}."
        )
    else:
        print(
            f"  Linux:   UFW:       sudo ufw allow from 192.168.0.0/16 to any port {port} proto tcp\n"
            f"           firewalld: sudo firewall-cmd --add-port={port}/tcp"
        )


def run_doctor(args: argparse.Namespace) -> int:
    print(f"tabgpu {__version__} doctor")
    print("=" * 40)
    print(f"Python:   {sys.version.split()[0]} ({platform.system()} {platform.release()})")

    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        print(f"ORT CPU:  onnxruntime {ort.__version__} (providers: {', '.join(providers)})")
    except Exception as exc:
        print(f"ORT CPU:  UNAVAILABLE ({exc})")

    try:
        preferred_ip = validate_ipv4_bind_address(args.bind) if args.bind else detect_lan_ip()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    lan_ips = list_lan_ipv4_addresses()
    print(f"LAN IP:   {preferred_ip} (detected interfaces: {', '.join(lan_ips) or 'loopback only'})")

    port_ok, port_msg = check_port_available(preferred_ip, args.port)
    status_str = "OK (available)" if port_ok else f"BUSY/BLOCKED ({port_msg})"
    print(f"Port:     {preferred_ip}:{args.port} -> {status_str}")

    assets = AssetManager(cache_dir=args.cache_dir)
    status = assets.cache_status()
    print(f"Cache:    {status['cache_dir']}")
    print("Models:")
    for mid, st in status["models"].items():
        spec = assets.manifest.models[mid]
        print(f"  - {mid:<8} ({spec.filename}): {st}")
    req_cached = status["ort_required_cached"]
    req_total = status["ort_required_total"]
    print(f"ORT-Web:  v{status['ort_version']} ({req_cached}/{req_total} required WebGPU runtime files cached)")

    _print_firewall_hints(args.port)
    return 0 if port_ok else 1


def _format_ms(val: float | None) -> str:
    return f"{val:.1f} ms" if val is not None else "n/a"


def _print_bench(coordinator: JobCoordinator, model_id: str, ort_version: str) -> None:
    summary = coordinator.benchmark_summary()
    cold = summary["cold_start"]
    warm = summary["warm"]
    print("\n--- Benchmark Summary (--bench) ---")
    print(f"Model:                 {model_id}")
    print(f"Worker provider:       {cold['provider'] or 'n/a'}")
    if cold["adapter_name"]:
        print(f"GPU adapter:           {cold['adapter_name']}")
    print(f"ORT-Web version:       {ort_version}")
    print("Cold-start (excluded from warm percentiles when >= 2 LAN jobs):")
    print(f"  Model LAN transfer:  {_format_ms(cold['model_fetch_ms'])}")
    print(f"  Session compile:     {_format_ms(cold['session_compile_ms'])}")
    print(f"Warm per-image ({warm['sample_count']} samples):")
    print(f"  Inference p50 / p95: {_format_ms(warm['inference_p50_ms'])} / {_format_ms(warm['inference_p95_ms'])}")
    print(f"  Round-trip p50 / p95:{_format_ms(warm['round_trip_p50_ms'])} / {_format_ms(warm['round_trip_p95_ms'])}")


def run_cutout(args: argparse.Namespace) -> int:
    try:
        parse_bg_color(args.bg)
        if args.feather < 0:
            raise ValueError("--feather must be >= 0")
        if args.bind is not None:
            validate_ipv4_bind_address(args.bind)
        sources = discover_inputs(args.inputs)
        if not sources:
            raise ValueError("No supported input images found.")
        resolve_output_format(args.output_format, args.bg, sources[0])
        planned = plan_outputs(
            sources,
            args.output,
            fmt=args.output_format,
            bg=args.bg,
            overwrite=args.overwrite,
        )
    except (ValueError, FileNotFoundError, OutputSafetyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    task = TASKS["cutout"]
    assets = AssetManager(cache_dir=args.cache_dir)

    try:
        model_spec = assets.get_model_spec(args.model)
        print(f"Preparing model '{model_spec.id}' ({model_spec.display_name})...")
        model_path = assets.ensure_model(model_spec.id)
        if args.mode in ("auto", "lan"):
            print(f"Ensuring pinned onnxruntime-web v{assets.manifest.ort_version} runtime...")
            assets.ensure_ort_runtime()
    except (UnknownAssetError, AssetIntegrityError, OSError) as exc:
        print(f"Asset error: {exc}", file=sys.stderr)
        return 1

    total = len(planned)
    print(f"Queued {total} image(s) -> {args.output.resolve()} (mode={args.mode}, bg={args.bg})")

    def _on_progress(job: JobRecord) -> None:
        counts = coordinator.counts()
        completed = counts["done"] + counts["failed"]
        if job.status == "done":
            rt = f"{job.round_trip_ms:.0f}ms" if job.round_trip_ms is not None else "?"
            print(f"[{completed}/{total}] OK   ({job.worker_type}) {job.src_path.name} -> {job.out_path.name} ({rt})")
        elif job.status == "failed":
            print(f"[{completed}/{total}] FAIL ({job.worker_type or 'worker'}) {job.src_path.name}: {job.error}")

    coordinator = JobCoordinator(
        task=task,
        model_spec=model_spec,
        planned_jobs=planned,
        finalize_options={"bg": args.bg, "feather": args.feather},
        lease_seconds=args.lease_seconds,
        on_progress=_on_progress,
    )
    cpu_runner = LocalOrtRunner(lambda: model_path)
    report_path = args.report or (args.output.resolve() / "tabgpu_report.csv")

    if args.mode == "cpu":
        try:
            coordinator.run_remaining_on_cpu(cpu_runner)
        except KeyboardInterrupt:
            print("\nInterrupted by user; writing partial report...", file=sys.stderr)
            coordinator.mark_remaining_failed("interrupted by user")
    else:
        bind_ip = validate_ipv4_bind_address(args.bind) if args.bind else detect_lan_ip()
        advertised_host = args.host or (detect_lan_ip() if bind_ip == "0.0.0.0" else bind_ip)
        san_hosts = [advertised_host]
        if bind_ip != "0.0.0.0" and bind_ip not in san_hosts:
            san_hosts.append(bind_ip)

        try:
            tls_mat = ensure_tls_cert(
                assets.certs_dir,
                san_hosts,
                cert_path=args.cert,
                key_path=args.key,
            )
            token = generate_session_token()
            server = TabGpuHTTPSServer(
                bind_host=bind_ip,
                port=args.port,
                advertised_host=advertised_host,
                cert_path=tls_mat.cert_path,
                key_path=tls_mat.key_path,
                token=token,
                coordinator=coordinator,
                asset_manager=assets,
                model_spec=model_spec,
                model_path=model_path,
            )
        except (TLSError, ValueError, OSError) as exc:
            print(f"Server startup error: {exc}", file=sys.stderr)
            return 1

        server_thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.2},
            daemon=True,
        )
        server_thread.start()

        print("\nOpen this URL in Chrome (Incognito recommended) on your GPU laptop:")
        print(f"  {server.worker_url}")
        print(f"TLS cert SHA-256: {tls_mat.fingerprint_sha256}")
        if args.mode == "auto":
            print(f"Waiting up to {args.wait_seconds:.0f}s for LAN GPU worker (will fall back to host CPU)...\n")
        else:
            print(f"Waiting up to {args.wait_seconds:.0f}s for LAN GPU worker (--mode lan)...\n")

        try:
            _coordinate_lan_run(
                coordinator=coordinator,
                cpu_runner=cpu_runner,
                mode=args.mode,
                wait_seconds=args.wait_seconds,
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user; shutting down cleanly...", file=sys.stderr)
            coordinator.mark_remaining_failed("interrupted by user")
        finally:
            coordinator.close_to_workers()
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=3.0)

    coordinator.write_csv_report(report_path)
    counts = coordinator.counts()
    print(
        f"\nSummary: {counts['done']} done, {counts['failed']} failed "
        f"(report: {report_path})"
    )
    if args.bench:
        _print_bench(coordinator, model_spec.id, assets.manifest.ort_version)

    return 1 if coordinator.has_failures() else 0


def _coordinate_lan_run(
    *,
    coordinator: JobCoordinator,
    cpu_runner: LocalOrtRunner,
    mode: str,
    wait_seconds: float,
) -> None:
    """Wait for LAN worker progress, reclaim expired leases, and fall back to CPU in auto mode."""
    deadline = time.monotonic() + max(0.0, wait_seconds)
    disconnected_since: float | None = None

    while not coordinator.is_finished():
        coordinator.reclaim_expired_leases()
        if coordinator.is_finished():
            break

        now = time.monotonic()
        if not coordinator.ever_connected:
            if now >= deadline:
                if mode == "auto":
                    print("No LAN worker connected within wait window; falling back to host CPU...")
                    coordinator.run_remaining_on_cpu(cpu_runner)
                else:
                    coordinator.mark_remaining_failed(
                        f"no LAN worker connected within {wait_seconds:.0f}s"
                    )
                break
        elif not coordinator.is_worker_alive():
            if mode == "auto":
                print("LAN worker disconnected or timed out; finishing remaining jobs on host CPU...")
                coordinator.run_remaining_on_cpu(cpu_runner)
                break
            else:
                if disconnected_since is None:
                    disconnected_since = now
                    print(
                        f"LAN worker disconnected; waiting up to {wait_seconds:.0f}s for reconnect..."
                    )
                if now - disconnected_since >= wait_seconds:
                    coordinator.mark_remaining_failed(
                        "LAN worker disconnected and did not reconnect within wait window"
                    )
                    break
        else:
            disconnected_since = None

        coordinator.all_done_event.wait(timeout=0.25)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return run_doctor(args)
    if args.command == "cutout":
        return run_cutout(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
