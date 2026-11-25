"""Tests for CLI entry point, help output, and `tabgpu doctor`."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tabgpu.cli import main


def test_cli_help_via_module() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "tabgpu.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "cutout" in proc.stdout
    assert "doctor" in proc.stdout


def test_cli_cutout_help_flags() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "tabgpu.cli", "cutout", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    for flag in ("--model", "--mode", "--bg", "--format", "--feather", "--bench", "--bind", "--host", "--cert", "--key", "--wait-seconds"):
        assert flag in proc.stdout


def test_doctor_command(tmp_path: Path, capsys) -> None:
    rc = main(["doctor", "--bind", "127.0.0.1", "--port", "0", "--cache-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "tabgpu" in captured.out
    assert "ORT CPU:" in captured.out
    assert "LAN IP:" in captured.out
    assert "Firewall hints" in captured.out


def test_cutout_cli_cpu_bench_and_nonzero_on_failed_job(tmp_path: Path, monkeypatch, capsys) -> None:
    from types import SimpleNamespace
    import numpy as np
    from PIL import Image
    from tabgpu import assets as assets_mod
    from tabgpu import pipeline as pipeline_mod

    # Missing input path -> exit code 2
    rc_missing = main(["cutout", str(tmp_path / "does_not_exist.jpg"), "-o", str(tmp_path / "out0"), "--mode", "cpu"])
    assert rc_missing == 2

    dummy_model = tmp_path / "dummy.onnx"
    dummy_model.write_bytes(b"fake")
    monkeypatch.setattr(assets_mod.AssetManager, "ensure_model", lambda self, mid: dummy_model)
    monkeypatch.setattr(
        pipeline_mod.LocalOrtRunner,
        "get_session",
        lambda self: SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="input_image")],
            run=lambda _o, f: [np.ones((1, 1, 320, 320), dtype=np.float32)],
        ),
    )

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    good_img = in_dir / "good.png"
    bad_img = in_dir / "broken.png"
    Image.new("RGB", (32, 32), (10, 120, 200)).save(good_img)
    bad_img.write_bytes(b"not a valid png")

    # Good image alone -> exit 0 + --bench output + CSV report
    rc_ok = main(
        [
            "cutout",
            str(good_img),
            "-o",
            str(out_dir),
            "--model",
            "fast",
            "--mode",
            "cpu",
            "--bg",
            "white",
            "--bench",
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    captured = capsys.readouterr()
    assert rc_ok == 0
    assert "Benchmark Summary (--bench)" in captured.out
    assert (out_dir / "good.png").is_file()
    assert (out_dir / "tabgpu_report.csv").is_file()

    # Mix of good + corrupted image -> exit 1 + both recorded in CSV
    out_dir2 = tmp_path / "out2"
    rc_fail = main(
        [
            "cutout",
            str(good_img),
            str(bad_img),
            "-o",
            str(out_dir2),
            "--model",
            "fast",
            "--mode",
            "cpu",
            "--cache-dir",
            str(tmp_path / "cache"),
        ]
    )
    assert rc_fail == 1
    report_text = (out_dir2 / "tabgpu_report.csv").read_text(encoding="utf-8")
    assert ",done,cpu," in report_text
    assert ",failed,cpu," in report_text


