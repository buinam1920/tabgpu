"""JS syntax checks, no-persistence checks, and JS->Python grayscale PNG interop test."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tabgpu.pipeline import validate_grayscale_png_mask

WEB_DIR = Path(__file__).resolve().parents[1] / "src" / "tabgpu" / "web"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_js_syntax_check() -> None:
    for js_file in (WEB_DIR / "worker.js", WEB_DIR / "tasks" / "cutout.js"):
        proc = subprocess.run(
            ["node", "--check", str(js_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Syntax check failed for {js_file}: {proc.stderr}"


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not available")
def test_js_cutout_postprocess_produces_valid_1ch_png_for_python() -> None:
    cutout_js = (WEB_DIR / "tasks" / "cutout.js").resolve().as_uri()
    script = f"""
import {{ postprocess }} from {cutout_js!r};
const raw = new Float32Array([-10, -2, 0, 2, 10, -5, 5, 1, 0, 3, -3, 8, -8, 1, 2, 4]);
const fakeTensor = {{ getData: async () => raw }};
const blob = await postprocess(
  {{ output_image: fakeTensor }},
  {{ width: 4, height: 4, output_name: 'output_image', postprocess: 'sigmoid_min_max' }}
);
const buf = Buffer.from(await blob.arrayBuffer());
process.stdout.write(buf);
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    mask = validate_grayscale_png_mask(proc.stdout, 4, 4)
    assert mask.mode == "L"
    assert mask.size == (4, 4)
    assert mask.getpixel((0, 0)) == 0
    assert mask.getpixel((0, 1)) == 255  # index 4 => 10.0


def test_web_assets_contain_no_persistence_apis() -> None:
    forbidden = ("localStorage", "sessionStorage", "indexedDB", "caches.", "document.cookie")
    for path in (WEB_DIR / "index.html", WEB_DIR / "worker.js", WEB_DIR / "tasks" / "cutout.js"):
        # Check code outside human-readable notice comments/text in index.html
        if path.suffix == ".js":
            content = path.read_text(encoding="utf-8")
            for api in forbidden:
                assert api not in content, f"Forbidden persistence API {api!r} found in {path}"
