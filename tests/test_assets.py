"""Tests for pinned asset manifest and atomic SHA-256 verification."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tabgpu.assets import (
    AssetIntegrityError,
    AssetManager,
    AssetManifest,
    ModelSpec,
    OrtAssetSpec,
    UnknownAssetError,
    load_manifest,
)


def test_bundled_manifest_integrity_schema() -> None:
    manifest = load_manifest()
    assert manifest.schema_version == 1
    assert set(manifest.models.keys()) == {"fast", "balanced", "best"}
    for spec in manifest.models.values():
        assert len(spec.sha256) == 64
        assert int(spec.sha256, 16) >= 0
        assert spec.url.startswith("https://")
        assert spec.postprocess in ("min_max", "sigmoid_min_max", "sigmoid")

    # Verify required ORT WebGPU variants (including asyncify and jspi) are pinned.
    ort_names = set(manifest.ort_files.keys())
    for required in (
        "ort.webgpu.min.js",
        "ort-wasm-simd-threaded.mjs",
        "ort-wasm-simd-threaded.wasm",
        "ort-wasm-simd-threaded.jsep.mjs",
        "ort-wasm-simd-threaded.jsep.wasm",
        "ort-wasm-simd-threaded.asyncify.mjs",
        "ort-wasm-simd-threaded.asyncify.wasm",
        "ort-wasm-simd-threaded.jspi.mjs",
        "ort-wasm-simd-threaded.jspi.wasm",
    ):
        assert required in ort_names
        assert len(manifest.ort_files[required].sha256) == 64


def _make_test_manifest(payload: bytes) -> AssetManifest:
    digest = hashlib.sha256(payload).hexdigest()
    return AssetManifest(
        schema_version=1,
        ort_version="1.31.1",
        ort_license="MIT",
        ort_files={
            "ort.webgpu.min.js": OrtAssetSpec(
                filename="ort.webgpu.min.js",
                url="https://example.invalid/ort.webgpu.min.js",
                sha256=digest,
                size=len(payload),
                content_type="application/javascript",
            )
        },
        models={
            "fast": ModelSpec(
                id="fast",
                task="cutout",
                display_name="Test Fast",
                filename="test_fast.onnx",
                url="https://example.invalid/test_fast.onnx",
                revision="test@1",
                sha256=digest,
                size=len(payload),
                license="MIT",
                upstream="https://example.invalid",
                status="stable",
                input_name="input",
                output_name="output",
                input_shape=(1, 3, 16, 16),
                input_size=(16, 16),
                mean=(0.5, 0.5, 0.5),
                std=(1.0, 1.0, 1.0),
                postprocess="min_max",
            )
        },
    )


def test_atomic_download_and_reuse(tmp_path: Path) -> None:
    payload = b"valid-model-bytes-12345"
    manifest = _make_test_manifest(payload)
    calls = 0

    def fake_download(url: str, part: Path) -> str:
        nonlocal calls
        calls += 1
        part.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    mgr = AssetManager(cache_dir=tmp_path, manifest=manifest, downloader=fake_download)
    p1 = mgr.ensure_model("fast")
    assert p1.read_bytes() == payload
    assert calls == 1
    assert list(p1.parent.glob("*.part")) == []

    # Second call reuses verified cache without downloading again
    p2 = mgr.ensure_model("fast")
    assert p2 == p1
    assert calls == 1


def test_sha256_mismatch_aborts_without_corrupting_cache(tmp_path: Path) -> None:
    expected_payload = b"expected-good-bytes"
    tampered_payload = b"tampered-bad-bytes!"
    manifest = _make_test_manifest(expected_payload)

    def bad_download(url: str, part: Path) -> str:
        part.write_bytes(tampered_payload)
        return hashlib.sha256(tampered_payload).hexdigest()

    mgr = AssetManager(cache_dir=tmp_path, manifest=manifest, downloader=bad_download)
    with pytest.raises(AssetIntegrityError):
        mgr.ensure_model("fast")

    # Target file must not exist and no .part leftovers remain
    assert not mgr.model_path("fast").exists()
    assert list(mgr.models_dir.glob("*.part")) == []


def test_corrupted_cached_file_detected_on_reuse(tmp_path: Path) -> None:
    payload = b"good-bytes"
    manifest = _make_test_manifest(payload)
    mgr = AssetManager(cache_dir=tmp_path, manifest=manifest)
    target = mgr.model_path("fast")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"corrupted-on-disk")

    with pytest.raises(AssetIntegrityError):
        mgr.ensure_model("fast")

    with pytest.raises(UnknownAssetError):
        mgr.get_model_spec("nonexistent")
