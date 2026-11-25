"""Tests for CPU cutout inference, EXIF/ICC compositing, and path safety."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageCms

from tabgpu.assets import ModelSpec
from tabgpu.pipeline import (
    InvalidMaskError,
    JobCoordinator,
    LocalOrtRunner,
    OutputSafetyError,
    apply_cutout_postprocess,
    plan_outputs,
    resolve_output_format,
    validate_grayscale_png_mask,
)
from tabgpu.tasks.cutout import CutoutTask


def _test_model_spec(postprocess: str = "min_max") -> ModelSpec:
    return ModelSpec(
        id="test",
        task="cutout",
        display_name="Test Cutout",
        filename="test.onnx",
        url="https://example.invalid/test.onnx",
        revision="test",
        sha256="0" * 64,
        size=10,
        license="MIT",
        upstream="https://example.invalid",
        status="stable",
        input_name="input_image",
        output_name="output_image",
        input_shape=(1, 3, 32, 32),
        input_size=(32, 32),
        mean=(0.5, 0.5, 0.5),
        std=(1.0, 1.0, 1.0),
        postprocess=postprocess,
    )


class _FakeOrtSession:
    """Deterministic fake ONNX session predicting foreground in the center square."""

    def __init__(self, logits: bool = False) -> None:
        self._logits = logits

    def get_inputs(self):
        return [SimpleNamespace(name="input_image")]

    def run(self, _output_names, feeds):
        tensor = next(iter(feeds.values()))
        _, _, h, w = tensor.shape
        out = np.full((1, 1, h, w), -8.0 if self._logits else 0.05, dtype=np.float32)
        out[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 8.0 if self._logits else 0.95
        return [out]


def test_exif_rotation_stripped_and_composited_transparent_and_white(tmp_path: Path) -> None:
    # Create a 40x20 image with EXIF Orientation=6 (90° CW -> becomes 20x40 after transpose).
    img = Image.new("RGB", (40, 20), (220, 40, 40))
    exif = Image.Exif()
    exif[274] = 6  # Orientation = Rotate 90 CW
    exif[271] = "SecretCameraMake"  # Should be stripped from output & worker payload
    src = tmp_path / "rotated.jpg"
    img.save(src, format="JPEG", exif=exif.tobytes())

    task = CutoutTask()
    spec = _test_model_spec("min_max")

    # Worker payload must be resized and EXIF-free
    payload, meta = task.prepare(src, spec)
    with Image.open(io.BytesIO(payload)) as prep_img:
        assert prep_img.size == (32, 32)
        assert not prep_img.getexif()
    assert "rotated.jpg" not in str(meta)

    # Local inference produces a valid 1-channel grayscale PNG
    mask_bytes = task.infer_local(payload, meta, _FakeOrtSession(logits=False))
    mask_img = validate_grayscale_png_mask(mask_bytes, 32, 32)
    assert mask_img.mode == "L"

    # Transparent cutout preserves transposed dimensions (20x40) and strips EXIF
    out_trans = tmp_path / "out_trans.png"
    task.finalize(src, mask_bytes, meta, out_trans, {"bg": "transparent", "format": "png", "feather": 0.0})
    with Image.open(out_trans) as res:
        assert res.size == (20, 40)
        assert res.mode == "RGBA"
        assert not res.getexif()
        # Corner transparent, center opaque
        assert res.getpixel((0, 0))[3] < 20
        assert res.getpixel((10, 20))[3] > 235

    # White background produces opaque RGB with white corner and red center
    out_white = tmp_path / "out_white.png"
    task.finalize(src, mask_bytes, meta, out_white, {"bg": "white", "format": "png", "feather": 1.0})
    with Image.open(out_white) as res_w:
        assert res_w.size == (20, 40)
        assert res_w.mode == "RGB"
        assert res_w.getpixel((0, 0)) == (255, 255, 255)
        r, g, b = res_w.getpixel((10, 20))
        assert r > 180 and g < 80 and b < 80


def test_rgba_and_icc_profile_handling(tmp_path: Path) -> None:
    # Build an RGBA image with an embedded sRGB ICC profile and pre-existing transparent pixel
    img = Image.new("RGBA", (24, 24), (30, 120, 210, 255))
    img.putpixel((12, 12), (30, 120, 210, 0))  # Transparent hole in center
    srgb_bytes = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    src = tmp_path / "with_icc.png"
    img.save(src, format="PNG", icc_profile=srgb_bytes)

    task = CutoutTask()
    spec = _test_model_spec("sigmoid_min_max")
    payload, meta = task.prepare(src, spec)
    mask_bytes = task.infer_local(payload, meta, _FakeOrtSession(logits=True))

    out_path = tmp_path / "out_rgba.png"
    task.finalize(src, mask_bytes, meta, out_path, {"bg": "transparent", "format": "png", "feather": 0.0})
    with Image.open(out_path) as res:
        assert res.mode == "RGBA"
        # Existing alpha=0 hole stays alpha=0 even where mask is foreground
        assert res.getpixel((12, 12))[3] == 0
        # Neighboring center pixel stays opaque
        assert res.getpixel((10, 10))[3] > 235
        # Output does not carry stale ICC/EXIF baggage
        assert not res.getexif()


def test_postprocess_sigmoid_and_min_max() -> None:
    logits = np.array([[[[-10.0, 0.0], [2.0, 10.0]]]], dtype=np.float32)
    mask = apply_cutout_postprocess(logits, "sigmoid_min_max")
    assert mask.dtype == np.uint8
    assert mask[0, 0] == 0
    assert mask[1, 1] == 255
    assert 100 < mask[0, 1] < 160


def test_source_overwrite_and_output_collision_rejected(tmp_path: Path) -> None:
    src_a = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(src_a)

    # Writing back into same dir with same name would overwrite source -> rejected
    with pytest.raises(OutputSafetyError, match="overwrite source"):
        plan_outputs([src_a], tmp_path, fmt="png", bg="transparent")

    # Two distinct inputs mapping to the same output filename -> rejected
    dir1 = tmp_path / "d1"
    dir2 = tmp_path / "d2"
    out_dir = tmp_path / "out"
    dir1.mkdir()
    dir2.mkdir()
    p1 = dir1 / "item.jpg"
    p2 = dir2 / "item.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(p1)
    Image.new("RGB", (8, 8), (4, 5, 6)).save(p2)

    with pytest.raises(OutputSafetyError, match="collision"):
        plan_outputs([p1, p2], out_dir, fmt="png", bg="transparent")

    # JPEG with transparent bg rejected
    with pytest.raises(ValueError, match="cannot preserve --bg transparent"):
        resolve_output_format("jpg", "transparent", p1)


def test_validate_mask_rejects_bad_inputs() -> None:
    # Not a PNG
    with pytest.raises(InvalidMaskError):
        validate_grayscale_png_mask(b"not a png at all" * 4, 16, 16)

    # Wrong dimensions
    buf = io.BytesIO()
    Image.new("L", (8, 8), 128).save(buf, format="PNG")
    with pytest.raises(InvalidMaskError, match="do not match"):
        validate_grayscale_png_mask(buf.getvalue(), 16, 16)

    # RGB (color type 2) rejected
    buf_rgb = io.BytesIO()
    Image.new("RGB", (16, 16), (128, 128, 128)).save(buf_rgb, format="PNG")
    with pytest.raises(InvalidMaskError, match="grayscale"):
        validate_grayscale_png_mask(buf_rgb.getvalue(), 16, 16)


def test_coordinator_cpu_run_and_csv_report(tmp_path: Path) -> None:
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    img_path = in_dir / "sample.png"
    Image.new("RGB", (32, 32), (80, 160, 240)).save(img_path)

    task = CutoutTask()
    spec = _test_model_spec()
    planned = plan_outputs([img_path], out_dir, fmt="png", bg="#00FF00")

    coord = JobCoordinator(
        task=task,
        model_spec=spec,
        planned_jobs=planned,
        finalize_options={"bg": "#00FF00", "feather": 0.0},
    )

    class _FakeRunner(LocalOrtRunner):
        def __init__(self) -> None:
            pass

        def get_session(self):
            return _FakeOrtSession()

    coord.run_remaining_on_cpu(_FakeRunner())
    assert coord.is_finished()
    assert not coord.has_failures()
    assert (out_dir / "sample.png").is_file()

    report_path = out_dir / "report.csv"
    coord.write_csv_report(report_path)
    csv_text = report_path.read_text(encoding="utf-8")
    assert "sample.png" in csv_text
    assert ",done,cpu," in csv_text
