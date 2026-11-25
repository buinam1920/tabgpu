"""Cutout (background removal) task implementation."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tabgpu.assets import ModelSpec, load_manifest
from tabgpu.pipeline import (
    apply_cutout_postprocess,
    atomic_save_image,
    composite_mask,
    encode_clean_rgb_png,
    load_image_srgb,
    validate_grayscale_png_mask,
)


class CutoutTask:
    """Batch AI image background removal task."""

    name: str = "cutout"
    default_model: str = "balanced"
    js_module: str = "/tasks/cutout.js"

    def __init__(self, models: dict[str, ModelSpec] | None = None) -> None:
        if models is None:
            manifest = load_manifest()
            self.models = {
                mid: spec for mid, spec in manifest.models.items() if spec.task == "cutout"
            }
        else:
            self.models = dict(models)

    def job_meta(self, model_spec: ModelSpec) -> dict[str, Any]:
        """Metadata sent to the browser worker (strictly free of filenames/paths/EXIF)."""
        return {
            "task": self.name,
            "model_id": model_spec.id,
            "width": model_spec.input_size[0],
            "height": model_spec.input_size[1],
            "input_name": model_spec.input_name,
            "output_name": model_spec.output_name,
            "mean": list(model_spec.mean),
            "std": list(model_spec.std),
            "postprocess": model_spec.postprocess,
        }

    def prepare(self, src: Path, model_spec: ModelSpec) -> tuple[bytes, dict[str, Any]]:
        """Load source with EXIF+ICC->sRGB, strip metadata, and resize to model input."""
        srgb = load_image_srgb(src)
        payload = encode_clean_rgb_png(srgb, model_spec.input_size)
        return payload, self.job_meta(model_spec)

    def infer_local(
        self,
        payload_bytes: bytes,
        meta: dict[str, Any],
        session: Any,
    ) -> bytes:
        """Run host CPU ONNX inference and return a lossless 1-channel grayscale PNG mask."""
        width = int(meta["width"])
        height = int(meta["height"])
        mean = meta["mean"]
        std = meta["std"]
        input_name = str(meta["input_name"])
        postprocess = str(meta["postprocess"])

        with Image.open(io.BytesIO(payload_bytes)) as img:
            rgb = img.convert("RGB")
            if rgb.size != (width, height):
                rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0

        norm = np.empty_like(arr, dtype=np.float32)
        norm[:, :, 0] = (arr[:, :, 0] - float(mean[0])) / float(std[0])
        norm[:, :, 1] = (arr[:, :, 1] - float(mean[1])) / float(std[1])
        norm[:, :, 2] = (arr[:, :, 2] - float(mean[2])) / float(std[2])
        tensor = np.expand_dims(norm.transpose((2, 0, 1)), axis=0).astype(np.float32)

        # Resolve input name defensively against actual ONNX session inputs.
        session_inputs = session.get_inputs()
        actual_input = (
            input_name
            if any(inp.name == input_name for inp in session_inputs)
            else session_inputs[0].name
        )
        outputs = session.run(None, {actual_input: tensor})
        mask_u8 = apply_cutout_postprocess(outputs[0], postprocess)

        mask_img = Image.fromarray(mask_u8, mode="L")
        if mask_img.size != (width, height):
            mask_img = mask_img.resize((width, height), Image.Resampling.LANCZOS)

        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        return buf.getvalue()

    def validate_result(self, result_bytes: bytes, meta: dict[str, Any]) -> None:
        """Strictly verify uploaded mask is an 8-bit 1-channel PNG of exact expected size."""
        validate_grayscale_png_mask(
            result_bytes,
            expected_width=int(meta["width"]),
            expected_height=int(meta["height"]),
        )

    def finalize(
        self,
        src: Path,
        result_bytes: bytes,
        meta: dict[str, Any],
        out: Path,
        options: dict[str, Any],
    ) -> None:
        """Composite validated grayscale mask onto full-res original and write atomically."""
        mask_l = validate_grayscale_png_mask(
            result_bytes,
            expected_width=int(meta["width"]),
            expected_height=int(meta["height"]),
        )
        original_srgb = load_image_srgb(src)
        composited = composite_mask(
            original_srgb,
            mask_l,
            bg=str(options.get("bg", "transparent")),
            feather=float(options.get("feather", 0.0)),
        )
        fmt = options.get("format", "png")
        atomic_save_image(composited, out, fmt)
