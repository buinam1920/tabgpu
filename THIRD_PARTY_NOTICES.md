# Third-Party Notices & Model Provenance

`tabgpu` itself is licensed under the **MIT License** (see [LICENSE](LICENSE)).

`tabgpu` does **not** bundle model weights or `onnxruntime-web` binaries inside this repository or Python wheel. Instead, the host CLI downloads pinned artifacts on demand into the host user's OS cache directory and verifies each file against the SHA-256 checksums in `src/tabgpu/assets_manifest.json`.

Each downloaded third-party component remains governed by its own upstream license and terms:

---

## 1. ONNX Runtime Web (`onnxruntime-web` v1.31.1)

- **Upstream:** <https://github.com/microsoft/onnxruntime>
- **Distribution:** `onnxruntime-web@1.31.1` via npm / jsDelivr
- **License:** MIT License (Copyright (c) Microsoft Corporation)
- **Pinned files:** `ort.webgpu.min.js`, `ort-wasm-simd-threaded.{mjs,wasm}`, `ort-wasm-simd-threaded.jsep.{mjs,wasm}`, `ort-wasm-simd-threaded.asyncify.{mjs,wasm}`, `ort-wasm-simd-threaded.jspi.{mjs,wasm}`.

---

## 2. `fast` Model — `u2netp.onnx` (U²-NetP, 320×320)

- **Upstream architecture & weights:** Xuebin Qin et al., [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)
- **ONNX distribution:** [danielgatis/rembg `v0.0.0`](https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx)
- **SHA-256:** `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8`
- **License:** **Apache License 2.0**
- **Verified ONNX spec:**
  - Input: `input.1` (`float32[1, 3, 320, 320]`), RGB `/ 255.0`, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
  - Output: `1959` (`float32[1, 1, 320, 320]`), probabilities already in `[0, 1]` (sigmoid inside graph).
  - Postprocessing: `min_max` normalization `(pred - min) / (max - min)` matching `rembg`'s `U2netpSession`.

---

## 3. `balanced` Model (Default) — `isnet-general-use.onnx` (IS-Net / DIS, 1024×1024)

- **Upstream architecture & weights:** Xuebin Qin et al., [xuebinqin/DIS](https://github.com/xuebinqin/DIS) (ECCV 2022)
- **ONNX distribution:** [danielgatis/rembg `v0.0.0`](https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx)
- **SHA-256:** `60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a`
- **License & Important Provenance Note:**
  - The upstream `xuebinqin/DIS` repository carries the **Apache License 2.0** for its code and evaluation metrics, and publishes `isnet-general-use.pth` ("optimized model weights for general use") alongside the repository without a separate non-commercial restriction on the weights.
  - However, the **DIS5K dataset** (`DIS5K-Dataset-Terms-of-Use.pdf` in `xuebinqin/DIS`) is governed by a **non-commercial research/educational** dataset license. If your organization requires strict commercial dataset-provenance clearance for trained weights, please review `xuebinqin/DIS` before commercial deployment, or select `--model fast` (`u2netp`, Apache-2.0) or `--model best` (`BiRefNet_lite`, MIT).
- **Verified ONNX spec:**
  - Input: `input_image` (`float32[1, 3, 1024, 1024]`), RGB `/ 255.0`, mean `[0.5, 0.5, 0.5]`, std `[1.0, 1.0, 1.0]` (maps pixel values to `[-0.5, 0.5]`).
  - Output: `output_image` (`float32[1, 1, 1024, 1024]`), probabilities already in `[0, 1]` (sigmoid inside graph).
  - Postprocessing: exact `rembg` `DisSession` `min_max` normalization `(pred - min) / (max - min)` with zero-range guard.

---

## 4. `best` Model — `BiRefNet_lite` ONNX (1024×1024)

- **Upstream architecture & weights:** Peng Zheng et al., [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
- **ONNX conversion & pinned revision:** [onnx-community/BiRefNet_lite-ONNX](https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/tree/de15b22ba131738a16dff04aab8bdf8dc32e3ac1) (`onnx/model.onnx`, commit `de15b22ba131738a16dff04aab8bdf8dc32e3ac1`; byte-for-byte identical to `rembg`'s `BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx`)
- **SHA-256:** `5600024376f572a557870a5eb0afb1e5961636bef4e1e22132025467d0f03333`
- **License:** **MIT License**
- **Verified ONNX spec (verified on real ONNX Runtime inference):**
  - Input: `input_image` (`float32[1, 3, 1024, 1024]`), RGB `/ 255.0`, mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
  - Output: `output_image` (`float32[1, 1, 1024, 1024]`), **raw unbounded logits** (verified range approx. `[-18.0, +14.6]` on real inference).
  - Postprocessing: `sigmoid_min_max` (`1 / (1 + exp(-x))` followed by `min_max` normalization with zero-range guard, matching `rembg`'s `BiRefNetSessionGeneral`).

---

## 5. Python Runtime Dependencies

`tabgpu` relies on the following open-source Python packages installed via `pip`/`pipx`/`uvx`:
- **cryptography** (Apache-2.0 / BSD-3-Clause)
- **NumPy** (BSD-3-Clause)
- **ONNX Runtime** (MIT)
- **Pillow** (MIT-CMU / HPND)
- **platformdirs** (MIT)
