// tabgpu cutout browser task module: preprocess -> ONNX feeds, postprocess -> 1-channel grayscale PNG.

const PNG_SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

// Precomputed IEEE 802.3 CRC-32 table for PNG chunks.
const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) {
      c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) {
    c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  }
  return (c ^ 0xffffffff) >>> 0;
}

function makePngChunk(typeStr, data) {
  const chunk = new Uint8Array(12 + data.length);
  const view = new DataView(chunk.buffer);
  view.setUint32(0, data.length, false);
  for (let i = 0; i < 4; i += 1) {
    chunk[4 + i] = typeStr.charCodeAt(i);
  }
  chunk.set(data, 8);
  const crc = crc32(chunk.subarray(4, 8 + data.length));
  view.setUint32(8 + data.length, crc, false);
  return chunk;
}

async function zlibDeflate(rawBytes) {
  const cs = new CompressionStream("deflate");
  const writer = cs.writable.getWriter();
  const writePromise = writer.write(rawBytes).then(() => writer.close());
  const reader = cs.readable.getReader();
  const parts = [];
  let total = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    total += value.byteLength;
  }
  await writePromise;
  const out = new Uint8Array(total);
  let offset = 0;
  for (const p of parts) {
    out.set(p, offset);
    offset += p.byteLength;
  }
  return out;
}

/**
 * Encode an 8-bit 1-channel grayscale (PNG color type 0) lossless PNG in pure JS.
 * Uses native CompressionStream('deflate') (RFC 1950 zlib format) with zero dependencies.
 */
export async function encodeGrayscalePng(grayBytes, width, height) {
  if (grayBytes.length !== width * height) {
    throw new Error(`Expected ${width * height} grayscale pixels, got ${grayBytes.length}.`);
  }

  // PNG scanlines: 1 filter byte (0 = None) + width pixel bytes per row.
  const stride = width + 1;
  const rawScanlines = new Uint8Array(stride * height);
  for (let y = 0; y < height; y += 1) {
    const rowOff = y * stride;
    rawScanlines[rowOff] = 0; // Filter type 0 (None)
    rawScanlines.set(grayBytes.subarray(y * width, (y + 1) * width), rowOff + 1);
  }

  const compressed = await zlibDeflate(rawScanlines);

  const ihdrData = new Uint8Array(13);
  const ihdrView = new DataView(ihdrData.buffer);
  ihdrView.setUint32(0, width, false);
  ihdrView.setUint32(4, height, false);
  ihdrData[8] = 8; // Bit depth = 8
  ihdrData[9] = 0; // Color type = 0 (Grayscale, 1 channel)
  ihdrData[10] = 0; // Compression = deflate
  ihdrData[11] = 0; // Filter = adaptive
  ihdrData[12] = 0; // Interlace = none

  const ihdr = makePngChunk("IHDR", ihdrData);
  const idat = makePngChunk("IDAT", compressed);
  const iend = makePngChunk("IEND", new Uint8Array(0));

  const png = new Uint8Array(
    PNG_SIGNATURE.length + ihdr.length + idat.length + iend.length,
  );
  let pos = 0;
  png.set(PNG_SIGNATURE, pos); pos += PNG_SIGNATURE.length;
  png.set(ihdr, pos); pos += ihdr.length;
  png.set(idat, pos); pos += idat.length;
  png.set(iend, pos);
  return png;
}

function extractRgbPixels(imageBitmap, width, height) {
  let canvas;
  let ctx;
  if (typeof OffscreenCanvas !== "undefined") {
    canvas = new OffscreenCanvas(width, height);
    ctx = canvas.getContext("2d", { willReadFrequently: true });
  } else {
    canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    ctx = canvas.getContext("2d", { willReadFrequently: true });
  }
  ctx.drawImage(imageBitmap, 0, 0, width, height);
  const imgData = ctx.getImageData(0, 0, width, height);
  return imgData.data;
}

/**
 * Convert an ImageBitmap into normalized NCHW Float32 ONNX Runtime feeds.
 */
export function preprocess(imageBitmap, meta, ort) {
  const width = Number(meta.width);
  const height = Number(meta.height);
  const [meanR, meanG, meanB] = meta.mean;
  const [stdR, stdG, stdB] = meta.std;
  const rgba = extractRgbPixels(imageBitmap, width, height);

  const hw = width * height;
  const chw = new Float32Array(3 * hw);
  const inv255 = 1.0 / 255.0;

  for (let i = 0; i < hw; i += 1) {
    const p = i * 4;
    chw[i] = (rgba[p] * inv255 - meanR) / stdR;
    chw[hw + i] = (rgba[p + 1] * inv255 - meanG) / stdG;
    chw[2 * hw + i] = (rgba[p + 2] * inv255 - meanB) / stdB;
  }

  const inputTensor = new ort.Tensor("float32", chw, [1, 3, height, width]);
  return {
    feeds: { [meta.input_name]: inputTensor },
    inputTensor,
  };
}

/**
 * Apply model-specific postprocessing (min_max / sigmoid_min_max) and return a 1-channel PNG Blob.
 */
export async function postprocess(outputs, meta) {
  const width = Number(meta.width);
  const height = Number(meta.height);
  const hw = width * height;

  const outTensor = outputs[meta.output_name] || Object.values(outputs)[0];
  if (!outTensor) {
    throw new Error("ONNX session returned no output tensors.");
  }

  const rawData = typeof outTensor.getData === "function"
    ? await outTensor.getData()
    : outTensor.data;

  if (!rawData || rawData.length < hw) {
    throw new Error(
      `Unexpected output tensor size: expected at least ${hw} elements, got ${rawData ? rawData.length : 0}.`,
    );
  }

  const probs = new Float32Array(hw);
  const mode = meta.postprocess || "min_max";
  const applySigmoid = mode === "sigmoid_min_max" || mode === "sigmoid";
  const applyMinMax = mode === "min_max" || mode === "sigmoid_min_max";

  let minVal = Infinity;
  let maxVal = -Infinity;

  for (let i = 0; i < hw; i += 1) {
    let v = rawData[i];
    if (applySigmoid) {
      if (v > 60) v = 60;
      else if (v < -60) v = -60;
      v = 1.0 / (1.0 + Math.exp(-v));
    }
    probs[i] = v;
    if (v < minVal) minVal = v;
    if (v > maxVal) maxVal = v;
  }

  const gray = new Uint8Array(hw);
  const span = maxVal - minVal;
  for (let i = 0; i < hw; i += 1) {
    let v = probs[i];
    if (applyMinMax && span >= 1e-6) {
      v = (v - minVal) / span;
    }
    if (v < 0) v = 0;
    else if (v > 1) v = 1;
    gray[i] = Math.round(v * 255);
  }

  const pngBytes = await encodeGrayscalePng(gray, width, height);
  return new Blob([pngBytes], { type: "image/png" });
}
