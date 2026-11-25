// tabgpu generic LAN WebGPU worker shell (zero build step, RAM/VRAM only).

const MAX_LOG_LINES = 100;
const HEARTBEAT_INTERVAL_MS = 5000;

const els = {
  statusPill: document.getElementById("status-pill"),
  tokenBanner: document.getElementById("token-banner"),
  gpuWarning: document.getElementById("gpu-warning"),
  gpuAdapter: document.getElementById("gpu-adapter"),
  execProvider: document.getElementById("exec-provider"),
  modelName: document.getElementById("model-name"),
  progressText: document.getElementById("progress-text"),
  progressBar: document.getElementById("progress-bar"),
  btnStart: document.getElementById("btn-start"),
  btnStop: document.getElementById("btn-stop"),
  wakeLockStatus: document.getElementById("wakelock-status"),
  log: document.getElementById("log"),
};

// Extract per-run secret token from #fragment and scrub the address bar immediately.
let authToken = "";
if (window.location.hash && window.location.hash.length > 1) {
  authToken = decodeURIComponent(window.location.hash.slice(1));
  try {
    history.replaceState(null, "", window.location.pathname + window.location.search);
  } catch (_) {
    // Ignore history restriction in non-standard contexts.
  }
}

const logLines = [];
function log(msg) {
  const ts = new Date().toLocaleTimeString();
  logLines.push(`[${ts}] ${msg}`);
  if (logLines.length > MAX_LOG_LINES) {
    logLines.splice(0, logLines.length - MAX_LOG_LINES);
  }
  els.log.textContent = logLines.join("\n");
  els.log.scrollTop = els.log.scrollHeight;
}

function setStatus(text, variant = "") {
  els.statusPill.textContent = text;
  els.statusPill.className = variant ? `pill ${variant}` : "pill";
}

function formatAdapterName(info) {
  const rawParts = [info?.description, info?.vendor, info?.architecture, info?.device]
    .map((s) => (s || "").trim())
    .filter(Boolean);
  const unique = [];
  for (const part of rawParts) {
    if (!unique.some((u) => u.toLowerCase().includes(part.toLowerCase()))) {
      unique.push(part);
    }
  }
  return unique.length > 0 ? unique.join(" / ") : "WebGPU Adapter";
}

function isLikelyIntegratedOrSoftwareGpu(info) {
  const vendor = (info?.vendor || "").toLowerCase();
  const arch = (info?.architecture || "").toLowerCase();
  const desc = (info?.description || "").toLowerCase();
  const device = (info?.device || "").toLowerCase();
  const combined = `${vendor} ${arch} ${desc} ${device}`.trim();

  if (!combined || combined === "unknown") return true;

  // Software / fallback renderers
  const softwarePatterns = ["swiftshader", "llvmpipe", "microsoft basic", "software", "warp"];
  if (softwarePatterns.some((p) => combined.includes(p))) return true;

  // Discrete NVIDIA and Apple Silicon unified GPUs are high-performance
  if (vendor.includes("nvidia") || desc.includes("nvidia") || desc.includes("geforce") || desc.includes("rtx")) {
    return false;
  }
  if (vendor.includes("apple") || desc.includes("apple")) {
    return false;
  }

  // Intel: discrete Arc vs integrated UHD/Iris/Gen/Xe-LP
  if (vendor.includes("intel") || desc.includes("intel")) {
    const discreteIntel = ["arc a", "arc b", "xe-hpg", "xe2-hpg", "alchemist", "battlemage"];
    if (discreteIntel.some((p) => combined.includes(p))) return false;
    return true;
  }

  // AMD: distinguish integrated APU graphics from discrete Radeon RX / Navi dGPUs
  if (vendor.includes("amd") || vendor.includes("ati") || desc.includes("radeon")) {
    const discreteAmd = ["rx ", "radeon pro", "navi", "rdna-3", "rdna-4", "7900", "7800", "7700", "7600", "6900", "6800", "6700", "6600"];
    if (discreteAmd.some((p) => combined.includes(p))) return false;
    const integratedAmd = [
      "radeon(tm) graphics",
      "radeon graphics",
      "vega",
      "renoir",
      "cezanne",
      "rembrandt",
      "phoenix",
      "hawk",
      "strix",
      "raphael",
      "van gogh",
      "gcn",
      "integrated",
    ];
    if (integratedAmd.some((p) => combined.includes(p))) return true;
  }

  return false;
}

async function detectGpuAdapter() {
  if (!("gpu" in navigator) || !navigator.gpu) {
    els.gpuAdapter.textContent = "WebGPU unavailable";
    els.gpuWarning.textContent =
      "Warning: navigator.gpu is unavailable in this browser/context. " +
      "Inference will fall back to slower WASM CPU execution. Use Chrome over HTTPS on this LAN URL.";
    els.gpuWarning.classList.remove("hidden");
    return { available: false, adapterName: "WebGPU unavailable" };
  }

  try {
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) {
      els.gpuAdapter.textContent = "No WebGPU adapter";
      els.gpuWarning.textContent =
        "Warning: Chrome could not acquire a WebGPU adapter and will fall back to slower WASM CPU execution.";
      els.gpuWarning.classList.remove("hidden");
      return { available: false, adapterName: "No WebGPU adapter" };
    }

    const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    const adapterName = formatAdapterName(info);
    els.gpuAdapter.textContent = adapterName;

    if (isLikelyIntegratedOrSoftwareGpu(info)) {
      els.gpuWarning.innerHTML =
        `<strong>Integrated/software GPU detected (${escapeHtml(adapterName)}):</strong> ` +
        "On Windows multi-GPU laptops, Windows may assign Chrome to the integrated GPU by default. " +
        "Open <em>Windows Settings &rarr; System &rarr; Display &rarr; Graphics</em>, select Chrome, and choose <strong>High performance</strong>.";
      els.gpuWarning.classList.remove("hidden");
    } else {
      els.gpuWarning.classList.add("hidden");
    }

    return { available: true, adapterName };
  } catch (err) {
    els.gpuAdapter.textContent = "Adapter probe failed";
    els.gpuWarning.textContent =
      `Warning: WebGPU adapter check failed (${err.message || err}). Falling back to WASM.`;
    els.gpuWarning.classList.remove("hidden");
    return { available: false, adapterName: "Adapter probe failed" };
  }
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function authHeaders(extra = {}) {
  return Object.assign(
    {
      Authorization: `Bearer ${authToken}`,
    },
    extra,
  );
}

async function apiFetch(url, options = {}) {
  const headers = authHeaders(options.headers || {});
  const resp = await fetch(url, {
    ...options,
    headers,
    cache: "no-store",
    credentials: "omit",
  });
  return resp;
}

// Runtime worker state
let running = false;
let stopRequested = false;
let abortController = null;
let wakeLock = null;
let heartbeatTimer = null;
let currentWorkerId = null;
let activeSession = null;
let inFlightInference = null;
let gpuProbe = { available: false, adapterName: "Detecting..." };

async function acquireWakeLock() {
  if (!("wakeLock" in navigator) || !navigator.wakeLock) {
    els.wakeLockStatus.textContent = "Wake Lock: unsupported";
    return;
  }
  try {
    wakeLock = await navigator.wakeLock.request("screen");
    els.wakeLockStatus.textContent = "Wake Lock: active";
    wakeLock.addEventListener("release", () => {
      els.wakeLockStatus.textContent = "Wake Lock: released";
    });
  } catch (_) {
    els.wakeLockStatus.textContent = "Wake Lock: denied";
  }
}

async function releaseWakeLock() {
  if (wakeLock) {
    try {
      await wakeLock.release();
    } catch (_) {
      // Ignore release errors.
    }
    wakeLock = null;
  }
  els.wakeLockStatus.textContent = "Wake Lock: released";
}

document.addEventListener("visibilitychange", () => {
  if (running && !stopRequested && document.visibilityState === "visible") {
    acquireWakeLock();
  }
});

// Release worker session immediately if the user reloads or closes the tab.
window.addEventListener("pagehide", () => {
  if (!currentWorkerId || !authToken) return;
  const wid = currentWorkerId;
  currentWorkerId = null;
  try {
    fetch("/api/disconnect", {
      method: "POST",
      keepalive: true,
      headers: authHeaders({ "Content-Type": "application/json", "X-Worker-Id": wid }),
      body: JSON.stringify({ worker_id: wid, reason: "pagehide" }),
      cache: "no-store",
      credentials: "omit",
    }).catch(() => {});
  } catch (_) {
    // Ignore unload errors.
  }
});

async function sendDisconnect(reason) {
  if (!currentWorkerId || !authToken) return;
  const wid = currentWorkerId;
  currentWorkerId = null;
  try {
    await apiFetch("/api/disconnect", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Worker-Id": wid },
      body: JSON.stringify({ worker_id: wid, reason }),
    });
  } catch (_) {
    // Host may already have shut down.
  }
}

function updateProgress(counts, total) {
  const done = (counts?.done || 0) + (counts?.failed || 0);
  els.progressText.textContent = `${done} / ${total}`;
  els.progressBar.max = Math.max(1, total);
  els.progressBar.value = Math.min(total, done);
}

async function fetchLeasedJob(workerId, signal) {
  const resp = await apiFetch("/api/jobs/next", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Worker-Id": workerId },
    body: JSON.stringify({ worker_id: workerId }),
    signal,
  });
  if (!resp.ok) {
    throw new Error(`Lease request failed (HTTP ${resp.status})`);
  }
  const data = await resp.json();
  if (data.status !== "job") {
    return data;
  }

  const imgResp = await apiFetch(data.input_url, {
    method: "GET",
    headers: {
      "X-Worker-Id": workerId,
      "X-Lease-Id": data.lease_id,
    },
    signal,
  });
  if (imgResp.status === 422) {
    let counts = data.counts;
    try {
      const errBody = await imgResp.json();
      if (errBody?.counts) counts = errBody.counts;
    } catch (_) {}
    log(`Skipped unreadable host input ${data.job_id}; continuing batch.`);
    return { status: "skip", jobId: data.job_id, counts };
  }
  if (!imgResp.ok) {
    throw new Error(`Input fetch failed for ${data.job_id} (HTTP ${imgResp.status})`);
  }
  const blob = await imgResp.blob();
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
  return {
    status: "job",
    jobId: data.job_id,
    leaseId: data.lease_id,
    meta: data.meta,
    resultUrl: data.result_url,
    bitmap,
    counts: data.counts,
  };
}

function disposePreparedJob(job) {
  if (job && job.bitmap) {
    try {
      job.bitmap.close();
    } catch (_) {
      // Ignore bitmap close errors.
    }
    job.bitmap = null;
  }
}

async function runWorker() {
  if (running) return;
  if (!authToken) {
    els.tokenBanner.classList.remove("hidden");
    return;
  }

  running = true;
  stopRequested = false;
  abortController = new AbortController();
  els.btnStart.disabled = true;
  els.btnStop.disabled = false;
  setStatus("Starting…");

  let prefetchedPromise = null;
  let prefetchedJob = null;
  let disconnectReason = "client_disconnect";
  let modelBuffer = null;
  let modelBytes = null;

  try {
    await acquireWakeLock();

    const cfgResp = await apiFetch("/api/config", { signal: abortController.signal });
    if (!cfgResp.ok) {
      throw new Error(`Failed to fetch host config (HTTP ${cfgResp.status})`);
    }
    const config = await cfgResp.json();
    els.modelName.textContent = config.model_display_name || config.model_id;
    updateProgress({ done: 0, failed: 0 }, config.total_jobs);
    log(`Connected to host. Task=${config.task}, model=${config.model_id}, jobs=${config.total_jobs}`);

    const taskModule = await import(config.task_module);
    if (typeof taskModule.preprocess !== "function" || typeof taskModule.postprocess !== "function") {
      throw new Error(`Task module ${config.task_module} missing preprocess/postprocess exports.`);
    }

    const ort = window.ort;
    if (!ort) {
      throw new Error("onnxruntime-web failed to load from /vendor/ort/ort.webgpu.min.js.");
    }
    ort.env.wasm.wasmPaths = "/vendor/ort/";
    if (ort.env.webgpu) {
      ort.env.webgpu.powerPreference = "high-performance";
    }

    const initialProvider = gpuProbe.available ? "webgpu" : "wasm";
    const regResp = await apiFetch("/api/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        adapter_name: gpuProbe.adapterName,
        provider: initialProvider,
        webgpu_available: gpuProbe.available,
      }),
      signal: abortController.signal,
    });
    if (regResp.status === 409) {
      throw new Error("Another GPU worker tab is already active on this host.");
    }
    if (regResp.status === 410) {
      throw new Error("Host already switched to CPU fallback or finished this batch.");
    }
    if (!regResp.ok) {
      throw new Error(`Worker registration failed (HTTP ${regResp.status})`);
    }
    const reg = await regResp.json();
    currentWorkerId = reg.worker_id;

    heartbeatTimer = window.setInterval(async () => {
      if (!currentWorkerId || stopRequested) return;
      try {
        await apiFetch("/api/heartbeat", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Worker-Id": currentWorkerId },
          body: JSON.stringify({ worker_id: currentWorkerId }),
        });
      } catch (_) {
        // Handled by main loop.
      }
    }, HEARTBEAT_INTERVAL_MS);

    // Fetch ONNX model bytes into RAM only (no-store, Bearer auth).
    setStatus("Downloading model…");
    log(`Fetching model over LAN (${(config.model_size_bytes / (1024 * 1024)).toFixed(1)} MB)…`);
    const tFetch0 = performance.now();
    const modelResp = await apiFetch(config.model_url, { signal: abortController.signal });
    if (!modelResp.ok) {
      throw new Error(`Model download failed (HTTP ${modelResp.status})`);
    }
    modelBuffer = await modelResp.arrayBuffer();
    const modelFetchMs = performance.now() - tFetch0;
    log(`Model loaded into RAM in ${modelFetchMs.toFixed(0)} ms.`);

    // Create ONNX Runtime session (prefer WebGPU, fall back to WASM with proxy worker so heartbeats keep firing).
    setStatus("Compiling WebGPU session…");
    const tCompile0 = performance.now();
    let activeProvider = "webgpu";
    modelBytes = new Uint8Array(modelBuffer);

    if (gpuProbe.available) {
      try {
        ort.env.wasm.proxy = false;
        activeSession = await ort.InferenceSession.create(modelBytes, {
          executionProviders: ["webgpu"],
          graphOptimizationLevel: "all",
        });
      } catch (webgpuErr) {
        log(`WebGPU session creation failed (${webgpuErr.message || webgpuErr}); falling back to WASM.`);
        activeProvider = "wasm";
        els.gpuWarning.textContent =
          `Warning: WebGPU session failed (${webgpuErr.message || webgpuErr}). Running on WASM CPU fallback.`;
        els.gpuWarning.classList.remove("hidden");
        ort.env.wasm.proxy = true;
        activeSession = await ort.InferenceSession.create(modelBytes, {
          executionProviders: ["wasm"],
        });
      }
    } else {
      activeProvider = "wasm";
      ort.env.wasm.proxy = true;
      activeSession = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ["wasm"],
      });
    }

    // Drop both references immediately so the 178-224 MB model buffer is eligible for GC.
    modelBytes = null;
    modelBuffer = null;

    const sessionCompileMs = performance.now() - tCompile0;
    els.execProvider.textContent = activeProvider.toUpperCase();
    log(`Session ready (${activeProvider}) in ${sessionCompileMs.toFixed(0)} ms.`);

    await apiFetch("/api/metrics", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Worker-Id": currentWorkerId },
      body: JSON.stringify({
        worker_id: currentWorkerId,
        model_fetch_ms: modelFetchMs,
        session_compile_ms: sessionCompileMs,
        provider: activeProvider,
        adapter_name: gpuProbe.adapterName,
      }),
      signal: abortController.signal,
    });

    setStatus(`Running (${activeProvider.toUpperCase()})`, "ok");

    // Main prefetch + inference loop.
    let currentJob = await fetchLeasedJob(currentWorkerId, abortController.signal);
    while (!stopRequested) {
      if (currentJob.counts) {
        updateProgress(currentJob.counts, config.total_jobs);
      }

      if (currentJob.status === "done") {
        disconnectReason = "completed";
        log("All jobs completed.");
        break;
      }

      if (currentJob.status === "skip") {
        currentJob = await fetchLeasedJob(currentWorkerId, abortController.signal);
        continue;
      }

      if (currentJob.status === "wait") {
        await new Promise((r) => setTimeout(r, 250));
        currentJob = await fetchLeasedJob(currentWorkerId, abortController.signal);
        continue;
      }

      // Start prefetching next job while GPU inference runs on currentJob.
      prefetchedPromise = stopRequested
        ? null
        : fetchLeasedJob(currentWorkerId, abortController.signal).catch((err) => ({
            status: "prefetch_error",
            error: err,
          }));

      let inputTensor = null;
      let outputs = null;
      try {
        const tInf0 = performance.now();
        const prep = taskModule.preprocess(currentJob.bitmap, currentJob.meta, ort);
        inputTensor = prep.inputTensor;
        disposePreparedJob(currentJob);

        inFlightInference = activeSession.run(prep.feeds);
        outputs = await inFlightInference;
        inFlightInference = null;

        const maskBlob = await taskModule.postprocess(outputs, currentJob.meta);
        const infMs = performance.now() - tInf0;

        if (!stopRequested) {
          const uploadResp = await apiFetch(currentJob.resultUrl, {
            method: "POST",
            headers: {
              "Content-Type": "image/png",
              "X-Worker-Id": currentWorkerId,
              "X-Lease-Id": currentJob.leaseId,
              "X-Inference-Ms": infMs.toFixed(2),
            },
            body: maskBlob,
            signal: abortController.signal,
          });
          if (uploadResp.ok) {
            const ack = await uploadResp.json();
            updateProgress(ack.counts, config.total_jobs);
            log(`Completed ${currentJob.jobId} (${infMs.toFixed(0)} ms, ${(maskBlob.size / 1024).toFixed(1)} KB mask)`);
          } else {
            const errText = await uploadResp.text();
            log(`Result upload rejected for ${currentJob.jobId}: HTTP ${uploadResp.status} ${errText}`);
          }
        }
      } catch (jobErr) {
        inFlightInference = null;
        if (stopRequested) break;
        const msg = jobErr?.message || String(jobErr);
        log(`Inference error on ${currentJob.jobId}: ${msg}`);
        const isDeviceLost = msg.toLowerCase().includes("device") && msg.toLowerCase().includes("lost");
        try {
          await apiFetch(`/api/jobs/${encodeURIComponent(currentJob.jobId)}/fail`, {
            method: "POST",
            headers: { "Content-Type": "application/json", "X-Worker-Id": currentWorkerId },
            body: JSON.stringify({
              worker_id: currentWorkerId,
              lease_id: currentJob.leaseId,
              error: msg,
              fatal: isDeviceLost,
            }),
          });
        } catch (_) {
          // Ignore failure reporting errors.
        }
        if (isDeviceLost) {
          disconnectReason = "webgpu_device_lost";
          break;
        }
      } finally {
        disposePreparedJob(currentJob);
        if (inputTensor && typeof inputTensor.dispose === "function") {
          try { inputTensor.dispose(); } catch (_) {}
        }
        if (outputs) {
          for (const tensor of Object.values(outputs)) {
            if (tensor && typeof tensor.dispose === "function") {
              try { tensor.dispose(); } catch (_) {}
            }
          }
        }
      }

      if (stopRequested) break;

      prefetchedJob = prefetchedPromise ? await prefetchedPromise : null;
      prefetchedPromise = null;
      if (!prefetchedJob) {
        currentJob = await fetchLeasedJob(currentWorkerId, abortController.signal);
      } else if (prefetchedJob.status === "prefetch_error") {
        if (stopRequested) break;
        throw prefetchedJob.error;
      } else {
        currentJob = prefetchedJob;
        prefetchedJob = null;
      }
    }
  } catch (err) {
    if (!stopRequested) {
      log(`Worker stopped due to error: ${err?.message || err}`);
      setStatus("Error — releasing…");
    }
  } finally {
    if (heartbeatTimer !== null) {
      clearInterval(heartbeatTimer);
      heartbeatTimer = null;
    }

    if (inFlightInference) {
      try {
        await inFlightInference;
      } catch (_) {}
      inFlightInference = null;
    }

    if (prefetchedPromise) {
      try {
        const leftover = await prefetchedPromise;
        disposePreparedJob(leftover);
      } catch (_) {}
      prefetchedPromise = null;
    }
    disposePreparedJob(prefetchedJob);
    prefetchedJob = null;

    if (activeSession) {
      try {
        await activeSession.release();
      } catch (_) {}
      activeSession = null;
    }
    modelBytes = null;
    modelBuffer = null;

    await sendDisconnect(disconnectReason);
    await releaseWakeLock();

    running = false;
    abortController = null;
    els.btnStart.disabled = false;
    els.btnStop.disabled = true;
    setStatus("Released — safe to close tab", "released");
    log("All WebGPU sessions, tensors, bitmaps, and wake lock released — safe to close tab.");
  }
}

function requestStop() {
  if (!running || stopRequested) return;
  stopRequested = true;
  els.btnStop.disabled = true;
  setStatus("Stopping & releasing…");
  log("Stop requested — aborting pending fetches and waiting for in-flight inference to settle…");
  if (abortController) {
    abortController.abort();
  }
}

els.btnStart.addEventListener("click", () => {
  runWorker();
});

els.btnStop.addEventListener("click", () => {
  requestStop();
});

// Initialize page state
(async function init() {
  if (!authToken) {
    els.tokenBanner.classList.remove("hidden");
    els.btnStart.disabled = true;
  }
  gpuProbe = await detectGpuAdapter();
  log(`Ready. Detected adapter: ${gpuProbe.adapterName}`);
})();
