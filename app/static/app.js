(() => {
  const $ = (id) => document.getElementById(id);

  const dropzone = $("dropzone");
  const fileInput = $("file-input");
  const dzIdle = dropzone.querySelector(".dz-idle");
  const dzFile = dropzone.querySelector(".dz-file");
  const fileName = $("file-name");
  const fileMeta = $("file-meta");
  const clearBtn = $("clear-file");
  const presetList = $("preset-list");
  const ownerCheck = $("owner-check");
  const protectBtn = $("protect-btn");
  const statusEl = $("status");
  const errorEl = $("error");
  const result = $("result");
  const resultNote = $("result-note");
  const downloadLink = $("download-link");
  const exportButtons = $("export-buttons");
  const exportStatus = $("export-status");
  const statsEl = $("stats");
  const audioOriginal = $("audio-original");
  const audioProtected = $("audio-protected");
  const limitsInline = $("limits-inline");
  const demoBtn = $("demo-btn");
  const consentText = $("consent-text");
  const originalLabel = $("original-label");

  let selectedFile = null;
  let isDemo = false;
  let info = { accepted_extensions: [".wav", ".flac"], presets: [], max_upload_mb: 80, max_duration_minutes: 15, max_stereo_minutes_44k: null };
  let originalUrl = null;
  let busy = false;
  let currentJob = null;
  let exporting = false;

  const fmtBytes = (n) => (n >= 1024 * 1024 ? `${(n / 1024 / 1024).toFixed(1)} MB` : `${Math.round(n / 1024)} KB`);
  const extOf = (name) => {
    const i = name.lastIndexOf(".");
    return i >= 0 ? name.slice(i).toLowerCase() : "";
  };

  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.hidden = !msg;
  }

  function setStatus(msg) {
    statusEl.textContent = msg || "";
  }

  function updateButton() {
    protectBtn.disabled = busy || !selectedFile || !ownerCheck.checked;
  }

  function setDemoMode(on) {
    isDemo = on;
    consentText.textContent = on ? consentText.dataset.demo : consentText.dataset.owner;
    originalLabel.textContent = on ? "Original (synthetic demo clip)" : "Original (your upload, not stored)";
    // Attestation is per-file: switching between demo and a real upload resets it.
    ownerCheck.checked = false;
  }

  function setFile(file, { demo = false } = {}) {
    showError("");
    if (!file) {
      selectedFile = null;
      dzIdle.hidden = false;
      dzFile.hidden = true;
      fileInput.value = "";
      setDemoMode(false);
      updateButton();
      return;
    }
    setDemoMode(demo);
    const ext = extOf(file.name);
    if (!info.accepted_extensions.includes(ext)) {
      setFile(null);
      showError(`"${file.name}" is not a supported type. Upload ${info.accepted_extensions.join(", ")}.`);
      return;
    }
    if (file.size > info.max_upload_mb * 1024 * 1024) {
      setFile(null);
      showError(`That file is ${fmtBytes(file.size)}; the limit is ${info.max_upload_mb} MB.`);
      return;
    }
    selectedFile = file;
    fileName.textContent = file.name;
    fileMeta.textContent = `${fmtBytes(file.size)} · ${ext.slice(1).toUpperCase()}${demo ? " · synthetic demo, generated on the server" : ""}`;
    dzIdle.hidden = true;
    dzFile.hidden = false;
    result.hidden = true;
    updateButton();
  }

  function renderPresets() {
    presetList.innerHTML = "";
    info.presets.forEach((p) => {
      const id = `preset-${p.name}`;
      const label = document.createElement("label");
      label.className = "preset";
      label.innerHTML = `
        <input type="radio" name="strength" id="${id}" value="${p.name}" ${p.default ? "checked" : ""} />
        <span class="preset-body">
          <span class="preset-name">${p.label}</span>
          <span class="preset-desc">${p.description}</span>
        </span>`;
      presetList.appendChild(label);
    });
  }

  function stemOf(name) {
    const i = name.lastIndexOf(".");
    return i > 0 ? name.slice(0, i) : name;
  }

  function setExportStatus(msg) {
    exportStatus.textContent = msg || "";
  }

  function updateExportButtons() {
    exportButtons.querySelectorAll("button").forEach((b) => {
      b.disabled = exporting || !currentJob || b.dataset.available !== "1";
    });
  }

  function renderExportButtons() {
    exportButtons.innerHTML = "";
    (info.export_formats || [])
      .filter((f) => f.lossy)
      .forEach((f) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "button secondary";
        btn.dataset.format = f.name;
        btn.dataset.available = f.available ? "1" : "0";
        btn.textContent = f.label;
        btn.title = f.available ? f.note : f.unavailable_reason || "Not available on this server.";
        btn.addEventListener("click", () => downloadExport(f));
        exportButtons.appendChild(btn);
      });
    updateExportButtons();
  }

  async function downloadExport(fmt) {
    if (!currentJob || exporting) return;
    exporting = true;
    updateExportButtons();
    showError("");
    const short = fmt.label.split(" ·")[0];
    setExportStatus(`Encoding ${short} from the protected file… a few seconds, longer on a small server.`);
    try {
      const res = await fetch(`${currentJob.download_url}?format=${encodeURIComponent(fmt.name)}`);
      if (!res.ok) {
        let detail = `Server returned ${res.status}.`;
        try {
          const payload = await res.json();
          if (payload && payload.detail) detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
        } catch {
          /* non-JSON error body */
        }
        throw new Error(detail);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${stemOf(currentJob.output_name)}${fmt.extension}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
      setExportStatus(`${short} ready (${fmtBytes(blob.size)}). Lossy: keep the ${currentJob.output_ext.slice(1).toUpperCase()} as your reference copy.`);
    } catch (err) {
      setExportStatus("");
      showError(err && err.message ? err.message : `Could not export ${short}.`);
    } finally {
      exporting = false;
      updateExportButtons();
    }
  }

  function renderStats(data) {
    const s = data.stats;
    const rows = [
      ["Strength", s.preset],
      ["Change level (SNR)", Number.isFinite(s.snr_db) ? `${s.snr_db.toFixed(1)} dB — higher means a smaller change` : "n/a"],
      ["Duration", `${s.duration_s.toFixed(1)} s`],
      ["Format", `${s.sample_rate} Hz · ${s.channels === 1 ? "mono" : s.channels === 2 ? "stereo" : s.channels + " ch"} · ${data.output_ext.slice(1).toUpperCase()}`],
      ["Peak", `${s.original_peak.toFixed(3)} → ${s.protected_peak.toFixed(3)}${s.peak_limited ? " (scaled down to avoid clipping)" : ""}`],
      ["High-band component", s.hf_component_applied ? "applied" : "skipped (sample rate too low)"],
    ];
    statsEl.innerHTML = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  }

  async function loadInfo() {
    try {
      const res = await fetch("/api/info");
      if (!res.ok) throw new Error();
      info = await res.json();
    } catch {
      info.presets = [{ name: "light", label: "Light (default)", description: "Conservative default.", default: true }];
    }
    renderPresets();
    renderExportButtons();
    const minutes = info.max_stereo_minutes_44k
      ? `~${info.max_stereo_minutes_44k} min stereo (${info.max_mono_minutes_44k} min mono)`
      : `${info.max_duration_minutes} min`;
    limitsInline.textContent = `up to ${info.max_upload_mb} MB / ${minutes}`;
  }

  async function loadDemo() {
    if (busy) return;
    showError("");
    demoBtn.disabled = true;
    setStatus("Generating demo clip…");
    try {
      const demo = info.demo || { url: "/api/demo-clip", filename: "music-shield-demo.wav" };
      const res = await fetch(demo.url);
      if (!res.ok) throw new Error(`Could not fetch the demo clip (server returned ${res.status}).`);
      const blob = await res.blob();
      const file = new File([blob], demo.filename, { type: "audio/wav" });
      setFile(file, { demo: true });
      setStatus("Demo clip loaded. Pick a strength, tick the box, then protect.");
    } catch (err) {
      showError(err && err.message ? err.message : "Could not load the demo clip.");
      setStatus("");
    } finally {
      demoBtn.disabled = false;
    }
  }

  async function submit(ev) {
    ev.preventDefault();
    if (!selectedFile || busy) return;
    busy = true;
    updateButton();
    showError("");
    result.hidden = true;
    setStatus("Uploading and processing… longer tracks take longer.");

    const form = new FormData();
    form.append("file", selectedFile);
    const strength = presetList.querySelector("input[name=strength]:checked");
    form.append("strength", strength ? strength.value : "light");

    try {
      const res = await fetch("/api/protect", { method: "POST", body: form });
      let payload = null;
      try {
        payload = await res.json();
      } catch {
        payload = null;
      }
      if (!res.ok) {
        const detail = payload && payload.detail ? payload.detail : `Server returned ${res.status}.`;
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      currentJob = payload;
      downloadLink.href = payload.download_url;
      downloadLink.download = payload.output_name;
      downloadLink.textContent = `Download ${payload.output_ext.slice(1).toUpperCase()} (lossless, recommended)`;
      setExportStatus("");
      updateExportButtons();
      renderStats(payload);
      resultNote.textContent = payload.converted_to_lossless
        ? `Your ${payload.source_ext.slice(1).toUpperCase()} was decoded and returned as lossless ${payload.output_ext.slice(1).toUpperCase()} so the perturbation is not smoothed away by re-encoding. Download expires in ${payload.expires_in_minutes} minutes.`
        : `Returned as ${payload.output_ext.slice(1).toUpperCase()}, same sample rate as your upload. Download expires in ${payload.expires_in_minutes} minutes.`;

      if (originalUrl) URL.revokeObjectURL(originalUrl);
      originalUrl = URL.createObjectURL(selectedFile);
      audioOriginal.src = originalUrl;
      audioProtected.src = payload.download_url;
      result.hidden = false;
      setStatus("Done.");
      result.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (err) {
      showError(err && err.message ? err.message : "Something went wrong. Try again with a different file.");
      setStatus("");
    } finally {
      busy = false;
      updateButton();
    }
  }

  dropzone.addEventListener("click", (e) => {
    if (e.target === clearBtn) return;
    fileInput.click();
  });
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("over");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) setFile(file);
  });
  fileInput.addEventListener("change", () => setFile(fileInput.files[0] || null));
  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    setFile(null);
  });
  ownerCheck.addEventListener("change", updateButton);
  demoBtn.addEventListener("click", loadDemo);
  $("protect-form").addEventListener("submit", submit);

  loadInfo();
})();
