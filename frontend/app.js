// Manga Translator frontend.
// One image -> a single job. Several files, or a ZIP/CBZ/PDF -> a batch of page jobs.
// Either way, one page at a time is shown in the viewer + translation editor.
const STAGES = ["detecting", "reading", "translating", "cleaning", "typesetting"];
const IMAGE_RE = /\.(png|jpe?g|webp|gif|bmp|avif|heic|heif)$/i;
const SUPPORTED_RE = /\.(png|jpe?g|webp|gif|bmp|avif|heic|heif|zip|cbz|epub|rar|cbr|pdf)$/i;

let files = [];          // what the user picked
let job = null;          // the page currently shown
let batch = null;        // current batch, if any
let selectedId = null;   // job id of the selected page in a batch
let shownId = null;      // job id whose result is currently rendered
let scrolledFor = null;  // batch id whose first page we already scrolled into view
let shownStatus = null;  // status of the selected page when we last rendered it

// ------------------------------------------------------------------ setup

let appCfg = null;  // /api/config, kept for language names etc.

async function loadConfig() {
  const cfg = await (await fetch("/api/config")).json();
  appCfg = cfg;
  const fill = (el, langs, def) => {
    el.innerHTML = Object.entries(langs)
      .map(([code, name]) => `<option value="${code}" ${code === def ? "selected" : ""}>${name}</option>`)
      .join("");
  };
  fill($("#source-lang"), cfg.source_langs, "auto");
  fill($("#target-lang"), cfg.target_langs, "en");
  fill($("#font-style"), cfg.font_styles, "manga");
  // Quality: show what 1,000 pages would roughly cost with each
  $("#quality").innerHTML = Object.entries(cfg.qualities).map(([k, q]) =>
    `<option value="${k}" ${k === cfg.default_quality ? "selected" : ""}>` +
    `${q.label} · ≈ ${fmtUsd(q.est_page_usd * 1000)} / 1,000 pages</option>`).join("");
  fill($("#edit-font"), cfg.font_styles, "manga");
  $("#again-quality").innerHTML = Object.entries(cfg.qualities)
    .map(([k, q]) => `<option value="${k}">${q.label.split(" (")[0]}</option>`).join("");
  if (cfg.translator === "mock") {
    $("#mode-note").hidden = false;
    $("#mode-note").textContent =
      "Demo mode: no ANTHROPIC_API_KEY set, so translations are placeholders. Detection, OCR and typesetting are real.";
  }
}

// ------------------------------------------------------------------ file picker

const LOW_RES_HEIGHT = 1100;  // pages smaller than this give noticeably worse OCR and lettering

function fmtSize(bytes) {
  return bytes > 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function addFiles(list) {
  const added = [...list].filter((f) => SUPPORTED_RE.test(f.name) || f.type.startsWith("image/"));
  for (const f of added) {
    if (!files.some((x) => x.name === f.name && x.size === f.size)) files.push(f);
  }
  renderFileList();
}

function renderFileList() {
  const grid = $("#file-grid");
  grid.innerHTML = "";
  $("#file-list").hidden = files.length === 0;
  $("#translate-btn").disabled = files.length === 0;
  $("#dropzone").classList.toggle("has-file", files.length > 0);

  const images = files.filter((f) => IMAGE_RE.test(f.name) || (f.type.startsWith("image/") && !f.name));
  const docs = files.length - images.length;
  $("#file-count").textContent = [
    images.length && `${images.length} image${images.length > 1 ? "s" : ""}`,
    docs && `${docs} document${docs > 1 ? "s" : ""} (pages are counted after upload)`,
  ].filter(Boolean).join(" · ");

  files.forEach((f, i) => {
    const card = document.createElement("div");
    card.className = "file-card";
    const ext = (f.name.split(".").pop() || "img").toUpperCase();
    card.innerHTML = `<div class="preview"><span class="doc">${ext}</span></div>
      <div class="meta"><b></b><span>${fmtSize(f.size)}</span></div>
      <button class="remove" type="button" title="Remove">✕</button>`;
    card.querySelector("b").textContent = f.name || "pasted image";
    card.querySelector(".remove").addEventListener("click", () => { files.splice(i, 1); renderFileList(); });
    if (IMAGE_RE.test(f.name) || f.type.startsWith("image/")) {
      const img = new Image();
      img.onload = () => {
        card.querySelector(".preview").replaceChildren(img);
        if (img.naturalHeight < LOW_RES_HEIGHT && img.naturalWidth < LOW_RES_HEIGHT) {
          card.classList.add("lowres");
          card.insertAdjacentHTML("beforeend",
            `<span class="badge" title="${img.naturalWidth}×${img.naturalHeight}px: small pages give worse OCR and lettering. Tip: tick 'Clean up low-quality scans'.">⚠ Low-res</span>`);
        }
      };
      img.src = URL.createObjectURL(f);  // HEIC won't decode in most browsers: the card keeps its label
    }
    grid.appendChild(card);
  });
}

$("#clear-all").addEventListener("click", () => { files = []; renderFileList(); $("#file-input").value = ""; });

const dz = $("#dropzone");
$("#file-input").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
["dragenter", "dragover"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
document.addEventListener("paste", (e) => {
  const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
  if (item) addFiles([new File([item.getAsFile()], `pasted-${Date.now()}.png`, { type: "image/png" })]);
});

// ------------------------------------------------------------------ start

$("#translate-btn").addEventListener("click", async () => {
  // Every upload is a batch (one page or many), so it also shows on the dashboard
  const form = new FormData();
  files.forEach((f) => form.append("files", f, f.name || "pasted.png"));
  form.append("source_lang", $("#source-lang").value);
  form.append("target_lang", $("#target-lang").value);
  form.append("style", $("#style").value);
  form.append("glossary", $("#glossary").value);
  form.append("font_style", $("#font-style").value);
  form.append("quality", $("#quality").value);
  form.append("enhance", $("#enhance").checked ? "true" : "false");

  $("#translate-btn").disabled = true;
  $("#result-panel").hidden = true;
  $("#error").hidden = true;
  $("#progress-panel").hidden = false;
  renderStages("queued");
  resetBatchView();

  const res = await fetch("/api/batches", { method: "POST", body: form });
  if (!res.ok) return showError((await res.json()).detail || res.statusText);
  batch = await res.json();
  history.replaceState(null, "", `?batch=${batch.id}`);  // reload/bookmark reopens this batch
  pollBatch();
});

function resetBatchView() {
  batch = null; job = null; selectedId = null; shownId = null;
  $("#page-strip").innerHTML = "";
  $("#page-chips").innerHTML = "";
}

function renderStages(stage) {
  // "waiting" = queued for translation behind the previous page; show it on the Translate step
  if (stage === "waiting") stage = "translating";
  if (stage === "enhancing") stage = "detecting";  // scan cleanup happens before detection
  const idx = stage === "done" ? STAGES.length : STAGES.indexOf(stage);
  document.querySelectorAll("#stages li").forEach((li, i) => {
    li.className = i < idx ? "done" : i === idx ? "active" : "";
  });
}

function showError(msg) {
  $("#error").hidden = false;
  $("#error").textContent = msg;
  $("#translate-btn").disabled = false;
}

// ------------------------------------------------------------------ batch

let pollTimer = null;

// Start polling again (e.g. after a retry) unless a poll is already scheduled
function ensurePolling() {
  if (!pollTimer) pollBatch();
}

async function pollBatch() {
  pollTimer = null;
  batch = await (await fetch(`/api/batches/${batch.id}`)).json();
  const multi = batch.total > 1;

  renderHead();
  renderProgress();
  renderSummary();
  renderChips();
  renderDownloadCard();
  renderCost();
  renderStrip();
  $("#page-strip").hidden = !multi;
  // Results open once there's a page to show (or a failed single page, for its Retry buttons)
  $("#result-panel").hidden = !multi && batch.done === 0 && batch.status !== "done";

  // Show the first finished page automatically, and refresh the selected one when it finishes
  if (!selectedId) {
    const first = batch.pages.find((p) => p.status === "done")
      || (batch.status === "done" && batch.pages.find((p) => p.status === "error"));
    if (first) selectPage(first.id);
  } else if (selectedId !== shownId) {
    const sel = batch.pages.find((p) => p.id === selectedId);
    if (sel && (sel.status === "done" || sel.status === "error") && sel.status !== shownStatus) selectPage(selectedId);
  }

  if (batch.status !== "done") { pollTimer = setTimeout(pollBatch, 1500); return; }
  $("#translate-btn").disabled = false;
}

// ------------------------------------------------------------------ progress page parts

const STAGE_LABEL = { queued: "Pending", enhancing: "Cleaning scan", detecting: "Detecting text",
  reading: "Reading", waiting: "In line", translating: "Translating", cleaning: "Erasing",
  typesetting: "Lettering" };
// How many of the 5 steps a page has finished, by its current stage
const STAGE_STEP = { queued: 0, enhancing: 0, detecting: 0, reading: 1, waiting: 2, translating: 2,
  cleaning: 3, typesetting: 4, done: 5 };

function langName(code) {
  if (code === "auto") return "Auto-detect";
  const all = { ...(appCfg?.source_langs || {}), ...(appCfg?.target_langs || {}) };
  return (all[code] || code).split(" (")[0];
}

// Title line: swap the upload box for "Translation Progress · Auto-detect → English · 29 pages"
function renderHead() {
  $("#upload-panel").hidden = true;
  $("#job-head").hidden = false;
  document.querySelector("main").classList.add("results");
  const quality = batch.quality === "best" ? "Best quality" : "Economy";
  $("#job-sub").textContent = `${batch.name} · ${langName(batch.source_lang)} → ${langName(batch.target_lang)}`
    + ` · ${batch.total} page${batch.total === 1 ? "" : "s"} · ${quality}`;
  $("#stop-btn").hidden = batch.status === "done";
}

function renderProgress() {
  const finished = batch.done + batch.failed;
  const done = batch.status === "done";
  $("#progress-panel").hidden = false;
  $("#progress-panel").classList.toggle("done", done);
  $("#pc-icon").textContent = done ? "✓" : "文";
  $("#pc-title").textContent = done ? "Overall Progress" : "Processing";
  $("#pc-count").textContent = `${finished}/${batch.total}`;
  $("#bar-fill").style.width = `${(100 * finished) / batch.total}%`;
  // "2 Translating · 3 Erasing · 6 Pending"
  const counts = {};
  for (const p of batch.pages) {
    if (p.status === "queued" || p.status === "running") {
      const label = STAGE_LABEL[p.status === "queued" ? "queued" : p.stage] || "Working";
      counts[label] = (counts[label] || 0) + 1;
    }
  }
  const order = Object.values(STAGE_LABEL);
  $("#pc-detail").textContent = done ? "" : Object.entries(counts)
    .sort((a, b) => order.indexOf(b[0]) - order.indexOf(a[0])).map(([l, n]) => `${n} ${l}`).join(" · ");
  $("#pc-detail").hidden = done;
  // Single pages show the 5 steps; chapters show per-page chips instead
  const single = batch.total === 1;
  $("#stages").hidden = !single || done;
  if (single) {
    const p = batch.pages[0];
    renderStages(p.status === "done" ? "done" : p.status === "queued" ? "queued" : p.stage);
  }
}

function renderSummary() {
  const show = batch.done + batch.failed > 0;
  $("#summary-card").hidden = !show;
  if (!show) return;
  const parts = [`${batch.total} page${batch.total === 1 ? "" : "s"}`, `${batch.done} translated`];
  if (batch.failed) parts.push(`${batch.failed} failed`);
  parts.push(`est. cost ${fmtUsd(batch.cost_usd)}`);
  $("#summary-text").textContent = `🗒 ${parts.join(" · ")}`;
  $("#retry-failed").hidden = !batch.failed;
  $("#retry-failed").textContent = `↻ Retry failed (${batch.failed})`;
}

function renderChips() {
  const box = $("#page-chips");
  box.hidden = batch.total < 2;
  if (box.hidden) return;
  batch.pages.forEach((p, i) => {
    let chip = box.querySelector(`[data-id="${p.id}"]`);
    if (!chip) {
      chip = document.createElement("button");
      chip.type = "button";
      chip.dataset.id = p.id;
      chip.addEventListener("click", () => {
        const page = batch.pages.find((x) => x.id === p.id);
        if (page && (page.status === "done" || page.status === "error")) {
          selectPage(p.id);
          scrollToWorkspace();
        }
      });
      box.appendChild(chip);
    }
    const n = i + 1;
    if (p.status === "done") {
      chip.className = "chip done";
      chip.innerHTML = `✓ ${n}`;
      chip.title = `Page ${n}: done (${fmtUsd(p.cost_usd)}). Click to open.`;
    } else if (p.status === "error") {
      chip.className = "chip error";
      chip.innerHTML = `! ${n}`;
      chip.title = `Page ${n}: failed. Click to see why and retry.`;
    } else {
      const step = STAGE_STEP[p.status === "queued" ? "queued" : p.stage] ?? 0;
      const dots = [0, 1, 2, 3, 4].map((k) => `<i class="${k < step ? "on" : k === step && p.status === "running" ? "now" : ""}"></i>`).join("");
      const label = p.status === "running" ? `<span class="lbl">${STAGE_LABEL[p.stage] || "Working"}</span>` : "";
      chip.className = `chip ${p.status}`;
      chip.innerHTML = `${n} <span class="dots">${dots}</span>${label}`;
      chip.title = `Page ${n}: ${STAGE_LABEL[p.status === "queued" ? "queued" : p.stage] || p.status}`;
    }
    chip.classList.toggle("selected", p.id === selectedId);
  });
}

function renderDownloadCard() {
  $("#download-card").hidden = batch.done === 0;
  $("#download-all").href = batch.download_url;
  $("#keep-days").textContent = appCfg?.retention_days ?? 10;
  $("#keep-until").textContent = new Date(batch.expires_at * 1000)
    .toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" });
}

$("#stop-btn").addEventListener("click", async () => {
  $("#stop-btn").disabled = true;
  await fetch(`/api/batches/${batch.id}/stop`, { method: "POST" });
  $("#stop-btn").disabled = false;
  ensurePolling();
});

// ‹ 3 / 26 › between finished pages
// Pages the arrows step through: finished and failed ones (failed pages show Retry)
function donePageIds() {
  return batch ? batch.pages.filter((p) => p.status === "done" || p.status === "error").map((p) => p.id) : [];
}

function renderPageNav() {
  const ids = batch ? batch.pages.map((p) => p.id) : [];
  $("#page-nav").hidden = ids.length < 2 && !(job && job.status === "done" && !job.regions.length);
  const i = ids.indexOf(selectedId || shownId);
  $("#page-pos").textContent = i >= 0 ? `${i + 1} / ${ids.length}` : `– / ${ids.length}`;
  // Short note about the page on screen, e.g. "No text found on this page"
  let note = "";
  if (job && job.id === selectedId) {
    if (job.status === "error") note = "Failed";
    else if (job.status === "done" && !job.regions.length) note = "No text found on this page";
  }
  $("#page-note").hidden = !note;
  $("#page-note").textContent = note;
}

function scrollToWorkspace() {
  const top = $("#toolbar").getBoundingClientRect().top + window.scrollY;
  window.scrollTo({ top });
}

function stepPage(step) {
  const ids = donePageIds();
  if (!ids.length) return;
  const i = ids.indexOf(selectedId);
  const next = ids[Math.min(ids.length - 1, Math.max(0, (i < 0 ? 0 : i) + step))];
  if (next && next !== selectedId) {
    selectPage(next);
    // Slide the thumbnail strip sideways to the current page (without scrolling the window)
    const thumb = document.querySelector(`.thumb[data-id="${next}"]`);
    const strip = $("#page-strip");
    if (thumb) strip.scrollTo({ left: thumb.offsetLeft - strip.clientWidth / 2 + thumb.clientWidth / 2 });
  }
}

$("#prev-page").addEventListener("click", () => stepPage(-1));
$("#next-page").addEventListener("click", () => stepPage(1));
document.addEventListener("keydown", (e) => {
  // Arrow keys flip pages, unless you're typing or the fullscreen reader is open
  if (!$("#reader").hidden || ["TEXTAREA", "INPUT", "SELECT"].includes(document.activeElement?.tagName)) return;
  if (e.key === "ArrowRight") stepPage(1);
  else if (e.key === "ArrowLeft") stepPage(-1);
});

function renderCost() {
  if (!batch) return;
  const page = batch.pages.find((p) => p.id === shownId);
  const parts = [];
  if (page && batch.total > 1) parts.push(`This page: ${fmtUsd(page.cost_usd)}`);
  parts.push(`${batch.total > 1 ? "All pages" : "Cost"}: ${fmtUsd(batch.cost_usd)}`);
  $("#cost-info").textContent = `${parts.join(" · ")} (est.)`;
}

function renderStrip() {
  const strip = $("#page-strip");
  batch.pages.forEach((p, i) => {
    let el = strip.querySelector(`[data-id="${p.id}"]`);
    if (!el) {
      el = document.createElement("button");
      el.className = "thumb";
      el.dataset.id = p.id;
      el.title = p.filename;
      el.innerHTML = `<img alt="" loading="lazy"><span class="num">${i + 1}</span><span class="state"></span>`;
      el.querySelector("img").src = `/api/jobs/${p.id}/image/original`;
      el.addEventListener("click", () => selectPage(p.id));
      strip.appendChild(el);
    }
    const label = p.stage === "waiting" ? "In line…"
      : { queued: "Waiting", running: "Working…", done: "Done", error: "Failed" }[p.status];
    el.querySelector(".state").textContent = label;
    el.classList.remove("queued", "running", "done", "error");
    el.classList.add(p.status);
    el.classList.toggle("selected", p.id === selectedId);
  });
}

async function selectPage(id) {
  selectedId = id;
  document.querySelectorAll(".thumb, .chip").forEach((t) => t.classList.toggle("selected", t.dataset.id === id));
  const page = await (await fetch(`/api/jobs/${id}`)).json();
  shownStatus = page.status;
  $("#page-msg").hidden = true;
  if (page.status === "done") {
    job = page;
    showResult(page);
    return;
  }
  // Not translated (yet): show the original, and for a failure, why + Retry buttons
  job = page;
  shownId = null;
  $("#img-original").src = page.original_url;
  $("#cmp-original").src = page.original_url;
  $("#img-translated").removeAttribute("src");
  $("#bubbles").innerHTML = page.status === "error" ? ""
    : '<p class="note">This page is being translated…</p>';
  $("#retry-box").hidden = page.status !== "error";
  $("#retry-reason").textContent = `This page failed: ${page.error || "unknown error"}`;
  $("#edit-actions").hidden = true;
  $("#again-box").hidden = true;
  renderPageNav();
}

async function retryPage(id, quality) {
  const res = await fetch(`/api/jobs/${id}/retry`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(quality ? { quality } : {}),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    return pageMessage(err.detail || "Couldn't start the retry", true);
  }
  if (id === selectedId) {
    shownId = null;
    shownStatus = "queued";
    $("#retry-box").hidden = true;
    $("#edit-actions").hidden = true;
    $("#again-box").hidden = true;
    $("#bubbles").innerHTML = "";
    pageMessage(`Translating this page again${quality === "best" ? " with Best quality" : ""}…`);
  }
  ensurePolling();
}

function pageMessage(text, bad = false) {
  $("#page-msg").hidden = false;
  $("#page-msg").textContent = text;
  $("#page-msg").classList.toggle("bad", bad);
}

$("#retry-economy").addEventListener("click", () => retryPage(selectedId, job?.quality || "economy"));
$("#retry-best").addEventListener("click", () => retryPage(selectedId, "best"));
$("#again-btn").addEventListener("click", () => retryPage(selectedId, $("#again-quality").value));
$("#retry-failed").addEventListener("click", async () => {
  $("#retry-failed").disabled = true;
  for (const p of batch.pages.filter((p) => p.status === "error")) await retryPage(p.id);
  $("#retry-failed").disabled = false;
});

// ------------------------------------------------------------------ viewer + editor

function bust(url) { return `${url}?t=${Date.now()}`; }

function showResult(page) {
  shownId = page.id;
  shownStatus = page.status;
  $("#retry-box").hidden = true;
  $("#edit-actions").hidden = false;
  $("#again-box").hidden = false;
  $("#again-quality").value = page.quality === "best" ? "economy" : "best";  // suggest the other one
  renderCost();
  renderPageNav();
  $("#result-panel").hidden = false;
  $("#img-original").src = page.original_url;
  $("#img-translated").src = bust(page.translated_url);
  $("#cmp-original").src = page.original_url;
  $("#cmp-translated").src = $("#img-translated").src;
  // Long webtoon strips scroll at full width instead of shrinking to the window height
  $("#img-translated").onload = (e) => {
    $("#compare").classList.toggle("strip", isStrip(e.target));
  };
  $("#edit-font").value = page.font_style || "manga";
  $("#download").href = page.translated_url;
  const stem = page.filename.split(" / ").pop().replace(/\.\w+$/, "");
  $("#download").download = stem + "_translated.png";

  const list = $("#bubbles");
  list.innerHTML = "";
  if (!page.regions.length) list.innerHTML = '<p class="note">No text found on this page.</p>';
  page.regions.forEach((r) => {
    const row = document.createElement("div");
    row.className = "bubble";
    row.innerHTML = `<div class="num">${r.id + 1}</div><div class="src"></div><textarea rows="2"></textarea>`;
    row.querySelector(".src").textContent = r.source_text;
    const ta = row.querySelector("textarea");
    ta.value = r.translated_text;
    ta.dataset.id = r.id;
    list.appendChild(row);
  });
  // The first page shown for a batch: scroll so the toolbar sits at the top, then
  // toolbar + whole page + editor fill the screen
  // (no auto-scroll: the progress cards above stay in view, the page is just below)
}

$("#rerender-btn").addEventListener("click", async () => {
  if (!job || job.status !== "done") return;
  const translations = [...document.querySelectorAll("#bubbles textarea")]
    .map((ta) => ({ id: Number(ta.dataset.id), text: ta.value }));
  $("#rerender-btn").disabled = true;
  const res = await fetch(`/api/jobs/${job.id}/translations`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ translations, font_style: $("#edit-font").value }),
  });
  $("#rerender-btn").disabled = false;
  if (!res.ok) return pageMessage("Re-render failed: this page isn't finished.", true);
  job = await res.json();
  $("#img-translated").src = bust(job.translated_url);
  $("#cmp-translated").src = $("#img-translated").src;
});

// Changing the font of a finished page re-renders it right away
$("#edit-font").addEventListener("change", () => $("#rerender-btn").click());

// Live preview of the chosen lettering font
$("#font-style").addEventListener("change", (e) => {
  $("#font-preview").className = `font-preview font-${e.target.value}`;
});

// ------------------------------------------------------------------ before/after slider

(function setupSlider() {
  const cmp = $("#compare");
  let dragging = false;
  const move = (e) => {
    const rect = cmp.getBoundingClientRect();
    const pct = Math.min(100, Math.max(0, ((e.clientX - rect.left) / rect.width) * 100));
    cmp.style.setProperty("--pos", `${pct}%`);
  };
  cmp.addEventListener("pointerdown", (e) => { dragging = true; cmp.setPointerCapture(e.pointerId); move(e); });
  cmp.addEventListener("pointermove", (e) => dragging && move(e));
  cmp.addEventListener("pointerup", () => { dragging = false; });
})();

function isStrip(img) {
  return img.naturalHeight > 2.5 * img.naturalWidth;
}

// ------------------------------------------------------------------ fullscreen reader

const reader = { pages: [], index: 0, original: false };

function readerPages() {
  // Finished pages of the current batch, or just the page on screen
  if (batch) return batch.pages.filter((p) => p.status === "done").map((p) => p.id);
  return job ? [job.id] : [];
}

function renderReader() {
  const id = reader.pages[reader.index];
  if (!id) return;
  const img = $("#reader-img");
  img.onload = () => {
    // Strips: full width, scroll down to read. Pages: fit the whole page on screen.
    $("#reader").classList.toggle("scroll", isStrip(img));
  };
  img.src = `/api/jobs/${id}/image/${reader.original ? "original" : "translated"}`;
  $("#reader-count").textContent = `${reader.index + 1} / ${reader.pages.length}`;
  $("#reader-orig").classList.toggle("on", reader.original);
  // Preload the next page so flipping is instant
  const next = reader.pages[reader.index + 1];
  if (next) new Image().src = `/api/jobs/${next}/image/translated`;
}

function openReader() {
  reader.pages = readerPages();
  if (!reader.pages.length) return;
  reader.index = Math.max(0, reader.pages.indexOf(shownId));
  reader.original = false;
  $("#reader").hidden = false;
  try { $("#reader").requestFullscreen?.().catch(() => {}); } catch (e) { /* not allowed: overlay still works */ }
  renderReader();
}

function closeReader() {
  $("#reader").hidden = true;
  if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
}

function flip(step) {
  reader.index = Math.min(reader.pages.length - 1, Math.max(0, reader.index + step));
  $("#reader").scrollTop = 0;
  renderReader();
}

$("#read-btn").addEventListener("click", openReader);
$("#reader-close").addEventListener("click", closeReader);
$("#reader-prev").addEventListener("click", () => flip(-1));
$("#reader-next").addEventListener("click", () => flip(1));
$(".reader-zone.prev").addEventListener("click", () => flip(-1));
$(".reader-zone.next").addEventListener("click", () => flip(1));
$("#reader-orig").addEventListener("click", () => { reader.original = !reader.original; renderReader(); });
document.addEventListener("fullscreenchange", () => {
  if (!document.fullscreenElement && !$("#reader").hidden) $("#reader").hidden = true;
});
document.addEventListener("keydown", (e) => {
  if ($("#reader").hidden) return;
  if (e.key === "ArrowRight" || e.key === " ") { e.preventDefault(); flip(1); }
  else if (e.key === "ArrowLeft") flip(-1);
  else if (e.key === "Escape") closeReader();
  else if (e.key.toLowerCase() === "o" && !reader.original) { reader.original = true; renderReader(); }
});
document.addEventListener("keyup", (e) => {
  if (!$("#reader").hidden && e.key.toLowerCase() === "o") { reader.original = false; renderReader(); }
});

document.querySelectorAll(".seg button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("active", x === b));
    $("#viewer").className = b.dataset.view;
  }));

// Opened from the dashboard (or reloaded): /index.html?batch=<id>
async function openBatchFromUrl() {
  const id = new URLSearchParams(location.search).get("batch");
  if (!id) return;
  const res = await fetch(`/api/batches/${id}`);
  if (!res.ok) return history.replaceState(null, "", location.pathname);
  resetBatchView();
  batch = await res.json();
  $("#progress-panel").hidden = false;
  pollBatch();
}

currentUser.then(async () => {
  await loadConfig();
  openBatchFromUrl();
});
