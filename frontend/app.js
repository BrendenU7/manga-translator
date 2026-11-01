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

// ------------------------------------------------------------------ setup

async function loadConfig() {
  const cfg = await (await fetch("/api/config")).json();
  const fill = (el, langs, def) => {
    el.innerHTML = Object.entries(langs)
      .map(([code, name]) => `<option value="${code}" ${code === def ? "selected" : ""}>${name}</option>`)
      .join("");
  };
  fill($("#source-lang"), cfg.source_langs, "auto");
  fill($("#target-lang"), cfg.target_langs, "en");
  fill($("#font-style"), cfg.font_styles, "manga");
  fill($("#edit-font"), cfg.font_styles, "manga");
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
            `<span class="badge" title="${img.naturalWidth}×${img.naturalHeight}px: small pages give worse OCR and lettering">⚠ Low-res</span>`);
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
}

function renderStages(stage) {
  // "waiting" = queued for translation behind the previous page; show it on the Translate step
  if (stage === "waiting") stage = "translating";
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

async function pollBatch() {
  batch = await (await fetch(`/api/batches/${batch.id}`)).json();
  const finished = batch.done + batch.failed;
  $("#bar-fill").style.width = `${(100 * finished) / batch.total}%`;
  $("#batch-count").textContent =
    `${batch.done} / ${batch.total} pages translated` + (batch.failed ? ` · ${batch.failed} failed` : "");

  const running = batch.pages.find((p) => p.status === "running" && p.stage !== "waiting")
    || batch.pages.find((p) => p.status === "running");
  renderStages(running ? running.stage : batch.status === "done" ? "done" : "queued");

  const multi = batch.total > 1;
  $("#batch-progress").hidden = !multi;
  renderStrip();
  $("#page-strip").hidden = !multi;
  $("#result-panel").hidden = false;
  $("#download-all").hidden = !multi || batch.done === 0;
  $("#download-all").href = batch.download_url;

  // Show the first finished page automatically, and refresh the selected one when it finishes
  if (!selectedId) {
    const first = batch.pages.find((p) => p.status === "done");
    if (first) selectPage(first.id);
  } else if (selectedId !== shownId) {
    const sel = batch.pages.find((p) => p.id === selectedId);
    if (sel && sel.status === "done") selectPage(selectedId);
  }

  const failedPage = !multi && batch.pages[0].status === "error";
  if (failedPage) {
    const page = await (await fetch(`/api/jobs/${batch.pages[0].id}`)).json();
    showError(page.error);
  }
  if (batch.status !== "done") return setTimeout(pollBatch, 1500);
  $("#translate-btn").disabled = false;
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
  document.querySelectorAll(".thumb").forEach((t) => t.classList.toggle("selected", t.dataset.id === id));
  const page = await (await fetch(`/api/jobs/${id}`)).json();
  if (page.status === "done") {
    job = page;
    showResult(page);
  } else {
    // Not translated yet: show the original only
    $("#img-original").src = page.original_url;
    $("#img-translated").removeAttribute("src");
    $("#bubbles").innerHTML = `<p class="note">${
      page.status === "error" ? "This page failed: " + page.error : "This page hasn't been translated yet."}</p>`;
    shownId = null;
  }
}

// ------------------------------------------------------------------ viewer + editor

function bust(url) { return `${url}?t=${Date.now()}`; }

function showResult(page) {
  const firstShow = $("#result-panel").hidden;
  shownId = page.id;
  $("#result-panel").hidden = false;
  $("#img-original").src = page.original_url;
  $("#img-translated").src = bust(page.translated_url);
  $("#cmp-original").src = page.original_url;
  $("#cmp-translated").src = $("#img-translated").src;
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
  if (firstShow) $("#result-panel").scrollIntoView({ behavior: "smooth" });
}

$("#rerender-btn").addEventListener("click", async () => {
  if (!job) return;
  const translations = [...document.querySelectorAll("#bubbles textarea")]
    .map((ta) => ({ id: Number(ta.dataset.id), text: ta.value }));
  $("#rerender-btn").disabled = true;
  const res = await fetch(`/api/jobs/${job.id}/translations`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ translations, font_style: $("#edit-font").value }),
  });
  $("#rerender-btn").disabled = false;
  if (!res.ok) return showError("Re-render failed");
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
  $("#reader-img").src = `/api/jobs/${id}/image/${reader.original ? "original" : "translated"}`;
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
