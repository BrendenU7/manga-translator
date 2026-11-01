// Manga Translator frontend.
// One image -> a single job. Several files, or a ZIP/CBZ/PDF -> a batch of page jobs.
// Either way, one page at a time is shown in the viewer + translation editor.
const $ = (sel) => document.querySelector(sel);
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
  const isSingleImage = files.length === 1 && IMAGE_RE.test(files[0].name || "x.png");
  const form = new FormData();
  files.forEach((f) => form.append(isSingleImage ? "file" : "files", f, f.name || "pasted.png"));
  form.append("source_lang", $("#source-lang").value);
  form.append("target_lang", $("#target-lang").value);
  form.append("style", $("#style").value);
  form.append("glossary", $("#glossary").value);

  $("#translate-btn").disabled = true;
  $("#result-panel").hidden = true;
  $("#error").hidden = true;
  $("#progress-panel").hidden = false;
  $("#batch-progress").hidden = isSingleImage;
  renderStages("queued");
  batch = null; job = null; selectedId = null; shownId = null;

  const res = await fetch(isSingleImage ? "/api/jobs" : "/api/batches", { method: "POST", body: form });
  if (!res.ok) return showError((await res.json()).detail || res.statusText);
  if (isSingleImage) {
    job = await res.json();
    pollJob();
  } else {
    batch = await res.json();
    pollBatch();
  }
});

function renderStages(stage) {
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

// ------------------------------------------------------------------ single page

async function pollJob() {
  job = await (await fetch(`/api/jobs/${job.id}`)).json();
  renderStages(job.stage);
  if (job.status === "error") return showError(job.error);
  if (job.status !== "done") return setTimeout(pollJob, 800);
  $("#translate-btn").disabled = false;
  $("#page-strip").hidden = true;
  $("#download-all").hidden = true;
  showResult(job);
}

// ------------------------------------------------------------------ batch

async function pollBatch() {
  batch = await (await fetch(`/api/batches/${batch.id}`)).json();
  const finished = batch.done + batch.failed;
  $("#bar-fill").style.width = `${(100 * finished) / batch.total}%`;
  $("#batch-count").textContent =
    `${batch.done} / ${batch.total} pages translated` + (batch.failed ? ` · ${batch.failed} failed` : "");

  const running = batch.pages.find((p) => p.status === "running");
  renderStages(running ? running.stage : batch.status === "done" ? "done" : "queued");

  renderStrip();
  $("#result-panel").hidden = false;
  $("#download-all").hidden = batch.done === 0;
  $("#download-all").href = batch.download_url;

  // Show the first finished page automatically, and refresh the selected one when it finishes
  if (!selectedId) {
    const first = batch.pages.find((p) => p.status === "done");
    if (first) selectPage(first.id);
  } else if (selectedId !== shownId) {
    const sel = batch.pages.find((p) => p.id === selectedId);
    if (sel && sel.status === "done") selectPage(selectedId);
  }

  if (batch.status !== "done") return setTimeout(pollBatch, 1500);
  $("#translate-btn").disabled = false;
}

function renderStrip() {
  const strip = $("#page-strip");
  strip.hidden = false;
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
    const label = { queued: "Waiting", running: "Working…", done: "Done", error: "Failed" }[p.status];
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
    body: JSON.stringify({ translations }),
  });
  $("#rerender-btn").disabled = false;
  if (!res.ok) return showError("Re-render failed");
  job = await res.json();
  $("#img-translated").src = bust(job.translated_url);
});

document.querySelectorAll(".seg button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("active", x === b));
    $("#viewer").className = b.dataset.view;
  }));

// ------------------------------------------------------------------ dark mode

function renderThemeButton() {
  const dark = document.documentElement.dataset.theme === "dark";
  $("#theme-toggle").textContent = dark ? "☀" : "☾";
  $("#theme-toggle").title = dark ? "Switch to light mode" : "Switch to dark mode";
}

$("#theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) { /* storage blocked: still works this visit */ }
  renderThemeButton();
});
renderThemeButton();

loadConfig();
