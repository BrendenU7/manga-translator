// Manga Translator frontend: upload a page, poll the job, show results, edit translations.
const $ = (sel) => document.querySelector(sel);
const STAGES = ["detecting", "reading", "translating", "cleaning", "typesetting"];
let file = null;
let job = null;

async function loadConfig() {
  const cfg = await (await fetch("/api/config")).json();
  const fill = (el, langs, def) => {
    el.innerHTML = Object.entries(langs)
      .map(([code, name]) => `<option value="${code}" ${code === def ? "selected" : ""}>${name}</option>`)
      .join("");
  };
  fill($("#source-lang"), cfg.source_langs, "ja");
  fill($("#target-lang"), cfg.target_langs, "en");
  if (cfg.translator === "mock") {
    $("#mode-note").hidden = false;
    $("#mode-note").textContent =
      "Demo mode: no ANTHROPIC_API_KEY set, so translations are placeholders. Detection, OCR and typesetting are real.";
  }
}

function setFile(f) {
  if (!f || !f.type.startsWith("image/")) return;
  file = f;
  $("#dropzone strong").textContent = f.name;
  $("#dropzone").classList.add("has-file");
  $("#translate-btn").disabled = false;
}

const dz = $("#dropzone");
$("#file-input").addEventListener("change", (e) => setFile(e.target.files[0]));
["dragenter", "dragover"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
document.addEventListener("paste", (e) => {
  const item = [...e.clipboardData.items].find((i) => i.type.startsWith("image/"));
  if (item) setFile(item.getAsFile());
});

$("#translate-btn").addEventListener("click", async () => {
  const form = new FormData();
  form.append("file", file);
  form.append("source_lang", $("#source-lang").value);
  form.append("target_lang", $("#target-lang").value);
  form.append("style", $("#style").value);
  form.append("glossary", $("#glossary").value);
  $("#translate-btn").disabled = true;
  $("#result-panel").hidden = true;
  $("#error").hidden = true;
  $("#progress-panel").hidden = false;
  renderStages("queued");

  const res = await fetch("/api/jobs", { method: "POST", body: form });
  if (!res.ok) return showError((await res.json()).detail || res.statusText);
  job = await res.json();
  poll();
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

async function poll() {
  job = await (await fetch(`/api/jobs/${job.id}`)).json();
  renderStages(job.stage);
  if (job.status === "error") return showError(job.error);
  if (job.status !== "done") return setTimeout(poll, 800);
  $("#translate-btn").disabled = false;
  showResult();
}

function bust(url) { return `${url}?t=${Date.now()}`; }

function showResult() {
  $("#result-panel").hidden = false;
  $("#img-original").src = job.original_url;
  $("#img-translated").src = bust(job.translated_url);
  $("#download").href = job.translated_url;
  $("#download").download = job.filename.replace(/\.\w+$/, "") + "_translated.png";

  const list = $("#bubbles");
  list.innerHTML = "";
  if (!job.regions.length) list.innerHTML = '<p class="note">No text found on this page.</p>';
  job.regions.forEach((r) => {
    const row = document.createElement("div");
    row.className = "bubble";
    row.innerHTML = `<div class="num">${r.id + 1}</div><div class="src"></div><textarea rows="2"></textarea>`;
    row.querySelector(".src").textContent = r.source_text;
    const ta = row.querySelector("textarea");
    ta.value = r.translated_text;
    ta.dataset.id = r.id;
    list.appendChild(row);
  });
  $("#result-panel").scrollIntoView({ behavior: "smooth" });
}

$("#rerender-btn").addEventListener("click", async () => {
  const translations = [...document.querySelectorAll("#bubbles textarea")]
    .map((ta) => ({ id: Number(ta.dataset.id), text: ta.value }));
  $("#rerender-btn").disabled = true;
  const res = await fetch(`/api/jobs/${job.id}/translations`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ translations }),
  });
  $("#rerender-btn").disabled = false;
  if (!res.ok) return alertInline("Re-render failed");
  job = await res.json();
  $("#img-translated").src = bust(job.translated_url);
});

function alertInline(msg) {
  $("#error").hidden = false;
  $("#progress-panel").hidden = false;
  $("#error").textContent = msg;
}

document.querySelectorAll(".seg button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("active", x === b));
    $("#viewer").className = b.dataset.view;
  }));

loadConfig();
