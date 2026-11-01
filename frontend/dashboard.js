// Dashboard: lists your uploads from the last RETENTION_DAYS days.
let langNames = {};

async function loadDashboard() {
  await currentUser;
  let cfg, data;
  try {
    [cfg, data] = await Promise.all([
      fetch("/api/config").then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }),
      fetch("/api/batches").then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); }),
    ]);
  } catch (e) {
    // Never fail silently: say so, and point at the server terminal for details
    $("#retention-note").textContent =
      `Couldn't load your translations (server error ${e.message}). Check the terminal running the app for details, then refresh.`;
    $("#retention-note").classList.add("error");
    return;
  }
  langNames = { ...cfg.source_langs, ...cfg.target_langs };
  $("#retention-note").textContent =
    `Uploads and results are private to your account and deleted automatically after ${data.retention_days} days.`;
  render(data.batches);
  const total = data.batches.reduce((sum, b) => sum + (b.cost_usd || 0), 0);
  $("#retention-note").textContent += ` Estimated Claude cost of everything here: ${fmtUsd(total)}.`;
  // Keep refreshing while something is still translating
  if (data.batches.some((b) => b.status !== "done")) setTimeout(loadDashboard, 3000);
}

function render(batches) {
  const grid = $("#dash-grid");
  grid.innerHTML = "";
  $("#dash-empty").hidden = batches.length > 0;
  for (const b of batches) {
    const card = document.createElement("article");
    card.className = "dash-card";
    const progress = b.status === "done"
      ? `${b.done} page${b.done === 1 ? "" : "s"}${b.failed ? ` · ${b.failed} failed` : ""}`
      : `Translating… ${b.done} / ${b.total}`;
    const from = b.source_lang === "auto" ? "Auto" : (langNames[b.source_lang] || b.source_lang).split(" (")[0];
    card.innerHTML = `
      <a class="dash-thumb" href="/index.html?batch=${b.id}">
        ${b.thumb_url ? `<img src="${b.thumb_url}" alt="" loading="lazy">` : ""}
      </a>
      <div class="dash-body">
        <b class="dash-name"></b>
        <span class="note">${progress}</span>
        <span class="note">${escapeHtml(from)} → ${escapeHtml(langNames[b.target_lang] || b.target_lang)}</span>
        <span class="note">${fmtDate(b.created_at)}</span>
        <span class="note">${b.quality === "best" ? "Best quality" : "Economy"} · est. ${fmtUsd(b.cost_usd)}</span>
        <span class="expires">${fmtExpires(b.expires_at)}</span>
        <div class="dash-actions">
          <a class="button" href="/index.html?batch=${b.id}">Open</a>
          <a class="button secondary" href="${b.download_url}" ${b.done ? "" : 'aria-disabled="true"'}>ZIP</a>
          <button class="link-btn danger" type="button">Delete</button>
        </div>
      </div>`;
    card.querySelector(".dash-name").textContent = b.name;
    card.querySelector(".dash-name").title = b.name;
    card.querySelector(".danger").addEventListener("click", async () => {
      if (!confirmDelete(card)) return;
      await fetch(`/api/batches/${b.id}`, { method: "DELETE" });
      // ------------------------------------------------------------------ extension key

async function renderKeyStatus() {
  const me = await (await fetch("/api/auth/me")).json();
  const has = !!me.api_key_created_at;
  $("#key-status").textContent = has
    ? `You have an active key, created ${fmtDate(me.api_key_created_at)}. Creating a new one replaces it.`
    : "No key yet.";
  $("#key-new").textContent = has ? "Replace key" : "Create extension key";
  $("#key-revoke").hidden = !has;
}

$("#key-new").addEventListener("click", async () => {
  const res = await fetch("/api/auth/api-key", { method: "POST" });
  if (!res.ok) return;
  const { key } = await res.json();
  $("#key-value").value = key;
  $("#key-row").hidden = false;
  $("#key-warning").hidden = false;
  renderKeyStatus();
});

$("#key-copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("#key-value").value); } catch (e) { $("#key-value").select(); }
  $("#key-copy").textContent = "Copied";
  setTimeout(() => { $("#key-copy").textContent = "Copy"; }, 1500);
});

$("#key-revoke").addEventListener("click", async () => {
  await fetch("/api/auth/api-key", { method: "DELETE" });
  $("#key-row").hidden = true;
  $("#key-warning").hidden = true;
  renderKeyStatus();
});

currentUser.then(renderKeyStatus);
loadDashboard();
    });
    grid.appendChild(card);
  }
}

// Two-click delete instead of a browser confirm() popup
function confirmDelete(card) {
  const btn = card.querySelector(".danger");
  if (btn.dataset.armed) return true;
  btn.dataset.armed = "1";
  btn.textContent = "Click again to delete";
  setTimeout(() => { delete btn.dataset.armed; btn.textContent = "Delete"; }, 3000);
  return false;
}

// ------------------------------------------------------------------ extension key

async function renderKeyStatus() {
  const me = await (await fetch("/api/auth/me")).json();
  const has = !!me.api_key_created_at;
  $("#key-status").textContent = has
    ? `You have an active key, created ${fmtDate(me.api_key_created_at)}. Creating a new one replaces it.`
    : "No key yet.";
  $("#key-new").textContent = has ? "Replace key" : "Create extension key";
  $("#key-revoke").hidden = !has;
}

$("#key-new").addEventListener("click", async () => {
  const res = await fetch("/api/auth/api-key", { method: "POST" });
  if (!res.ok) return;
  const { key } = await res.json();
  $("#key-value").value = key;
  $("#key-row").hidden = false;
  $("#key-warning").hidden = false;
  renderKeyStatus();
});

$("#key-copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("#key-value").value); } catch (e) { $("#key-value").select(); }
  $("#key-copy").textContent = "Copied";
  setTimeout(() => { $("#key-copy").textContent = "Copy"; }, 1500);
});

$("#key-revoke").addEventListener("click", async () => {
  await fetch("/api/auth/api-key", { method: "DELETE" });
  $("#key-row").hidden = true;
  $("#key-warning").hidden = true;
  renderKeyStatus();
});

currentUser.then(renderKeyStatus);
loadDashboard();
