// Popup: connect to your app, choose language/font, enable per site, translate all.
const $ = (s) => document.querySelector(s);
const DEFAULTS = { server: "http://localhost:8000", apiKey: "", target: "en", fontStyle: "manga",
                   enabledSites: [] };
let s = {};
let tab = null;
let host = "";

const base = () => s.server.replace(/\/$/, "");

async function api(path) {
  const res = await fetch(base() + path, { headers: { Authorization: `Bearer ${s.apiKey}` } });
  if (!res.ok) throw new Error(res.status === 401 ? "Key invalid or revoked" : `Error ${res.status}`);
  return res.json();
}

function setConn(text, kind) {
  $("#conn").textContent = text;
  $("#conn").className = `muted ${kind || ""}`;
}

function fill(select, options, value) {
  select.innerHTML = Object.entries(options)
    .map(([k, v]) => `<option value="${k}" ${k === value ? "selected" : ""}>${v}</option>`).join("");
}

async function connect() {
  $("#dash-link").href = `${base()}/dashboard.html#extension`;
  $("#open-dash").href = `${base()}/dashboard.html`;
  if (!s.apiKey) {
    setConn("Not connected");
    $("#setup").hidden = false;
    $("#main").hidden = true;
    return;
  }
  try {
    const [me, cfg] = await Promise.all([api("/api/auth/me"), api("/api/config")]);
    setConn(`Connected as ${me.username}`, "ok");
    fill($("#target"), cfg.target_langs, s.target);
    fill($("#font-style"), cfg.font_styles, s.fontStyle);
    $("#setup").hidden = true;
    $("#main").hidden = false;
  } catch (e) {
    setConn(e.message === "Failed to fetch" ? `Can't reach ${s.server}. Is the app running?` : e.message, "bad");
    $("#setup").hidden = false;
    $("#main").hidden = true;
  }
}

async function init() {
  s = { ...DEFAULTS, ...(await chrome.storage.local.get(Object.keys(DEFAULTS))) };
  [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  try { host = new URL(tab.url).hostname; } catch (e) { host = ""; }
  $("#site").textContent = host || "this page";
  $("#site-toggle").checked = s.enabledSites.includes(host);
  $("#site-toggle").disabled = !host;
  $("#server").value = s.server;
  $("#api-key").value = s.apiKey;
  connect();
}

$("#save").addEventListener("click", async () => {
  s.server = $("#server").value.trim() || DEFAULTS.server;
  s.apiKey = $("#api-key").value.trim();
  await chrome.storage.local.set({ server: s.server, apiKey: s.apiKey });
  connect();
});

$("#edit-setup").addEventListener("click", () => { $("#setup").hidden = false; });

$("#target").addEventListener("change", (e) => chrome.storage.local.set({ target: e.target.value }));
$("#font-style").addEventListener("change", (e) => chrome.storage.local.set({ fontStyle: e.target.value }));

$("#site-toggle").addEventListener("change", async (e) => {
  const sites = new Set(s.enabledSites);
  e.target.checked ? sites.add(host) : sites.delete(host);
  s.enabledSites = [...sites];
  await chrome.storage.local.set({ enabledSites: s.enabledSites });
});

$("#translate-all").addEventListener("click", async () => {
  $("#translate-all").disabled = true;
  try {
    const res = await chrome.tabs.sendMessage(tab.id, { type: "translate-all" });
    $("#result").textContent = res.count
      ? `Translating ${res.count} image${res.count > 1 ? "s" : ""}… they'll update on the page as they finish.`
      : "No manga-sized images found (or they're already translated).";
  } catch (e) {
    $("#result").textContent = "Can't run on this page. Try reloading it.";
  }
  $("#translate-all").disabled = false;
});

init();
