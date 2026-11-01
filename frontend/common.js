// Shared by every page: dark mode toggle, login check, and the header's user menu.
const $ = (sel) => document.querySelector(sel);

// ------------------------------------------------------------------ dark mode
// (The saved theme is applied by a tiny inline script in each page's <head>, before
// anything draws, so there's no flash of the wrong theme.)

function renderThemeButton() {
  const btn = $("#theme-toggle");
  if (!btn) return;
  const dark = document.documentElement.dataset.theme === "dark";
  btn.textContent = dark ? "☀" : "☾";
  btn.title = dark ? "Switch to light mode" : "Switch to dark mode";
}

$("#theme-toggle")?.addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) { /* storage blocked: still works this visit */ }
  renderThemeButton();
});
renderThemeButton();

// ------------------------------------------------------------------ login

// Resolves to {username, retention_days}. Sends you to the login page if you aren't logged in.
const currentUser = (async () => {
  if (document.body.dataset.public === "true") return null;
  const res = await fetch("/api/auth/me");
  if (res.status === 401) {
    const next = encodeURIComponent(location.pathname + location.search);
    location.replace(`/login.html?next=${next}`);
    return new Promise(() => {});  // never resolves: we're leaving the page
  }
  const me = await res.json();
  const name = $("#user-name");
  if (name) name.textContent = me.username;
  return me;
})();

$("#logout")?.addEventListener("click", async () => {
  await fetch("/api/auth/logout", { method: "POST" });
  location.href = "/login.html";
});

// Highlight the current page in the nav
document.querySelectorAll("nav a").forEach((a) => {
  const here = location.pathname === "/" ? "/index.html" : location.pathname;
  a.classList.toggle("active", a.getAttribute("href") === here);
});

// ------------------------------------------------------------------ helpers

function fmtDate(ts) {
  return new Date(ts * 1000).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function fmtExpires(ts) {
  const days = (ts * 1000 - Date.now()) / 86400000;
  if (days > 1.5) return `Deletes in ${Math.round(days)} days`;
  if (days > 1 / 24) return `Deletes in ${Math.round(days * 24)} hours`;
  return "Deleting soon";
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// Money: small amounts are shown precisely ("$0.0006"), larger ones normally ("$1.24")
function fmtUsd(x) {
  if (!x) return "$0.00";
  if (x < 0.01) return `$${x.toFixed(4)}`;
  return `$${x.toFixed(2)}`;
}
