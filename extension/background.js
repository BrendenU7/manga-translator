// Background worker: talks to your Manga Translator server.
//
// The content script (on the manga site) sends us the images to translate. We fetch each
// image, upload them as one batch (so pages translate in order with context), poll until
// each page is done, and send the translated images back to the page as they finish.
// Runs as a service worker in Chrome and as a background script in Firefox.

const DEFAULTS = { server: "http://localhost:8000", apiKey: "", target: "en", fontStyle: "manga",
                   style: "standard", quality: "economy", enabledSites: [] };

async function settings() {
  return { ...DEFAULTS, ...(await chrome.storage.local.get(Object.keys(DEFAULTS))) };
}

async function api(path, options = {}) {
  const { server, apiKey } = await settings();
  if (!apiKey) throw new Error("Add your extension key in the Manga Translator popup first.");
  let res;
  try {
    res = await fetch(server.replace(/\/$/, "") + path, {
      ...options,
      headers: { ...(options.headers || {}), Authorization: `Bearer ${apiKey}` },
    });
  } catch (e) {
    throw new Error(`Can't reach ${server}. Is the app running?`);
  }
  if (res.status === 401) throw new Error("Your extension key is invalid or was revoked. Create a new one on the Dashboard.");
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Server error ${res.status}`);
  }
  return res;
}

// ------------------------------------------------------------------ fetching images

let ruleId = 1000;

// Some sites block image requests that don't come from their own pages ("hotlink
// protection"). If a plain fetch fails, retry once with the page as the Referer.
async function fetchImage(src, pageUrl) {
  let res = await fetch(src, { credentials: "include" }).catch(() => null);
  if (res && res.ok) return res.blob();

  const id = ++ruleId;
  try {
    await chrome.declarativeNetRequest.updateSessionRules({
      addRules: [{
        id, priority: 1,
        action: { type: "modifyHeaders",
                  requestHeaders: [{ header: "Referer", operation: "set", value: pageUrl }] },
        condition: { urlFilter: src, resourceTypes: ["xmlhttprequest", "other"] },
      }],
    });
    res = await fetch(src, { credentials: "include" });
    if (res.ok) return res.blob();
    throw new Error(`The site refused the image (${res.status}).`);
  } finally {
    await chrome.declarativeNetRequest.updateSessionRules({ removeRuleIds: [id] }).catch(() => {});
  }
}

function dataUrlToBlob(dataUrl) {
  const [head, b64] = dataUrl.split(",");
  const mime = head.match(/data:(.*?);/)[1];
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return new Blob([bytes], { type: mime });
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

const EXT = { "image/jpeg": "jpg", "image/png": "png", "image/webp": "webp", "image/gif": "gif",
              "image/avif": "avif", "image/bmp": "bmp" };

// ------------------------------------------------------------------ translating

async function translateImages(tabId, requestId, images, pageUrl, title) {
  const send = (msg) => chrome.tabs.sendMessage(tabId, { requestId, ...msg }).catch(() => {});
  try {
    const s = await settings();
    send({ type: "status", text: "Fetching images…" });
    const form = new FormData();
    for (let i = 0; i < images.length; i++) {
      const img = images[i];
      const blob = img.dataUrl ? dataUrlToBlob(img.dataUrl) : await fetchImage(img.src, pageUrl);
      const ext = EXT[blob.type] || "png";
      form.append("files", blob, `page${String(i + 1).padStart(3, "0")}.${ext}`);
    }
    form.append("source_lang", "auto");
    form.append("target_lang", s.target);
    form.append("style", s.style);
    form.append("font_style", s.fontStyle);
    form.append("quality", s.quality);
    form.append("name", title || new URL(pageUrl).hostname);  // shown on the Dashboard

    send({ type: "status", text: "Uploading…" });
    const batch = await (await api("/api/batches", { method: "POST", body: form })).json();

    // Poll; send back each page as soon as it's done
    const delivered = new Set();
    while (true) {
      const b = await (await api(`/api/batches/${batch.id}`)).json();
      for (let i = 0; i < b.pages.length; i++) {
        const page = b.pages[i];
        if (delivered.has(i)) continue;
        if (page.status === "done") {
          const blob = await (await api(`/api/jobs/${page.id}/image/translated`)).blob();
          send({ type: "translated", index: i, dataUrl: await blobToDataUrl(blob) });
          delivered.add(i);
        } else if (page.status === "error") {
          send({ type: "error", index: i, text: "This image failed to translate." });
          delivered.add(i);
        } else {
          const stage = page.stage === "waiting" ? "in line" : page.stage;
          send({ type: "status", index: i, text: `Translating… (${stage})` });
        }
      }
      if (delivered.size === b.pages.length) break;
      await new Promise((r) => setTimeout(r, 1500));
    }
  } catch (e) {
    send({ type: "error", text: e.message });
  }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "translate-images" && sender.tab) {
    translateImages(sender.tab.id, msg.requestId, msg.images, msg.pageUrl, msg.title);
    sendResponse({ ok: true });
  }
});

// ------------------------------------------------------------------ right-click menu

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: "mt-translate", title: "Translate this image", contexts: ["image"] });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "mt-translate" && tab?.id != null) {
    chrome.tabs.sendMessage(tab.id, { type: "translate-src", src: info.srcUrl }).catch(() => {});
  }
});
