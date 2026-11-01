// Runs on every page. On sites you've enabled, a "Translate" button appears when you
// hover a manga-sized image. "Translate all" (popup) and the right-click menu work on
// any site. Translated images replace the originals in place; hover one to switch back.
(() => {
  const MIN_SIZE = 300;  // ignore icons, avatars, ads: manga pages are big
  let enabledHere = false;
  const pending = new Map();  // requestId -> [img, ...]

  chrome.storage.local.get({ enabledSites: [] }).then(({ enabledSites }) => {
    enabledHere = enabledSites.includes(location.hostname);
  });
  chrome.storage.onChanged.addListener((changes) => {
    if (changes.enabledSites) enabledHere = (changes.enabledSites.newValue || []).includes(location.hostname);
    if (!enabledHere) hoverBtn.hidden = true;
  });

  const isMangaImage = (img) =>
    img.naturalWidth >= MIN_SIZE && img.naturalHeight >= MIN_SIZE && img.offsetWidth > 100;

  // All manga-sized images, top to bottom (reading order for chapters and webtoons)
  function findImages() {
    return [...document.images]
      .filter((img) => isMangaImage(img) && !img.dataset.mtState)
      .sort((a, b) => a.getBoundingClientRect().top - b.getBoundingClientRect().top);
  }

  // ---------------------------------------------------------------- badges

  function badge(img) {
    let b = img._mtBadge;
    if (!b) {
      b = document.createElement("div");
      b.className = "mt-badge";
      document.body.appendChild(b);
      img._mtBadge = b;
    }
    const r = img.getBoundingClientRect();
    b.style.top = `${window.scrollY + r.top + 8}px`;
    b.style.left = `${window.scrollX + r.left + 8}px`;
    return b;
  }

  function setBadge(img, text, kind = "busy") {
    const b = badge(img);
    b.textContent = text;
    b.dataset.kind = kind;
    b.hidden = false;
  }

  // ---------------------------------------------------------------- translate

  async function toPayload(img) {
    const src = img.currentSrc || img.src;
    // blob: and data: URLs only exist inside this page, so read them here
    if (src.startsWith("blob:") || src.startsWith("data:")) {
      const blob = await (await fetch(src)).blob();
      const dataUrl = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.readAsDataURL(blob);
      });
      return { dataUrl };
    }
    return { src };
  }

  async function translate(images) {
    images = images.filter((img) => !img.dataset.mtState);
    if (!images.length) return 0;
    const requestId = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    pending.set(requestId, images);
    images.forEach((img) => { img.dataset.mtState = "busy"; setBadge(img, "Translating…"); });
    const payload = await Promise.all(images.map(toPayload));
    chrome.runtime.sendMessage({ type: "translate-images", requestId, images: payload,
                                 pageUrl: location.href, title: document.title });
    return images.length;
  }

  function showTranslated(img, dataUrl) {
    img.dataset.mtOriginal = img.currentSrc || img.src;
    img.dataset.mtTranslated = dataUrl;
    img.removeAttribute("srcset");  // otherwise the browser may switch back to the original
    img.closest("picture")?.querySelectorAll("source").forEach((s) => s.remove());
    img.src = dataUrl;
    img.dataset.mtState = "done";
    if (img._mtBadge) img._mtBadge.hidden = true;
    if (hovered === img && !hoverBtn.hidden) hoverBtn.textContent = "Show original";
  }

  function toggleOriginal(img) {
    const showingOriginal = img.src === img.dataset.mtOriginal;
    img.src = showingOriginal ? img.dataset.mtTranslated : img.dataset.mtOriginal;
    hoverBtn.textContent = showingOriginal ? "Show original" : "Show translation";
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg.type === "translate-all") {
      translate(findImages()).then((n) => sendResponse({ count: n }));
      return true;  // async response
    }
    if (msg.type === "translate-src") {
      const img = [...document.images].find((i) => (i.currentSrc || i.src) === msg.src);
      if (img) translate([img]);
      return;
    }
    const images = pending.get(msg.requestId);
    if (!images) return;
    if (msg.type === "translated") {
      showTranslated(images[msg.index], msg.dataUrl);
    } else if (msg.type === "status") {
      (msg.index == null ? images : [images[msg.index]])
        .filter((img) => img.dataset.mtState === "busy")
        .forEach((img) => setBadge(img, msg.text));
    } else if (msg.type === "error") {
      (msg.index == null ? images : [images[msg.index]])
        .filter((img) => img.dataset.mtState === "busy")
        .forEach((img) => { setBadge(img, msg.text, "error"); delete img.dataset.mtState; });
    }
  });

  // ---------------------------------------------------------------- hover button

  const hoverBtn = document.createElement("button");
  hoverBtn.className = "mt-hover-btn";
  hoverBtn.hidden = true;
  document.documentElement.appendChild(hoverBtn);
  let hovered = null;

  document.addEventListener("mouseover", (e) => {
    if (!enabledHere || e.target === hoverBtn) return;
    const img = e.target.closest?.("img");
    if (!img || !isMangaImage(img) || img.dataset.mtState === "busy") return;
    hovered = img;
    const r = img.getBoundingClientRect();
    hoverBtn.style.top = `${window.scrollY + r.top + 8}px`;
    hoverBtn.style.left = `${window.scrollX + r.right - 8}px`;
    hoverBtn.textContent = img.dataset.mtState === "done"
      ? (img.src === img.dataset.mtOriginal ? "Show translation" : "Show original")
      : "訳 Translate";
    hoverBtn.hidden = false;
  }, true);

  hoverBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!hovered) return;
    if (hovered.dataset.mtState === "done") toggleOriginal(hovered);
    else { translate([hovered]); hoverBtn.hidden = true; }
  });
})();
