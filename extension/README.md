# Manga Translator browser extension

Translate manga, manhwa and manhua images **in place** on the sites you read. The extension
sends images to your own Manga Translator app (running on your PC) and swaps in the
translated versions. Works in Chrome, Edge, Brave and Firefox (121+).

## Install

1. Start the app (`uvicorn backend.app.main:app` from the project folder) and log in.
2. On the **Dashboard**, under *Browser extension*, click **Create extension key** and copy it.
3. Load the extension:
   - **Chrome / Edge / Brave:** open `chrome://extensions` (or `edge://extensions`), turn on
     **Developer mode**, click **Load unpacked**, and pick this `extension` folder.
   - **Firefox:** open `about:debugging#/runtime/this-firefox`, click
     **Load Temporary Add-on…**, and pick `extension/manifest.json`. (Temporary add-ons are
     removed when Firefox restarts.) Then in `about:addons` → Manga Translator →
     Permissions, allow access to all sites.
4. Click the extension's 訳 icon, paste your key, and click **Save & connect**. It should
   say *Connected as <you>*.

## Use

- **Translate all images on this page**: popup button. The images are sent as one batch,
  top to bottom, so each page gets the previous page's dialogue as context. They update
  on the page as each one finishes.
- **Per-site buttons**: turn on *Translate buttons on <site>* in the popup, then hover any
  manga-sized image and click **訳 Translate**. Hover a translated image to switch
  between the original and the translation.
- **Right-click** any image → *Translate this image* (works on every site).
- Target language and lettering font are set in the popup. Every translation also shows
  up on your Dashboard, so you can edit or download it later.

## How it works

`content.js` finds manga-sized images (300px+) on the page. `background.js` fetches them,
uploads them to `POST /api/batches` with your key (`Authorization: Bearer mtk_…`), polls
until each page is done, and sends the translated image back to the page.

## Limitations

- The app must be running on your PC while you read.
- **Hotlink protection:** if a site refuses the image request, the extension retries
  with the page as the Referer, which fixes most sites.
- **Canvas readers / scrambled pages:** some official readers draw pages on a `<canvas>`
  or as shuffled tiles, so there's no image to grab. Those aren't supported.
- Some sites' terms forbid modifying their pages, and extension stores reject
  extensions aimed at unofficial manga sites, so this is meant to be loaded unpacked for
  personal use.
- Your key lets anyone who has it translate as you. Revoke it on the Dashboard if it leaks.
