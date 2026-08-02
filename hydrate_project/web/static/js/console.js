// console.js -- Console page: session-only action log (COMSOL "Messages"
// analogue) + hosts the Result Cache panel (panels.js's initCachePanel,
// logic unchanged, just relocated here from the old sidebar).
import { initCachePanel } from "./panels.js";

const $ = (root, sel) => root.querySelector(sel);

export function initConsolePage(page) {
  const logEl = $(page, '[data-testid="console-log"]');
  const cachePanel = initCachePanel($(page, '[data-testid="panel-cache"]'));

  function logEvent(level, message) {
    const entry = document.createElement("div");
    entry.className = `console-log-entry level-${level}`;
    entry.dataset.testid = "console-log-entry";
    const ts = new Date().toLocaleTimeString();
    entry.innerHTML = `<span class="ts">${ts}</span><span class="msg">${message}</span>`;
    logEl.appendChild(entry);
  }

  logEvent("info", "Console ready.");

  return { logEvent, cachePanel };
}
