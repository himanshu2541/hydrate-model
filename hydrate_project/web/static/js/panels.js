// panels.js -- config panels: gas, liquid, temperature, eos, exp-data, cache.
import { api } from "./api.js";

const $ = (root, sel) => root.querySelector(sel);
const $$ = (root, sel) => Array.from(root.querySelectorAll(sel));

// ── Gas panel ────────────────────────────────────────────────────────────

export function initGasPanel(panel, availableGases) {
  const rowsEl = $(panel, '[data-testid="gas-rows"]');
  const sumEl = $(panel, '[data-testid="gas-sum"]');
  const addBtn = $(panel, '[data-testid="gas-add-row"]');
  let rows = [];

  function updateSum() {
    let total = 0;
    for (const r of rows) {
      const v = parseFloat(r.fracInput.value);
      if (!Number.isNaN(v)) total += v;
    }
    const ok = Math.abs(total - 1.0) < 1e-6;
    sumEl.textContent = `Sum: ${total.toFixed(4)}  ${ok ? "✓" : "⚠ must equal 1.000"}`;
    sumEl.className = "sum-label " + (ok ? "ok" : "bad");
  }

  function addRow(gas = availableGases[0], frac = 0.5) {
    const row = document.createElement("div");
    row.className = "gas-row";
    row.dataset.testid = "gas-row";

    const select = document.createElement("select");
    select.dataset.testid = "gas-row-select";
    for (const g of availableGases) {
      const opt = document.createElement("option");
      opt.value = g; opt.textContent = g;
      if (g === gas) opt.selected = true;
      select.appendChild(opt);
    }

    const fracInput = document.createElement("input");
    fracInput.type = "text";
    fracInput.dataset.testid = "gas-row-frac";
    fracInput.value = String(Math.round(frac * 10000) / 10000);
    fracInput.style.width = "6em";

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.textContent = "✕";
    removeBtn.dataset.testid = "gas-row-remove";
    removeBtn.addEventListener("click", () => {
      if (rows.length <= 1) { alert("At least one gas component is required."); return; }
      row.remove();
      rows = rows.filter((r) => r !== entry);
      updateSum();
    });

    row.append("Gas: ", select, " Mole fraction: ", fracInput, removeBtn);
    rowsEl.appendChild(row);

    const entry = { gas: select, fracInput, row };
    select.addEventListener("change", updateSum);
    fracInput.addEventListener("input", updateSum);
    rows.push(entry);
    updateSum();
  }

  addBtn.addEventListener("click", () => {
    const used = new Set(rows.map((r) => r.gas.value));
    const avail = availableGases.filter((g) => !used.has(g));
    if (!avail.length) { alert("All available gases are already added."); return; }
    addRow(avail[0], 0.0);
  });

  function getComposition() {
    const comp = {};
    for (const r of rows) {
      const f = parseFloat(r.fracInput.value);
      if (Number.isNaN(f)) throw new Error(`Invalid mole fraction for ${r.gas.value}.`);
      comp[r.gas.value] = f;
    }
    return comp;
  }

  function setComposition(comp) {
    for (const r of [...rows]) r.row.remove();
    rows = [];
    const entries = Object.entries(comp || {});
    if (!entries.length) { addRow(availableGases[0], 1.0); return; }
    for (const [gas, frac] of entries) addRow(gas, frac);
  }

  return { addRow, getComposition, setComposition };
}

// ── Liquid panel ─────────────────────────────────────────────────────────

export function initLiquidPanel(panel, promoters) {
  const select = $(panel, '[data-testid="liquid-promoter"]');
  const fracRow = $(panel, '[data-testid="liquid-frac-row"]');
  const fracInput = $(panel, '[data-testid="liquid-frac"]');
  const hintEl = $(panel, '[data-testid="liquid-hint"]');
  const display = $(panel, '[data-testid="liquid-display"]');

  for (const [key, info] of Object.entries(promoters)) {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = `${info.display_name} (${key})`;
    select.appendChild(opt);
  }

  function refreshDisplay() {
    if (select.value === "none") {
      display.textContent = "H₂O: 1.0000";
      return;
    }
    let xp = parseFloat(fracInput.value);
    if (Number.isNaN(xp)) xp = 0.0;
    xp = Math.max(0.0, Math.min(xp, 0.9999));
    const xw = Math.round((1.0 - xp) * 1e6) / 1e6;
    const name = promoters[select.value].display_name;
    display.textContent = `H₂O: ${xw.toFixed(4)}    ${name}: ${xp.toFixed(4)}`;
  }

  select.addEventListener("change", () => {
    if (select.value === "none") {
      fracRow.hidden = true;
    } else {
      const info = promoters[select.value];
      fracInput.value = String(Math.round(info.stoichiometric_x * 10000) / 10000);
      hintEl.textContent = info.hint;
      fracRow.hidden = false;
    }
    refreshDisplay();
  });
  fracInput.addEventListener("input", refreshDisplay);

  function getComposition() {
    if (select.value === "none") return { H2O: 1.0 };
    const xp = parseFloat(fracInput.value);
    if (Number.isNaN(xp)) throw new Error("Promoter mole fraction must be a number.");
    if (!(xp > 0.0 && xp < 1.0)) throw new Error("Promoter mole fraction must be between 0 and 1.");
    const comp = {};
    comp["H2O"] = Math.round((1.0 - xp) * 1e8) / 1e8;
    comp[select.value] = Math.round(xp * 1e8) / 1e8;
    return comp;
  }

  function setComposition(comp) {
    const promoterKey = Object.keys(comp || {}).find((k) => k !== "H2O");
    if (!promoterKey) {
      select.value = "none";
      fracRow.hidden = true;
      refreshDisplay();
      return;
    }
    select.value = promoterKey;
    fracInput.value = String(comp[promoterKey]);
    const info = promoters[promoterKey];
    if (info) hintEl.textContent = info.hint;
    fracRow.hidden = false;
    refreshDisplay();
  }

  return { getComposition, setComposition };
}

// ── Temperature panel ────────────────────────────────────────────────────

export function initTemperaturePanel(panel) {
  const minEl = $(panel, '[data-testid="temp-min"]');
  const maxEl = $(panel, '[data-testid="temp-max"]');
  const stepEl = $(panel, '[data-testid="temp-step"]');
  const preview = $(panel, '[data-testid="temp-preview"]');

  function updatePreview() {
    const tmin = parseFloat(minEl.value), tmax = parseFloat(maxEl.value), tstep = parseFloat(stepEl.value);
    if ([tmin, tmax, tstep].some(Number.isNaN) || tstep <= 0) { preview.textContent = ""; return; }
    const n = Math.max(0, Math.round((tmax - tmin) / tstep) + 1);
    preview.textContent = `→ ${n} temperature points`;
  }
  [minEl, maxEl, stepEl].forEach((el) => el.addEventListener("input", updatePreview));
  updatePreview();

  function getTRange() {
    const tmin = parseFloat(minEl.value), tmax = parseFloat(maxEl.value), tstep = parseFloat(stepEl.value);
    if ([tmin, tmax, tstep].some(Number.isNaN)) throw new Error("Temperature inputs must be numeric.");
    if (tmin >= tmax) throw new Error("T_min must be less than T_max.");
    if (tstep <= 0) throw new Error("T_step must be positive.");
    return { T_min: tmin, T_max: tmax, T_step: tstep };
  }

  function setTRange({ T_min, T_max, T_step }) {
    if (T_min != null) minEl.value = String(T_min);
    if (T_max != null) maxEl.value = String(T_max);
    if (T_step != null) stepEl.value = String(T_step);
    updatePreview();
  }

  return { getTRange, setTRange };
}

// ── EOS panel ────────────────────────────────────────────────────────────

export function initEosPanel(panel, eosNames) {
  const container = $(panel, '[data-testid="eos-checkboxes"]');
  const boxes = [];
  for (const name of eosNames) {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox"; cb.checked = true; cb.value = name;
    cb.dataset.testid = "eos-checkbox";
    label.append(cb, " " + name);
    container.appendChild(label);
    boxes.push(cb);
  }

  function getSelected() {
    const selected = boxes.filter((b) => b.checked).map((b) => b.value);
    if (!selected.length) throw new Error("Select at least one EOS model.");
    return selected;
  }

  function setSelected(names) {
    if (!names || !names.length) return;
    for (const b of boxes) b.checked = names.includes(b.value);
  }

  return { getSelected, setSelected };
}

// ── Experimental data panel ──────────────────────────────────────────────

export function initExpDataPanel(panel, literature) {
  const presetRow = $(panel, '[data-testid="expdata-preset-row"]');
  const presetSelect = $(panel, '[data-testid="expdata-preset"]');
  const customRow = $(panel, '[data-testid="expdata-custom-row"]');
  const customText = $(panel, '[data-testid="expdata-custom"]');

  for (const name of Object.keys(literature)) {
    const opt = document.createElement("option");
    opt.value = name; opt.textContent = name;
    presetSelect.appendChild(opt);
  }

  $$(panel, 'input[name="expdata-mode"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      presetRow.hidden = radio.value !== "preset" || !radio.checked;
      customRow.hidden = radio.value !== "custom" || !radio.checked;
    });
  });

  function currentMode() {
    return $(panel, 'input[name="expdata-mode"]:checked').value;
  }

  async function getExpData() {
    const mode = currentMode();
    if (mode === "preset") {
      const preset = literature[presetSelect.value];
      return { "T (K)": preset.T, "P_eq (MPa)": preset.P };
    }
    if (mode === "custom") {
      const parsed = await api.parseCustomExpData(customText.value);
      if (parsed.T == null) return null;
      return { "T (K)": parsed.T, "P_eq (MPa)": parsed.P };
    }
    return null;
  }

  function getState() {
    return {
      expdata_mode: currentMode(),
      expdata_preset: presetSelect.value || null,
      expdata_custom_text: customText.value,
    };
  }

  function setState({ expdata_mode, expdata_preset, expdata_custom_text }) {
    const mode = expdata_mode || "none";
    const radio = $(panel, `input[name="expdata-mode"][value="${mode}"]`);
    if (radio) {
      radio.checked = true;
      presetRow.hidden = mode !== "preset";
      customRow.hidden = mode !== "custom";
    }
    if (expdata_preset && [...presetSelect.options].some((o) => o.value === expdata_preset)) {
      presetSelect.value = expdata_preset;
    }
    if (expdata_custom_text != null) customText.value = expdata_custom_text;
  }

  return { getExpData, getState, setState };
}

// ── Cache panel ───────────────────────────────────────────────────────────

export function initCachePanel(panel) {
  const infoEl = $(panel, '[data-testid="cache-info"]');
  const clearBtn = $(panel, '[data-testid="cache-clear"]');

  async function refresh() {
    try {
      const info = await api.cacheInfo();
      infoEl.textContent =
        `${info.total_entries} / ${info.max_entries} entries  •  ` +
        `session: ${info.hits_session} hits, ${info.misses_session} misses`;
    } catch (exc) {
      infoEl.textContent = `Cache unavailable: ${exc.message}`;
    }
  }

  clearBtn.addEventListener("click", async () => {
    if (!confirm("Delete all cached results?\nThis cannot be undone.")) return;
    await api.cacheClear();
    await refresh();
  });

  refresh();
  return { refresh };
}
