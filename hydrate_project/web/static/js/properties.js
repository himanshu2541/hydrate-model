// properties.js -- port of ui/properties_window.py's table tab.
// (The Tkinter app's separate "Data Extractor" tab -- picking a column
// subset before export -- is folded into this single view: Export CSV
// exports exactly the columns/rows currently shown, filtered by the same
// T range. Full parity on everything else.)

const $ = (root, sel) => root.querySelector(sel);

const PRIORITY_COLS = [
  "T (K)", "P_eq (MPa)", "Optimum_Structure", "Z",
  "dG_diss (kJ/mol)", "dS_diss (kJ/mol.K)", "dH_diss (kJ/mol)",
  "a_w", "gamma_w", "Delta_Mu_w", "Delta_Mu_H",
];

function orderedColumns(rows) {
  if (!rows.length) return [];
  const all = Object.keys(rows[0]);
  const rest = all.filter((c) => !PRIORITY_COLS.includes(c));
  return PRIORITY_COLS.filter((c) => all.includes(c)).concat(rest);
}

function fmt(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (Number.isNaN(v)) return "—";
    return Number(v.toPrecision(5)).toString();
  }
  return String(v);
}

function median(values) {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function toCsv(rows, cols) {
  const lines = [cols.join(",")];
  for (const r of rows) {
    lines.push(cols.map((c) => {
      const v = r[c];
      if (v === null || v === undefined) return "";
      if (typeof v === "number") return String(Number(v.toPrecision(6)));
      const s = String(v);
      return s.includes(",") ? `"${s}"` : s;
    }).join(","));
  }
  return lines.join("\n");
}

function downloadText(filename, text) {
  const blob = new Blob([text], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

export function initPropertiesView(view) {
  const eosSelect = $(view, '[data-testid="props-eos"]');
  const tminEl = $(view, '[data-testid="props-tmin"]');
  const tmaxEl = $(view, '[data-testid="props-tmax"]');
  const applyBtn = $(view, '[data-testid="props-apply"]');
  const clearBtn = $(view, '[data-testid="props-clear"]');
  const summaryEl = $(view, '[data-testid="props-summary"]');
  const table = $(view, '[data-testid="props-table"]');
  const thead = table.querySelector("thead tr");
  const tbody = table.querySelector("tbody");
  const rowLabel = $(view, '[data-testid="props-rowlabel"]');
  const copyBtn = $(view, '[data-testid="props-copy"]');
  const exportBtn = $(view, '[data-testid="props-export"]');

  let resultsDict = {};
  let sortState = { col: null, dir: 1 };

  function filteredRows() {
    const rows = resultsDict[eosSelect.value] || [];
    const tmin = parseFloat(tminEl.value), tmax = parseFloat(tmaxEl.value);
    return rows.filter((r) => {
      if (!Number.isNaN(tmin) && r["T (K)"] < tmin) return false;
      if (!Number.isNaN(tmax) && r["T (K)"] > tmax) return false;
      return true;
    });
  }

  function card(label, value) {
    const div = document.createElement("div");
    div.className = "summary-card";
    div.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>`;
    return div;
  }

  function updateSummary(rows) {
    summaryEl.innerHTML = "";
    if (!rows.length) { summaryEl.appendChild(card("No data", "—")); return; }
    const T = rows.map((r) => r["T (K)"]).filter((v) => typeof v === "number");
    const P = rows.map((r) => r["P_eq (MPa)"]).filter((v) => typeof v === "number");
    const Z = rows.map((r) => r["Z"]).filter((v) => typeof v === "number");
    const nOk = rows.filter((r) => typeof r["P_eq (MPa)"] === "number").length;

    if (T.length) {
      const lo = Math.min(...T), hi = Math.max(...T);
      summaryEl.appendChild(card("T range", lo === hi ? `${lo.toFixed(2)} K` : `${lo.toFixed(2)} – ${hi.toFixed(2)} K`));
    }
    if (P.length) summaryEl.appendChild(card("P range", `${Math.min(...P).toFixed(3)} – ${Math.max(...P).toFixed(3)} MPa`));
    if (Z.length) summaryEl.appendChild(card("Z̄ (avg)", (Z.reduce((a, b) => a + b, 0) / Z.length).toFixed(4)));
    summaryEl.appendChild(card("Converged", `${nOk}/${rows.length}`));

    for (const [key, label] of [["dH_diss (kJ/mol)", "ΔH_diss"], ["dS_diss (kJ/mol.K)", "ΔS_diss"], ["dG_diss (kJ/mol)", "ΔG_diss"]]) {
      const vals = rows.map((r) => r[key]).filter((v) => typeof v === "number");
      if (vals.length) {
        const m = median(vals);
        const unit = key.includes("kJ/mol.K") ? "kJ/mol.K" : "kJ/mol";
        summaryEl.appendChild(card(label, `${m.toFixed(key.includes("dS") ? 3 : 1)} ${unit}`));
      }
    }
  }

  function renderTable() {
    let rows = filteredRows();
    const cols = orderedColumns(rows);

    if (sortState.col) {
      rows = [...rows].sort((a, b) => {
        const av = a[sortState.col], bv = b[sortState.col];
        if (av == null) return 1;
        if (bv == null) return -1;
        if (av < bv) return -1 * sortState.dir;
        if (av > bv) return 1 * sortState.dir;
        return 0;
      });
    }

    thead.innerHTML = "";
    for (const c of cols) {
      const th = document.createElement("th");
      th.textContent = c + (sortState.col === c ? (sortState.dir === 1 ? " ▲" : " ▼") : "");
      th.addEventListener("click", () => {
        sortState = { col: c, dir: sortState.col === c ? -sortState.dir : 1 };
        renderTable();
      });
      thead.appendChild(th);
    }

    tbody.innerHTML = "";
    let nOk = 0;
    for (const r of rows) {
      const tr = document.createElement("tr");
      let hasNan = false;
      for (const c of cols) {
        const td = document.createElement("td");
        const v = r[c];
        if (v === null || v === undefined) hasNan = true;
        td.textContent = fmt(v);
        tr.appendChild(td);
      }
      if (hasNan) tr.classList.add("has-nan"); else nOk += 1;
      tbody.appendChild(tr);
    }

    rowLabel.textContent = `${rows.length} rows  •  ${nOk} converged  •  ${rows.length - nOk} failed`;
    updateSummary(rows);
    return { rows, cols };
  }

  applyBtn.addEventListener("click", renderTable);
  clearBtn.addEventListener("click", () => { tminEl.value = ""; tmaxEl.value = ""; renderTable(); });
  eosSelect.addEventListener("change", renderTable);

  exportBtn.addEventListener("click", () => {
    const { rows, cols } = renderTable();
    if (!rows.length) { alert("Nothing to export."); return; }
    downloadText("hydrate_properties.csv", toCsv(rows, cols));
  });
  copyBtn.addEventListener("click", async () => {
    const { rows, cols } = renderTable();
    await navigator.clipboard.writeText(toCsv(rows, cols));
    rowLabel.textContent = "✓  Copied to clipboard";
  });

  function setResults(newResultsDict) {
    resultsDict = newResultsDict;
    const names = Object.keys(resultsDict);
    eosSelect.innerHTML = "";
    for (const n of names) {
      const o = document.createElement("option"); o.value = n; o.textContent = n;
      eosSelect.appendChild(o);
    }
    sortState = { col: null, dir: 1 };
    renderTable();
  }

  return { setResults };
}
